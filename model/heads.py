"""
model/heads.py

Policy head and Value head for the chess network.

Policy head:
    Input:  (B, 256, 8, 8) feature map from ResNet tower
    Output: (B, 4672) logits over the AlphaZero 64×73 action space
    Masking: legal-move mask is applied OUTSIDE this module (in MCTS/trainer)

Value head:
    Input:  (B, 256, 8, 8) feature map
    Output: (B, 3)  WDL probabilities [win, draw, loss]
    Also:   (B, 1)  auxiliary scalar for phase supervision (active only
            during the first `aux_steps` training steps)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """
    Maps the residual tower output to action logits.
    Uses a 2-filter conv projection followed by a fully-connected layer.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_actions: int = 4672,   # 64 × 73
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=2, num_channels=2)
        self.fc   = nn.Linear(2 * 8 * 8, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 256, 8, 8)
        Returns:
            logits: (B, num_actions)  — unnormalised, no softmax
        """
        x = F.gelu(self.norm(self.conv(x)))   # (B, 2, 8, 8)
        x = x.flatten(start_dim=1)             # (B, 128)
        return self.fc(x)                      # (B, 4672)

    def masked_softmax(
        self,
        logits: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply legal-move mask and compute probabilities.

        Args:
            logits:     (B, num_actions)
            legal_mask: (B, num_actions) bool — True for legal moves
        Returns:
            probs: (B, num_actions)
        """
        neg = torch.finfo(torch.float32).min / 4
        masked = torch.where(
            legal_mask,
            logits.float(),
            torch.full_like(logits, neg, dtype=torch.float32),
        )
        return F.softmax(masked, dim=-1)


class ValueHead(nn.Module):
    """
    Maps the residual tower output to WDL probabilities and an auxiliary scalar.

    The auxiliary scalar approximates material balance (used only during the
    first `aux_steps` training steps to accelerate early-training value accuracy).
    After `aux_steps`, the auxiliary branch is detached and effectively ignored
    in the loss computation.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()

        self.conv   = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.norm   = nn.GroupNorm(num_groups=1, num_channels=1)
        self.fc1    = nn.Linear(64, hidden_size)
        self.fc_wdl = nn.Linear(hidden_size, 3)    # win / draw / loss
        self.fc_aux = nn.Linear(hidden_size, 1)    # auxiliary phase signal

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 256, 8, 8)
        Returns:
            wdl: (B, 3)  softmax probabilities [win, draw, loss]
            aux: (B, 1)  tanh scalar in [-1, 1] (material-balance proxy)
        """
        x = F.gelu(self.norm(self.conv(x)))    # (B, 1, 8, 8)
        x = x.flatten(start_dim=1)              # (B, 64)
        x = F.gelu(self.fc1(x))                 # (B, 256)
        wdl = F.softmax(self.fc_wdl(x), dim=-1) # (B, 3)
        aux = torch.tanh(self.fc_aux(x))         # (B, 1)
        return wdl, aux

    def scalar_value(self, wdl: torch.Tensor) -> torch.Tensor:
        """
        Convert WDL probabilities to a scalar value in [-1, 1].
        value = P(win) - P(loss)
        """
        return wdl[:, 0] - wdl[:, 2]  # (B,)
