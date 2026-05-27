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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """
    Maps the residual tower output to action logits.

    Uses a 1×1 conv projection (256 → `mid_channels`) followed by GroupNorm,
    GELU, and a fully-connected layer producing the AlphaZero 64×73 logit
    field. `mid_channels` defaults to 32 — wider than the original 2-channel
    bottleneck (which throttled the FC layer to 128 features for the entire
    4672-dim policy) and small enough that the FC weight matrix stays at
    ~10M params. The AlphaZero paper uses 73 (one per move-type plane);
    32 is a memory-cheaper compromise that still gives the FC ~16× more
    information than the legacy width.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_actions: int = 4672,   # 64 × 73
        mid_channels: int = 32,
    ) -> None:
        super().__init__()

        # GroupNorm needs num_channels % num_groups == 0; pick groups that
        # divide cleanly for any reasonable mid_channels (2, 8, 16, 32, 73…).
        groups = math.gcd(8, mid_channels) or 1

        self.mid_channels = mid_channels
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=mid_channels)
        self.fc   = nn.Linear(mid_channels * 8 * 8, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 256, 8, 8)
        Returns:
            logits: (B, num_actions)  — unnormalised, no softmax
        """
        x = F.gelu(self.norm(self.conv(x)))   # (B, mid, 8, 8)
        x = x.flatten(start_dim=1)             # (B, mid * 64)
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
        # Numerical stabilisation: subtract the per-row max BEFORE masking so
        # the dynamic range of legal logits is preserved relative to itself.
        # The previous hard clamp([-30, 30]) flattened the top of the
        # distribution (a 50-vs-48 logit became indistinguishable after
        # clamp, destroying the network's confidence ordering exactly when
        # the policy was certain). The mask then injects a large finite
        # negative for illegal moves so softmax → 0 there with no NaN risk.
        logits_f = logits.float()
        # Replace inf/nan defensively so the max reduction below stays finite.
        logits_f = torch.nan_to_num(logits_f, nan=0.0, posinf=50.0, neginf=-50.0)
        # Per-row max over legal moves only (or the full row if a sample has
        # no legal moves — caller's invariant violation, but don't NaN here).
        masked_for_max = torch.where(
            legal_mask, logits_f, torch.full_like(logits_f, -1e30)
        )
        row_max = masked_for_max.max(dim=-1, keepdim=True).values
        # Guard against the "no legal moves" row (row_max == -1e30 → finite
        # below) by zeroing the offset in that pathological case.
        row_max = torch.where(row_max < -1e29,
                              torch.zeros_like(row_max), row_max)
        shifted = logits_f - row_max
        neg = -1e9
        masked = torch.where(legal_mask, shifted, torch.full_like(shifted, neg))
        return F.softmax(masked, dim=-1)


class ValueHead(nn.Module):
    """
    Maps the residual tower output to WDL probabilities and an auxiliary scalar.

    The auxiliary scalar is a **learnable material-balance signal**:
    aux = tanh((white_count − black_count) · piece_values / material_scale),
    where ``piece_values`` is a (6,) ``nn.Parameter`` initialised to the
    classical chess values [P=1, N=3, B=3, R=5, Q=9, K=0]. The values adapt
    during training data-driven — final inspection of
    ``value_head.piece_values`` reveals what the net considers each piece worth.

    Rationale (early training): self-play games are draw-dominated (random
    policy + 150-ply cap), so the WDL target collapses to (0, 1, 0) and the
    value loss flatlines. Material differential is non-trivial on most
    positions, so this aux head feeds the value tower a dense supervisory
    signal in the phase where WDL is information-poor. The aux output does
    NOT feed MCTS — it is loss-only, so it never biases self-play priors.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_size: int = 256,
        material_scale: float = 12.0,
    ) -> None:
        super().__init__()

        # Default 12.0 mirrors the AlphaZero-paper sweet spot for tanh
        # saturation (max realistic |Σ piece_values × diff| ≈ 39; tanh
        # saturates near ±4, so 12.0 keeps an 8-point material edge at
        # tanh(8/12) ≈ 0.58 — informative, not saturated). Lower for more
        # decisive auxiliary signals; higher for finer granularity.
        # Stored as a buffer so it (a) moves with .to(device) and (b)
        # round-trips through state_dict for reproducibility.
        self.register_buffer("_material_scale",
                             torch.tensor(float(material_scale)))

        self.conv   = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.norm   = nn.GroupNorm(num_groups=1, num_channels=1)
        self.fc1    = nn.Linear(64, hidden_size)
        self.fc_wdl = nn.Linear(hidden_size, 3)    # win / draw / loss
        # Learnable piece values: [Pawn, Knight, Bishop, Rook, Queen, King].
        # King kept at 0 — both sides always have one, so it never contributes
        # to material differential in legal positions. It stays a parameter
        # (not a buffer) only so the optimizer reads it uniformly, but with
        # zero gradient signal it stays at 0 unless the data says otherwise.
        self.piece_values = nn.Parameter(
            torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0], dtype=torch.float32)
        )

    def forward(
        self,
        x: torch.Tensor,
        material_diff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 256, 8, 8) residual tower features.
            material_diff: (B, 6) piece-type count difference oriented to the
                side-to-move (white_count − black_count when white-to-move,
                negated when black-to-move). ``None`` (back-compat / inference
                paths without board context) → aux output is zeros.
        Returns:
            wdl: (B, 3)  softmax probabilities [win, draw, loss]
            aux: (B, 1)  tanh scalar in [-1, 1] (material-balance from
                 side-to-move's perspective)
        """
        x = F.gelu(self.norm(self.conv(x)))    # (B, 1, 8, 8)
        x = x.flatten(start_dim=1)              # (B, 64)
        x = F.gelu(self.fc1(x))                 # (B, 256)
        wdl = F.softmax(self.fc_wdl(x), dim=-1) # (B, 3)

        if material_diff is None:
            aux = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        else:
            # Cast piece_values to the same dtype as material_diff to stay
            # safely inside BF16 autocast contexts. The parameter itself is
            # held in fp32 (network is cast head-by-head; piece_values may
            # already be bf16 if .to(bf16) was applied — both are fine).
            pv = self.piece_values.to(material_diff.dtype)
            scale = self._material_scale.to(material_diff.dtype)
            score = (material_diff @ pv) / scale  # (B,)
            aux = torch.tanh(score).unsqueeze(1)  # (B, 1)
        return wdl, aux

    def scalar_value(self, wdl: torch.Tensor) -> torch.Tensor:
        """
        Convert WDL probabilities to a scalar value in [-1, 1].
        value = P(win) - P(loss)
        """
        return wdl[:, 0] - wdl[:, 2]  # (B,)
