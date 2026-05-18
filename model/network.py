"""
model/network.py

Full chess neural network: GRU History Encoder → Input Projection →
20 Residual Blocks (with SE attention every 4 blocks) → Policy Head + Value Head.

Architecture summary:
  Input:   (B, 25, 8, 8)   [21 board planes + 4 GRU context planes]
  Tower:   20 × ResBlock (256 channels, GroupNorm, GELU)
           SE-block (Squeeze-Excitation) inserted at blocks 4, 8, 12, 16, 20
  Policy:  (B, 4672) logits
  Value:   (B, 3) WDL probs + (B, 1) aux scalar

CUDA optimisations applied:
  - torch.compile(mode="reduce-overhead") on the forward pass (CUDA Graphs; Triton-free)
  - BF16 autocast (externally, in the trainer)
  - Gradient checkpointing on ResBlocks 5–20
  - CUDA Graph capture for pure-inference path (in MCTS)
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.attention import GRUHistoryEncoder
from model.heads import PolicyHead, ValueHead


# ── Building blocks ───────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-Excitation channel attention (after ~4 ResBlocks)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))              # Global avg pool: (B, C)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))      # (B, C)
        return x * s.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    """
    Residual block:
      Conv3×3 → GroupNorm → GELU → Conv3×3 → GroupNorm → (+residual) [→ SE]
    """

    def __init__(
        self,
        channels: int = 256,
        with_se: bool = False,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.se    = SEBlock(channels, se_reduction) if with_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.se is not None:
            x = self.se(x)
        return F.gelu(x + residual)


# ── Main network ──────────────────────────────────────────────────────────

class ChessNet(nn.Module):
    """
    Full chess neural network.

    Parameters
    ----------
    input_planes    : int  — board encoding planes from chess engine (21)
    gru_context     : int  — planes added by GRU encoder (4)  → total input = 25
    channels        : int  — residual tower width (256)
    num_blocks      : int  — number of residual blocks (20)
    se_every        : int  — SE block every N residual blocks (4)
    gru_hidden      : int  — GRU hidden size (256)
    gru_layers      : int  — GRU depth (2)
    history_len     : int  — number of history boards for GRU (8)
    num_actions     : int  — policy head output size (4672)
    grad_ckpt_from  : int  — apply gradient checkpointing from this block index (4)
    """

    def __init__(
        self,
        input_planes: int = 21,
        gru_context: int = 4,
        channels: int = 256,
        num_blocks: int = 20,
        se_every: int = 4,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        history_len: int = 8,
        num_actions: int = 4672,
        grad_ckpt_from: int = 4,
    ) -> None:
        super().__init__()

        self.input_planes   = input_planes
        self.gru_context    = gru_context
        self.channels       = channels
        self.num_blocks     = num_blocks
        self.history_len    = history_len
        self.grad_ckpt_from = grad_ckpt_from

        total_input = input_planes + gru_context  # 25

        # ── GRU History Encoder ────────────────────────
        self.gru_encoder = GRUHistoryEncoder(
            input_planes=input_planes,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            history_len=history_len,
            output_planes=gru_context,
        )

        # ── Input projection: 25 → 256 channels ───────
        self.input_proj = nn.Sequential(
            nn.Conv2d(total_input, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.GELU(),
        )

        # ── Residual tower: 20 blocks ──────────────────
        blocks = []
        for i in range(1, num_blocks + 1):
            with_se = (i % se_every == 0)  # SE at blocks 4, 8, 12, 16, 20
            blocks.append(ResBlock(channels, with_se=with_se))
        self.blocks = nn.ModuleList(blocks)

        # ── Heads ─────────────────────────────────────
        self.policy_head = PolicyHead(in_channels=channels, num_actions=num_actions)
        self.value_head  = ValueHead(in_channels=channels, hidden_size=256)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        board: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            board:   (B, 21, 8, 8)  current board encoding
            history: (B, T, 21, 8, 8) or None.  If None, no GRU context is added.

        Returns:
            policy_logits: (B, 4672)
            wdl:           (B, 3)
            aux:           (B, 1)
        """
        if history is not None:
            context = self.gru_encoder(history)           # (B, 4, 8, 8)
            x = torch.cat([board, context], dim=1)        # (B, 25, 8, 8)
        else:
            # Pad with zeros if no history (e.g. early in a game)
            B = board.shape[0]
            pad = torch.zeros(B, self.gru_context, 8, 8,
                              device=board.device, dtype=board.dtype)
            x = torch.cat([board, pad], dim=1)            # (B, 25, 8, 8)

        x = self.input_proj(x)  # (B, 256, 8, 8)

        for i, block in enumerate(self.blocks):
            if self.training and i >= self.grad_ckpt_from:
                # Gradient checkpointing saves activation memory at cost of recompute
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        policy_logits = self.policy_head(x)
        wdl, aux      = self.value_head(x)
        return policy_logits, wdl, aux

    # ── Inference-only forward (no GRU, no checkpointing) ─────────────────

    @torch.no_grad()
    def infer(
        self,
        board: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fast inference path used by MCTS leaf evaluation.
        Returns (policy_logits, value_scalar) — no aux head, no gradients.
        """
        policy_logits, wdl, _ = self.forward(board, history)
        value = self.value_head.scalar_value(wdl)  # (B,) in [-1, 1]
        return policy_logits, value

    # ── Parameter group helpers (for separate LR per component) ──────────

    def parameter_groups(
        self,
        lr: float,
        backbone_mult: float = 1.0,
        head_mult: float = 0.5,
        gru_mult: float = 0.3,
    ) -> list[dict]:
        """
        Returns a list of parameter-group dicts for AdamW with separate
        learning rates for backbone / heads / GRU encoder.
        """
        gru_params   = list(self.gru_encoder.parameters())
        head_params  = (list(self.policy_head.parameters())
                      + list(self.value_head.parameters()))
        backbone_params = [
            p for p in self.parameters()
            if not any(p is q for q in gru_params)
            and not any(p is q for q in head_params)
        ]
        return [
            {"params": backbone_params, "lr": lr * backbone_mult, "name": "backbone"},
            {"params": head_params,     "lr": lr * head_mult,     "name": "heads"},
            {"params": gru_params,      "lr": lr * gru_mult,      "name": "gru"},
        ]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Factory with torch.compile ────────────────────────────────────────────

def build_model(cfg, compile_model: bool = True) -> ChessNet:
    """Build a ChessNet from config and optionally compile it."""
    net = ChessNet(
        input_planes  = cfg.input_planes,
        gru_context   = cfg.gru_context_planes,
        channels      = cfg.num_channels,
        num_blocks    = cfg.num_res_blocks,
        se_every      = cfg.se_every_n,
        gru_hidden    = cfg.gru_hidden,
        gru_layers    = cfg.gru_layers,
        history_len   = cfg.gru_history_len,
        num_actions   = cfg.num_actions,
    )

    device = torch.device(cfg.device)
    net = net.to(device)

    if cfg.precision == "bf16":
        net = net.to(dtype=torch.bfloat16)
    elif cfg.precision == "fp16":
        net = net.to(dtype=torch.float16)

    # The GRU history encoder must stay fp32: cuDNN won't use the contiguous
    # RNN fast path for bf16 weights (flatten_parameters() no-ops and it
    # recompacts on every forward). It casts internally and back.
    net.gru_encoder.float()

    if compile_model and cfg.torch_compile:
        # torch.compile's inductor backend requires Triton for GPU codegen,
        # and Triton has no official Windows build. Detect it up front and
        # fall back to eager rather than crashing at first forward pass.
        _triton_ok = False
        try:
            import triton  # noqa: F401
            _triton_ok = True
        except ImportError:
            _triton_ok = False

        if _triton_ok:
            try:
                net = torch.compile(net, mode="reduce-overhead")
                print("[ChessNet] torch.compile(mode='reduce-overhead') applied.")
            except Exception as e:
                print(f"[ChessNet] torch.compile failed ({e}), using eager mode.")
        else:
            cfg.torch_compile = False
            print("[ChessNet] Triton unavailable (no Windows build) — running "
                  "in eager mode. BF16 on GPU is still fast.")

    total = net.param_count() if not hasattr(net, '_orig_mod') else \
        sum(p.numel() for p in net.parameters())
    print(f"[ChessNet] Parameters: {total:,}  "
          f"({total/1e6:.1f}M)  device={device}  precision={cfg.precision}")

    return net
