"""
model/attention.py

GRU History Encoder.

Takes the last `history_len` board tensors (each 21×8×8) and produces
a context vector of shape (B, 4, 8, 8) that is concatenated to the
current board's 21 planes before entering the residual tower.

The GRU processes: (B, T, 21*64) → last hidden state (B, 256)
Then reshape → (B, 4, 8, 8) to match the spatial format.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUHistoryEncoder(nn.Module):
    """
    Encodes a sequence of `history_len` board tensors into a spatial
    context map that is concatenated to the current board encoding.

    Input:
        history: Tensor of shape (B, T, 21, 8, 8)  — the T positions
        strictly BEFORE the current board, right-aligned, zero-padded
        (the most recent past position is history[:, -1]; the current
        board itself is NOT included — audit P3-4 convention)

    Output:
        context: Tensor of shape (B, 4, 8, 8)
    """

    def __init__(
        self,
        input_planes: int = 21,
        hidden_size: int = 256,
        num_layers: int = 2,
        history_len: int = 8,
        output_planes: int = 4,
    ) -> None:
        super().__init__()

        self.input_planes  = input_planes
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.history_len   = history_len
        self.output_planes = output_planes

        # Flatten each 21×8×8 board to a 1344-dim vector before feeding the GRU
        self.input_dim = input_planes * 8 * 8  # 1344

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,  # No dropout — stabilises BF16 training
        )

        # Project the final hidden state to (output_planes * 64) spatial features
        self.proj = nn.Linear(hidden_size, output_planes * 8 * 8)
        self.norm = nn.GroupNorm(num_groups=min(4, output_planes), num_channels=output_planes)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, T, 21, 8, 8) — T board tensors in chronological order.
                     Padding with zeros if fewer than T boards are available.
        Returns:
            context: (B, 4, 8, 8)
        """
        B, T, C, H, W = history.shape
        out_dtype = history.dtype
        # cuDNN rejects bf16 RNN weights for the contiguous fast path
        # (flatten_parameters() then no-ops and it warns + recompacts every
        # call), so this encoder runs in fp32. Autocast is explicitly
        # DISABLED for the whole block: with fp32 master weights + autocast
        # compute (audit P2-3 fix) an enclosing autocast region would cast
        # the RNN back to bf16 and reintroduce the cuDNN slow path.
        device_type = "cuda" if history.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x = history.reshape(B, T, C * H * W).float()

            # `flatten_parameters()` MUST NOT run inside a CUDA-graph
            # capture: there it allocates the cuDNN flat weight buffer from
            # the graph's PRIVATE memory pool and re-points every gru.*
            # Parameter at a view into that pool. Consequences, all
            # measured on step-116185 (2026-08-04):
            #   * the parameters read as exact zeros outside a replay, so
            #     every eager forward on that model (web GUI, oversize-batch
            #     fallback, a state_dict() save before the first replay) sees
            #     a DEAD encoder;
            #   * each replay re-runs the captured copy from the pre-capture
            #     storage, so optimiser updates written after capture are
            #     reverted — the encoder froze at its capture-time weights;
            #   * `load_state_dict` on such a model writes into graph-private
            #     memory → cudaErrorIllegalAddress (the web-GUI second-model
            #     crash).
            # Outside a capture it must still run. `build_model` compacts
            # the weights once, but that layout does not survive forever —
            # the 2026-08-11 self-play run started warning "RNN module
            # weights are not part of single contiguous chunk of memory"
            # right after the first arena/SF-Elo gate, and from then on
            # cuDNN recompacted on EVERY forward. The call is a cheap no-op
            # when the weights are already flat, so guarding it on the
            # capture state gives us both: no graph-pool capture, no
            # silent per-call recompaction.
            if not (x.is_cuda and torch.cuda.is_current_stream_capturing()):
                self.gru.flatten_parameters()
            _, hidden = self.gru(x)
            # hidden: (num_layers, B, hidden_size) — take the last layer
            h_last = hidden[-1]  # (B, hidden_size)

            # Project to spatial context
            context = self.proj(h_last)             # (B, 4*64)
            context = context.reshape(B, self.output_planes, 8, 8)
            context = self.norm(context)
        return context.to(out_dtype)

    def make_history_buffer(
        self,
        boards: list,           # list of np.ndarray (21,8,8)
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Utility: convert a Python list of board tensors (most recent last)
        to a padded batch tensor of shape (1, T, 21, 8, 8).
        """
        import numpy as np

        T = self.history_len
        buf = np.zeros((T, self.input_planes, 8, 8), dtype=np.float32)
        n = min(len(boards), T)
        for i in range(n):
            buf[T - n + i] = boards[-(n - i)]
        t = torch.from_numpy(buf).unsqueeze(0).to(device=device, dtype=dtype)
        return t
