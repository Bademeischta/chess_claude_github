"""
model/net2net.py

Net2Net depth growth: warm-start a deeper ChessNet from a shallower
checkpoint without throwing away learned weights.

How the warm-start works (and its limit)
----------------------------------------
`ResBlock.forward` is  ``gelu(norm2(conv2(gelu(norm1(conv1(x))))) + x)``.
If a freshly inserted block's ``conv2.weight`` is zeroed, its conv path is
exactly 0 (conv2 has no bias; norm2 of an all-zero input is 0), so the block
collapses to ``gelu(x)``. The original blocks / input projection / GRU
encoder / heads are copied verbatim (identical shapes — `with_se` depends only
on the block index and `se_every`, which are unchanged).

This is a strong WARM-START, not a bit-identity extension: because the block
is *post-activation* (the GELU wraps the sum), an appended block computes
``gelu(residual)`` rather than ``residual`` — there is no zero-init that makes
a post-activation residual block an exact identity. Each appended block mildly
squashes positive activations; this compounds over the inserted depth
(empirically ~0.15 policy-logit drift for +6 blocks). That is far better than
training the new capacity from scratch (heads, GRU and the first N blocks stay
fully trained) and SGD removes the drift within a short fine-tune. Fresh
optimizer + step 0 by design.

This only grows DEPTH (block count). Channel widening is intentionally not
implemented — depth (10→20) is the documented path here and widening is rarely
needed; a half-built widen() would be worse than none.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model.network import ChessNet


def _count_blocks(state_dict: dict) -> int:
    """Highest blocks.<i>.* index in a model state_dict, + 1."""
    idx = -1
    for k in state_dict:
        if k.startswith("blocks."):
            idx = max(idx, int(k.split(".")[1]))
    return idx + 1


def grow_model(
    old_ckpt_path: str,
    cfg,
    new_blocks: int,
    out_ckpt_path: str,
) -> tuple[int, int]:
    """
    Load `old_ckpt_path` (a Trainer checkpoint), build a ChessNet with
    `new_blocks` residual blocks, transfer the shared weights, identity-init
    the appended blocks, and write `out_ckpt_path`.

    The optimizer / scheduler / step are intentionally NOT carried over: the
    parameter set changed, so training resumes with a fresh optimizer from
    step 0 (warm-started weights only). Returns (old_blocks, new_blocks).
    """
    state = torch.load(old_ckpt_path, map_location="cpu", weights_only=True)
    old_sd = state["model"]
    old_blocks = _count_blocks(old_sd)
    if new_blocks <= old_blocks:
        raise ValueError(
            f"--grow target ({new_blocks}) must exceed the checkpoint's "
            f"block count ({old_blocks})."
        )

    target = ChessNet(
        input_planes = cfg.input_planes,
        gru_context  = cfg.gru_context_planes,
        channels     = cfg.num_channels,
        num_blocks   = new_blocks,
        se_every     = cfg.se_every_n,
        gru_hidden   = cfg.gru_hidden,
        gru_layers   = cfg.gru_layers,
        history_len  = cfg.gru_history_len,
        num_actions  = cfg.num_actions,
    )

    # Shared keys (blocks 0..old_blocks-1, input_proj, gru_encoder, heads)
    # match by shape; the appended blocks' keys are simply absent from old_sd
    # and keep their fresh init.
    missing, unexpected = target.load_state_dict(old_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when growing: {unexpected[:5]} …")

    # Identity warm-start for every appended block: zero its conv2 weight.
    with torch.no_grad():
        for j in range(old_blocks, new_blocks):
            nn.init.zeros_(target.blocks[j].conv2.weight)

    out = {"step": 0, "model": target.state_dict()}
    Path(out_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_ckpt_path)
    return old_blocks, new_blocks
