"""
utils/ckpt_arch.py

Make the model architecture follow the checkpoint that is about to be loaded.

The residual-block count is a config default (10), but checkpoints on disk may
have a different depth (older 20-block runs, or Net2Net-grown nets). Building
the model before adjusting the config → state_dict load fails. These helpers
peek the block count from a checkpoint and align `cfg.num_res_blocks` *before*
`build_model`, so every entry point (training resume, --play, web GUI, lichess
bot, ELO tool) loads correctly regardless of the file's depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Newest checkpoints/step_*.pt path, or None if there is none."""
    d = Path(ckpt_dir)
    ckpts = sorted(d.glob("step_*.pt")) if d.is_dir() else []
    return str(ckpts[-1]) if ckpts else None


def _load_state_dict(path: str) -> Optional[dict]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return state["model"]
    except Exception:
        return None


def checkpoint_block_count(path: str) -> Optional[int]:
    """How many residual blocks the checkpoint's model has, or None."""
    sd = _load_state_dict(path)
    if sd is None:
        return None
    idx = max((int(k.split(".")[1]) for k in sd
               if k.startswith("blocks.")), default=-1)
    return idx + 1 if idx >= 0 else None


def checkpoint_policy_mid_channels(path: str) -> Optional[int]:
    """Width of the policy head's conv bottleneck in the checkpoint, or None."""
    sd = _load_state_dict(path)
    if sd is None:
        return None
    # Look for the conv weight under either the bare or torch.compile-wrapped
    # name. Shape is (out_channels, in_channels, 1, 1).
    for k in ("policy_head.conv.weight", "_orig_mod.policy_head.conv.weight"):
        if k in sd:
            return int(sd[k].shape[0])
    return None


def checkpoint_material_scale(path: str) -> Optional[float]:
    """Material-scale buffer from the checkpoint, or None if absent (legacy)."""
    sd = _load_state_dict(path)
    if sd is None:
        return None
    for k in ("value_head._material_scale", "_orig_mod.value_head._material_scale"):
        if k in sd:
            return float(sd[k].item())
    return None


def match_arch(cfg, ckpt_path: Optional[str]) -> None:
    """
    Align `cfg` to architectural choices baked into a checkpoint.

    Without this, the freshly-built model has a different shape than the
    saved state_dict and `load_state_dict` raises. Adjusted fields:
      * `num_res_blocks`        — depth of the residual tower
      * `policy_mid_channels`   — width of the policy head bottleneck
      * `material_scale`        — denominator for the aux material head

    Each is only adjusted when the checkpoint actually carries the value
    AND it differs from the config (older checkpoints predating a feature
    keep the config default). Called from every checkpoint-loading entry
    point: training resume, --play, ELO tool, Lichess bot.
    """
    if not ckpt_path or not Path(ckpt_path).exists():
        return

    bc = checkpoint_block_count(ckpt_path)
    if bc is not None and bc != cfg.num_res_blocks:
        print(f"[ckpt] Checkpoint has {bc} residual blocks "
              f"(config {cfg.num_res_blocks}) — matching architecture.")
        cfg.num_res_blocks = bc

    pmc = checkpoint_policy_mid_channels(ckpt_path)
    if pmc is not None and pmc != getattr(cfg, "policy_mid_channels", 32):
        print(f"[ckpt] Checkpoint policy head is {pmc}-channel "
              f"(config {getattr(cfg, 'policy_mid_channels', '?')}) — "
              f"matching architecture.")
        cfg.policy_mid_channels = pmc

    ms = checkpoint_material_scale(ckpt_path)
    if ms is not None and abs(ms - getattr(cfg, "material_scale", 12.0)) > 1e-6:
        print(f"[ckpt] Checkpoint material_scale={ms:.3f} "
              f"(config {getattr(cfg, 'material_scale', '?')}) — matching.")
        cfg.material_scale = ms
