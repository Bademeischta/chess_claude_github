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


def checkpoint_block_count(path: str) -> Optional[int]:
    """How many residual blocks the checkpoint's model has, or None."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        sd = state["model"]
        idx = max((int(k.split(".")[1]) for k in sd
                   if k.startswith("blocks.")), default=-1)
        return idx + 1 if idx >= 0 else None
    except Exception:
        return None


def match_arch(cfg, ckpt_path: Optional[str]) -> None:
    """
    If `ckpt_path` exists and its block count differs from the config,
    align `cfg.num_res_blocks` so the model is built to match it.
    """
    if not ckpt_path or not Path(ckpt_path).exists():
        return
    bc = checkpoint_block_count(ckpt_path)
    if bc is not None and bc != cfg.num_res_blocks:
        print(f"[ckpt] Checkpoint has {bc} residual blocks "
              f"(config {cfg.num_res_blocks}) — matching architecture.")
        cfg.num_res_blocks = bc
