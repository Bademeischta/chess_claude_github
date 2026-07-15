"""
tools/webgui/monitor.py — live training monitor for the web UI.

Reads the newest TensorBoard event file + pool_state.json straight from
disk (no coupling to the training process). Moved verbatim out of
tools/web_gui.py (plan B1 step 2).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from config import CONFIG

_root = Path(__file__).resolve().parent.parent.parent

_EA_CACHE: dict = {"path": None, "ea": None}
_EA_LOCK = threading.Lock()


def _training_snapshot() -> dict:
    """LIVE training stats for the web monitor. Loss / lr / entropy /
    throughput / buffer sizes / VRAM come straight from the newest TensorBoard
    event file (written every step by the trainer); ELO comes from
    pool_state.json. The EventAccumulator is cached and reloaded incrementally
    so the 4-second poll stays cheap. Missing sources just yield nulls so the
    panel degrades gracefully."""
    out = {k: None for k in (
        "step", "loss_total", "p", "v", "kl", "aux", "lr", "entropy",
        "pos_s", "games_hr", "replay", "teacher", "vram",
        "elo", "best_elo")}
    out["ts"] = time.time()
    root = Path(_root)

    # ELO + best ELO from the opponent pool.
    try:
        ps = root / str(CONFIG.checkpoint_dir) / "pool_state.json"
        if ps.exists():
            d = json.loads(ps.read_text(encoding="utf-8", errors="ignore"))
            ratings = {k: float(v) for k, v in d.get("elo_ratings", {}).items()}
            cur = d.get("current_id")
            if cur in ratings:
                out["elo"] = round(ratings[cur], 1)
            if ratings:
                out["best_elo"] = round(max(ratings.values()), 1)
    except Exception:
        pass

    # Live scalars from the newest tfevents file (incrementally reloaded).
    try:
        import glob
        import os
        from tensorboard.backend.event_processing.event_accumulator \
            import EventAccumulator
        evs = glob.glob(str(root / "runs" / "events.out.tfevents.*"))
        if evs:
            newest = max(evs, key=os.path.getmtime)
            with _EA_LOCK:
                ea = _EA_CACHE.get("ea")
                if _EA_CACHE.get("path") != newest or ea is None:
                    ea = EventAccumulator(newest, size_guidance={"scalars": 64})
                    _EA_CACHE["path"] = newest
                    _EA_CACHE["ea"] = ea
                ea.Reload()
                avail = set(ea.Tags().get("scalars", []))

                def last(tag):
                    if tag not in avail:
                        return None
                    try:
                        pts = ea.Scalars(tag)
                        return max(pts, key=lambda s: s.step).value if pts else None
                    except Exception:
                        return None

                tagmap = {
                    "step": "global_step", "loss_total": "loss/total",
                    "p": "loss/policy", "v": "loss/value",
                    "kl": "loss/teacher_kl", "aux": "loss/aux", "lr": "lr",
                    "entropy": "policy_entropy",
                    "pos_s": "throughput/positions_per_sec",
                    "games_hr": "throughput/games_per_hour",
                    "replay": "replay_buf_size", "teacher": "teacher_buf_size",
                    "vram": "memory/vram_used_mb",
                }
                for key, tag in tagmap.items():
                    v = last(tag)
                    if v is not None:
                        out[key] = v
            if out["step"] is not None:
                out["step"] = int(out["step"])
    except Exception:
        pass

    # Fallback: derive the step from the newest checkpoint filename.
    if out["step"] is None:
        try:
            import glob
            import os
            import re
            cks = glob.glob(str(root / str(CONFIG.checkpoint_dir) / "step_*.pt"))
            if cks:
                m = re.search(r"step_(\d+)\.pt",
                              max(cks, key=os.path.getmtime))
                if m:
                    out["step"] = int(m.group(1))
        except Exception:
            pass
    return out
