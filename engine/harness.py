"""
engine/harness.py — shared engine bring-up for every "play against the AI"
front-end (audit 2026-07-15, plan A3/A7).

Before this module, web_gui.py, uci_engine.py, run_arena.py,
elo_vs_stockfish.py, lichess_bot.py and main.py --play each carried their
own copy of the same bring-up sequence — system probe, Syzygy auto-detect,
checkpoint discovery, match_arch, build_model, state-dict load (with three
different strictness policies!), inference-engine build — plus three
slightly different time-control formulas. Fixes never landed in all
copies at once; this module is the single implementation.

Public API:
    load_engine(...)        → EngineHandle (model + inference engine + meta)
    list_checkpoints(cfg)   → checkpoint inventory incl. known Elo labels
    time_budget_s(...)      → the UCI clock heuristic (clock/25 + inc·3/4)
    run_search(tree, ...)   → batched, interruptible search driver

Nothing here mutates training state; all loads are read-only
(weights_only=True).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from config import CONFIG


# Keys tolerated as missing on old checkpoints (buffers added after the
# fact). Everything else missing/unexpected is a REAL architecture
# mismatch and must fail loudly — silent strict=False loads have already
# masked a policy-head migration bug once (audit 2026-06 history).
_ALLOWED_MISSING = {"value_head._material_scale"}


@dataclass
class EngineHandle:
    """A ready-to-play engine: model + batched inference engine + metadata."""
    model: torch.nn.Module
    engine: object                 # inference engine (infer(board_t, hist_t))
    device: torch.device
    cfg: object                    # the (possibly probe-updated) config
    ckpt_path: Optional[str]       # None = untrained net
    step: Optional[int]            # training step of the checkpoint, if known


def init_syzygy(cfg=CONFIG, repo_root: Optional[Path] = None) -> bool:
    """Auto-detect + initialise Syzygy tablebases (same rule as main.py).

    Play-time search must probe the TB like training does — without it the
    net trades into tablebase-lost ≤5-man endgames it cannot evaluate
    (observed in a real game: Qxf3?? into a lost 5-man)."""
    from engine.rules import init_tablebase
    root = repo_root or Path(__file__).resolve().parent.parent
    if not cfg.syzygy_path:
        auto = root / "syzygy"
        if auto.is_dir() and any(auto.glob("*.rtbw")):
            cfg.syzygy_path = str(auto)
    if cfg.syzygy_path:
        return bool(init_tablebase(cfg.syzygy_path))
    return False


def load_state_into(model: torch.nn.Module, ckpt: str,
                    device: torch.device) -> Optional[int]:
    """Load a checkpoint's model weights; return its training step (if any).

    Accepts both full trainer checkpoints ({'model': state_dict, ...}) and
    raw state_dicts (opponent-pool entries). Uniform strictness policy:
    strict=False load, then verify that nothing beyond the known-benign
    allowlist is missing/unexpected (the audited policy from
    elo_vs_stockfish; uci_engine's strict=True broke on old checkpoints,
    web_gui's bare strict=False hid real mismatches)."""
    st = torch.load(ckpt, map_location=device, weights_only=True)
    state = st["model"] if isinstance(st, dict) and "model" in st else st
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    missing, unexpected = base.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if k not in _ALLOWED_MISSING]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {ckpt}.\n"
            f"  missing   : {real_missing}\n"
            f"  unexpected: {list(unexpected)}")
    step = st.get("step") if isinstance(st, dict) else None
    return int(step) if step is not None else None


def load_engine(
    ckpt: Optional[str] = None,
    *,
    cfg=CONFIG,
    probe: bool = True,
    syzygy: bool = True,
    build_engine: bool = True,
    compile_model: bool = False,
    log: Callable[[str], None] = print,
) -> EngineHandle:
    """One-call engine bring-up shared by every play/eval front-end.

    ckpt=None → newest checkpoint in cfg.checkpoint_dir (untrained net if
    none exists). `probe` runs the hardware probe config update (skip when
    the caller already did). `build_engine=False` skips the inference-
    engine build (callers that only need the model)."""
    from utils.ckpt_arch import latest_checkpoint, match_arch
    from model.network import build_model
    from model.inference import build_inference_engine

    if probe:
        from utils.system_probe import run_probe
        from config import update_config_from_dict
        update_config_from_dict(run_probe(write=False))

    device = torch.device(cfg.device)

    if syzygy:
        ok = init_syzygy(cfg)
        if cfg.syzygy_path:
            log(f"[harness] Syzygy tablebases: {'loaded' if ok else 'not found'}")

    ckpt = ckpt or latest_checkpoint(cfg.checkpoint_dir)
    match_arch(cfg, ckpt)
    model = build_model(cfg, compile_model=compile_model).eval()

    step: Optional[int] = None
    if ckpt and Path(ckpt).exists():
        step = load_state_into(model, ckpt, device)
        log(f"[harness] Loaded {ckpt}" + (f" (step {step})" if step else ""))
    else:
        ckpt = None
        log("[harness] No checkpoint — UNTRAINED net.")

    engine = build_inference_engine(cfg, model, device) if build_engine else None
    return EngineHandle(model=model, engine=engine, device=device, cfg=cfg,
                        ckpt_path=ckpt, step=step)


# ── Checkpoint inventory (web UI dropdown, tooling) ───────────────────────

def list_checkpoints(cfg=CONFIG) -> list[dict]:
    """Inventory of playable checkpoints with any known strength labels.

    Returns newest-first dicts:
      {path, name, step, mtime, pool_elo (opt), sf_elo (opt)}
    pool_elo comes from checkpoints/pool_state.json (self-referential
    scale); sf_elo from checkpoints/sf_elo_history.csv (objective anchor,
    nearest measurement within ±2500 steps)."""
    import json

    ckpt_dir = Path(cfg.checkpoint_dir)
    if not ckpt_dir.is_dir():
        return []

    pool_elo: dict[str, float] = {}
    try:
        pool = json.loads((ckpt_dir / "pool_state.json").read_text("utf-8"))
        for ent in pool.get("pool", []):
            name = Path(ent.get("path", "")).name
            if name:
                pool_elo[name] = float(ent.get("elo", 0.0))
    except (OSError, ValueError):
        pass

    sf_hist: list[tuple[int, float]] = []
    try:
        lines = (ckpt_dir / "sf_elo_history.csv").read_text("utf-8").splitlines()
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) >= 7:
                sf_hist.append((int(parts[0]), float(parts[6])))
    except (OSError, ValueError, IndexError):
        pass

    def _sf_elo_for(step: Optional[int]) -> Optional[float]:
        if step is None or not sf_hist:
            return None
        near = min(sf_hist, key=lambda t: abs(t[0] - step))
        return near[1] if abs(near[0] - step) <= 2500 else None

    def _step_of(p: Path) -> Optional[int]:
        import re
        m = re.search(r"(?:net_step|step_0*|ref_)(\d+)", p.stem)
        return int(m.group(1)) if m else None

    out: list[dict] = []
    for p in sorted(ckpt_dir.glob("*.pt"),
                    key=lambda q: q.stat().st_mtime, reverse=True):
        if p.suffix != ".pt" or p.name.endswith(".collapsed"):
            continue
        step = _step_of(p)
        out.append({
            "path": str(p),
            "name": p.name,
            "step": step,
            "mtime": p.stat().st_mtime,
            "pool_elo": pool_elo.get(p.name),
            "sf_elo": _sf_elo_for(step),
        })
    return out


# ── Time control (single implementation for UCI / web / terminal) ─────────

def time_budget_s(clock_ms: Optional[int], inc_ms: int = 0,
                  movetime_ms: Optional[int] = None) -> Optional[float]:
    """Per-move think budget in seconds.

    movetime dominates; else the conservative UCI-clock heuristic
    clock/25 + increment·0.75 (uci_engine's audited formula — safe against
    flagfall at any remaining-time level). None = no time limit given."""
    if movetime_ms is not None:
        return max(0.05, movetime_ms / 1000.0)
    if clock_ms is None:
        return None
    budget_ms = clock_ms / 25.0 + (inc_ms or 0) * 0.75
    return max(0.05, budget_ms / 1000.0)


# ── Search driver (batched, interruptible) ─────────────────────────────────

def run_search(
    tree,
    *,
    sims: Optional[int] = None,
    deadline: Optional[float] = None,
    stop=None,
    batch_size: int = 0,
    info_cb: Optional[Callable[[int], None]] = None,
    info_every: float = 0.3,
) -> int:
    """Drive a search on an already-begun tree until a budget is exhausted.

    Runs `run_batched_sims` one round (`batch_size` leaves sharing a single
    NN forward) at a time so `stop`/`deadline` stay responsive at ~one
    round's latency. batch_size<=0 falls back to single-leaf sims.

    sims     — total simulation budget (None = unbounded),
    deadline — absolute time.time() to stop at (None = none),
    stop     — threading.Event-like (checked between rounds),
    info_cb  — called with the running sim count every `info_every` s.
    Returns simulations completed. At least one round runs even if the
    deadline already passed (a move must never be produced from an
    unsearched root)."""
    done = 0
    last_info = time.time()
    while True:
        if sims is not None and done >= sims:
            break
        if done > 0:  # always complete at least one round
            if stop is not None and stop.is_set():
                break
            if deadline is not None and time.time() >= deadline:
                break
        if batch_size > 0:
            want = batch_size if sims is None else min(batch_size, sims - done)
            adv = tree.run_batched_sims(want, batch_size=batch_size)
            if adv <= 0:
                break  # search saturated (all lines terminal)
            done += adv
        else:
            tree.run_one_simulation()
            done += 1
        if info_cb is not None and time.time() - last_info >= info_every:
            try:
                info_cb(done)
            except Exception:
                pass
            last_info = time.time()
    return done
