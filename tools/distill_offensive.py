#!/usr/bin/env python3
"""
tools/distill_offensive.py — one-command Stockfish-distillation cycle
(audit 2026-07-02 Stufe 2.1 / plan A5, first executed 2026-07-18).

Motivation (measured): ~10k steps of pure self-play (steps 112k-123k)
moved the objective SF-anchor Elo from ~568 to ~550 and LOST the pool
arena -102 against the net's own 110k self — at this compute budget,
distillation is the strength driver, self-play the finisher. This runner
executes the whole cycle unattended:

  1. HARVEST : N parallel tools/stockfish_distill.py workers (CPU-only)
               play SF-vs-SF games at mixed skill levels.
  2. BASELINE: 40-game SF-anchor probe of the current net (GPU, ~5 min).
  3. MERGE   : worker outputs -> one replay_buffer/sf_distill.pkl.
  4. RESEED  : tools/seed_teacher.py rebuilds checkpoints/teacher.pkl.gz
               from the fresh harvest (replaces the memorised old set).
  5. DISTILL : main.py --phase=sf_distill (merges the buffer, trains
               `--epochs` passes; the in-training SF-Elo probe fires on
               its 5k-step gates).
  6. POST    : 40-game SF-anchor probe again -> the before/after pair IS
               the Elo-per-GPU-hour datapoint the A5 decision needs.

Deliberately does NOT restart self-play afterwards — evaluate first.

    python tools/distill_offensive.py --workers 6 --games-per-worker 2500
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PY = sys.executable


def _log(msg: str) -> None:
    print(f"[offensive {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _probe(tag: str, games: int) -> dict:
    """SF-anchor probe of checkpoints/latest.pt via the shared harness."""
    import torch
    from config import CONFIG
    from engine.harness import load_engine
    from utils.sf_anchor import play_anchor_match

    handle = load_engine(str(_root / "checkpoints" / "latest.pt"))
    res = play_anchor_match(
        handle.model, handle.device, CONFIG,
        games=games, sf_skill=int(CONFIG.sf_elo_skill),
        sims=128, batched_bs=32, engine_inf=handle.engine)
    _log(f"probe[{tag}]: {res}")
    # Free the GPU for the next stage (the distill trainer builds its own).
    del handle
    torch.cuda.empty_cache()
    return res or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel harvest processes (CPU; each runs its own "
                         "Stockfish with --sf-threads threads)")
    ap.add_argument("--games-per-worker", type=int, default=2500)
    ap.add_argument("--skills", default="10,12,14,16",
                    help="skill pool, one drawn per game (stockfish_distill "
                         "--skill-mix format)")
    ap.add_argument("--movetime", type=float, default=0.12)
    ap.add_argument("--sf-threads", type=int, default=2)
    ap.add_argument("--probe-games", type=int, default=40)
    ap.add_argument("--skip-harvest", action="store_true",
                    help="reuse existing replay_buffer/sf_distill_p*.pkl")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = _root / "replay_buffer"
    out_dir.mkdir(exist_ok=True)
    parts = [out_dir / f"sf_distill_p{i}.pkl" for i in range(args.workers)]

    # ── 1. Harvest (parallel, CPU) ────────────────────────────────────
    if not args.skip_harvest:
        procs = []
        for i, part in enumerate(parts):
            cmd = [PY, "-u", str(_root / "tools" / "stockfish_distill.py"),
                   "--games", str(args.games_per_worker),
                   "--skill-mix", args.skills,
                   "--movetime", str(args.movetime),
                   "--threads", str(args.sf_threads),
                   "--seed", str(1000 + i),
                   "--print-every", "100",
                   "--out", str(part.relative_to(_root))]
            lf = open(_root / "audit" / "tmp" / f"harvest_p{i}.log",
                      "w", encoding="utf-8")
            procs.append((subprocess.Popen(cmd, cwd=_root, stdout=lf,
                                           stderr=subprocess.STDOUT), lf))
            _log(f"harvest worker {i} started (games={args.games_per_worker})")
        for i, (p, lf) in enumerate(procs):
            rc = p.wait()
            lf.close()
            _log(f"harvest worker {i} exited rc={rc}")
    else:
        _log("harvest skipped (--skip-harvest)")

    # ── 2. Baseline probe ─────────────────────────────────────────────
    baseline = _probe("baseline", args.probe_games)

    # ── 3. Merge parts ────────────────────────────────────────────────
    records = []
    for part in parts:
        if not part.exists():
            _log(f"WARNING: missing harvest part {part.name}")
            continue
        with open(part, "rb") as f:
            st = pickle.load(f)
        n = int(st.get("size", 0))
        recs = [r for r in st.get("data", [])[:n] if r is not None]
        records.extend(recs)
        _log(f"merged {part.name}: {len(recs)} positions")
    merged = out_dir / "sf_distill.pkl"
    with open(merged, "wb") as f:
        # _merge_buffer_file consumes only "data" + "size".
        pickle.dump({"data": records, "size": len(records)}, f, protocol=4)
    _log(f"merged buffer -> {merged.name} ({len(records)} positions)")
    if not records:
        _log("ABORT: no harvested positions.")
        return 1

    # ── 4. Reseed the teacher buffer ──────────────────────────────────
    rc = subprocess.call(
        [PY, "-u", str(_root / "tools" / "seed_teacher.py"),
         "--sources", str(merged.relative_to(_root)),
         "--capacity", "150000"], cwd=_root)
    _log(f"seed_teacher rc={rc}")

    # ── 5. Distill training ───────────────────────────────────────────
    t_gpu = time.time()
    rc = subprocess.call(
        [PY, "-u", str(_root / "main.py"), "--phase=sf_distill",
         "--resume", str(_root / "checkpoints" / "latest.pt")], cwd=_root)
    gpu_hours = (time.time() - t_gpu) / 3600.0
    _log(f"sf_distill rc={rc} ({gpu_hours:.2f} GPU-h)")

    # ── 6. Post probe + summary ───────────────────────────────────────
    post = _probe("post", args.probe_games)
    summary = {
        "harvested_positions": len(records),
        "baseline": baseline, "post": post,
        "distill_gpu_hours": round(gpu_hours, 2),
        "elo_delta": (round(post.get("estimated_elo", 0)
                            - baseline.get("estimated_elo", 0), 1)
                      if baseline and post else None),
        "total_hours": round((time.time() - t0) / 3600.0, 2),
    }
    out = _root / "audit" / "tmp" / "distill_offensive_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _log(f"SUMMARY: {json.dumps(summary, default=str)}")
    _log(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
