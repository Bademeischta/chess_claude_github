#!/usr/bin/env python3
"""
tools/seed_teacher.py — make Stockfish/puzzle distillation PERSISTENT by
writing it into the TEACHER buffer (checkpoints/teacher.pkl.gz).

Why: 30 % of every training batch is a KL distillation from the teacher buffer
(config.teacher_batch_ratio=0.30, teacher_loss_weight=1.0, applied EVERY step
in trainer.train_step). But the Stockfish/puzzle data was merged into the
*rotating* replay buffer (washes out in ~1-2 days of self-play), while the
teacher buffer held only the net's OWN ~1000-Elo self-play — so 30 % of every
step was the net imitating itself. Filling the teacher buffer with the strong
positions turns that channel into a permanent pull toward 2000-2500-Elo moves
(Stockfish never hangs a piece → the policy is continuously taught not to).

Memory-safe: the source pickles are ~8 GB each, so they are loaded ONE AT A
TIME, down-sampled to a per-source budget, and freed before the next — peak
RAM stays at roughly one source plus the kept sample.

Run with TRAINING STOPPED (the trainer overwrites teacher.pkl.gz on every
checkpoint), then restart training so it loads the seeded buffer. Pair with
config.teacher_game_ratio=0.0 so self-play won't re-dilute the channel.

    python tools/seed_teacher.py
    python tools/seed_teacher.py --sources replay_buffer/sf_distill.pkl --capacity 120000
"""

from __future__ import annotations

import argparse
import gc
import gzip
import pickle
import random
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import CONFIG                          # noqa: E402
import training.replay_buffer as _rb               # noqa: E402,F401  (registers PositionRecord)
from training.replay_buffer import (                # noqa: E402
    TeacherBuffer, iter_record_chunks)


def _sample_records(path: Path, budget: int) -> tuple[list, int]:
    """Up to `budget` records drawn uniformly at random, plus the total seen.

    Reservoir sampling (Algorithm R) over `iter_record_chunks`, which handles
    both the chunked stream format and the legacy single-pickle buffer, gzip
    or plain. For a streamed source, peak RAM is one chunk plus the reservoir
    — previously an ~8 GB part was loaded whole just to draw 25k records from
    it. Legacy files still come back as one chunk (they cannot be read
    incrementally), so the win applies to freshly harvested parts.
    """
    if not path.exists():
        print(f"[seed] not found: {path}")
        return [], 0
    budget = max(0, int(budget))
    reservoir: list = []
    seen = 0
    for chunk in iter_record_chunks(str(path)):
        for rec in chunk:
            seen += 1
            if len(reservoir) < budget:
                reservoir.append(rec)
            else:
                j = random.randrange(seen)
                if j < budget:
                    reservoir[j] = rec
    return reservoir, seen


def _free_avail_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="checkpoints/teacher.pkl.gz",
                    help="teacher buffer snapshot the trainer loads on startup")
    ap.add_argument("--sources",
                    default="replay_buffer/sf_distill.pkl,"
                            "replay_buffer/lichess_puzzles.pkl",
                    help="comma-separated source pickles (Stockfish / puzzles)")
    ap.add_argument("--capacity", type=int, default=150_000,
                    help="total teacher positions to seed (default 150k; raise "
                         "if RAM allows, lower if you hit memory pressure)")
    ap.add_argument("--keep-existing-frac", type=float, default=0.0,
                    help="fraction of the CURRENT teacher buffer to retain "
                         "(default 0 = replace; reloads the ~12 GB snapshot if >0)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    srcs = [s.strip() for s in args.sources.split(",") if s.strip()]
    if not srcs:
        print("[seed] no sources given.")
        return 1
    cap = int(args.capacity)
    teacher_path = (_root / args.teacher)

    avail = _free_avail_gb()
    if avail >= 0:
        print(f"[seed] free RAM: {avail:.1f} GB  (sources are ~8 GB each; "
              f"make sure training is stopped)")
        if avail < 12.0:
            print("[seed] WARNING: <12 GB free — stop training / close apps "
                  "first to avoid an out-of-memory kill.")

    per_src = max(1, cap // len(srcs))
    pool: list = []
    for i, s in enumerate(srcs):
        remaining = cap - len(pool)
        budget = remaining if i == len(srcs) - 1 else min(per_src, remaining)
        if budget <= 0:
            break
        recs, seen = _sample_records(_root / s, budget)
        print(f"[seed] {s}: {seen:,} records → sampling {len(recs):,}")
        pool.extend(recs)
        del recs
        gc.collect()

    if not pool:
        print("[seed] nothing loaded — aborting.")
        return 1

    # Optionally retain a slice of the existing (weak) teacher data.
    if args.keep_existing_frac > 0 and teacher_path.exists():
        keep_n = int(cap * args.keep_existing_frac)
        existing, _seen = _sample_records(teacher_path, keep_n)
        keep_n = len(existing)
        if keep_n:
            pool.extend(existing)
        del existing
        gc.collect()
        print(f"[seed] retained {keep_n:,} existing teacher records")

    random.shuffle(pool)
    if len(pool) > cap:
        pool = pool[:cap]

    tb = TeacherBuffer(capacity=cap)
    tb.add_batch(pool)
    del pool
    gc.collect()

    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    if teacher_path.exists():
        bak = teacher_path.with_name(teacher_path.name + ".bak")
        try:
            teacher_path.replace(bak)
            print(f"[seed] backed up old teacher buffer → {bak.name}")
        except Exception:
            pass
    tmp = teacher_path.with_name(teacher_path.name + ".tmp")
    state = {"data": tb._data, "ptr": tb._ptr, "size": tb._size}
    with gzip.open(tmp, "wb", compresslevel=3) as f:
        pickle.dump(state, f, protocol=4)
    tmp.replace(teacher_path)

    print(f"[seed] wrote {teacher_path}  (size={tb._size:,})")
    print("[seed] DONE.")
    print("[seed] Next: restart training so it loads this teacher buffer.")
    print("[seed] Recommended: config.teacher_game_ratio is now 0.0 so "
          "self-play won't re-dilute the teacher channel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
