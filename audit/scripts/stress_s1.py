"""
stress_s1.py — Repro harness for the S1 native crash (audit 2026-07-02).

Drives the exact production path that crashes (ParallelMCTS.game_stream →
RoundDriver.select, tree.py:1153) with a STUB inference engine instead of
the GPU network: random policy logits + zero values. The NN quality is
irrelevant to the crash class (heap corruption in the C++ tree lifetime
management), so removing the GPU forward multiplies the native
select/expand/backup/advance_root/new_root churn per wall-clock second by
~50-100x versus production. Short games (small move cap) additionally
maximise the new_root() full-tree-free frequency.

Usage:
    python audit/scripts/stress_s1.py [--minutes 30] [--trees 64]
        [--sims 24] [--threads 1] [--max-moves 60] [--seed 0]

Run against an ASan build (CHESS_AI_ASAN=1 python setup_ext.py build_ext
--inplace) with clang_rt.asan_dynamic-x86_64.dll on PATH: any
use-after-free / double-free / heap overflow inside chess_ext aborts with
a symbolised report at the faulting instruction. Against the production
build it is still a crash reproducer (0xC0000005 under a debugger).

Progress + counters go to stdout; on a clean run the script prints a
summary and exits 0.
"""
from __future__ import annotations

import argparse
import dataclasses
import random
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

# Windows cp1252 console guard (project convention).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os  # noqa: E402

# ASan runtime (clang_rt.asan_dynamic-x86_64.dll): Python >= 3.8 resolves
# extension dependencies via add_dll_directory, not PATH. Look next to the
# newest MSVC toolset if present; harmless no-op for a production build.
_msvc_root = Path(r"C:\Program Files (x86)\Microsoft Visual Studio")
for _cand in sorted(_msvc_root.glob(r"*\BuildTools\VC\Tools\MSVC\*")):
    _bin = _cand / "bin" / "Hostx64" / "x64"
    if (_bin / "clang_rt.asan_dynamic-x86_64.dll").exists():
        os.add_dll_directory(str(_bin))

import numpy as np  # noqa: E402
import torch        # noqa: E402

from config import CONFIG  # noqa: E402


class StubEngine:
    """Random-policy inference stub with the InferenceEngine interface.

    Returns CPU tensors shaped like the real engine's output. Random logits
    make the search visit realistic-breadth trees; value 0 keeps games long
    enough to exercise advance_root chains, the move cap bounds them.
    """

    def __init__(self, num_actions: int, seed: int = 0) -> None:
        self.num_actions = num_actions
        self._gen = torch.Generator().manual_seed(seed)

    def infer(self, board_t: torch.Tensor, hist_t: torch.Tensor):
        b = board_t.shape[0]
        policy = torch.randn((b, self.num_actions), generator=self._gen)
        values = torch.zeros((b,))
        return policy, values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0,
                    help="wall-clock budget (default 30)")
    ap.add_argument("--trees", type=int, default=64,
                    help="parallel trees (default 64)")
    ap.add_argument("--sims", type=int, default=24,
                    help="MCTS sims per move (default 24 — small: more "
                         "advance/select cycles per second)")
    ap.add_argument("--threads", type=int, default=1,
                    help="cpp_round_threads (default 1 = the serialised "
                         "production setting the crash survives)")
    ap.add_argument("--max-moves", type=int, default=60,
                    help="move cap (default 60 — short games = frequent "
                         "full-tree frees via new_root)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = dataclasses.replace(
        CONFIG,
        device="cpu",
        precision="fp32",
        inference_backend="torch",
        parallel_games=args.trees,
        mcts_sims=args.sims,
        mcts_sims_teacher=args.sims * 2,   # DCA ceiling
        mcts_sims_critical=args.sims * 2,
        cpp_round_threads=args.threads,
        cpp_round_driver=True,
        max_game_moves=args.max_moves,
        random_opening_plies=6,
        endgame_curriculum_fraction=0.0,   # no Syzygy dependency
        syzygy_path="",
        gru_history_len=CONFIG.gru_history_len,
        resign_q=0.55, resign_streak=6, resign_disable_fraction=0.15,
    )

    import chess_ext  # noqa: F401 — fail fast if the C++ backend is absent
    from mcts.tree import ParallelMCTS
    from model.network import build_model

    print(f"[stress_s1] backend=chess_ext trees={args.trees} "
          f"sims={args.sims} threads={args.threads} "
          f"cap={args.max_moves} budget={args.minutes:.0f}min", flush=True)

    # Tiny real model only to satisfy constructor signatures; every forward
    # goes through the stub engine below.
    tiny = dataclasses.replace(cfg, num_res_blocks=1, num_channels=16)
    model = build_model(tiny, compile_model=False).eval()

    pool = ParallelMCTS(cfg, model, torch.device("cpu"), n_games=args.trees)
    stub = StubEngine(cfg.num_actions, seed=args.seed)
    pool.engine = stub
    for t in pool.trees:
        t.engine = stub
    assert pool._round_driver is not None, \
        "RoundDriver unavailable — this harness must exercise the C++ path"

    t0 = time.time()
    deadline = t0 + args.minutes * 60.0
    games = 0
    positions = 0
    last_report = t0

    stream = pool.game_stream()
    for rec in stream:
        games += 1
        positions += len(rec.move_numbers)
        now = time.time()
        if now - last_report >= 30.0:
            stale = int(chess_ext.stale_move_count())
            print(f"[stress_s1] t={now - t0:7.0f}s games={games:6d} "
                  f"pos={positions:8d} stale={stale} "
                  f"({positions / (now - t0):.0f} pos/s)", flush=True)
            last_report = now
            if stale:
                print("[stress_s1] WARNING: stale_move_count nonzero — "
                      "native corruption indicator fired!", flush=True)
        if now >= deadline:
            break

    dt = time.time() - t0
    print(f"[stress_s1] CLEAN: {games} games / {positions} positions in "
          f"{dt:.0f}s ({positions / dt:.0f} pos/s), "
          f"stale_move_count={int(chess_ext.stale_move_count())}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
