"""
audit/scripts/profile_selfplay.py — find the real self-play per-move bottleneck.

Runs a short production-shaped self-play burst through ParallelMCTS.game_stream
under cProfile, then prints the functions with the highest CUMULATIVE and TOTAL
(self) time. This is what gates the "move per-move bookkeeping to C++" work
(lever C): we only rewrite what the profile proves is hot.

Run with the training STOPPED (it needs the GPU + the chess_ext .pyd):
    python audit/scripts/profile_selfplay.py            # 30 games, prod width
    python audit/scripts/profile_selfplay.py 60 128     # 60 games, 128 sims

Output: top-25 by cumulative time and top-25 by self time, plus a phase
breakdown (NN forward vs C++ driver vs Python glue) estimated from the
profile. A dump is written to audit/scripts/profile_selfplay.prof for
`snakeviz` / `pstats` inspection.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from config import CONFIG  # noqa: E402
from model.network import build_model  # noqa: E402
from engine.rules import init_tablebase, is_tablebase_available  # noqa: E402


def main() -> int:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sims    = int(sys.argv[2]) if len(sys.argv) > 2 else CONFIG.mcts_sims
    # Production width comes from system_probe at runtime (this script does
    # NOT run main.py's probe), so the config default (6) would otherwise
    # give a wildly unrepresentative profile. Force the real pool width.
    width   = int(sys.argv[3]) if len(sys.argv) > 3 else 160

    # Match production: tablebases on (the driver TB path is part of the loop).
    sz = CONFIG.syzygy_path or str(_ROOT / "syzygy")
    init_tablebase(sz)

    CONFIG.mcts_sims = sims
    CONFIG.parallel_games = width
    dev = torch.device(CONFIG.device if torch.cuda.is_available() else "cpu")
    model = build_model(CONFIG).to(dev).eval()

    from mcts.tree import ParallelMCTS, _USE_CPP
    pool = ParallelMCTS(CONFIG, model, dev)
    driver_on = pool._round_driver is not None
    print(f"[profile] width={pool.n} sims={sims} cpp={_USE_CPP} "
          f"driver={driver_on} tablebases={is_tablebase_available()} "
          f"device={dev}")

    stream = pool.game_stream(is_teacher=False)

    # Warm up (CUDA-graph capture, cuDNN autotune) OUTSIDE the profile.
    print("[profile] warming up (graph capture)...")
    for _ in range(2):
        next(stream)

    print(f"[profile] profiling {n_games} games...")
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    positions = 0
    for _ in range(n_games):
        rec = next(stream)
        positions += len(rec.move_numbers)
    pr.disable()
    wall = time.perf_counter() - t0

    pos_s = positions / wall
    print(f"\n[profile] {n_games} games | {positions} positions | "
          f"{wall:.1f}s | {pos_s:.0f} pos/s | "
          f"{n_games/wall*3600:.0f} games/h\n")

    out = _ROOT / "audit" / "scripts" / "profile_selfplay.prof"
    pr.dump_stats(str(out))

    def _report(sortkey: str, label: str, n: int = 25) -> None:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortkey)
        ps.print_stats(n)
        print(f"================ TOP {n} BY {label} ================")
        # Trim pstats' verbose path noise to the tail (func + times).
        for line in s.getvalue().splitlines():
            print(line)
        print()

    _report("cumulative", "CUMULATIVE TIME (includes callees)")
    _report("tottime", "SELF TIME (the actual hot code)")

    print(f"[profile] raw stats -> {out}")
    print("[profile] inspect interactively:  python -m pstats "
          f"{out}   (then: sort tottime / stats 30)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
