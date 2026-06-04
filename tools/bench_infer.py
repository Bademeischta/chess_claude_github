#!/usr/bin/env python3
"""
tools/bench_infer.py  —  self-play throughput across inference backends.

Drives ParallelMCTS.game_stream for a fixed wall-clock window with each
backend and reports:

  * forward-pass throughput (positions evaluated / second on the GPU — the
    quantity that scales with sims_per_move) — this is the load-bearing
    number for self-play speed and the one to compare across backends.
  * game-finalisation rate (records / second yielded by game_stream) —
    informational, naturally near 0 for any sensible window since a single
    game takes many seconds to complete.

Use the forward-pass number when choosing between backends — the
"completed games" metric used previously gave 0.x pos/s as a measurement
artefact whenever the window was shorter than one game's wall time.

    python tools/bench_infer.py [seconds]   (default 60)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch  # noqa: F401  (ensures CUDA init before timing)

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe
from model.network import build_model
from mcts.tree import ParallelMCTS


def _run(backend: str, seconds: float) -> tuple[float, float]:
    """Returns (forward_pos_per_s, finished_game_pos_per_s)."""
    CONFIG.inference_backend = backend
    # Bootstrap-style fast sims so the harness moves quickly.
    CONFIG.mcts_sims = min(50, CONFIG.mcts_sims)
    model = build_model(CONFIG, compile_model=False).eval()
    pm = ParallelMCTS(CONFIG, model, torch.device(CONFIG.device),
                      n_games=CONFIG.parallel_games)

    # Wrap engine.infer to count positions actually evaluated on the GPU —
    # the metric that scales with `sims_per_move` and is independent of
    # whether any individual game happens to finish inside the window.
    eval_counter = [0]
    real_infer = pm.engine.infer
    def _counting_infer(b, h):
        eval_counter[0] += int(b.shape[0])
        return real_infer(b, h)
    pm.engine.infer = _counting_infer

    stream = pm.game_stream(is_teacher=False)
    finished_pos = 0
    t0 = time.time()
    # Loop until window expires. game_stream yields once per completed game;
    # we count those for the informational metric and ignore the rest.
    while True:
        elapsed = time.time() - t0
        if elapsed >= seconds:
            break
        try:
            rec = next(stream)
            finished_pos += len(rec.board_tensors)
        except StopIteration:
            break
    dt = time.time() - t0
    return eval_counter[0] / dt, finished_pos / dt


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    update_config_from_dict(run_probe(write=False))

    backends = ["torch", "cudagraph", "onnx"]
    print(f"\nBenchmarking {len(backends)} backends for ~{seconds:.0f}s "
          f"(pool={CONFIG.parallel_games}, "
          f"sims={min(50, CONFIG.mcts_sims)})\n")
    print(f"  {'backend':<12} {'fwd pos/s':>12} {'fin pos/s':>12}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12}")

    results: dict[str, tuple[float, float]] = {}
    for be in backends:
        try:
            fwd, fin = _run(be, seconds)
            results[be] = (fwd, fin)
            print(f"  {be:<12} {fwd:>12.1f} {fin:>12.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {be:<12} FAILED: {type(e).__name__}: {e}")

    if "torch" in results and results:
        torch_fwd = results["torch"][0] or 1.0
        print()
        for be, (fwd, _) in results.items():
            if be == "torch":
                continue
            ratio = fwd / torch_fwd
            verdict = (f"{ratio:.2f}x torch — "
                       + ("WORTH switching." if ratio > 1.05
                          else "no real gain." if ratio > 0.95
                          else "slower; keep torch."))
            print(f"  {be:<10}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
