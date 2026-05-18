#!/usr/bin/env python3
"""
tools/bench_infer.py  —  self-play throughput: Torch vs ONNX backend.

Drives ParallelMCTS.game_stream for a fixed wall-clock window with each
backend and reports positions/s. Use this to decide whether flipping
config.inference_backend to "onnx" is actually a win on this machine
(only accept ONNX as default if parity passes AND pos/s strictly improves).

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


def _run(backend: str, seconds: float) -> float:
    CONFIG.inference_backend = backend
    # Bootstrap-style fast sims so the harness moves quickly.
    CONFIG.mcts_sims = min(50, CONFIG.mcts_sims)
    model = build_model(CONFIG, compile_model=False).eval()
    pm = ParallelMCTS(CONFIG, model, torch.device(CONFIG.device),
                      n_games=CONFIG.parallel_games)
    stream = pm.game_stream(is_teacher=False)

    positions = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        rec = next(stream)
        positions += len(rec.board_tensors)
    dt = time.time() - t0
    return positions / dt


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    update_config_from_dict(run_probe(write=False))

    print(f"\nBenchmarking each backend for ~{seconds:.0f}s "
          f"(pool={CONFIG.parallel_games}, sims={min(50, CONFIG.mcts_sims)})\n")
    torch_ps = _run("torch", seconds)
    print(f"  torch : {torch_ps:6.1f} pos/s")
    onnx_ps = _run("onnx", seconds)
    print(f"  onnx  : {onnx_ps:6.1f} pos/s")

    if onnx_ps > torch_ps:
        print(f"\nONNX is {onnx_ps / torch_ps:.2f}x faster — "
              f"consider config.inference_backend = 'onnx'.")
    else:
        print(f"\nONNX is NOT faster ({onnx_ps / torch_ps:.2f}x) — "
              f"keep 'torch'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
