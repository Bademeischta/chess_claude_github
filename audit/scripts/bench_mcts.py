"""Audit P3.9/P6.2: inference latency sweep + MCTS sims/s + time split.

Read-only (loads latest.pt weights; no training state written).
Run: python audit/scripts/bench_mcts.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe
from utils.ckpt_arch import match_arch, latest_checkpoint

derived = run_probe(write=False)
update_config_from_dict(derived)
ckpt = latest_checkpoint(str(ROOT / "checkpoints"))
match_arch(CONFIG, ckpt)

from model.network import build_model  # noqa: E402
from model.inference import (  # noqa: E402
    TorchInferenceEngine, CudaGraphInferenceEngine,
)
from engine.board import Board  # noqa: E402
from mcts.tree import MCTSTree, ParallelMCTS  # noqa: E402
import chess_ext  # noqa: E402

device = torch.device(CONFIG.device)
model = build_model(CONFIG, compile_model=False)
state = torch.load(ckpt, map_location=device, weights_only=True)
base = model._orig_mod if hasattr(model, "_orig_mod") else model
base.load_state_dict(state["model"], strict=False)
model.eval()
print(f"loaded {ckpt} step={state.get('step')}")

dt = torch.bfloat16 if CONFIG.precision == "bf16" else torch.float32
OUT: dict = {"checkpoint_step": state.get("step"),
             "parallel_games": CONFIG.parallel_games,
             "mcts_sims": CONFIG.mcts_sims}


def timed_infer(engine, B, iters=60, warmup=10):
    b = torch.randn(B, 21, 8, 8, device=device, dtype=dt)
    h = torch.randn(B, CONFIG.gru_history_len, 21, 8, 8, device=device, dtype=dt)
    with torch.inference_mode():
        for _ in range(warmup):
            engine.infer(b, h)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            engine.infer(b, h)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    return {"batch": B, "median_ms": round(med * 1e3, 3),
            "positions_per_s": round(B / med)}


# ── eager latency sweep ──────────────────────────────────────────────────
eng_eager = TorchInferenceEngine(model)
OUT["eager_sweep"] = [timed_infer(eng_eager, B)
                      for B in (1, 8, 32, 128, 256, 512, 1024)]

# ── CUDA-graph engine at pool width ─────────────────────────────────────
try:
    eng_cg = CudaGraphInferenceEngine(model, CONFIG, device,
                                      CONFIG.parallel_games, dt)
    OUT["cudagraph_at_pool_width"] = timed_infer(eng_cg, CONFIG.parallel_games)
    OUT["cudagraph_batch1_padded"] = timed_infer(eng_cg, 1)
except Exception as e:  # noqa: BLE001
    OUT["cudagraph_error"] = f"{type(e).__name__}: {e}"

# ── single-tree MCTS sims/s ─────────────────────────────────────────────
chess_ext.reset_stale_move_count()


def bench_single_tree(engine, label, batched, sims=256):
    tree = MCTSTree(CONFIG, model, device, engine=engine)
    board = Board.startpos()
    tree.reset(board)
    t0 = time.perf_counter()
    if batched:
        done = tree.run_batched_sims(sims, batch_size=CONFIG.parallel_games)
    else:
        for _ in range(sims):
            tree.run_one_simulation()
        done = sims
    dtm = time.perf_counter() - t0
    OUT[label] = {"sims": done, "secs": round(dtm, 3),
                  "sims_per_s": round(done / dtm),
                  "ms_per_128sim_move": round(dtm / done * 128 * 1e3, 1),
                  "ms_per_800sim_move": round(dtm / done * 800 * 1e3, 1)}


bench_single_tree(eng_eager, "single_tree_unbatched_eager", batched=False)
bench_single_tree(eng_eager, "single_tree_batched_eager", batched=True)
if "cudagraph_error" not in OUT:
    bench_single_tree(eng_cg, "single_tree_batched_cudagraph", batched=True)

# ── ParallelMCTS pool: production self-play shape, with time split ──────
pool = ParallelMCTS(CONFIG, model, device)  # builds its own engine per config
n = pool.n
boards = [pool._fresh_board() for _ in range(n)]
active = list(range(n))
pool._batch_begin_search(active, boards, fresh=set(active))

t_select = t_infer = t_finish = 0.0
rounds = CONFIG.mcts_sims
t_all0 = time.perf_counter()
for _ in range(rounds):
    pend = []
    t0 = time.perf_counter()
    for i in active:
        p = pool.trees[i].select_leaf_only()
        if p is not None:
            pend.append((i, p))
    t_select += time.perf_counter() - t0
    if not pend:
        continue
    t0 = time.perf_counter()
    values, logits = pool._evaluate_batch(pend)
    t_infer += time.perf_counter() - t0
    t0 = time.perf_counter()
    for k, (i, p) in enumerate(pend):
        pool.trees[i].apply_eval(p, float(values[k]), logits[k])
    t_finish += time.perf_counter() - t0
t_all = time.perf_counter() - t_all0
sims_done = rounds * n
OUT["pool_move_timesplit"] = {
    "n_trees": n, "rounds": rounds, "total_sims": sims_done,
    "wall_s": round(t_all, 3),
    "sims_per_s": round(sims_done / t_all),
    "select_pct": round(t_select / t_all * 100, 1),
    "infer_pct": round(t_infer / t_all * 100, 1),
    "expand_backup_pct": round(t_finish / t_all * 100, 1),
    "ms_per_move_all_trees": round(t_all * 1e3, 1),
    "positions_per_hour_at_this_rate": round(n * 3600 / t_all),
}
OUT["stale_moves_after_bench"] = int(chess_ext.stale_move_count())

# ── MCTS determinism probe ──────────────────────────────────────────────
def visit_dist():
    tree = MCTSTree(CONFIG, model, device, engine=eng_eager)
    tree.reset(Board.startpos())
    tree._expand_root()
    for _ in range(100):
        tree.run_one_simulation()
    _, pt = tree.select_move(1.0)
    return pt

d1, d2 = visit_dist(), visit_dist()
OUT["determinism"] = {"same_visit_dist_without_seeding":
                      bool(np.allclose(d1, d2))}

out = Path(__file__).with_name("bench_mcts.json")
out.write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print(json.dumps(OUT, indent=2, default=str))
print(f"-> {out}")
