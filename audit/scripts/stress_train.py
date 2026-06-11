"""Audit P10.3/P10.4/P4.6/P6: bounded self-play window, 1000-step from-scratch
training, resume test, step-time benchmark, GPU-util sampling.

ISOLATION: CONFIG.checkpoint_dir / tensorboard_dir point into audit/tmp/.
Never touches checkpoints/, runs/, replay_buffer/.
Run: python audit/scripts/stress_train.py
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if sys.platform == "win32":  # logger.py prints U+2192; console may be cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe

TMP = ROOT / "audit" / "tmp"
(TMP / "checkpoints").mkdir(parents=True, exist_ok=True)
(TMP / "runs").mkdir(parents=True, exist_ok=True)

derived = run_probe(write=False)
update_config_from_dict(derived)
# ── isolation + stress-test shaping ─────────────────────────────────────
CONFIG.checkpoint_dir = str(TMP / "checkpoints")
CONFIG.tensorboard_dir = str(TMP / "runs")
CONFIG.replay_start_training = 20_000
CONFIG.mcts_sims = 50            # phase-1 bootstrap budget (the historic regime)
CONFIG.mcts_sims_teacher = 50
CONFIG.checkpoint_every = 10_000_000   # disable cadence; we save manually
OUT: dict = {"config": {"batch_size": CONFIG.batch_size,
                        "parallel_games": CONFIG.parallel_games,
                        "sims": CONFIG.mcts_sims,
                        "precision": CONFIG.precision,
                        "policy_mid_channels": CONFIG.policy_mid_channels,
                        "num_res_blocks": CONFIG.num_res_blocks}}

from model.network import build_model  # noqa: E402
from training.replay_buffer import PrioritizedReplayBuffer, TeacherBuffer  # noqa: E402
from training.self_play import SelfPlayWorker  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from training.pbt import OpponentPool  # noqa: E402
from utils.logger import TrainingLogger  # noqa: E402
import chess_ext  # noqa: E402

device = torch.device(CONFIG.device)

# GPU utilisation sampler
util_samples: list[int] = []
_stop = threading.Event()


def _sampler():
    while not _stop.is_set():
        try:
            util_samples.append(torch.cuda.utilization())
        except Exception:
            pass
        time.sleep(0.25)


def util_stats(tag, samples):
    if samples:
        OUT[tag] = {"mean_pct": round(statistics.mean(samples), 1),
                    "max_pct": max(samples), "n": len(samples)}


torch.manual_seed(1234)
np.random.seed(1234)

model = build_model(CONFIG, compile_model=False)
replay = PrioritizedReplayBuffer(capacity=120_000, alpha=CONFIG.per_alpha,
                                 beta_start=CONFIG.per_beta_start,
                                 beta_end=CONFIG.per_beta_end,
                                 total_steps=CONFIG.total_steps)
teacher = TeacherBuffer(capacity=20_000)
worker = SelfPlayWorker(CONFIG, model, replay, teacher, device)
chess_ext.reset_stale_move_count()

# ── 1) bounded self-play window (pure generation, no training) ──────────
import pickle
cache = TMP / "stress_buffer.pkl"
if cache.exists():
    with open(cache, "rb") as f:
        cached = pickle.load(f)
    replay._data, replay._ptr, replay._size = cached["rd"], cached["rp"], cached["rs"]
    for i in range(replay._size):
        replay.tree.update(i, 1.0)
    teacher._data, teacher._ptr, teacher._size = cached["td"], cached["tp"], cached["ts"]
    OUT["selfplay_window"] = cached["stats"]
    print(f"[stress] reusing cached buffer: {len(replay)}/{len(teacher)}")
else:
    th = threading.Thread(target=_sampler, daemon=True); th.start()
    t0 = time.time()
    games = 0
    gen_target = 24_000
    while len(replay) + len(teacher) < gen_target:
        worker.generate_batch(CONFIG.parallel_games)
        games += CONFIG.parallel_games
        if time.time() - t0 > 1200:
            break
    gen_secs = time.time() - t0
    _stop.set(); th.join(timeout=2)
    total_pos = len(replay) + len(teacher)
    OUT["selfplay_window"] = {
        "games": games, "positions": total_pos,
        "secs": round(gen_secs, 1),
        "positions_per_s": round(total_pos / gen_secs, 1),
        "positions_per_hour": round(total_pos / gen_secs * 3600),
        "games_per_hour": round(games / gen_secs * 3600),
        "avg_positions_per_game": round(total_pos / max(games, 1), 1),
        "stale_moves": int(chess_ext.stale_move_count()),
    }
    util_stats("gpu_util_selfplay", util_samples)
    with open(cache, "wb") as f:
        pickle.dump({"rd": replay._data, "rp": replay._ptr, "rs": replay._size,
                     "td": teacher._data, "tp": teacher._ptr, "ts": teacher._size,
                     "stats": OUT["selfplay_window"]}, f, protocol=4)

# label distribution of generated data (draw share — P4-1 context)
labels = [r.wdl_label for r in replay._data[: len(replay)] if r is not None]
OUT["selfplay_window"]["label_histogram"] = {
    "win(1.0)": labels.count(1.0), "draw(0.5)": labels.count(0.5),
    "loss(0.0)": labels.count(0.0)}

# ── 2) 1000-step training from scratch ──────────────────────────────────
pool = OpponentPool(checkpoint_dir=CONFIG.checkpoint_dir)
logger = TrainingLogger(log_dir=CONFIG.tensorboard_dir, log_every=100)
trainer = Trainer(CONFIG, model, replay, teacher, pool, logger, device)

loss_at: dict = {}
step_times: list[float] = []
grad_norms: list[float] = []
nan_count = 0
util_samples = []; _stop.clear()
th = threading.Thread(target=_sampler, daemon=True); th.start()
ckpt100 = str(TMP / "checkpoints" / "stress_step100.pt")
post100_losses: list[float] = []

for i in range(1000):
    t0 = time.perf_counter()
    m = trainer.train_step()
    step_times.append(time.perf_counter() - t0)
    loss = m.get("loss/total", float("nan"))
    if loss != loss:
        nan_count += 1
    if trainer.global_step in (1, 100, 500, 1000) or i in (0, 99, 499, 999):
        loss_at[trainer.global_step] = round(loss, 4)
        with torch.no_grad():
            g = [p.grad.float().norm() ** 2 for p in model.parameters()
                 if p.grad is not None]
            if g:
                grad_norms.append(float(torch.stack(g).sum().sqrt()))
    if trainer.global_step == 100:
        trainer.save_checkpoint(ckpt100)
    if 100 < trainer.global_step <= 110:
        post100_losses.append(round(loss, 5))
_stop.set(); th.join(timeout=2)
util_stats("gpu_util_training", util_samples)

OUT["training_1000"] = {
    "loss_at": loss_at,
    "nan_steps": nan_count,
    "median_step_ms": round(statistics.median(step_times) * 1e3, 1),
    "steps_per_hour": round(3600 / statistics.median(step_times)),
    "grad_norm_samples": [round(g, 2) for g in grad_norms],
    "policy_value_aux_at_end": {
        k: round(v, 4) for k, v in trainer.train_step().items()
        if k.startswith("loss/")},
    "param_group_lrs_after_run": [g["lr"] for g in trainer.optimizer.param_groups],
    "post100_losses_original": post100_losses,
}

# ── 3) resume test: reload step-100 ckpt, repeat steps 101..110 ─────────
torch.manual_seed(1234)
np.random.seed(1234)
model2 = build_model(CONFIG, compile_model=False)
trainer2 = Trainer(CONFIG, model2, replay, teacher, pool, logger, device)
trainer2.load_checkpoint(ckpt100)
resumed_losses = []
for _ in range(10):
    m = trainer2.train_step()
    resumed_losses.append(round(m.get("loss/total", float("nan")), 5))
OUT["resume_test"] = {
    "original_steps_101_110": post100_losses,
    "resumed_steps_101_110": resumed_losses,
    "identical": post100_losses == resumed_losses,
    "note": "checkpoint stores no RNG states; PER sampling uses np.random -> "
            "exact reproduction not expected; magnitudes should match",
}

out = Path(__file__).with_name("stress_train.json")
out.write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print(json.dumps(OUT, indent=2, default=str))
print(f"-> {out}")
