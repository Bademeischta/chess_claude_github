"""
utils/logger.py

TensorBoard + console logger for the training run.

Tracked metrics:
  - loss/total, loss/policy, loss/value, loss/teacher_kl, loss/aux
  - policy_entropy (moving average)
  - elo/current, elo/best_pool
  - mcts/avg_sims_per_move
  - throughput/positions_per_sec, throughput/games_per_hour
  - memory/vram_used_mb, memory/vram_peak_mb
  - training/lr, training/dirichlet_eps, training/alpha_v_global
  - buffer/replay_size, buffer/teacher_size
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch


class TrainingLogger:
    """Unified TensorBoard + console logger."""

    def __init__(
        self,
        log_dir: str = "runs",
        log_every: int = 100,
        use_tensorboard: bool = True,
    ) -> None:
        self.log_every = log_every
        self._writer  = None
        self._start   = time.time()
        self._step    = 0

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=log_dir)
                print(f"[Logger] TensorBoard logging → {log_dir}")
                print(f"         Run: tensorboard --logdir={log_dir}")
            except ImportError:
                print("[Logger] tensorboard not installed — console logging only.")

        self._metrics_buffer: dict = {}

    # ── Logging ───────────────────────────────────────────────────────────

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log a dict of {name: value} metrics."""
        step = step if step is not None else self._step
        self._step = step

        for k, v in metrics.items():
            self._metrics_buffer[k] = v
            if self._writer and isinstance(v, (int, float)):
                self._writer.add_scalar(k, float(v), global_step=step)

        if step % self.log_every == 0:
            self._print_summary(step)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a single scalar."""
        if self._writer:
            self._writer.add_scalar(name, value, global_step=step)

    def log_histogram(self, name: str, tensor: torch.Tensor, step: int) -> None:
        """Log a histogram (expensive — use sparingly)."""
        if self._writer:
            try:
                self._writer.add_histogram(name, tensor, global_step=step)
            except Exception:
                pass

    def log_text(self, name: str, text: str, step: int) -> None:
        if self._writer:
            self._writer.add_text(name, text, global_step=step)

    # ── Console output ────────────────────────────────────────────────────

    def _print_summary(self, step: int) -> None:
        m = self._metrics_buffer
        elapsed = time.time() - self._start
        h = int(elapsed // 3600)
        s = int(elapsed % 60)
        mi = int((elapsed % 3600) // 60)
        time_str = f"{h:02d}:{mi:02d}:{s:02d}"

        # Core metrics
        loss  = m.get("loss/total",  float("nan"))
        p_los = m.get("loss/policy", float("nan"))
        v_los = m.get("loss/value",  float("nan"))
        lr    = m.get("lr",          0.0)
        entr  = m.get("policy_entropy", float("nan"))
        pos_s = m.get("throughput/positions_per_sec", 0.0)
        vram  = m.get("memory/vram_used_mb", 0.0)
        # Trainer emits `replay_buf_size`; accept the TB-style key too.
        buf   = m.get("buffer/replay_size", m.get("replay_buf_size", 0))

        # ELO only exists after the first arena match — show "--" (not a
        # misleading 0) until it has actually been measured.
        elo = m.get("elo/current")
        elo_str = f"{elo:.0f}" if elo is not None else "--"
        h_str = f"{entr:.2f}" if entr == entr else "--"  # nan-safe

        line = (
            f"[{time_str}]  step={step:>7,}  "
            f"loss={loss:.4f}  (p={p_los:.4f} v={v_los:.4f})  "
            f"ELO={elo_str}  H={h_str}  "
            f"lr={lr:.2e}  pos/s={pos_s:.0f}  "
            f"VRAM={vram:.0f}MB  buf={buf:,}"
        )
        print(line, flush=True)

    # ── VRAM monitoring ───────────────────────────────────────────────────

    def get_vram_metrics(self) -> dict:
        if not torch.cuda.is_available():
            return {}
        used = torch.cuda.memory_allocated(0) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        return {
            "memory/vram_used_mb": used,
            "memory/vram_peak_mb": peak,
        }

    # ── ELO logging ───────────────────────────────────────────────────────

    def log_elo(self, current_elo: float, best_pool_elo: float, step: int) -> None:
        self.log({"elo/current": current_elo, "elo/best_pool": best_pool_elo}, step)

    # ── Throughput ────────────────────────────────────────────────────────

    def log_throughput(
        self,
        positions_per_sec: float,
        games_per_hour: float,
        step: int,
    ) -> None:
        self.log({
            "throughput/positions_per_sec": positions_per_sec,
            "throughput/games_per_hour":    games_per_hour,
        }, step)

    def close(self) -> None:
        if self._writer:
            self._writer.close()
