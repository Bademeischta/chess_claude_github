"""
utils/load_governor.py

Adaptive training throttle: when ANOTHER process starts using the GPU
heavily (e.g. the user launches a game), training/self-play automatically
backs off to a configurable duty cycle, and ramps back to full speed once
the foreign load disappears.

Detection signal: "foreign VRAM" = total GPU memory in use (nvidia-smi)
minus our own process' reserved memory (torch). At startup a baseline is
recorded (desktop/browser/DWM typically hold a few hundred MB); a game
launching pushes foreign VRAM **above the baseline** by a GB or more within
seconds. This needs no extra dependencies (no pynvml) and is robust against
our own allocation growth, which is subtracted out.

Throttle mechanics: the training loop calls ``maybe_throttle(work_seconds)``
once per outer iteration (one self-play batch + its train steps). While
foreign load is present, the governor sleeps so that our duty cycle is
roughly ``throttle_duty`` (e.g. 0.25 → work 25 % of wall time). Ramp-up
uses hysteresis: ONE dirty sample throttles immediately (fast reaction when
a game starts), but ``clear_samples_to_resume`` consecutive clean samples
are required to resume (no flapping on loading screens).

The check itself costs one ``nvidia-smi`` subprocess every
``check_every_s`` seconds (cached in between) — microseconds of overhead
per training batch. All failures degrade to "never throttle": a broken
nvidia-smi must not stall training.
"""

from __future__ import annotations

import subprocess
import time


class LoadGovernor:
    def __init__(
        self,
        enabled: bool = True,
        foreign_vram_mb: float = 1000.0,
        duty: float = 0.25,
        check_every_s: float = 5.0,
        clear_samples_to_resume: int = 3,
        max_sleep_s: float = 60.0,
        _sampler=None,          # injectable for tests
        _torch_reserved=None,   # injectable for tests
    ) -> None:
        self.enabled        = bool(enabled)
        self.threshold_mb   = float(foreign_vram_mb)
        self.duty           = min(1.0, max(0.05, float(duty)))
        self.check_every_s  = float(check_every_s)
        self.clear_needed   = int(max(1, clear_samples_to_resume))
        self.max_sleep_s    = float(max_sleep_s)

        self._sampler        = _sampler or self._nvidia_smi_used_mb
        self._torch_reserved = _torch_reserved or self._torch_reserved_mb
        self._baseline_mb: float | None = None
        self._last_check     = 0.0
        self._throttled      = False
        self._clear_streak   = 0
        self._was_announced  = False

    # ── Sampling ──────────────────────────────────────────────────────────

    @staticmethod
    def _nvidia_smi_used_mb() -> float | None:
        """Total VRAM in use across ALL processes (MB), or None on failure."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode != 0:
                return None
            return float(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    @staticmethod
    def _torch_reserved_mb() -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_reserved() / (1024 ** 2)
        except Exception:
            pass
        return 0.0

    def _foreign_mb(self) -> float | None:
        total = self._sampler()
        if total is None:
            return None
        return max(0.0, total - self._torch_reserved())

    # ── State machine ─────────────────────────────────────────────────────

    def _sample(self) -> None:
        foreign = self._foreign_mb()
        if foreign is None:
            # Sampling broken → fail open (never throttle).
            self._throttled = False
            return
        if self._baseline_mb is None:
            self._baseline_mb = foreign
            return
        over = foreign - self._baseline_mb > self.threshold_mb
        if over:
            self._clear_streak = 0
            self._throttled = True
        else:
            # Track the baseline downward only (apps closing shrink it; a
            # creeping upward baseline would mask a slowly-loading game).
            self._baseline_mb = min(self._baseline_mb, foreign)
            self._clear_streak += 1
            if self._throttled and self._clear_streak >= self.clear_needed:
                self._throttled = False

    def is_throttled(self) -> bool:
        """Cached check — samples at most once per check_every_s."""
        if not self.enabled:
            return False
        now = time.monotonic()
        if now - self._last_check >= self.check_every_s:
            self._last_check = now
            self._sample()
        return self._throttled

    def maybe_throttle(self, work_seconds: float) -> float:
        """Call after each unit of work (one batch+train cycle).

        If foreign GPU load is present, sleeps so our duty cycle is roughly
        ``duty``: sleep = work × (1−duty)/duty, capped at max_sleep_s.
        Returns the seconds slept (0.0 when running at full speed).
        """
        if not self.is_throttled():
            if self._was_announced:
                print("[Governor] Foreign GPU load gone — resuming full "
                      "speed.", flush=True)
                self._was_announced = False
            return 0.0
        if not self._was_announced:
            print(f"[Governor] Foreign GPU load detected (> "
                  f"{self.threshold_mb:.0f} MB over baseline) — throttling "
                  f"to {self.duty:.0%} duty cycle. Training resumes "
                  f"automatically.", flush=True)
            self._was_announced = True
        sleep_s = min(self.max_sleep_s,
                      max(0.0, work_seconds) * (1.0 - self.duty) / self.duty)
        if sleep_s > 0:
            time.sleep(sleep_s)
        return sleep_s
