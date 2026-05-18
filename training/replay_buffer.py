"""
training/replay_buffer.py

Prioritized Experience Replay (PER) buffer + dedicated teacher buffer.

PER implementation:
  - Priority ∝ |TD-error|^α  (α=0.6)
  - Importance-sampling correction β annealed from 0.4 → 1.0
  - Ring-buffer backed by NumPy arrays for memory efficiency
  - Separate `TeacherBuffer` for 800-sim self-distillation data

Each stored position is a dict-like PositionRecord containing:
  - board_tensor:    (21, 8, 8) float32
  - history_tensor:  (T, 21, 8, 8) float32
  - policy_target:   (num_actions,) float32
  - wdl_label:       float  (1.0=win, 0.5=draw, 0.0=loss from current side's view)
  - phase:           int    (0=opening, 1=mid, 2=endgame)
  - piece_count:     int
  - move_number:     int
  - is_teacher:      bool
"""

from __future__ import annotations

import os
import math
import pickle
import numpy as np
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# ── Position record ────────────────────────────────────────────────────────

class PositionRecord:
    __slots__ = (
        "board_tensor",
        "history_tensor",
        "policy_target",
        "wdl_label",
        "mcts_q",
        "phase",
        "piece_count",
        "move_number",
        "is_teacher",
    )

    def __init__(
        self,
        board_tensor: np.ndarray,    # (21, 8, 8)
        history_tensor: np.ndarray,  # (T, 21, 8, 8)
        policy_target: np.ndarray,   # (4672,)
        wdl_label: float,
        phase: int,
        piece_count: int,
        move_number: int,
        is_teacher: bool = False,
        mcts_q: float = 0.0,         # MCTS root Q in [-1,1] (TD bootstrap)
    ) -> None:
        self.board_tensor   = board_tensor
        self.history_tensor = history_tensor
        self.policy_target  = policy_target
        self.wdl_label      = wdl_label
        self.mcts_q         = mcts_q
        self.phase          = phase
        self.piece_count    = piece_count
        self.move_number    = move_number
        self.is_teacher     = is_teacher


# ── PER buffer ────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Ring-buffer with priority-based sampling.

    Priorities are stored as a flat NumPy array so that the sum-tree
    bookkeeping (optional) can be added later.  For now we use a simple
    proportional scheme: O(N) sampling, which is fast enough for 2M positions
    when sampled in a batch via `np.random.choice`.
    """

    def __init__(
        self,
        capacity: int = 2_000_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        total_steps: int = 500_000,
    ) -> None:
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_end    = beta_end
        self.total_steps = total_steps

        self._data:      list[Optional[PositionRecord]] = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._ptr        = 0          # Write pointer
        self._size       = 0          # Current fill
        self._step       = 0          # Training steps seen (for β annealing)
        self._max_prio   = 1.0        # Max priority seen (new entries get this)

    def __len__(self) -> int:
        return self._size

    def is_ready(self, min_size: int) -> bool:
        return self._size >= min_size

    def add(self, record: PositionRecord, td_error: Optional[float] = None) -> None:
        """Add a single position.  If no td_error given, uses max priority."""
        prio = (abs(td_error) + 1e-6) ** self.alpha if td_error is not None \
               else self._max_prio
        self._data[self._ptr]       = record
        self._priorities[self._ptr] = prio
        self._max_prio              = max(self._max_prio, prio)
        self._ptr   = (self._ptr + 1) % self.capacity
        self._size  = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        records: List[PositionRecord],
        td_errors: Optional[np.ndarray] = None,
    ) -> None:
        for i, rec in enumerate(records):
            err = float(td_errors[i]) if td_errors is not None else None
            self.add(rec, err)

    def sample(
        self, batch_size: int
    ) -> Tuple[List[PositionRecord], np.ndarray, np.ndarray]:
        """
        Sample `batch_size` positions.

        Returns:
            (records, indices, weights)
            weights: importance-sampling correction, shape (B,)
        """
        if self._size == 0:
            raise RuntimeError("Cannot sample from empty buffer")

        beta = self._current_beta()
        prios = self._priorities[: self._size]
        probs = prios / prios.sum()

        indices = np.random.choice(self._size, size=batch_size,
                                   replace=True, p=probs)
        records = [self._data[i] for i in indices]

        # IS weights
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalise
        weights = weights.astype(np.float32)

        return records, indices, weights

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        prios = (np.abs(td_errors) + 1e-6) ** self.alpha
        self._priorities[indices] = prios.astype(np.float32)
        self._max_prio = max(self._max_prio, float(prios.max()))

    def step(self) -> None:
        """Call once per training step to advance β annealing."""
        self._step += 1

    def _current_beta(self) -> float:
        frac = min(1.0, self._step / self.total_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        state = {
            "data":       self._data,
            "priorities": self._priorities,
            "ptr":        self._ptr,
            "size":       self._size,
            "step":       self._step,
            "max_prio":   self._max_prio,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=4)

    def load(self, path: str) -> None:
        # SECURITY: pickle.load executes arbitrary code embedded in the file.
        # Only ever load buffer dumps this process wrote itself / from a
        # trusted location — never an untrusted or downloaded .pkl.
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._data       = state["data"]
        self._priorities = state["priorities"]
        self._ptr        = state["ptr"]
        self._size       = state["size"]
        self._step       = state["step"]
        self._max_prio   = state["max_prio"]


# ── Teacher buffer ────────────────────────────────────────────────────────

class TeacherBuffer:
    """
    Dedicated ring-buffer for teacher (800-sim) rollout data.
    Same API as PrioritizedReplayBuffer but simpler — uniform sampling.
    """

    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = capacity
        self._data: list[Optional[PositionRecord]] = [None] * capacity
        self._ptr  = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, record: PositionRecord) -> None:
        self._data[self._ptr] = record
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, records: List[PositionRecord]) -> None:
        for rec in records:
            self.add(rec)

    def sample(self, batch_size: int) -> List[PositionRecord]:
        """Uniform random sampling from teacher buffer."""
        if self._size == 0:
            return []
        n = min(batch_size, self._size)
        indices = np.random.choice(self._size, size=n, replace=True)
        return [self._data[i] for i in indices]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"data": self._data, "ptr": self._ptr, "size": self._size}, f, protocol=4)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._data = state["data"]
        self._ptr  = state["ptr"]
        self._size = state["size"]


# ── PyTorch Dataset wrapper ───────────────────────────────────────────────

class ReplayDataset(Dataset):
    """
    Wraps a list of PositionRecord objects as a PyTorch Dataset.
    Used to feed the DataLoader in the training loop.
    """

    def __init__(
        self,
        records: List[PositionRecord],
        weights: Optional[np.ndarray] = None,
        precision: str = "bf16",
    ) -> None:
        self.records   = records
        self.weights   = weights
        self.precision = precision

    def __len__(self) -> int:
        return len(self.records)

    def _dtype(self) -> torch.dtype:
        if self.precision == "bf16":   return torch.bfloat16
        elif self.precision == "fp16": return torch.float16
        return torch.float32

    def __getitem__(self, idx: int) -> dict:
        rec  = self.records[idx]
        dt   = self._dtype()
        item = {
            "board":   torch.from_numpy(rec.board_tensor).to(dt),
            "history": torch.from_numpy(rec.history_tensor).to(dt),
            "policy":  torch.from_numpy(rec.policy_target).float(),  # Keep as fp32
            "wdl":     torch.tensor(rec.wdl_label, dtype=torch.float32),
            "mcts_q":  torch.tensor(rec.mcts_q, dtype=torch.float32),
            "phase":   torch.tensor(rec.phase, dtype=torch.long),
            "piece_count": torch.tensor(rec.piece_count, dtype=torch.long),
            "move_number": torch.tensor(rec.move_number, dtype=torch.long),
            "is_teacher":  torch.tensor(rec.is_teacher, dtype=torch.bool),
            "weight":  torch.tensor(
                self.weights[idx] if self.weights is not None else 1.0,
                dtype=torch.float32,
            ),
        }
        return item


def collate_records(
    records: List[PositionRecord],
    teacher_records: Optional[List[PositionRecord]],
    teacher_ratio: float = 0.30,
    is_weights: Optional[np.ndarray] = None,
    precision: str = "bf16",
) -> Tuple[ReplayDataset, "ReplayDataset | None"]:
    """
    Create mixed ReplayDataset (70 % replay + 30 % teacher) for one training batch.
    """
    dataset = ReplayDataset(records, weights=is_weights, precision=precision)
    teacher_dataset = None
    if teacher_records:
        teacher_dataset = ReplayDataset(teacher_records, precision=precision)
    return dataset, teacher_dataset
