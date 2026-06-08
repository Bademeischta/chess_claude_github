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


# ── SumTree for PER ─────────────────────────────────────────────────────────

class SumTree:
    """
    A binary tree data structure where the parent node is the sum of its children.
    Enables O(log N) sampling and O(log N) priority updates.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # The tree has 2*capacity - 1 nodes.
        # Leaves are in the range [capacity-1, 2*capacity-2].
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def update(self, idx: int, priority: float) -> None:
        """Update priority at leaf index `idx` (0-based relative to leaves)."""
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate the change up to the root
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float) -> Tuple[int, float]:
        """
        Find the leaf index corresponding to value `v` in [0, sum(tree)].
        Returns (leaf_idx, priority).
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # Check if we reached a leaf
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return data_idx, self.tree[leaf_idx]

    @property
    def total_priority(self) -> float:
        return self.tree[0]


# ── Position record ────────────────────────────────────────────────────────

class PositionRecord:
    """In-memory replay record.

    Storage optimisations vs. the dense PyTorch tensors used at training time:
      * Board / history kept as float16 (set externally by the producer in
        ``ParallelMCTS.game_stream``) — 2 bytes/cell instead of 4.
      * Policy target stored sparse: an int16 array of nonzero action indices
        and a float32 array of values. The MCTS target has ≤ ~50 nonzeros
        out of 4672, so this compresses ~47×. The ``policy_target`` property
        reconstructs the dense (num_actions,) array on read for the dataset.
    """

    __slots__ = (
        "board_tensor",
        "history_tensor",
        "_pol_idx",       # int16 (k,) nonzero indices
        "_pol_val",       # float32 (k,) values
        "_pol_size",      # int — num_actions (typically 4672)
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
        # Sparse-encode policy target. The index domain (≤ num_actions = 4672)
        # fits in int16 unsigned interpretation but we still want negative-safe
        # casting → int32 only adds ~150 B/pos; int16 is fine since 4672 < 32768.
        nz = np.flatnonzero(policy_target)
        pol_size = int(policy_target.shape[0])
        # Sanity: indices must be in-range AND fit in int16 (we use int16 to
        # halve memory). 4672 < 32768 so this is a guaranteed property of the
        # AlphaZero action encoding — but assert anyway because a silent
        # truncation here corrupts every record going forward.
        if nz.size and (nz.max() >= pol_size or nz.max() >= 32768):
            raise ValueError(
                f"PositionRecord: policy_target index out of range "
                f"(max idx={int(nz.max())}, size={pol_size})"
            )
        self._pol_idx  = nz.astype(np.int16, copy=False)
        self._pol_val  = policy_target[nz].astype(np.float32, copy=False)
        self._pol_size = pol_size
        self.wdl_label      = wdl_label
        self.mcts_q         = mcts_q
        self.phase          = phase
        self.piece_count    = piece_count
        self.move_number    = move_number
        self.is_teacher     = is_teacher

    @property
    def policy_target(self) -> np.ndarray:
        """Reconstruct dense (num_actions,) float32 policy target on demand."""
        out = np.zeros(self._pol_size, dtype=np.float32)
        if self._pol_idx.size:
            idx32 = self._pol_idx.astype(np.int32, copy=False)
            # Defensive: catch any corruption that slipped past __init__
            # (e.g. accidental in-place mutation of _pol_idx).
            if idx32.max() >= self._pol_size or idx32.min() < 0:
                raise IndexError(
                    f"PositionRecord.policy_target: idx out of bounds "
                    f"(min={int(idx32.min())}, max={int(idx32.max())}, "
                    f"size={self._pol_size})"
                )
            out[idx32] = self._pol_val
        return out


# ── PER buffer ────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Ring-buffer with priority-based sampling using a SumTree.
    Enables O(log N) sampling and updates.
    """

    def __init__(
        self,
        capacity: int = 2_000_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        total_steps: int = 500_000,
        draw_priority_mult: float = 1.0,
        draw_priority_start_mult: float = 1.0,
        draw_priority_ramp_steps: int = 50_000,
    ) -> None:
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_end    = beta_end
        self.total_steps = total_steps
        # Final (steady-state) draw-down multiplier. The effective value used
        # for sampling ramps from `draw_priority_start_mult` to this over the
        # first `draw_priority_ramp_steps` training steps — see
        # `_current_draw_mult`. Set start == end (both 0.5) to disable the
        # ramp and get the legacy behaviour.
        self.draw_priority_mult       = float(draw_priority_mult)
        self.draw_priority_start_mult = float(draw_priority_start_mult)
        self.draw_priority_ramp_steps = int(max(1, draw_priority_ramp_steps))

        self._data: list[Optional[PositionRecord]] = [None] * capacity
        self.tree  = SumTree(capacity)
        self._ptr  = 0          # Write pointer
        self._size = 0          # Current fill
        self._step = 0          # Training steps seen (for β annealing)
        self._max_prio = 1.0    # Max priority seen (new entries get this)

    def __len__(self) -> int:
        return self._size

    def is_ready(self, min_size: int) -> bool:
        return self._size >= min_size

    def _current_draw_mult(self) -> float:
        """Linearly interpolate draw multiplier from start → end over ramp."""
        if self.draw_priority_ramp_steps <= 0:
            return self.draw_priority_mult
        frac = min(1.0, self._step / self.draw_priority_ramp_steps)
        return (self.draw_priority_start_mult
                + frac * (self.draw_priority_mult
                          - self.draw_priority_start_mult))

    def add(self, record: PositionRecord, td_error: Optional[float] = None) -> None:
        """Add a single position. If no td_error given, uses max priority."""
        prio = (abs(td_error) + 1e-6) ** self.alpha if td_error is not None \
               else self._max_prio

        self._data[self._ptr] = record
        self.tree.update(self._ptr, prio)

        self._max_prio = max(self._max_prio, prio)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

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
        Sample `batch_size` positions using the SumTree.

        Returns:
            (records, indices, weights)
            weights: importance-sampling correction, shape (B,)
        """
        if self._size == 0:
            raise RuntimeError("Cannot sample from empty buffer")

        indices = []
        priorities = []
        beta = self._current_beta()
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, p = self.tree.get_leaf(v)
            indices.append(idx)
            priorities.append(p)

        indices = np.array(indices, dtype=np.int64)
        records = [self._data[i] for i in indices]

        # Importance sampling weights: w = (N * P(i))^-beta
        probs = np.array(priorities) / self.tree.total_priority
        weights = (self._size * probs) ** (-beta)
        weights /= weights.max()  # Normalise
        weights = weights.astype(np.float32)

        return records, indices, weights

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        prios = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, prio in zip(indices, prios):
            self.tree.update(idx, prio)
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
            "tree":       self.tree.tree,
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
        self._data      = state["data"]
        self._ptr       = state["ptr"]
        self._size      = state["size"]
        self._step      = state["step"]
        self._max_prio  = state.get("max_prio", 1.0)

        if "tree" in state:
            self.tree.tree = state["tree"]
        elif "priorities" in state:
            # Migration from old flat-array format to SumTree
            print(f"[Buffer] Migrating priorities from legacy format...")
            old_prios = state["priorities"]
            for i in range(self._size):
                self.tree.update(i, float(old_prios[i]))


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
