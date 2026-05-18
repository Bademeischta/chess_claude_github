"""
training/pbt.py

Opponent Pool Manager (replaces Population-Based Training).

Maintains a pool of the top-K checkpoints ranked by ELO.
Self-play: 70% vs. latest checkpoint, 30% vs. a random pool member.

Arena evaluation:
  - 200 games every 10K training steps
  - Current network vs. best pool member
  - ELO update after each arena
  - If current net beats best by > ELO_PROMOTION_THRESHOLD: promotes to pool
"""

from __future__ import annotations

import os
import copy
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from utils.elo import ELOSystem


ELO_PROMOTION_THRESHOLD = 10.0  # Minimum ELO gain required for promotion
ARENA_WIN_THRESHOLD     = 0.55  # Win rate (not draws) to trigger promotion


class CheckpointEntry:
    """Metadata for a single pool checkpoint."""

    __slots__ = ("path", "elo", "step", "player_id")

    def __init__(self, path: str, elo: float, step: int, player_id: str) -> None:
        self.path      = path
        self.elo       = elo
        self.step      = step
        self.player_id = player_id

    def __repr__(self) -> str:
        return f"CheckpointEntry(elo={self.elo:.0f}, step={self.step}, id={self.player_id})"


class OpponentPool:
    """
    Manages the top-K checkpoints and ELO ratings.

    Does NOT run arena matches itself — that is done in trainer.py.
    This class handles checkpoint storage, rotation, and ELO tracking.
    """

    def __init__(
        self,
        pool_size: int = 5,
        checkpoint_dir: str = "checkpoints",
        initial_elo: float = 1000.0,
    ) -> None:
        self.pool_size      = pool_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.elo_system = ELOSystem(initial_elo=initial_elo, k=32.0)
        self._pool: List[CheckpointEntry] = []
        self._current_id: str = "current_v0"
        self._step: int = 0

    # ── Pool management ───────────────────────────────────────────────────

    def register_current(
        self,
        model: nn.Module,
        step: int,
        elo: Optional[float] = None,
    ) -> str:
        """Save current model weights and register it as a new pool candidate."""
        player_id = f"net_step{step}"
        path = str(self.checkpoint_dir / f"{player_id}.pt")

        # Save only weights (no optimizer state) for pool entries
        state = model.state_dict() if not hasattr(model, "_orig_mod") \
                else model._orig_mod.state_dict()
        torch.save(state, path)

        if elo is None:
            elo = self.elo_system.get_elo(self._current_id)
        self.elo_system.register(player_id, elo)

        entry = CheckpointEntry(path, elo, step, player_id)
        self._pool.append(entry)
        self._pool.sort(key=lambda e: e.elo, reverse=True)

        # Keep only top-K entries; remove files for evicted checkpoints
        while len(self._pool) > self.pool_size:
            evicted = self._pool.pop()
            try:
                os.remove(evicted.path)
            except FileNotFoundError:
                pass

        self._current_id = player_id
        self._step = step
        return player_id

    def best_opponent_path(self) -> Optional[str]:
        """Return the file path of the strongest checkpoint in the pool."""
        if not self._pool:
            return None
        return self._pool[0].path

    def random_opponent_path(self) -> Optional[str]:
        """Sample a random checkpoint from the pool (uniform)."""
        if not self._pool:
            return None
        return random.choice(self._pool).path

    def should_play_pool(self, pool_ratio: float = 0.30) -> bool:
        """True if this game should be played against a pool opponent."""
        return len(self._pool) >= 1 and random.random() < pool_ratio

    def get_opponent_path(self, pool_ratio: float = 0.30) -> Optional[str]:
        """
        Returns the path of the opponent for this game:
          - 30% probability: random pool checkpoint
          - 70%: None (meaning play against current model itself)
        """
        if self.should_play_pool(pool_ratio):
            return self.random_opponent_path()
        return None

    # ── ELO management ───────────────────────────────────────────────────

    def update_elo(
        self,
        player_id: str,
        opponent_id: str,
        wins: int,
        draws: int,
        losses: int,
        step: int,
    ) -> Tuple[float, float]:
        """Update ELO from an arena series and return (new_elo_a, new_elo_b)."""
        self.elo_system.register(player_id)
        self.elo_system.register(opponent_id)
        new_a, new_b = self.elo_system.update_from_series(
            player_id, opponent_id, wins, draws, losses, step
        )
        # Update pool entry ELO
        for entry in self._pool:
            if entry.player_id == player_id:
                entry.elo = new_a
            if entry.player_id == opponent_id:
                entry.elo = new_b
        # Re-sort
        self._pool.sort(key=lambda e: e.elo, reverse=True)
        return new_a, new_b

    def current_elo(self) -> float:
        return self.elo_system.get_elo(self._current_id)

    def best_pool_elo(self) -> float:
        return self._pool[0].elo if self._pool else 1000.0

    # ── Persistence ───────────────────────────────────────────────────────

    def save_state(self, path: Optional[str] = None) -> None:
        path = path or str(self.checkpoint_dir / "pool_state.json")
        state = {
            "current_id": self._current_id,
            "step":       self._step,
            "pool": [
                {
                    "path": e.path,
                    "elo":  e.elo,
                    "step": e.step,
                    "player_id": e.player_id,
                }
                for e in self._pool
            ],
            "elo_ratings": self.elo_system.all_ratings(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Optional[str] = None) -> None:
        path = path or str(self.checkpoint_dir / "pool_state.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            state = json.load(f)
        self._current_id = state.get("current_id", self._current_id)
        self._step       = state.get("step", 0)
        self._pool = []
        for e in state.get("pool", []):
            if os.path.exists(e["path"]):
                self._pool.append(
                    CheckpointEntry(e["path"], e["elo"], e["step"], e["player_id"])
                )
        for pid, elo in state.get("elo_ratings", {}).items():
            self.elo_system.register(pid, elo)

    # ── Model loading ────────────────────────────────────────────────────

    def load_model_from_path(
        self,
        model: nn.Module,
        checkpoint_path: str,
        device: torch.device,
    ) -> nn.Module:
        """Load weights from `checkpoint_path` into a copy of `model`."""
        import copy
        opponent = copy.deepcopy(model)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Handle compiled model state dict
        target = opponent._orig_mod if hasattr(opponent, "_orig_mod") else opponent
        target.load_state_dict(state_dict, strict=True)
        opponent.eval()
        return opponent

    def __repr__(self) -> str:
        top = [(e.player_id, f"{e.elo:.0f}") for e in self._pool[:3]]
        return f"OpponentPool(size={len(self._pool)}, top3={top})"
