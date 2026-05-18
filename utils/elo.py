"""
ELO rating system for the internal opponent pool.
Supports K=32 classical ELO with CSV export for TensorBoard.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Optional


class ELOSystem:
    """Tracks ELO ratings for a pool of players identified by string IDs."""

    def __init__(self, initial_elo: float = 1000.0, k: float = 32.0) -> None:
        self.initial_elo = initial_elo
        self.k = k
        self._ratings: Dict[str, float] = {}
        self._games_played: Dict[str, int] = {}
        self._history: list[dict] = []  # [{step, player_id, elo}, ...]

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_elo(self, player_id: str) -> float:
        return self._ratings.get(player_id, self.initial_elo)

    def register(self, player_id: str, elo: Optional[float] = None) -> None:
        if player_id not in self._ratings:
            self._ratings[player_id] = elo if elo is not None else self.initial_elo
            self._games_played[player_id] = 0

    def all_ratings(self) -> Dict[str, float]:
        return dict(self._ratings)

    # ── Core ELO maths ────────────────────────────────────────────────────

    def expected_score(self, elo_a: float, elo_b: float) -> float:
        """Probability that player A beats player B."""
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def update(
        self,
        player_a: str,
        player_b: str,
        result: float,
        step: int = 0,
    ) -> tuple[float, float]:
        """
        Update both players' ELO after one match.

        result: 1.0 = A wins, 0.5 = draw, 0.0 = B wins.
        Returns (new_elo_a, new_elo_b).
        """
        self.register(player_a)
        self.register(player_b)

        ea = self.expected_score(self.get_elo(player_a), self.get_elo(player_b))
        eb = 1.0 - ea

        delta_a = self.k * (result - ea)
        delta_b = self.k * ((1.0 - result) - eb)

        self._ratings[player_a] += delta_a
        self._ratings[player_b] += delta_b
        self._games_played[player_a] += 1
        self._games_played[player_b] += 1

        self._history.append({"step": step, "player": player_a, "elo": self._ratings[player_a]})
        self._history.append({"step": step, "player": player_b, "elo": self._ratings[player_b]})

        return self._ratings[player_a], self._ratings[player_b]

    def update_from_series(
        self,
        player_a: str,
        player_b: str,
        wins: int,
        draws: int,
        losses: int,
        step: int = 0,
    ) -> tuple[float, float]:
        """Update ELO from a multi-game series result."""
        total = wins + draws + losses
        if total == 0:
            return self.get_elo(player_a), self.get_elo(player_b)
        result = (wins + 0.5 * draws) / total
        # Treat the series as a single aggregated result
        return self.update(player_a, player_b, result, step)

    # ── Confidence interval ───────────────────────────────────────────────

    @staticmethod
    def elo_difference_ci(wins: int, draws: int, losses: int) -> tuple[float, float, float]:
        """
        Compute point estimate and 95% CI for ELO difference from a series.
        Returns (elo_diff, lower_95, upper_95).
        Uses the Wald interval on the score fraction.
        """
        total = wins + draws + losses
        if total == 0:
            return 0.0, float("-inf"), float("inf")
        score = (wins + 0.5 * draws) / total
        # Wald SE
        se = math.sqrt(score * (1.0 - score) / total) if total > 0 else 0.0
        # Convert score to ELO
        def to_elo(s: float) -> float:
            s = max(1e-6, min(1.0 - 1e-6, s))
            return -400.0 * math.log10(1.0 / s - 1.0)

        diff = to_elo(score)
        lo = to_elo(max(1e-6, score - 1.96 * se))
        hi = to_elo(min(1.0 - 1e-6, score + 1.96 * se))
        return diff, lo, hi

    # ── Persistence ───────────────────────────────────────────────────────

    def save_csv(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "player", "elo"])
            writer.writeheader()
            writer.writerows(self._history)

    def save_ratings(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["player_id", "elo", "games_played"])
            for pid, elo in sorted(self._ratings.items(), key=lambda x: -x[1]):
                writer.writerow([pid, f"{elo:.1f}", self._games_played.get(pid, 0)])

    def load_ratings(self, path: Path | str) -> None:
        path = Path(path)
        if not path.exists():
            return
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._ratings[row["player_id"]] = float(row["elo"])
                self._games_played[row["player_id"]] = int(row["games_played"])

    def __repr__(self) -> str:  # noqa: D105
        top = sorted(self._ratings.items(), key=lambda x: -x[1])[:5]
        return f"ELOSystem(players={len(self._ratings)}, top5={top})"
