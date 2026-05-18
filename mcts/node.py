"""
mcts/node.py

Pure-Python MCTS node — used as a fallback when chess_ext C++ is unavailable,
and for unit tests.

When chess_ext IS available, the MCTSTree in mcts/tree.py uses the C++
mcts_core.MCTSNode / MCTSTree objects directly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


class MCTSNode:
    """
    A node in the Monte Carlo Search Tree.

    __slots__ reduces per-node memory overhead (hundreds of thousands of nodes
    are created during a search).
    """

    __slots__ = (
        "N", "W", "Q", "P",
        "virtual_loss",
        "move",
        "parent",
        "children",
        "terminal",
        "terminal_value",
    )

    def __init__(
        self,
        move: int = 0,
        parent: Optional["MCTSNode"] = None,
        prior: float = 1.0,
    ) -> None:
        self.N: int   = 0          # Visit count
        self.W: float = 0.0        # Accumulated value
        self.Q: float = 0.0        # Mean value = W / N
        self.P: float = prior      # Prior probability from policy net
        self.virtual_loss: int = 0 # Temporarily reduces Q during parallel selection

        self.move: int = move      # Move integer that led to this node
        self.parent: Optional[MCTSNode] = parent
        self.children: Dict[int, MCTSNode] = {}

        self.terminal: bool = False
        self.terminal_value: float = 0.0

    # ── UCB score ─────────────────────────────────────────────────────────

    def ucb(self, c_puct: float, parent_n: int) -> float:
        """
        Upper Confidence Bound (UCB) score for node selection.
        UCB = Q + c_puct × P × √N_parent / (1 + N + virtual_loss)
        """
        u = c_puct * self.P * math.sqrt(parent_n) / (1.0 + self.N + self.virtual_loss)
        q_adj = self.Q - float(self.virtual_loss)  # Each VL reduces Q by 1
        return q_adj + u

    # ── Virtual loss ──────────────────────────────────────────────────────

    def add_virtual_loss(self) -> None:
        self.virtual_loss += 1

    def undo_virtual_loss(self) -> None:
        if self.virtual_loss > 0:
            self.virtual_loss -= 1

    # ── Tree state ────────────────────────────────────────────────────────

    def is_expanded(self) -> bool:
        return bool(self.children) or self.terminal

    def is_leaf(self) -> bool:
        return not self.is_expanded()

    # ── Child selection ───────────────────────────────────────────────────

    def best_child(self, c_puct: float) -> "MCTSNode":
        """Return child with highest UCB score."""
        assert self.children, "best_child called on unexpanded node"
        parent_n = self.N
        best: Optional[MCTSNode] = None
        best_score = float("-inf")
        for child in self.children.values():
            score = child.ucb(c_puct, parent_n)
            if score > best_score:
                best_score = score
                best = child
        return best  # type: ignore[return-value]

    def sample_move(self, temperature: float = 1.0) -> int:
        """
        Sample a move according to the MCTS visit-count distribution raised
        to the power 1/temperature.  Temperature ≈ 0 → greedy.
        """
        assert self.children, "sample_move called on unexpanded node"
        if temperature < 0.01:
            # Greedy: most visited child
            return max(self.children, key=lambda m: self.children[m].N)

        import random
        import numpy as np
        counts = np.array([c.N for c in self.children.values()], dtype=np.float64)
        moves  = list(self.children.keys())
        counts = np.power(counts, 1.0 / temperature)
        total  = counts.sum()
        if total < 1e-9:
            return random.choice(moves)
        probs = counts / total
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]

    def get_policy_target(self) -> List[Tuple[int, float]]:
        """Return (move, visit_fraction) pairs for all children."""
        total = float(sum(c.N for c in self.children.values()))
        if total < 1e-6:
            total = 1.0
        return [(m, c.N / total) for m, c in self.children.items()]

    def get_value(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def __repr__(self) -> str:
        return (f"MCTSNode(N={self.N}, Q={self.Q:.3f}, "
                f"P={self.P:.3f}, children={len(self.children)})")
