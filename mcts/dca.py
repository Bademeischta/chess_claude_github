"""
mcts/dca.py

Dynamic Compute Allocation (DCA).

Decides the simulation budget for a given search position.
More simulations are allocated to:
  - Ambiguous positions where the top-2 moves have similar Q-values
  - Check positions (king safety is critical)
  - Phase-transition positions (material dramatically changes)

No separate neural network is used — the heuristic is purely
based on statistics already computed by MCTS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from mcts.node import MCTSNode


# ── Compatibility helpers (Python dict node vs C++ chess_ext.MCTSNode) ────

def _child_list(root) -> List:
    """Return a list of child MCTSNode objects regardless of backend."""
    if hasattr(root, "children"):
        # Python fallback node: children is a dict {move: MCTSNode}
        return list(root.children.values())
    else:
        # C++ chess_ext.MCTSNode: use get_children() → [(move, node*), ...]
        return [child for _, child in root.get_children()]


def _has_children(root) -> bool:
    if hasattr(root, "children"):
        return bool(root.children)
    return root.get_children_count() > 0


# ── Budget computation ────────────────────────────────────────────────────

def get_sim_budget(
    root: "MCTSNode",
    base_sims: int = 200,
    sims_critical: int = 400,
    max_sims: int = 800,
    in_check: bool = False,
    q_threshold: float = 0.05,
    check_mult: float = 1.5,
) -> int:
    """
    Compute the simulation budget for the current root.

    Logic:
      1. If the root has no children yet (before any MCTS), return base_sims.
      2. If |Q_top1 - Q_top2| < q_threshold → position is ambiguous → ×2
      3. If in_check → ×1.5 (king safety critical)
      4. Hard cap at max_sims.

    This heuristic is applied AFTER the first pass of `base_sims` simulations
    to decide whether to run additional simulations.
    """
    budget = base_sims

    if not _has_children(root):
        return budget

    children = sorted(_child_list(root), key=lambda c: c.N, reverse=True)

    if len(children) >= 2:
        q1 = children[0].Q
        q2 = children[1].Q
        if abs(q1 - q2) < q_threshold:
            budget = sims_critical

    if in_check:
        budget = int(budget * check_mult)

    return min(budget, max_sims)


def is_critical(
    root: "MCTSNode",
    in_check: bool = False,
    q_threshold: float = 0.05,
) -> bool:
    """Returns True if DCA should allocate extra simulations."""
    if in_check:
        return True

    if not _has_children(root):
        return False

    children = sorted(_child_list(root), key=lambda c: c.N, reverse=True)
    return len(children) >= 2 and abs(children[0].Q - children[1].Q) < q_threshold


def compute_extra_sims(
    root: "MCTSNode",
    already_run: int,
    base_sims: int = 200,
    sims_critical: int = 400,
    max_sims: int = 800,
    in_check: bool = False,
    q_threshold: float = 0.05,
    check_mult: float = 1.5,
) -> int:
    """
    Returns the NUMBER OF ADDITIONAL simulations to run (may be 0).
    Call this after the base simulation budget is exhausted.
    """
    total_budget = get_sim_budget(
        root, base_sims, sims_critical, max_sims,
        in_check, q_threshold, check_mult,
    )
    return max(0, total_budget - already_run)
