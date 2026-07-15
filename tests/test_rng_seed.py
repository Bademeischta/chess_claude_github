"""Regression test for audit S9: the C++ MCTS RNG must be seedable.

Before the fix the C++ backend seeded from std::random_device per thread,
so temperature move-sampling and Dirichlet noise differed on every run —
the project's reproducibility claim held only for the Python fallback.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

chess_ext = pytest.importorskip("chess_ext")

pytestmark = pytest.mark.skipif(
    not hasattr(chess_ext, "set_rng_seed"),
    reason="chess_ext build predates set_rng_seed (audit S9)")


def _sample_sequence(seed: int, n: int = 40) -> list:
    """Expand a root, add Dirichlet noise, then temperature-sample moves."""
    chess_ext.set_rng_seed(seed)
    board = chess_ext.Board()
    tree = chess_ext.MCTSTree(2.0)
    root = tree.new_root(0, 1.0)
    legal = board.legal_moves()
    priors = (np.ones(len(legal)) / len(legal)).astype(np.float32)
    tree.expand(root, legal, priors)
    tree.add_dirichlet_noise(root, 0.3, 0.25)
    # Give children visits so sampling has a distribution to draw from.
    for mv in legal:
        child = root.get_child(mv)
        tree.backup([root, child], 0.1)
    noisy_priors = [c.P for _, c in root.get_children()]
    samples = [root.sample_move(1.0) for _ in range(n)]
    return [noisy_priors, samples]


def test_same_seed_same_noise_and_samples():
    a = _sample_sequence(1234)
    b = _sample_sequence(1234)
    assert a == b, "identical seeds must reproduce noise + move samples"


def test_different_seed_differs():
    a = _sample_sequence(1234)
    b = _sample_sequence(9999)
    # Dirichlet noise over 20 children colliding across seeds is
    # astronomically unlikely; a collision means seeding is inert.
    assert a[0] != b[0], "different seeds must produce different noise"
