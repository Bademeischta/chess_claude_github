"""Trainer-side correctness — TD(λ) blending, phase-weight lookup."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.trainer import Trainer


def test_td_lambda_inside_window_blends():
    """Inside [start, end] the target is the λ-blend of MC and Q."""
    mc = torch.tensor([1.0, -1.0])
    q = torch.tensor([0.0, 0.0])
    mn = torch.tensor([10, 10], dtype=torch.long)
    out = Trainer.compute_td_lambda_targets(
        mc, q, mn, td_lambda=0.8, td_start=3, td_end=100,
    )
    # blended = 0.8 * mc + 0.2 * q
    assert out[0].item() == pytest.approx(0.8)
    assert out[1].item() == pytest.approx(-0.8)


def test_td_lambda_outside_window_is_pure_mc():
    """Moves outside [start, end] keep the MC return untouched."""
    mc = torch.tensor([1.0])
    q = torch.tensor([-1.0])
    out = Trainer.compute_td_lambda_targets(
        mc, q, torch.tensor([1], dtype=torch.long),
        td_lambda=0.8, td_start=3, td_end=100,
    )
    # move=1 is below td_start=3 → pure MC.
    assert out[0].item() == pytest.approx(1.0)


def test_td_lambda_endgame_now_in_window():
    """Regression: window used to end at move 25; endgame must blend now."""
    mc = torch.tensor([1.0])
    q = torch.tensor([0.0])
    out = Trainer.compute_td_lambda_targets(
        mc, q, torch.tensor([80], dtype=torch.long),
        td_lambda=0.8, td_start=3, td_end=100,
    )
    # Move 80 is inside the widened window → blended.
    assert out[0].item() == pytest.approx(0.8)
