"""Adaptive training throttle (utils/load_governor.py).

State-machine tests with injected samplers — no GPU/nvidia-smi needed.
"""

from __future__ import annotations

from utils.load_governor import LoadGovernor


def _gov(samples, reserved=2000.0, **kw):
    """Governor whose VRAM sampler replays `samples` (then repeats last)."""
    it = iter(samples)
    last = [samples[-1]]

    def sampler():
        try:
            v = next(it)
            last[0] = v
            return v
        except StopIteration:
            return last[0]

    defaults = dict(enabled=True, foreign_vram_mb=1000.0, duty=0.25,
                    check_every_s=0.0,  # sample every call in tests
                    clear_samples_to_resume=2, max_sleep_s=0.0)
    defaults.update(kw)
    return LoadGovernor(_sampler=sampler, _torch_reserved=lambda: reserved,
                        **defaults)


def test_throttles_when_game_starts_and_resumes_after():
    # total VRAM: baseline 2.5 GB (ours 2.0 → foreign 0.5), then a game
    # pushes total to 6 GB (foreign 4.0 = +3.5 over baseline), then quits.
    g = _gov([2500, 2500, 6000, 6000, 2500, 2500, 2500])
    assert g.is_throttled() is False        # establishes baseline (500)
    assert g.is_throttled() is False        # clean
    assert g.is_throttled() is True         # game running → throttle
    assert g.is_throttled() is True         # still running
    assert g.is_throttled() is True         # 1st clean sample — hysteresis holds
    assert g.is_throttled() is False        # 2nd clean sample → resume


def test_own_vram_growth_does_not_throttle():
    # Our own reservation grows (training warms up) while foreign stays flat:
    # total rises but foreign = total - ours is constant → never throttle.
    totals = iter([2500, 4500, 7500, 9000])
    own = iter([2000, 4000, 7000, 8500])
    g = LoadGovernor(_sampler=lambda: next(totals),
                     _torch_reserved=lambda: next(own),
                     enabled=True, foreign_vram_mb=1000.0,
                     check_every_s=0.0)
    for _ in range(4):
        assert g.is_throttled() is False


def test_sampler_failure_fails_open():
    g = LoadGovernor(_sampler=lambda: None, _torch_reserved=lambda: 0.0,
                     enabled=True, check_every_s=0.0)
    assert g.is_throttled() is False
    assert g.maybe_throttle(10.0) == 0.0


def test_disabled_never_samples():
    def boom():
        raise AssertionError("sampler must not be called when disabled")
    g = LoadGovernor(_sampler=boom, _torch_reserved=lambda: 0.0,
                     enabled=False, check_every_s=0.0)
    assert g.is_throttled() is False
    assert g.maybe_throttle(5.0) == 0.0


def test_sleep_duration_matches_duty_cycle():
    g = _gov([2500, 9000, 9000], max_sleep_s=999.0)
    g.is_throttled()                         # baseline
    slept = g.maybe_throttle(work_seconds=1.0)
    # duty 0.25 → sleep = 1.0 * 0.75/0.25 = 3.0 s
    assert abs(slept - 3.0) < 0.2
