"""Tests for engine/harness.py — the shared engine bring-up module.

Full load_engine needs a checkpoint + GPU and is exercised by every play
front-end; here we pin the pure logic: time control, checkpoint
inventory, and the search driver's budget/stop semantics.
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.harness import list_checkpoints, run_search, time_budget_s  # noqa: E402


# ── time_budget_s ─────────────────────────────────────────────────────────

def test_time_budget_movetime_dominates():
    assert time_budget_s(60_000, 1_000, movetime_ms=2_000) == 2.0


def test_time_budget_clock_formula():
    # clock/25 + inc*0.75 (ms) → seconds
    assert abs(time_budget_s(60_000, 1_000) - (60_000 / 25 + 750) / 1000.0) < 1e-9


def test_time_budget_no_limits_is_none():
    assert time_budget_s(None) is None


def test_time_budget_floor():
    assert time_budget_s(100, 0) == 0.05  # never below 50 ms


# ── list_checkpoints ──────────────────────────────────────────────────────

class _Cfg:
    def __init__(self, d):
        self.checkpoint_dir = str(d)


def test_list_checkpoints_reads_steps_and_pool_elo(tmp_path):
    (tmp_path / "net_step100011.pt").write_bytes(b"x")
    (tmp_path / "step_0109400.pt").write_bytes(b"x")
    (tmp_path / "latest.pt").write_bytes(b"x")
    (tmp_path / "latest.pt.collapsed").write_bytes(b"x")  # must be skipped
    (tmp_path / "pool_state.json").write_text(
        '{"pool": [{"path": "checkpoints\\\\net_step100011.pt", '
        '"elo": 1018.5, "step": 100011}]}', encoding="utf-8")
    (tmp_path / "sf_elo_history.csv").write_text(
        "step,games,wins,draws,losses,anchor_elo,estimated_elo,ci_lo,ci_hi,seconds\n"
        "100000,30,10,5,15,950,890,800,980,120\n", encoding="utf-8")

    entries = {e["name"]: e for e in list_checkpoints(_Cfg(tmp_path))}
    assert "latest.pt.collapsed" not in entries
    assert entries["net_step100011.pt"]["step"] == 100011
    assert entries["net_step100011.pt"]["pool_elo"] == 1018.5
    # step 100011 is within 2500 of the 100000 measurement
    assert entries["net_step100011.pt"]["sf_elo"] == 890.0
    assert entries["step_0109400.pt"]["step"] == 109400
    assert entries["step_0109400.pt"]["sf_elo"] is None  # 9400 away
    assert entries["latest.pt"]["step"] is None


def test_list_checkpoints_missing_dir_is_empty():
    assert list_checkpoints(_Cfg(r"Z:\does\not\exist")) == []


# ── run_search ────────────────────────────────────────────────────────────

class _FakeTree:
    """Counts sims; batched calls advance by the requested amount."""

    def __init__(self, saturate_after=None, delay=0.0):
        self.total = 0
        self.saturate_after = saturate_after
        self.delay = delay

    def run_batched_sims(self, n, batch_size=0):
        if self.saturate_after is not None and self.total >= self.saturate_after:
            return 0
        if self.delay:
            time.sleep(self.delay)
        self.total += n
        return n

    def run_one_simulation(self):
        self.total += 1


def test_run_search_respects_sims_budget():
    t = _FakeTree()
    done = run_search(t, sims=100, batch_size=32)
    assert done == 100
    assert t.total == 100


def test_run_search_single_leaf_fallback():
    t = _FakeTree()
    assert run_search(t, sims=5, batch_size=0) == 5


def test_run_search_stops_on_saturation():
    t = _FakeTree(saturate_after=64)
    done = run_search(t, sims=1000, batch_size=32)
    assert done == 64


def test_run_search_stop_event_breaks_but_first_round_runs():
    stop = threading.Event()
    stop.set()  # pre-set: still must complete one round
    t = _FakeTree()
    done = run_search(t, sims=1000, batch_size=32, stop=stop)
    assert done == 32


def test_run_search_deadline_breaks():
    t = _FakeTree(delay=0.02)
    t0 = time.time()
    done = run_search(t, deadline=time.time() + 0.1, batch_size=16)
    assert done >= 16                # at least one round
    assert time.time() - t0 < 2.0    # actually stopped
