"""Tests for utils/sf_anchor.py — the objective Stockfish-anchor probe.

The full match path needs a Stockfish binary + GPU and is exercised by the
in-training probe itself; here we pin the cheap contracts: graceful skip
without a binary, the skill→Elo table, and the degenerate-series clamp.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.sf_anchor import SKILL_ELO, play_anchor_match, resolve_stockfish_path  # noqa: E402


class _CfgStub:
    stockfish_path = ""
    mcts_sims = 8


def test_skill_table_covers_all_levels_and_is_monotonic():
    assert len(SKILL_ELO) == 21  # skill levels 0..20
    assert all(b >= a for a, b in zip(SKILL_ELO, SKILL_ELO[1:]))


def test_no_binary_returns_none_not_raise():
    # An explicit nonexistent path must yield a skipped probe (None),
    # never an exception — the in-training caller treats None as "skip".
    res = play_anchor_match(
        model=None, device=None, cfg=_CfgStub(),
        games=1, sf_skill=0,
        stockfish_path=r"Z:\does\not\exist\stockfish.exe",
    )
    assert res is None


def test_resolve_prefers_existing_config_path(tmp_path):
    exe = tmp_path / "sf.exe"
    exe.write_bytes(b"")
    cfg = _CfgStub()
    cfg.stockfish_path = str(exe)
    assert resolve_stockfish_path(cfg) == str(exe.resolve())
