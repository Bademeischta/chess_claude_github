"""Endgame curriculum: a fraction of self-play games start from a WINNABLE
Syzygy endgame and play out without resignation.

Pins:
  1. The pre-sampled pool contains only positions the side to move WINS,
     all within the piece-count ceiling.
  2. _endgame_seed_fen() respects endgame_curriculum_fraction (0 → never,
     1 → always returns a pool FEN).
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from config import CONFIG
from engine.board import Board
from engine.rules import init_tablebase, is_tablebase_available, probe_tablebase
from mcts.tree import ParallelMCTS


def _bare_pool(fraction: float, max_pieces: int = 5, pool_size: int = 60):
    """A ParallelMCTS shell with only what the curriculum methods need."""
    cfg = copy.deepcopy(CONFIG)
    cfg.endgame_curriculum_fraction = fraction
    cfg.endgame_curriculum_max_pieces = max_pieces
    cfg.endgame_curriculum_pool_size = pool_size
    pool = ParallelMCTS.__new__(ParallelMCTS)
    pool.cfg = cfg
    pool._endgame_pool = None
    return pool


def _ensure_tb():
    sz = CONFIG.syzygy_path or "syzygy"
    init_tablebase(sz)
    if not is_tablebase_available():
        pytest.skip("no Syzygy tablebases available on this machine")


def test_pool_is_all_won_for_side_to_move():
    _ensure_tb()
    pool = _bare_pool(fraction=1.0, max_pieces=5, pool_size=40)
    fens = pool._build_endgame_pool()
    assert fens, "endgame pool should not be empty with tablebases present"
    for fen in fens:
        b = Board.from_fen(fen)
        assert b.piece_count() <= 5, f"too many pieces: {fen}"
        assert probe_tablebase(b) == 1.0, \
            f"pool FEN is not won for the side to move: {fen}"


def test_seed_fen_respects_fraction():
    _ensure_tb()
    # fraction 0 → never seeds an endgame
    off = _bare_pool(fraction=0.0)
    assert all(off._endgame_seed_fen() is None for _ in range(50))

    # fraction 1 → always seeds, and the FEN comes from the pool
    on = _bare_pool(fraction=1.0, pool_size=40)
    seeds = [on._endgame_seed_fen() for _ in range(20)]
    assert all(s is not None for s in seeds)
    assert set(seeds) <= set(on._get_endgame_pool())


def test_pool_built_once_and_cached():
    _ensure_tb()
    pool = _bare_pool(fraction=1.0, pool_size=30)
    first = pool._get_endgame_pool()
    second = pool._get_endgame_pool()
    assert first is second, "pool must be cached, not rebuilt each call"


def test_game_stream_plays_seeded_endgames():
    """Full game_stream path on CPU with a stub engine, forcing every game to
    start from a won endgame. Confirms seeded endgame boards flow through
    search -> record -> finalize -> re-seed without error."""
    _ensure_tb()
    import numpy as np
    from mcts.tree import MCTSTree

    cfg = copy.deepcopy(CONFIG)
    cfg.device = "cpu"
    cfg.precision = "fp32"
    cfg.parallel_games = 2
    cfg.cpp_round_driver = False
    cfg.inference_backend = "torch"
    cfg.mcts_contempt = 0.0
    cfg.endgame_curriculum_fraction = 1.0   # every game is an endgame seed
    cfg.endgame_curriculum_pool_size = 30
    cfg.mcts_sims = 16

    class _Stub:
        batch_size = 8

        def infer(self, board_t, hist_t):
            B = board_t.shape[0]
            return (torch.zeros(B, cfg.num_actions), torch.zeros(B))

    pool = ParallelMCTS.__new__(ParallelMCTS)
    pool.cfg = cfg
    pool.model = torch.nn.Linear(1, 1)
    pool.device = torch.device("cpu")
    pool.n = 2
    pool._opening_fens = None
    pool._endgame_pool = None
    pool.engine = _Stub()
    pool.trees = [MCTSTree(cfg, pool.model, pool.device, engine=pool.engine)
                  for _ in range(2)]
    pool._hist_scratch = np.zeros(
        (cfg.gru_history_len, cfg.input_planes, 8, 8), dtype=np.float32)
    pool._round_driver = None
    pool._driver_boards = None
    pool._driver_pending = None

    gen = pool.game_stream(is_teacher=False)
    games = [next(gen) for _ in range(3)]
    for g in games:
        assert g.move_numbers, "seeded game record should be non-empty"
        # First recorded position must be the ≤5-piece endgame seed.
        assert g.piece_counts[0] <= 5, \
            f"game did not start from an endgame seed: {g.piece_counts[0]} pieces"
