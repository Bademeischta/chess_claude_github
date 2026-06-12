"""C++ RoundDriver (batched simulation round in C++, GIL released).

Pins three properties:
  1. chess_ext.az_action_index is bit-equal to the Python
     move_to_action_index over random games (both colours, promotions).
  2. The driver path produces EXACTLY the same trees as the legacy Python
     round loop under a deterministic stub network.
  3. The driver path finds the mate-in-1 (search quality smoke test).
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
chess_ext = pytest.importorskip("chess_ext")

from config import CONFIG
from engine.board import Board, STARTPOS_FEN
from engine.movegen import move_to_action_index


# ── 1. Action-index parity C++ vs Python ─────────────────────────────────

def test_az_action_index_parity_random_games():
    rng = random.Random(20260612)
    positions = 0
    for _ in range(25):
        b = Board.from_fen(STARTPOS_FEN)
        for _ply in range(40):
            legal = b.legal_moves()
            if not legal:
                break
            stm = b.side_to_move
            for m in legal:
                assert chess_ext.az_action_index(m, stm) == \
                    move_to_action_index(m, stm), (
                        f"index mismatch for move {m} stm {stm} "
                        f"fen {b.to_fen()}")
            positions += 1
            b = b.apply_move(legal[rng.randrange(len(legal))])
            if b.is_draw():
                break
    assert positions > 400


def test_az_action_index_parity_promotions():
    # Promotion-rich position, both colour conjugates.
    for fen in [
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "rnbqk2r/ppp1nNpp/8/2b5/8/2P5/PP1pBPPP/RNBQ1K1R b kq - 1 8",
    ]:
        b = Board.from_fen(fen)
        stm = b.side_to_move
        for m in b.legal_moves():
            assert chess_ext.az_action_index(m, stm) == \
                move_to_action_index(m, stm)


# ── 2. Driver ≡ legacy Python round loop (deterministic stub net) ────────

class _StubEngine:
    """Zero logits, zero value — search behaviour is purely structural."""
    batch_size = 8

    def infer(self, board_t, hist_t):
        B = board_t.shape[0]
        return (torch.zeros(B, CONFIG.num_actions),
                torch.zeros(B))


def _make_pool(use_driver: bool, n: int = 2):
    from mcts.tree import ParallelMCTS

    cfg = copy.deepcopy(CONFIG)
    cfg.device = "cpu"
    cfg.precision = "fp32"
    cfg.parallel_games = n
    cfg.cpp_round_driver = use_driver
    cfg.inference_backend = "torch"
    cfg.mcts_contempt = 0.0

    model = torch.nn.Linear(1, 1)  # never used by the stub engine
    pool = ParallelMCTS.__new__(ParallelMCTS)
    # Minimal manual init to avoid building a real GPU engine.
    pool.cfg = cfg
    pool.model = model
    pool.device = torch.device("cpu")
    pool.n = n
    pool._opening_fens = None
    pool.engine = _StubEngine()
    from mcts.tree import MCTSTree
    pool.trees = [MCTSTree(cfg, model, torch.device("cpu"),
                           engine=pool.engine) for _ in range(n)]
    pool._hist_scratch = np.zeros(
        (cfg.gru_history_len, cfg.input_planes, 8, 8), dtype=np.float32)
    pool._round_driver = None
    pool._driver_boards = None
    pool._driver_pending = None
    if use_driver:
        pool._round_driver = chess_ext.RoundDriver(int(cfg.num_actions))
        pool._driver_boards = np.zeros(
            (n, cfg.input_planes, 8, 8), dtype=np.float32)
        pool._driver_pending = np.zeros(n, dtype=np.uint8)
    return pool


def _visit_dist(tree) -> dict:
    root = tree._root_node
    return {mv: child.N for mv, child in root.get_children()}


def test_driver_matches_python_round_loop_exactly():
    fens = [
        STARTPOS_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    ]
    dists = []
    for use_driver in (False, True):
        pool = _make_pool(use_driver, n=len(fens))
        boards = [Board.from_fen(f) for f in fens]
        for i, b in enumerate(boards):
            pool.trees[i].reset(b, keep_history=False)
        pool._batched_sims(list(range(len(fens))), 40)
        dists.append([_visit_dist(t) for t in pool.trees])

    for k, fen in enumerate(fens):
        assert dists[0][k] == dists[1][k], (
            f"visit distributions diverge (driver vs python) for {fen}\n"
            f"python: {dists[0][k]}\ndriver: {dists[1][k]}")
        assert sum(dists[1][k].values()) > 0


def test_driver_finds_mate_in_one():
    # White: Qh5xf7# available (scholar's-mate pattern).
    fen = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    pool = _make_pool(True, n=1)
    b = Board.from_fen(fen)
    pool.trees[0].reset(b, keep_history=False)
    pool._batched_sims([0], 300)
    dist = _visit_dist(pool.trees[0])
    # Identify the mating move: apply each candidate, mate = no legal + check.
    mate_moves = []
    for m in b.legal_moves():
        nb = b.apply_move(m)
        if not nb.legal_moves() and nb.in_check():
            mate_moves.append(m)
    assert mate_moves, "test position must contain a mate-in-1"
    mate_visits = sum(dist.get(m, 0) for m in mate_moves)
    total = sum(dist.values())
    assert mate_visits > total * 0.5, (
        f"mate move under-visited via driver: {mate_visits}/{total}")
