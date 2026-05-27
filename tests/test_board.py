"""Board correctness — perft, threefold, draw conditions, promotion.

Perft numbers below are the canonical values from the Chess Programming
wiki (https://www.chessprogramming.org/Perft_Results). They are exhaustive
move counts, so any off-by-one in legal_moves or apply_move breaks them
immediately.
"""

from __future__ import annotations

import pytest

from engine.board import Board, STARTPOS_FEN


# Canonical perft results for the standard starting position.
_PERFT_STARTPOS = {
    1: 20,
    2: 400,
    3: 8_902,
    4: 197_281,
}

# Kiwipete (CPW position 2) — broader move-generator coverage.
_KIWIPETE_FEN = (
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
)
_PERFT_KIWIPETE = {
    1: 48,
    2: 2_039,
    3: 97_862,
}


@pytest.mark.parametrize("depth,expected", list(_PERFT_STARTPOS.items()))
def test_perft_startpos(depth: int, expected: int) -> None:
    board = Board.from_fen(STARTPOS_FEN)
    assert board.perft(depth) == expected, (
        f"Perft mismatch at depth {depth}: got {board.perft(depth)}, "
        f"expected {expected}"
    )


@pytest.mark.parametrize("depth,expected", list(_PERFT_KIWIPETE.items()))
def test_perft_kiwipete(depth: int, expected: int) -> None:
    board = Board.from_fen(_KIWIPETE_FEN)
    assert board.perft(depth) == expected


def test_startpos_legal_move_count() -> None:
    board = Board.from_fen(STARTPOS_FEN)
    assert len(board.legal_moves()) == 20


def test_apply_move_returns_new_board() -> None:
    board = Board.from_fen(STARTPOS_FEN)
    moves = board.legal_moves()
    new_board = board.apply_move(moves[0])
    # apply_move must be pure — original board unchanged.
    assert board.to_fen() == STARTPOS_FEN
    # The new board has a different hash and the other side to move.
    assert new_board.hash != board.hash
    assert new_board.side_to_move != board.side_to_move


def test_insufficient_material_K_vs_K() -> None:
    # Bare kings — draw.
    board = Board.from_fen("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
    assert board.is_insufficient_material()
    assert board.is_draw()


def test_insufficient_material_KB_vs_K() -> None:
    # King + single bishop vs king — draw.
    board = Board.from_fen("8/8/8/4k3/8/8/8/3BK3 w - - 0 1")
    assert board.is_insufficient_material()


def test_sufficient_material_KQ_vs_K() -> None:
    # King + queen vs king — sufficient for mate.
    board = Board.from_fen("8/8/8/4k3/8/8/8/3QK3 w - - 0 1")
    assert not board.is_insufficient_material()


def test_stale_move_counter_starts_zero() -> None:
    """Engine starts with a clean stale-move counter."""
    try:
        import chess_ext
    except ImportError:
        pytest.skip("chess_ext native module not available")
    if not hasattr(chess_ext, "stale_move_count"):
        pytest.skip("Native binary predates stale_move_count export — rebuild .pyd")
    chess_ext.reset_stale_move_count()
    assert chess_ext.stale_move_count() == 0

    # A clean game-length walk must NOT trip the counter.
    board = Board.from_fen(STARTPOS_FEN)
    for _ in range(20):
        moves = board.legal_moves()
        if not moves:
            break
        board = board.apply_move(moves[0])
    assert chess_ext.stale_move_count() == 0
