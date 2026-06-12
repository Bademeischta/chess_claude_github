"""Canonical (side-to-move) board/action encoding — audit F1.

Property under test: a Black-to-move position must encode IDENTICALLY to its
colour-conjugated White twin (python-chess ``mirror()`` — vertical flip +
colour/castling/turn swap — is exactly that conjugation, so it serves as the
oracle), in both the board tensor (modulo the colour-identity plane 12) and
the action-index space. This is what lets the policy/value heads share
weights across colours instead of learning two mappings.
"""

from __future__ import annotations

import numpy as np
import pytest

pc = pytest.importorskip("chess")

from engine.board import Board
from engine.movegen import _move_uci, move_to_action_index

FENS = [
    # after 1.e4 (Black to move)
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    # Kiwipete, Black to move
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    # en passant available
    "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PP1/RNBQKBNR b KQkq e3 0 3",
    # K+P endgame with halfmove clock
    "8/5k2/8/8/8/3K4/4P3/8 b - - 7 44",
    # asymmetric castling rights
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",
    # promotion-rich position (Black to move variant)
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
]


def test_canonical_tensor_matches_mirror():
    for fen in FENS:
        b1 = Board.from_fen(fen)
        b2 = Board.from_fen(pc.Board(fen).mirror().fen())
        t1, t2 = b1.to_tensor(), b2.to_tensor()
        # plane 12 carries the true colour and SHOULD differ
        assert np.allclose(np.delete(t1, 12, 0), np.delete(t2, 12, 0)), (
            f"canonical tensors diverge for mirrored pair\n{fen}")
        assert t1[12].max() != t2[12].max(), "plane 12 must keep colour identity"


def test_canonical_actions_match_mirror():
    for fen in FENS:
        b1 = Board.from_fen(fen)
        b2 = Board.from_fen(pc.Board(fen).mirror().fen())
        a1 = sorted(move_to_action_index(m, b1.side_to_move)
                    for m in b1.legal_moves())
        a2 = sorted(move_to_action_index(m, b2.side_to_move)
                    for m in b2.legal_moves())
        assert a1 == a2, f"action sets diverge for mirrored pair\n{fen}"
        assert -1 not in a1


def test_castling_shares_action_index_across_colours():
    """e8g8 (Black O-O) must map to the same index as e1g1 (White O-O).
    The old 63-sq rotation mirrored files, sending Black's kingside castle
    to a queenside-looking pattern — no cross-colour pattern sharing."""
    wb = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    bb = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    by_uci_w = {_move_uci(m): m for m in wb.legal_moves()}
    by_uci_b = {_move_uci(m): m for m in bb.legal_moves()}
    assert (move_to_action_index(by_uci_w["e1g1"], 0)
            == move_to_action_index(by_uci_b["e8g8"], 1))
    assert (move_to_action_index(by_uci_w["e1c1"], 0)
            == move_to_action_index(by_uci_b["e8c8"], 1))


def test_white_positions_encode_as_before():
    """White-to-move tensors are unchanged by canonicalisation (flip is a
    no-op) — pawns on rank 2 in plane 0, own castling in 14/15."""
    b = Board.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    t = b.to_tensor()
    assert t[0, 1].sum() == 8.0          # white pawns on rank 2
    assert t[6, 6].sum() == 8.0          # black pawns on rank 7
    assert t[12].min() == 1.0            # white to move
    assert t[14].max() == 1.0 and t[16].max() == 1.0


def test_black_position_flipped_into_own_view():
    """Black to move: own pawns appear on canonical rank 2 in plane 0."""
    b = Board.from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    t = b.to_tensor()
    assert t[0, 1].sum() == 8.0          # Black's pawns, canonical rank 2
    assert t[12].max() == 0.0            # black to move
    # White's e-pawn (real e4) lands on canonical e5 in the OPPONENT block
    assert t[6, 4, 4] == 1.0
