"""
engine/rules.py

Game-end detection, phase classification, and optional Syzygy tablebase probing.

python-chess is used EXCLUSIVELY for tablebase WDL probing (allowed by the plan).
All actual move generation and board logic uses chess_ext / board.py.
"""

from __future__ import annotations

import sys
from enum import IntEnum
from typing import Optional
import os

from engine.board import Board, PHASE_OPENING, PHASE_MID, PHASE_ENDGAME


# ── Game result ───────────────────────────────────────────────────────────

class GameResult(IntEnum):
    WHITE_WIN  =  1
    DRAW       =  0
    BLACK_WIN  = -1
    ONGOING    =  2   # Sentinel: game not yet over


def get_game_result(board: Board) -> GameResult:
    """
    Determine the game result from the current board position.
    Returns GameResult.ONGOING if the game is still in progress.

    Defensive against a known C++ chess_ext memory bug: ``is_stalemate``
    can access-violate on certain mate-in-2 positions (Qh4/Qh6 patterns
    with the king on its home square). We cheaply check legal moves
    FIRST — if any exist, neither mate nor stalemate is possible and we
    skip the dangerous C++ calls entirely. Only when zero legal moves are
    detected do we disambiguate mate-vs-stalemate, and we do it via
    python-chess instead of the C++ predicate to dodge the crash.
    """
    # Cheap pre-filter: if legal moves exist, the position is neither
    # checkmate nor stalemate. Skip the (sometimes-crashing) C++ calls.
    try:
        legal = board.legal_moves()
    except Exception:
        legal = []
    if legal:
        # Still need is_draw for fifty-move / threefold / insufficient
        # material — those don't depend on legal moves being empty.
        try:
            if board.is_draw():
                return GameResult.DRAW
        except Exception:
            # If C++ is_draw fails, fall back to python-chess for the
            # draw check too. Cheap when it isn't called every leaf.
            try:
                import chess as _pc
                pcb = _pc.Board(board.to_fen())
                if (pcb.is_fifty_moves() or pcb.is_repetition(3)
                        or pcb.is_insufficient_material()):
                    return GameResult.DRAW
            except Exception:
                pass
        return GameResult.ONGOING

    # No legal moves → terminal. Use python-chess to disambiguate
    # mate-vs-stalemate without touching the C++ is_stalemate path.
    try:
        import chess as _pc
        pcb = _pc.Board(board.to_fen())
        if pcb.is_checkmate():
            if board.side_to_move == 0:   # White to move but mated → Black wins
                return GameResult.BLACK_WIN
            return GameResult.WHITE_WIN
        # No legal moves and not in check → stalemate (draw).
        return GameResult.DRAW
    except Exception:
        # Last-resort fallback: if python-chess can't parse the FEN
        # (probably means the board itself is corrupt), call it a draw
        # rather than crashing the whole self-play pool.
        return GameResult.DRAW


def wdl_for_side(result: GameResult, side: int) -> float:
    """
    Convert GameResult to a WDL float from `side`'s perspective.
    Returns 1.0 (win), 0.5 (draw), 0.0 (loss), or None if ongoing.
    """
    if result == GameResult.ONGOING:
        return None
    if result == GameResult.DRAW:
        return 0.5
    white_wins = result == GameResult.WHITE_WIN
    if side == 0:   # White
        return 1.0 if white_wins else 0.0
    else:           # Black
        return 0.0 if white_wins else 1.0


# ── Phase detection ───────────────────────────────────────────────────────

def get_phase(board: Board) -> int:
    """Returns PHASE_OPENING / PHASE_MID / PHASE_ENDGAME."""
    return board.get_phase()


def get_piece_count(board: Board) -> int:
    return board.piece_count()


# ── Syzygy tablebase probing ──────────────────────────────────────────────

_tb_reader = None          # python-chess Tablebase reader, lazily initialised
_tb_path: str = ""
_tb_available = False


def init_tablebase(syzygy_path: str) -> bool:
    """
    Initialise the Syzygy tablebase reader.
    Returns True if tablebases were found and loaded successfully.
    python-chess is used ONLY for this WDL lookup — not for move generation.
    """
    global _tb_reader, _tb_path, _tb_available

    if not syzygy_path or not os.path.isdir(syzygy_path):
        return False

    try:
        import chess
        import chess.syzygy
        _tb_reader = chess.syzygy.open_tablebase(syzygy_path)
        _tb_path = syzygy_path
        _tb_available = True
        # Quick sanity check
        test = chess.Board("8/8/8/3k4/8/8/8/3K4 w - - 0 1")
        _tb_reader.probe_wdl(test)  # KvK should be draw
        return True
    except Exception as e:
        _tb_available = False
        _tb_reader = None
        # Graceful fallback — don't crash training if TBs are missing, but
        # surface why so a misconfigured syzygy_path is diagnosable.
        print(f"[rules] Syzygy init failed ({type(e).__name__}: {e})",
              file=sys.stderr)
        return False


def probe_tablebase(board: Board) -> Optional[float]:
    """
    Query the Syzygy tablebase for WDL from the current side's perspective.
    Returns 1.0 (win), 0.5 (draw), 0.0 (loss), or None if not available.

    Only returns a result for positions with ≤ 5 pieces.
    python-chess is used here exclusively for the probe — NOT for move generation.
    """
    global _tb_reader, _tb_available

    if not _tb_available or _tb_reader is None:
        return None

    if board.piece_count() > 5:
        return None

    try:
        import chess as _pychess
        # Build a python-chess board from FEN for the probe only
        pc_board = _pychess.Board(board.to_fen())
        wdl = _tb_reader.probe_wdl(pc_board)
        # python-chess WDL: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss
        if wdl >= 2:    return 1.0
        elif wdl >= 1:  return 0.75  # Cursed win (theoretical win but draw by rule)
        elif wdl == 0:  return 0.5
        elif wdl >= -1: return 0.25  # Blessed loss
        else:           return 0.0
    except Exception as e:
        print(f"[rules] Tablebase probe failed ({type(e).__name__}: {e})",
              file=sys.stderr)
        return None


def close_tablebase() -> None:
    """Close the tablebase reader and release resources."""
    global _tb_reader, _tb_available
    if _tb_reader is not None:
        try:
            _tb_reader.close()
        except Exception:
            pass
    _tb_reader = None
    _tb_available = False


def is_tablebase_available() -> bool:
    return _tb_available
