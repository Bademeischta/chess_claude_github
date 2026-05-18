"""
engine/board.py

Python-level Board interface.  Attempts to import the C++ chess_ext module;
falls back to a pure-Python BitBoard implementation if the extension is not
available (slower but correct).

Public API is identical in both paths:
    Board.from_fen(fen)   → Board
    board.legal_moves()   → list[int]
    board.apply_move(m)   → Board
    board.to_tensor()     → np.ndarray shape (21, 8, 8) float32
    board.is_checkmate()  → bool
    board.is_stalemate()  → bool
    board.is_draw()       → bool
    board.in_check()      → bool
    board.piece_count()   → int
    board.get_phase()     → int (0=opening, 1=mid, 2=endgame)
    board.to_fen()        → str
    board.side_to_move    → int (0=white, 1=black)
    board.halfmove_clock  → int
    board.fullmove_number → int
    board.hash            → int (Zobrist)
"""

from __future__ import annotations

import sys
import os
import numpy as np
from pathlib import Path
from typing import List

# ── Try C++ extension ─────────────────────────────────────────────────────

_CPP_AVAILABLE = False

# Add project root to path so chess_ext.pyd is found
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

try:
    import chess_ext as _cx
    _CPP_AVAILABLE = True
except ImportError:
    _cx = None  # type: ignore

# ── Constants ─────────────────────────────────────────────────────────────

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

WHITE = 0
BLACK = 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5

PHASE_OPENING  = 0
PHASE_MID      = 1
PHASE_ENDGAME  = 2

# Phase thresholds (piece count)
_OPENING_THRESHOLD  = 24
_ENDGAME_THRESHOLD  = 12


# ============================================================
#  C++ wrapper
# ============================================================

if _CPP_AVAILABLE:

    class Board:
        """Thin wrapper around chess_ext.Board (C++ backend)."""

        __slots__ = ("_b",)

        def __init__(self, _b=None) -> None:
            self._b = _b if _b is not None else _cx.Board()

        # ── Constructors ──────────────────────────────────

        @classmethod
        def from_fen(cls, fen: str) -> "Board":
            return cls(_cx.Board.from_fen(fen))

        @classmethod
        def startpos(cls) -> "Board":
            return cls.from_fen(STARTPOS_FEN)

        # ── Core operations ───────────────────────────────

        def legal_moves(self) -> List[int]:
            return list(self._b.legal_moves())

        def apply_move(self, move: int) -> "Board":
            return Board(self._b.apply_move(move))

        def to_tensor(self) -> np.ndarray:
            return self._b.to_tensor()  # Already returns (21,8,8) float32

        def to_fen(self) -> str:
            return self._b.to_fen()

        # ── State queries ────────────────────────────────

        def in_check(self)           -> bool: return self._b.in_check()
        def is_checkmate(self)       -> bool: return self._b.is_checkmate()
        def is_stalemate(self)       -> bool: return self._b.is_stalemate()
        def is_fifty_move(self)      -> bool: return self._b.is_fifty_move()
        def is_threefold(self)       -> bool: return self._b.is_threefold()
        def is_insufficient_material(self) -> bool:
            return self._b.is_insufficient_material()
        def is_draw(self)            -> bool: return self._b.is_draw()

        def piece_count(self)        -> int:  return self._b.piece_count()
        def get_phase(self)          -> int:  return self._b.get_phase()
        def piece_at(self, sq: int)  -> int:  return self._b.piece_at(sq)
        def perft(self, depth: int)  -> int:  return self._b.perft(depth)

        # ── Properties ────────────────────────────────────

        @property
        def side_to_move(self)    -> int: return self._b.side_to_move
        @property
        def halfmove_clock(self)  -> int: return self._b.halfmove_clock
        @property
        def fullmove_number(self) -> int: return self._b.fullmove_number
        @property
        def hash(self)            -> int: return self._b.hash
        @property
        def ep_square(self)       -> int: return self._b.ep_square
        @property
        def castling(self)        -> int: return self._b.castling

        def __repr__(self) -> str:
            return f"Board(fen='{self.to_fen()}')"


# ============================================================
#  Pure-Python fallback (slower, used when C++ ext unavailable)
# ============================================================

else:
    import warnings
    warnings.warn(
        "chess_ext C++ extension not found - using pure-Python fallback "
        "(run `python setup_ext.py build_ext --inplace` to build it).",
        RuntimeWarning,
        stacklevel=2,
    )

    # We delegate to python-chess for correctness in fallback mode.
    # python-chess IS allowed as an optional validation / fallback dependency.
    try:
        import chess as _pychess
        _PYCHESS_AVAILABLE = True
    except ImportError:
        _PYCHESS_AVAILABLE = False
        warnings.warn(
            "python-chess not installed either.  The pure-Python fallback "
            "requires `pip install python-chess` to function.",
            RuntimeWarning,
            stacklevel=2,
        )

    class Board:  # type: ignore[no-redef]
        """Pure-Python Board backed by python-chess (fallback only)."""

        __slots__ = ("_b", "_hash_history")

        def __init__(self, _b=None) -> None:
            if not _PYCHESS_AVAILABLE:
                raise RuntimeError(
                    "No chess backend available. "
                    "Build chess_ext (C++) or install python-chess."
                )
            self._b = _b if _b is not None else _pychess.Board()
            self._hash_history: list = [self._b.fen()]

        # ── Constructors ──────────────────────────────────

        @classmethod
        def from_fen(cls, fen: str) -> "Board":
            b = _pychess.Board(fen)
            obj = cls.__new__(cls)
            object.__setattr__(obj, "_b", b)
            object.__setattr__(obj, "_hash_history", [fen])
            return obj

        @classmethod
        def startpos(cls) -> "Board":
            return cls.from_fen(STARTPOS_FEN)

        # ── Core operations ───────────────────────────────

        def legal_moves(self) -> List[int]:
            moves = []
            for m in self._b.legal_moves:
                from_sq = m.from_square
                to_sq   = m.to_square
                # Encode as simple int: from | (to << 6) | (flags << 12)
                flags = 0
                if self._b.is_capture(m): flags = 4
                if m.promotion:
                    promo_map = {
                        _pychess.QUEEN:  11, _pychess.ROOK:  10,
                        _pychess.BISHOP: 9,  _pychess.KNIGHT: 8,
                    }
                    flags = promo_map.get(m.promotion, 11)
                moves.append(from_sq | (to_sq << 6) | (flags << 12))
            return moves

        def apply_move(self, move: int) -> "Board":
            from_sq = move & 0x3F
            to_sq   = (move >> 6) & 0x3F
            flags   = (move >> 12) & 0xF
            promo   = None
            if flags in (8, 12):  promo = _pychess.KNIGHT
            elif flags in (9, 13): promo = _pychess.BISHOP
            elif flags in (10, 14): promo = _pychess.ROOK
            elif flags in (11, 15): promo = _pychess.QUEEN
            m = _pychess.Move(from_sq, to_sq, promotion=promo)
            new_b = _pychess.Board(self._b.fen())
            new_b.push(m)
            obj = Board.__new__(Board)
            object.__setattr__(obj, "_b", new_b)
            h = list(self._hash_history) + [new_b.fen()]
            object.__setattr__(obj, "_hash_history", h)
            return obj

        def to_tensor(self) -> np.ndarray:
            out = np.zeros((21, 8, 8), dtype=np.float32)
            piece_map = self._b.piece_map()
            for sq, piece in piece_map.items():
                color = 0 if piece.color else 1
                ptype = piece.piece_type - 1  # python-chess uses 1-6
                plane_idx = color * 6 + ptype
                r, f = sq // 8, sq % 8
                out[plane_idx, r, f] = 1.0
            # Side to move
            if self._b.turn: out[12] = 1.0
            # En passant
            if self._b.ep_square is not None:
                r, f = self._b.ep_square // 8, self._b.ep_square % 8
                out[13, r, f] = 1.0
            # Castling
            if self._b.has_kingside_castling_rights(_pychess.WHITE):  out[14] = 1.0
            if self._b.has_queenside_castling_rights(_pychess.WHITE): out[15] = 1.0
            if self._b.has_kingside_castling_rights(_pychess.BLACK):  out[16] = 1.0
            if self._b.has_queenside_castling_rights(_pychess.BLACK): out[17] = 1.0
            out[18] = min(self._b.halfmove_clock, 100) / 100.0
            out[19] = min(self._b.fullmove_number, 200) / 200.0
            return out

        def to_fen(self) -> str:
            return self._b.fen()

        def in_check(self)           -> bool: return self._b.is_check()
        def is_checkmate(self)       -> bool: return self._b.is_checkmate()
        def is_stalemate(self)       -> bool: return self._b.is_stalemate()
        def is_fifty_move(self)      -> bool: return self._b.is_fifty_moves()
        def is_threefold(self)       -> bool:
            # apply_move() rebuilds the board from FEN, so python-chess'
            # is_repetition() (which needs the move stack) never fires here.
            # Detect repetition from the maintained FEN history instead: a
            # position repeats when piece placement, side to move, castling
            # rights and the en-passant square (first 4 FEN fields) match.
            def _key(fen: str) -> str:
                return " ".join(fen.split(" ")[:4])
            cur = _key(self._b.fen())
            return sum(1 for f in self._hash_history if _key(f) == cur) >= 3
        def is_insufficient_material(self) -> bool:
            return self._b.is_insufficient_material()
        def is_draw(self)            -> bool:
            return (self.is_fifty_move() or self.is_threefold()
                    or self.is_insufficient_material())

        def piece_count(self) -> int:
            return bin(self._b.occupied).count("1")

        def get_phase(self) -> int:
            pc = self.piece_count()
            if pc > _OPENING_THRESHOLD:  return PHASE_OPENING
            if pc >= _ENDGAME_THRESHOLD: return PHASE_MID
            return PHASE_ENDGAME

        def piece_at(self, sq: int) -> int:
            p = self._b.piece_at(sq)
            if p is None: return -1
            color = 0 if p.color else 1
            return color * 6 + (p.piece_type - 1)

        def perft(self, depth: int) -> int:
            if depth == 0: return 1
            moves = list(self._b.legal_moves)
            if depth == 1: return len(moves)
            count = 0
            for m in moves:
                self._b.push(m)
                count += self.perft(depth - 1)
                self._b.pop()
            return count

        @property
        def side_to_move(self)    -> int: return 0 if self._b.turn else 1
        @property
        def halfmove_clock(self)  -> int: return self._b.halfmove_clock
        @property
        def fullmove_number(self) -> int: return self._b.fullmove_number
        @property
        def hash(self) -> int: return self._b.zobrist_hash()
        @property
        def ep_square(self)  -> int:
            return self._b.ep_square if self._b.ep_square is not None else -1
        @property
        def castling(self)   -> int:
            c = 0
            if self._b.has_kingside_castling_rights(_pychess.WHITE):  c |= 1
            if self._b.has_queenside_castling_rights(_pychess.WHITE): c |= 2
            if self._b.has_kingside_castling_rights(_pychess.BLACK):  c |= 4
            if self._b.has_queenside_castling_rights(_pychess.BLACK): c |= 8
            return c

        def __repr__(self) -> str:
            return f"Board(fen='{self.to_fen()}', backend='python-chess')"


# ── Module-level helpers ──────────────────────────────────────────────────

def using_cpp_backend() -> bool:
    """Returns True if the C++ chess_ext extension is active."""
    return _CPP_AVAILABLE
