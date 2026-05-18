# Engine package — try C++ extension first, fall back to pure Python
from engine.board import Board, STARTPOS_FEN
from engine.movegen import generate_legal_moves, perft
from engine.rules import (
    get_game_result, get_phase, get_piece_count,
    probe_tablebase, GameResult,
)

__all__ = [
    "Board", "STARTPOS_FEN",
    "generate_legal_moves", "perft",
    "get_game_result", "get_phase", "get_piece_count",
    "probe_tablebase", "GameResult",
]
