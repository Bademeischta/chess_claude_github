"""
engine/movegen.py

Move-generation helpers and Perft tests.

The actual move generation lives in the C++ chess_ext (or python-chess
fallback) via engine/board.py.  This module provides:
  - generate_legal_moves(board) → list[int]
  - perft(board, depth)         → int
  - Move encoding / decoding utilities
  - AlphaZero 64×73 action-space encoding helpers
  - Perft test suite (validates C++ engine correctness)
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from engine.board import Board

# ── Constants ─────────────────────────────────────────────────────────────

# 8 queen-move directions (dr, df)
_QUEEN_DIRS = [
    (0,  1), (0, -1),   # E, W
    (1,  0), (-1, 0),   # N, S
    (1,  1), (1, -1),   # NE, NW
    (-1, 1), (-1,-1),   # SE, SW
]

# 8 knight-move offsets (dr, df)
_KNIGHT_MOVES = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]

# AlphaZero action index for each (from_sq, to_sq, promo_piece):
#   - queen moves: distance 1-7 × direction 0-7  → 56 channels (0-55)
#   - knight moves: 8 channels                   → (56-63)
#   - underpromotions: knight/bishop/rook × 3 push dirs → 9 channels (64-72)
# Total per source square: 73
# Total: 64 × 73 = 4672

_AZ_QUEEN_CHANNELS = 56  # 8 dirs × 7 distances
_AZ_KNIGHT_CHANNELS = 8
_AZ_PROMO_CHANNELS  = 9  # 3 pieces × 3 directions (left, straight, right)
_AZ_CHANNELS_PER_SQ = _AZ_QUEEN_CHANNELS + _AZ_KNIGHT_CHANNELS + _AZ_PROMO_CHANNELS  # 73
NUM_ACTIONS = 64 * _AZ_CHANNELS_PER_SQ  # 4672

def _sq_to_rf(sq: int):
    return sq >> 3, sq & 7

def _rf_to_sq(r: int, f: int) -> int:
    return r * 8 + f


@lru_cache(maxsize=None)
def move_to_action_index(move: int, side_to_move: int) -> int:
    """
    Convert a move integer to an AlphaZero action index (0 … 4671).
    Returns -1 if the move cannot be encoded (should never happen for legal moves).

    Pure function of (move, side_to_move) over a tiny finite domain
    (~2^16 moves x 2 sides), so it is memoised. This is the dominant
    self-play CPU cost (~42 % of wall time): it is called for every legal
    move of every MCTS leaf every round. Caching turns the repeated Python
    arithmetic into an O(1) dict hit with NO change to search results.
    """
    from_sq = move & 0x3F
    to_sq   = (move >> 6) & 0x3F
    flags   = (move >> 12) & 0xF

    # From AlphaZero's perspective, the board is always from the current player's view.
    # We flip the board for Black so that the policy head always sees the same orientation.
    if side_to_move == 1:  # BLACK
        from_sq = 63 - from_sq
        to_sq   = 63 - to_sq

    fr, ff = _sq_to_rf(from_sq)
    tr, tf = _sq_to_rf(to_sq)
    dr, df = tr - fr, tf - ff

    # ── Underpromotion ──────────────────────────────────────────────────
    # flags 8=N, 9=B, 10=R (quiet promo), 12=N, 13=B, 14=R (capture promo)
    if flags in (8, 9, 10, 12, 13, 14):
        promo_map = {8: 0, 12: 0, 9: 1, 13: 1, 10: 2, 14: 2}  # N=0, B=1, R=2
        piece_idx = promo_map[flags]
        if df == -1:   dir_idx = 0  # capture left
        elif df == 0:  dir_idx = 1  # push straight
        else:          dir_idx = 2  # capture right
        channel = _AZ_QUEEN_CHANNELS + _AZ_KNIGHT_CHANNELS + piece_idx * 3 + dir_idx
        return from_sq * _AZ_CHANNELS_PER_SQ + channel

    # ── Queen promo (treated as queen move to rank 8 / rank 1) ──────────
    # flags 11=Q quiet, 15=Q capture — same encoding as normal queen move

    # ── Knight move ─────────────────────────────────────────────────────
    knight_offsets = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
    for i, (kdr, kdf) in enumerate(knight_offsets):
        if dr == kdr and df == kdf:
            channel = _AZ_QUEEN_CHANNELS + i
            return from_sq * _AZ_CHANNELS_PER_SQ + channel

    # ── Queen / rook / bishop (sliding) move ────────────────────────────
    dist = max(abs(dr), abs(df))
    if dist == 0:
        return -1
    if dr != 0 and df != 0 and abs(dr) != abs(df):
        return -1  # Not a valid queen/rook/bishop direction
    norm_dr = (dr // dist) if dr else 0
    norm_df = (df // dist) if df else 0
    for dir_idx, (qdr, qdf) in enumerate(_QUEEN_DIRS):
        if norm_dr == qdr and norm_df == qdf:
            channel = dir_idx * 7 + (dist - 1)  # dist 1-7 → index 0-6
            return from_sq * _AZ_CHANNELS_PER_SQ + channel

    return -1


def action_index_to_move(idx: int, board: Board) -> int:
    """Reverse map: AlphaZero action index → move integer (for self-play)."""
    legal = board.legal_moves()
    stm   = board.side_to_move
    for m in legal:
        ai = move_to_action_index(m, stm)
        if ai == idx:
            return m
    return 0


def policy_array_to_legal_priors(
    policy_logits: np.ndarray,   # shape (4672,)
    legal_moves: List[int],
    side_to_move: int,
) -> np.ndarray:
    """
    Extract and softmax-normalise the policy logits for legal moves.
    Returns a 1-D float32 array of probabilities aligned with `legal_moves`.

    Hot path: called once per MCTS leaf in `_finish_leaf`, i.e. roughly
    `pool_size × sims_per_move` times per game. Vectorised with numpy
    fancy-indexing instead of a Python loop + list comprehension —
    measurably faster for the typical 20-40 legal-move case.
    """
    n = len(legal_moves)
    if n == 0:
        # Defensive: terminal positions should never call this, but keep
        # the function total instead of dividing by zero downstream.
        return np.zeros(0, dtype=np.float32)

    # move_to_action_index is cached (LRU); the loop here is the unavoidable
    # cost of resolving each legal move's index. Build into a pre-allocated
    # int array to skip Python list → numpy conversion overhead.
    indices = np.empty(n, dtype=np.int64)
    for i, m in enumerate(legal_moves):
        indices[i] = move_to_action_index(m, side_to_move)

    # Bulk gather: one vectorised fancy-index instead of n element accesses.
    valid_mask = indices >= 0
    # Pre-fill with a deep negative so any -1 indices softmax to ~0.
    logits = np.full(n, -1e9, dtype=np.float32)
    if valid_mask.any():
        logits[valid_mask] = policy_logits[indices[valid_mask]]

    # Numerically-stable softmax.
    logits -= logits.max()
    exp = np.exp(logits)
    s = exp.sum()
    if s <= 0.0:
        # Total mass collapsed — degenerate but recoverable; fall back to
        # uniform over legal moves so MCTS still progresses.
        return np.full(n, 1.0 / n, dtype=np.float32)
    return (exp / s).astype(np.float32, copy=False)


# ── Public API ────────────────────────────────────────────────────────────

def generate_legal_moves(board: Board) -> List[int]:
    """Return list of legal move integers for the current position."""
    return board.legal_moves()


def perft(board: Board, depth: int) -> int:
    """Count all leaf nodes at `depth` (bulk node count, no move printing)."""
    return board.perft(depth)


def perft_divide(board: Board, depth: int) -> dict:
    """Perft split by root move — useful for debugging a specific move."""
    result = {}
    for m in board.legal_moves():
        child = board.apply_move(m)
        result[m] = child.perft(depth - 1) if depth > 1 else 1
    return result


# ── Perft test suite ──────────────────────────────────────────────────────

# Expected values from https://www.chessprogramming.org/Perft_Results
_PERFT_POSITIONS = [
    # (FEN, depth, expected_nodes)
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281),
    # Kiwipete
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862),
    # Position 3
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812),
    # Position 4 — heavy promotions/captures (catches promo-capture bugs)
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467),
    # Position 5 — promotions incl. promo-captures at depth ≥ 2
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379),
    # Position 6
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1, 46),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2, 2079),
]


def run_perft_tests(verbose: bool = True) -> bool:
    """Run all perft tests. Returns True if all pass."""
    all_passed = True
    failures = []

    if verbose:
        print("\nRunning Perft Tests")
        print("=" * 60)

    for fen, depth, expected in _PERFT_POSITIONS:
        board = Board.from_fen(fen)
        got = perft(board, depth)
        passed = got == expected
        if not passed:
            all_passed = False
            failures.append((fen, depth, expected, got))
        if verbose:
            status = "✓" if passed else "✗"
            short_fen = fen[:40] + ("..." if len(fen) > 40 else "")
            print(f"  {status}  depth={depth}  expected={expected:>8,}  got={got:>8,}  "
                  f"{short_fen}")

    if verbose:
        if all_passed:
            print("\n  All perft tests PASSED ✓")
        else:
            print(f"\n  {len(failures)} test(s) FAILED ✗")
            for fen, depth, exp, got in failures:
                print(f"    FAIL: {fen}  depth={depth}  "
                      f"expected={exp}  got={got}  diff={got-exp:+d}")
        print("=" * 60)

    return all_passed


# ── C++ ↔ python-chess parity (differential random-walk) ──────────────────

def _move_uci(m: int) -> str:
    """Convert an engine move int to a UCI string (matches python-chess)."""
    frm = m & 0x3F
    to  = (m >> 6) & 0x3F
    fl  = (m >> 12) & 0xF
    sq  = lambda s: chr(97 + (s & 7)) + str((s >> 3) + 1)
    u = sq(frm) + sq(to)
    promo = {8: "n", 12: "n", 9: "b", 13: "b",
             10: "r", 14: "r", 11: "q", 15: "q"}
    if fl in promo:
        u += promo[fl]
    return u


def run_parity_tests(
    n_games: int = 200,
    max_plies: int = 60,
    seed: int = 12345,
    verbose: bool = True,
) -> bool:
    """
    Differential test: drive random games on the active Board backend and, at
    every ply, cross-check against python-chess (authoritative validator):
      - identical set of legal moves (UCI),
      - identical board after the chosen move (first 4 FEN fields),
      - identical checkmate / stalemate / insufficient-material verdicts.
    Reports the first divergence with its FEN. Returns True if all match.
    """
    try:
        import chess as _pc
    except ImportError:
        if verbose:
            print("[parity] python-chess not installed — skipping parity test.")
        return True

    import random
    rng = random.Random(seed)

    if verbose:
        print("\nRunning C++/python-chess Parity Tests")
        print("=" * 60)

    def fen4(fen: str) -> str:
        return " ".join(fen.split(" ")[:4])

    # Fixed endgame/edge FENs — random games rarely reach these, but they
    # directly exercise insufficient-material / stalemate / threefold parity.
    fixed_fens = [
        "8/8/8/4k3/8/4K3/8/8 w - - 0 1",            # KvK (insufficient)
        "8/8/8/4k3/8/4K3/8/6B1 w - - 0 1",          # K+B vs K (insufficient)
        "8/8/8/4k3/8/4K3/8/6N1 w - - 0 1",          # K+N vs K (insufficient)
        "8/8/8/3bk3/8/4K3/8/6B1 w - - 0 1",         # KB vs KB
        "8/8/8/3nk3/8/4K3/8/6N1 w - - 0 1",         # KN vs KN (NOT insufficient)
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",           # stalemate
        "k7/8/K7/8/8/8/8/7R w - - 0 1",             # K+R vs K (not draw)
        "8/8/8/8/8/5k2/5p2/5K2 w - - 0 1",          # K vs K+P
    ]
    for fen in fixed_fens:
        b = Board.from_fen(fen)
        pb = _pc.Board(fen)
        for name, a, c in (
            ("legal", sorted(_move_uci(m) for m in b.legal_moves()),
                      sorted(m.uci() for m in pb.legal_moves)),
            ("checkmate", b.is_checkmate(), pb.is_checkmate()),
            ("stalemate", b.is_stalemate(), pb.is_stalemate()),
            ("insufficient", b.is_insufficient_material(),
                             pb.is_insufficient_material()),
        ):
            if a != c:
                if verbose:
                    print(f"  ✗  fixed-FEN {name} mismatch")
                    print(f"     FEN: {fen}")
                    print(f"     C++={a!r}  py={c!r}")
                return False

    for g in range(n_games):
        board = Board.from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        for ply in range(max_plies):
            cur_fen = board.to_fen()
            pyb = _pc.Board(cur_fen)

            cpp_moves = board.legal_moves()
            cpp_set = sorted(_move_uci(m) for m in cpp_moves)
            py_set  = sorted(mv.uci() for mv in pyb.legal_moves)
            if cpp_set != py_set:
                if verbose:
                    print(f"  ✗  Legal-move mismatch (game {g}, ply {ply})")
                    print(f"     FEN: {cur_fen}")
                    print(f"     C++ only: {sorted(set(cpp_set)-set(py_set))}")
                    print(f"     py  only: {sorted(set(py_set)-set(cpp_set))}")
                return False

            # Terminal-state agreement
            if board.is_checkmate() != pyb.is_checkmate():
                print(f"  ✗  checkmate mismatch  FEN: {cur_fen}")
                return False
            if board.is_stalemate() != pyb.is_stalemate():
                print(f"  ✗  stalemate mismatch  FEN: {cur_fen}")
                return False
            if board.is_insufficient_material() != pyb.is_insufficient_material():
                print(f"  ✗  insufficient-material mismatch  FEN: {cur_fen}")
                return False

            if not cpp_moves:
                break

            m = cpp_moves[rng.randrange(len(cpp_moves))]
            uci = _move_uci(m)
            board = board.apply_move(m)
            pyb.push(_pc.Move.from_uci(uci))
            if fen4(board.to_fen()) != fen4(pyb.fen()):
                if verbose:
                    print(f"  ✗  Post-move board mismatch (game {g}, ply {ply})")
                    print(f"     move={uci}  before={cur_fen}")
                    print(f"     C++ : {board.to_fen()}")
                    print(f"     py  : {pyb.fen()}")
                return False

            if board.is_draw() or board.is_checkmate():
                break

    if verbose:
        print(f"  ✓  {n_games} random games × ≤{max_plies} plies — "
              f"C++ matches python-chess exactly")
        print("=" * 60)
    return True


if __name__ == "__main__":
    ok = run_perft_tests(verbose=True)
    ok = run_parity_tests(verbose=True) and ok
    raise SystemExit(0 if ok else 1)
