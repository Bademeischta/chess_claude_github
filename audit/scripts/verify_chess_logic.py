"""Audit P1 runtime verification: chess logic, encoding, parity, .pyd currency.

Writes JSON next to itself. Read-only w.r.t. the repo (no training state touched).
Run: python audit/scripts/verify_chess_logic.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

RESULTS: dict = {}


def section(name):
    def deco(fn):
        def wrapper():
            t0 = time.time()
            try:
                out = fn()
                RESULTS[name] = {"ok": True, "result": out,
                                 "secs": round(time.time() - t0, 2)}
                print(f"[OK]   {name}: {out}")
            except Exception as e:  # noqa: BLE001
                RESULTS[name] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                 "secs": round(time.time() - t0, 2)}
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return wrapper
    return deco


import chess_ext  # noqa: E402
from engine.board import Board, using_cpp_backend  # noqa: E402
from engine.movegen import (  # noqa: E402
    move_to_action_index, action_index_to_move, run_perft_tests,
    run_parity_tests, _move_uci,
)
from engine.rules import get_game_result, GameResult  # noqa: E402

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@section("backend")
def t_backend():
    return {"cpp": using_cpp_backend(),
            "has_stale_counter": hasattr(chess_ext, "stale_move_count")}


@section("startpos_20_moves")
def t_startpos():
    n = len(Board.startpos().legal_moves())
    assert n == 20, f"expected 20, got {n}"
    return n


@section("perft_suite")
def t_perft():
    ok = run_perft_tests(verbose=False)
    assert ok, "perft mismatch — see movegen suite"
    return "all standard positions pass"


@section("parity_200_games")
def t_parity():
    ok = run_parity_tests(n_games=200, max_plies=60, seed=12345, verbose=False)
    assert ok, "C++ vs python-chess divergence"
    return "200 random games x 60 plies match python-chess"


@section("tensor_determinism")
def t_det():
    fens = [STARTPOS,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8"]
    for f in fens:
        a = Board.from_fen(f).to_tensor()
        b = Board.from_fen(f).to_tensor()
        assert np.array_equal(a, b), f"non-deterministic tensor for {f}"
        assert a.shape == (21, 8, 8) and a.dtype == np.float32
    return "identical tensors across constructions"


@section("encoding_bijectivity_random_games")
def t_bijective():
    """Over random games: all legal moves of each position encode to unique
    indices in [0,4672) and decode back to the same move. Both colors."""
    rng = random.Random(7)
    positions = 0
    for g in range(60):
        b = Board.startpos()
        for ply in range(rng.randrange(10, 70)):
            legal = b.legal_moves()
            if not legal:
                break
            idxs = {}
            for m in legal:
                ai = move_to_action_index(m, b.side_to_move)
                assert 0 <= ai < 4672, f"index {ai} out of range ({_move_uci(m)})"
                assert ai not in idxs, (
                    f"collision: {_move_uci(m)} vs {_move_uci(idxs[ai])} "
                    f"idx={ai} fen={b.to_fen()}")
                idxs[ai] = m
                back = action_index_to_move(ai, b)
                assert back == m, (
                    f"decode mismatch {_move_uci(m)} -> {ai} -> {_move_uci(back)} "
                    f"fen={b.to_fen()}")
            positions += 1
            b = b.apply_move(legal[rng.randrange(len(legal))])
            if b.is_draw() or b.is_checkmate() or b.is_stalemate():
                break
    return f"{positions} positions verified, no collisions, decode==encode"


@section("underpromotion_encoding_both_colors")
def t_underpromo():
    # White pawn b7 can capture a8/c8 or push b8; Black pawn g2 mirrored.
    cases = [("n1n5/1P6/8/8/8/k7/8/4K3 w - - 0 1", 0),
             ("4k3/8/K7/8/8/8/6p1/5N1N b - - 0 1", 1)]
    seen = 0
    for fen, stm in cases:
        b = Board.from_fen(fen)
        promos = [m for m in b.legal_moves() if ((m >> 12) & 0xF) >= 8]
        # 3 destinations x 4 pieces = up to 12 promotion moves
        assert len(promos) >= 8, f"too few promo moves ({len(promos)}) for {fen}"
        idxs = set()
        for m in promos:
            ai = move_to_action_index(m, stm)
            assert ai >= 0 and ai not in idxs
            idxs.add(ai)
            assert action_index_to_move(ai, b) == m
            seen += 1
        # underpromos (N/B/R) must land in channels 64..72 of their square
        for m in promos:
            fl = (m >> 12) & 0xF
            if fl in (8, 9, 10, 12, 13, 14):
                ai = move_to_action_index(m, stm)
                assert ai % 73 >= 64, f"underpromo not in promo channels: {ai%73}"
    return f"{seen} promotion encodings bijective incl. underpromos both colors"


@section("pinned_ep_gate_P1_1")
def t_pinned_ep():
    """White: Ke1, Pe2, Rh4. Black: Ka4, Pd4. After e2e4 the ep-capture
    d4xe3 is ILLEGAL (horizontal pin) -> python-chess 'legal' FEN omits ep;
    does C++ set ep_square anyway (pseudo-legal gate)?"""
    import chess as pc
    fen = "8/8/8/8/k2p3R/8/4P3/4K3 w - - 0 1"
    b = Board.from_fen(fen)
    mv = next(m for m in b.legal_moves() if _move_uci(m) == "e2e4")
    after = b.apply_move(mv)
    pb = pc.Board(fen)
    pb.push_uci("e2e4")
    cpp_ep = after.ep_square
    py_fen_ep = pb.fen().split()[3]
    ep_legal_exists = any(pb.is_en_passant(m) for m in pb.legal_moves)
    # ep capture must NOT be in legal moves on either backend
    cpp_ep_moves = [m for m in after.legal_moves() if ((m >> 12) & 0xF) == 5]
    return {"cpp_ep_square": cpp_ep, "pychess_fen_ep_field": py_fen_ep,
            "ep_capture_legal_pychess": ep_legal_exists,
            "cpp_ep_capture_moves": [_move_uci(m) for m in cpp_ep_moves],
            "divergence": (cpp_ep != -1) != (py_fen_ep != "-")}


@section("plane20_parity_bug_P1_2")
def t_plane20():
    """Shuffle knights back to start. After 4 plies the position == startpos
    (2nd occurrence, history index 4 = even -> marker should fire). After 6
    plies the position == position-after-Nf3 (2nd occurrence, index 5 = odd
    -> current code misses it)."""
    seq = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]
    b = Board.startpos()
    out = {}
    snap = {1: None, 4: None, 5: None}
    for i, uci in enumerate(seq, start=1):
        mv = next(m for m in b.legal_moves() if _move_uci(m) == uci)
        b = b.apply_move(mv)
        if i in snap:
            snap[i] = b.to_tensor()[20].max()
    out["after4_repeat_of_startpos_marker"] = float(snap[4])   # expect 1.0
    out["after5_repeat_of_Nf3pos_marker"] = float(snap[5])     # expect 1.0, bug -> 0.0
    out["bug_confirmed"] = (snap[4] == 1.0 and snap[5] == 0.0)
    return out


@section("threefold_detection_cpp")
def t_threefold():
    seq = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2  # startpos occurs 3x
    b = Board.startpos()
    for uci in seq:
        mv = next(m for m in b.legal_moves() if _move_uci(m) == uci)
        b = b.apply_move(mv)
    assert b.is_threefold(), "threefold not detected after 2 full shuffles"
    assert get_game_result(b) == GameResult.DRAW
    return "threefold + draw result detected"


@section("fifty_move_and_insufficient")
def t_draws():
    b = Board.from_fen("8/8/8/4k3/8/4K3/8/6R1 w - - 99 80")
    mv = next(m for m in b.legal_moves() if _move_uci(m) == "g1h1")
    b2 = b.apply_move(mv)
    assert b2.halfmove_clock == 100 and b2.is_fifty_move(), "50-move clock"
    assert Board.from_fen("8/8/8/4k3/8/4K3/8/6B1 w - - 0 1").is_insufficient_material()
    assert not Board.from_fen("8/8/8/3nk3/8/4K3/8/6N1 w - - 0 1").is_insufficient_material()
    return "50-move tick + insufficient-material verdicts correct"


@section("checkmate_stalemate_terminals")
def t_terminals():
    mate = Board.from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert mate.is_checkmate() and get_game_result(mate) == GameResult.BLACK_WIN
    stale = Board.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stale.is_stalemate() and get_game_result(stale) == GameResult.DRAW
    return "fools-mate -> BLACK_WIN; stalemate -> DRAW"


@section("apply_move0_sentinel_P1_5")
def t_move0():
    chess_ext.reset_stale_move_count()
    b = Board.startpos()           # white rook on a1 -> move 0 is a1->a1
    after = b.apply_move(0)
    rook_pass = {"side_flipped": after.side_to_move,
                 "fen": after.to_fen(),
                 "stale_count": chess_ext.stale_move_count()}
    chess_ext.reset_stale_move_count()
    b2 = Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")  # empty a1
    _ = b2.apply_move(0)
    empty_sq = chess_ext.stale_move_count()
    chess_ext.reset_stale_move_count()
    return {"rook_on_a1": rook_pass, "stale_count_empty_a1": int(empty_sq)}


@section("fallback_hash_attr_P1_6")
def t_fallback_hash():
    import subprocess
    code = ("import os,sys; os.environ['CHESS_AI_NO_CPP']='1';"
            f"sys.path.insert(0,{str(ROOT)!r});"
            "from engine.board import Board;"
            "b=Board.startpos();"
            "print(b.hash)")
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=120)
    return {"returncode": r.returncode,
            "stdout": r.stdout.strip()[:100],
            "stderr_tail": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""}


@section("ep_unconditional_vs_gated_pyd_currency_T0_1")
def t_pyd_currency():
    """Current source sets ep_square only when capturable. If the shipped
    .pyd predates that, ep would be set after ANY double push."""
    b = Board.startpos()
    mv = next(m for m in b.legal_moves() if _move_uci(m) == "e2e4")
    after = b.apply_move(mv)  # no black pawn can capture e3
    return {"ep_after_lone_e4": after.ep_square,
            "gated_build": after.ep_square == -1,
            "has_stale_counter_binding": hasattr(chess_ext, "stale_move_count"),
            "pyd_mtime": time.strftime(
                "%Y-%m-%d %H:%M",
                time.localtime(os.path.getmtime(
                    ROOT / "chess_ext.cp311-win_amd64.pyd")))}


@section("movegen_benchmark")
def t_bench():
    kasparov_db_g6 = "1r1q1rk1/p1p2pp1/1pn1pn1p/3p4/Q2P4/2NBPN2/PP3PPP/R4RK1 w - - 0 12"
    res = {}
    for name, fen in [("startpos", STARTPOS), ("middlegame", kasparov_db_g6)]:
        b = Board.from_fen(fen)
        n = len(b.legal_moves())
        t0 = time.perf_counter()
        it = 0
        while time.perf_counter() - t0 < 1.0:
            b.legal_moves()
            it += 1
        dt = time.perf_counter() - t0
        res[name] = {"legal_moves": n,
                     "legal_moves_calls_per_s": round(it / dt),
                     "us_per_call": round(dt / it * 1e6, 1)}
    b = Board.startpos()
    t0 = time.perf_counter()
    nodes = b.perft(4)
    dt = time.perf_counter() - t0
    res["perft4"] = {"nodes": nodes, "secs": round(dt, 3),
                     "nodes_per_s": round(nodes / dt)}
    return res


if __name__ == "__main__":
    for fn in [t_backend, t_startpos, t_perft, t_parity, t_det, t_bijective,
               t_underpromo, t_pinned_ep, t_plane20, t_threefold, t_draws,
               t_terminals, t_move0, t_fallback_hash, t_pyd_currency, t_bench]:
        fn()
    out = Path(__file__).with_name("verify_chess_logic.json")
    out.write_text(json.dumps(RESULTS, indent=2, default=lambda o: o.item()
                              if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    print(f"\nResults -> {out}")
    n_fail = sum(1 for v in RESULTS.values() if not v["ok"])
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} sections OK")
