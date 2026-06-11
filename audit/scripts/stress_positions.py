"""Audit P10.1/P10.2: tricky chess positions through rules + MCTS.

Read-only. Run: python audit/scripts/stress_positions.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe
from utils.ckpt_arch import match_arch, latest_checkpoint

update_config_from_dict(run_probe(write=False))
ckpt = latest_checkpoint(str(ROOT / "checkpoints"))
match_arch(CONFIG, ckpt)

from model.network import build_model  # noqa: E402
from engine.board import Board  # noqa: E402
from engine.rules import get_game_result, GameResult  # noqa: E402
from engine.movegen import _move_uci  # noqa: E402
from mcts.tree import MCTSTree  # noqa: E402

device = torch.device(CONFIG.device)
model = build_model(CONFIG, compile_model=False)
st = torch.load(ckpt, map_location=device, weights_only=True)
base = model._orig_mod if hasattr(model, "_orig_mod") else model
base.load_state_dict(st["model"], strict=False)
model.eval()

OUT: dict = {"checkpoint_step": st.get("step")}


def search(fen, sims, temperature=0.01):
    b = Board.from_fen(fen)
    t = MCTSTree(CONFIG, model, device)
    mv, _ = t.search(b, sims, temperature=temperature, use_dca=False,
                     add_noise=False)
    return b, t, mv


CASES = [
    ("startpos_20", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     {"legal": 20}),
    ("en_passant_available", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
     {"contains_uci": "e5f6"}),
    ("check_limited_escapes", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
     {}),  # not mate yet? verify
    ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", {"terminal": "DRAW"}),
    ("checkmate", "rnb1kbnr/pppp1ppp/8/4p3/6PQ/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 3",
     {}),  # white Qh4#? actually verify mate state below
    ("fools_mate_terminal", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
     {"terminal_check": True}),
    ("promotion_wins", "8/5P2/8/8/8/1k6/8/4K3 w - - 0 1", {"expect_promo": True}),
    ("underpromotion_best", "8/2P5/8/8/8/4k3/8/4K3 w - - 0 1", {}),
    ("zugzwang", "6k1/6p1/7p/8/8/7P/6PK/8 b - - 0 1", {}),
    ("kp_endgame", "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", {}),
    ("halfmove_49", "8/8/8/4k3/8/4K3/8/6R1 w - - 98 80", {}),
]

for name, fen, expect in CASES:
    entry: dict = {"fen": fen}
    try:
        b = Board.from_fen(fen)
        legal = b.legal_moves()
        entry["n_legal"] = len(legal)
        res = get_game_result(b)
        entry["game_result"] = GameResult(res).name
        if "legal" in expect:
            entry["legal_count_ok"] = (len(legal) == expect["legal"])
        if "contains_uci" in expect:
            entry["ep_in_legal"] = expect["contains_uci"] in {
                _move_uci(m) for m in legal}
        if res == GameResult.ONGOING and legal:
            t0 = time.time()
            _, tree, mv = search(fen, 200)
            entry["chosen_move"] = _move_uci(mv)
            entry["root_q"] = round(tree.root_value(), 3)
            entry["search_ms"] = round((time.time() - t0) * 1e3)
            if expect.get("expect_promo"):
                entry["chose_promotion"] = entry["chosen_move"].endswith(
                    ("q", "n", "b", "r")) or entry["chosen_move"][2:4] == "f8"
    except Exception as e:  # noqa: BLE001
        entry["error"] = f"{type(e).__name__}: {e}"
    OUT[name] = entry
    print(name, "->", {k: v for k, v in entry.items() if k != "fen"})

# one legal move only
one_legal_fen = "k7/8/8/8/8/8/r6r/K7 w - - 0 1"  # Ka1, black rooks a2-h2... verify
b = Board.from_fen("7k/8/8/8/8/8/r7/K6r w - - 0 1")
OUT["one_legal_probe"] = {"n_legal": len(b.legal_moves()),
                          "moves": [_move_uci(m) for m in b.legal_moves()]}

# sim-count scaling on a tactical position (mate-in-1 for white: back rank)
m1_fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"  # Ra1-a8#
scal = {}
for sims in (1, 10, 100, 1000):
    t0 = time.time()
    _, tree, mv = search(m1_fen, sims)
    scal[sims] = {"move": _move_uci(mv), "q": round(tree.root_value(), 3),
                  "ms": round((time.time() - t0) * 1e3)}
OUT["mate_in_1_scaling"] = scal
OUT["mate_in_1_found_at"] = {s: v["move"] == "a1a8" for s, v in scal.items()}

# terminal-position search behavior
b = Board.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate, black to move
t = MCTSTree(CONFIG, model, device)
t0 = time.time()
mv, pt = t.search(b, 50, temperature=0.01, use_dca=False, add_noise=False)
OUT["terminal_search"] = {"returned_move": mv, "policy_sum": float(pt.sum()),
                          "ms": round((time.time() - t0) * 1e3)}

out = Path(__file__).with_name("stress_positions.json")
out.write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print(f"-> {out}")
