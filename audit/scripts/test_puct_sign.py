"""Decisive mechanical test for the PUCT Q-perspective inversion.

Uses the PYTHON MCTS fallback with a STUB evaluator (uniform priors,
value=0) so no trained weights are involved. In a mate-in-1 position a
correct PUCT (parent-perspective Q) must concentrate visits on the mating
move (its first visit returns a terminal win for the mover). An inverted
PUCT (child-perspective Q used directly) visits it once and then avoids it.

Run: set CHESS_AI_NO_CPP=1 implicitly via env in-process BEFORE imports.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CHESS_AI_NO_CPP"] = "1"  # force Python MCTS + python-chess board

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import CONFIG
from engine.board import Board
from engine.movegen import _move_uci, move_to_action_index
from mcts.tree import MCTSTree


class StubEngine:
    """value = 0 for every position, uniform logits."""

    def infer(self, board_t, hist_t):
        import torch
        B = board_t.shape[0]
        return (torch.zeros(B, CONFIG.num_actions),
                torch.zeros(B))


class StubModel:  # only needs .infer through the engine; tree wants nothing else
    pass


CONFIG.device = "cpu"
CONFIG.precision = "fp32"

import torch  # noqa: E402

device = torch.device("cpu")

# Mate in 1: white Ra1->a8#, back-rank. 16 legal root moves.
FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"

tree = MCTSTree(CONFIG, StubModel(), device, engine=StubEngine())
board = Board.from_fen(FEN)
tree.reset(board)
tree._expand_root()           # uniform priors via stub; no noise
for _ in range(300):
    tree.run_one_simulation()

root = tree._root_node
rows = []
for mv, child in root.children.items():
    rows.append({"move": _move_uci(mv), "N": child.N,
                 "Q_child_persp": round(child.Q, 3), "P": round(child.P, 4)})
rows.sort(key=lambda r: -r["N"])

mate = next(r for r in rows if r["move"] == "a1a8")
most_visited = rows[0]
sel_move, _ = tree.select_move(0.01)

result = {
    "fen": FEN,
    "root_N": root.N,
    "root_Q_stm_persp": round(root.Q, 3),
    "mate_move": mate,
    "most_visited": most_visited,
    "greedy_selection": _move_uci(sel_move),
    "top5": rows[:5],
    "verdict_inverted": mate["N"] <= 2 and most_visited["move"] != "a1a8",
}
print(json.dumps(result, indent=2))
out = Path(__file__).with_name("test_puct_sign.json")
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
