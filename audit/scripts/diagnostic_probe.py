"""
audit/scripts/diagnostic_probe.py — themed strength/weakness probe of the
trained net, with Stockfish as the oracle (2026-07-20).

For each themed FEN: the net searches (at one or more sim budgets) and we
record its best move + eval; Stockfish (fixed depth) gives the best move,
the position eval, and the eval AFTER the net's move -> centipawn loss
from the mover's POV. Themes isolate a skill:

  * mate / hanging piece  (control — must pass)
  * multi-move tactic     (horizon; run at rising sim budgets)
  * underpromotion        (only a knight-promo wins)
  * endgame technique     (K+R / K+Q vs K; must preserve the win)
  * zugzwang comprehension(value-head sign on a lost trebuchet)
  * opening breadth        (principled move vs a random one; h5 re-test)

Shares the GPU with any running training (the load governor throttles it).
Run: python audit/scripts/diagnostic_probe.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import chess
import chess.engine
import torch

from config import CONFIG
from engine.board import Board
from engine.harness import load_engine, run_search
from engine.movegen import _move_uci
from mcts.tree import MCTSTree
from model.inference import TorchInferenceEngine
from utils.sf_anchor import resolve_stockfish_path

SF_DEPTH = 18
MATE_CP = 100_000

# (theme, fen, [sim budgets], short human note on the expected idea)
TESTS = [
    ("Kontrolle · Matt in 1",
     "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", [800], "Ra8#"),
    ("Kontrolle · haengende Dame (1 Zug)",
     "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", [800], "exd5"),
    ("Taktik · Springergabel (2 Zuege)",
     "4r1k1/7p/8/3N4/8/8/6PP/6K1 w - - 0 1", [256, 1024, 4096], "Nf6+ dann Nxe8"),
    ("Taktik · Rueckrang-Matt mit Schlag",
     "3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", [256, 1024, 4096], "Rxd8#"),
    ("Unterverwandlung · e8=N Gabel",
     "8/4P3/3k1q2/8/8/8/8/6K1 w - - 0 1", [256, 1024, 4096], "e8=N+ gewinnt Dame"),
    ("Endspiel · K+T gegen K (Technik)",
     "4k3/8/8/8/8/8/8/R3K3 w - - 0 1", [2048], "Gewinn halten/verkuerzen"),
    ("Endspiel · K+D gegen K (Kontrolle)",
     "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", [2048], "Gewinn halten"),
    ("Zugzwang · Trebuchet, Weiss am Zug verloren",
     "8/8/8/3k4/3p4/3K4/8/8 w - - 0 1", [2048], "Eval muss negativ sein; Kc3/Ke3"),
    ("Eroeffnung · Grundstellung",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", [2048],
     "e4/d4/Nf3/c4"),
    ("Eroeffnung · Antwort auf 1.e4 (h5-Retest)",
     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", [2048],
     "e5/c5/e6/c6"),
    ("Eroeffnung · Antwort auf 1.d4",
     "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1", [2048],
     "Nf6/d5/e6/g6"),
]


def q_to_cp(q: float) -> int:
    w = max(1e-4, min(1.0 - 1e-4, (q + 1.0) / 2.0))
    return int(round(-400.0 * math.log10(1.0 / w - 1.0)))


def verdict(loss_cp: int) -> str:
    if loss_cp <= 20:
        return "TOP"
    if loss_cp <= 60:
        return "gut"
    if loss_cp <= 150:
        return "Ungenauigkeit"
    if loss_cp <= 400:
        return "Fehler"
    return "BLUNDER"


def ai_move(handle, fen: str, sims: int):
    ob = Board.from_fen(fen)
    tree = MCTSTree(handle.cfg, handle.model, handle.device,
                    engine=handle.engine)
    tree.reset(ob)
    tree.begin_search(ob, add_noise=False)
    run_search(tree, sims=sims, batch_size=32)
    a = tree.root_analysis(top_k=3, pv_len=6)
    top = a.get("top", [])
    if not top:
        return None, 0.0, []
    best = _move_uci(top[0][0])
    cands = [(_move_uci(m), int(n), q_to_cp(q)) for m, n, q in top]
    return best, q_to_cp(a.get("value", 0.0)), cands


def sf_eval(sf, board: chess.Board):
    info = sf.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
    score = info["score"].pov(board.turn)
    best = info["pv"][0] if info.get("pv") else None
    return score, best


def main() -> int:
    sf_path = resolve_stockfish_path(CONFIG)
    if not sf_path:
        print("[probe] kein Stockfish gefunden — abbruch.")
        return 1

    handle = load_engine(str(_root / "checkpoints" / "latest.pt"),
                         build_engine=False,
                         log=lambda m: print(m.replace("[harness]", "[probe]")))
    handle = handle.__class__(model=handle.model,
                              engine=TorchInferenceEngine(handle.model),
                              device=handle.device, cfg=handle.cfg,
                              ckpt_path=handle.ckpt_path, step=handle.step)
    print(f"[probe] Netz: {handle.ckpt_path} (step {handle.step})\n")

    sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    sf.configure({"Threads": 2, "Hash": 128})

    t0 = time.time()
    try:
        for theme, fen, budgets, note in TESTS:
            pyb = chess.Board(fen)
            score_before, sf_best = sf_eval(sf, pyb)
            sf_cp = score_before.score(mate_score=MATE_CP)
            print(f"── {theme}")
            print(f"   FEN {fen}")
            print(f"   Idee: {note}  |  SF-Bestzug: "
                  f"{sf_best.uci() if sf_best else '?'}  "
                  f"(SF-Eval {sf_cp/100:+.2f})")
            for sims in budgets:
                best, ai_cp, cands = ai_move(handle, fen, sims)
                if best is None:
                    print(f"   [{sims:>4} Sims] KEIN Zug (Terminalstellung)")
                    continue
                # Centipawn loss of the AI's move per Stockfish.
                pyb2 = chess.Board(fen)
                try:
                    mv = chess.Move.from_uci(best)
                    legal = mv in pyb2.legal_moves
                except Exception:
                    legal = False
                if not legal:
                    print(f"   [{sims:>4} Sims] {best} ILLEGAL?!")
                    continue
                pyb2.push(mv)
                score_after, _ = sf_eval(sf, pyb2)
                # score_after is POV of the side to move AFTER our move (the
                # opponent) — negate to get the original mover's POV.
                after_cp = -score_after.score(mate_score=MATE_CP)
                loss = max(0, sf_cp - after_cp)
                match = "=SF" if (sf_best and best == sf_best.uci()) else "   "
                cand_s = " ".join(f"{u}({n},{c/100:+.1f})" for u, n, c in cands)
                print(f"   [{sims:>4} Sims] {best} {match}  "
                      f"KI-Eval {ai_cp/100:+.2f}  "
                      f"SF-Verlust {loss/100:+.2f}  -> {verdict(loss)}")
                print(f"              Top: {cand_s}")
            print()
    finally:
        sf.quit()
    print(f"[probe] fertig in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
