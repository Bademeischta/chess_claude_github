#!/usr/bin/env python3
"""
tools/elo_vs_stockfish.py — objective ELO measurement against Stockfish.

Plays a series against Stockfish pinned to a known strength
(UCI_LimitStrength + UCI_Elo, which is a real ELO anchor), alternating
colours, and reports our estimated ELO with a 95 % confidence interval
(point estimate = Stockfish_Elo + elo_diff_from_series).

    python tools/elo_vs_stockfish.py --stockfish C:\\path\\stockfish.exe \\
        --games 30 --elo 1500 --resume checkpoints/step_0050000.pt

Read-only w.r.t. training (loads a checkpoint, plays games, prints a number).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import chess
import chess.engine

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe
from utils.elo import ELOSystem
from model.network import build_model
from model.inference import build_inference_engine
from engine.board import Board
from engine.movegen import _move_uci
from mcts.tree import MCTSTree


from utils.ckpt_arch import latest_checkpoint, match_arch


def _our_move(tree: MCTSTree, board: Board, sims: int) -> int:
    move, _ = tree.search(board, sims, temperature=CONFIG.temperature_final,
                          use_dca=False, keep_history=True)
    return move


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", default=CONFIG.stockfish_path,
                    help="Path to Stockfish UCI binary.")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--elo", type=int, default=1500,
                    help="Stockfish UCI_Elo anchor (≈1320–3190).")
    ap.add_argument("--sims", type=int, default=None,
                    help="Our MCTS sims/move (default: config).")
    ap.add_argument("--movetime", type=float, default=0.1,
                    help="Stockfish thinking time per move (s).")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint (default: newest in checkpoint dir).")
    args = ap.parse_args()

    if not args.stockfish or not Path(args.stockfish).exists():
        print("[elo] Need a valid --stockfish path (or set "
              "config.stockfish_path).")
        return 1

    update_config_from_dict(run_probe(write=False))
    device = torch.device(CONFIG.device)

    ckpt = args.resume or latest_checkpoint(CONFIG.checkpoint_dir)
    match_arch(CONFIG, ckpt)
    model = build_model(CONFIG, compile_model=False).eval()

    if ckpt and Path(ckpt).exists():
        st = torch.load(ckpt, map_location=device, weights_only=True)
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        base.load_state_dict(st["model"])
        print(f"[elo] Loaded {ckpt} (step {st.get('step', '?')})")
    else:
        print("[elo] No checkpoint — measuring an UNTRAINED net.")

    engine_inf = build_inference_engine(CONFIG, model, device)
    sims = args.sims if args.sims is not None else CONFIG.mcts_sims

    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo})
    except chess.engine.EngineError:
        print("[elo] Stockfish rejected UCI_Elo; using default strength.")

    wins = draws = losses = 0
    for g in range(args.games):
        we_white = (g % 2 == 0)
        pyb = chess.Board()
        ob = Board.from_fen(pyb.fen())
        tree = MCTSTree(CONFIG, model, device, engine=engine_inf)
        tree.reset(ob)

        while not pyb.is_game_over(claim_draw=True):
            our_turn = (pyb.turn == chess.WHITE) == we_white
            if our_turn:
                mv = _our_move(tree, ob, sims)
                uci = _move_uci(mv)
                pyb.push(chess.Move.from_uci(uci))
            else:
                res = sf.play(pyb, chess.engine.Limit(time=args.movetime))
                pyb.push(res.move)
                uci = res.move.uci()
            ob = Board.from_fen(pyb.fen())
            tree.reset(ob)  # re-sync tree to the authoritative position

        res = pyb.result(claim_draw=True)  # "1-0","0-1","1/2-1/2"
        if res == "1/2-1/2":
            draws += 1
        elif (res == "1-0") == we_white:
            wins += 1
        else:
            losses += 1
        print(f"  game {g+1}/{args.games}  "
              f"{'W' if we_white else 'B'}  result={res}  "
              f"(W{wins} D{draws} L{losses})", flush=True)

    sf.quit()

    diff, lo, hi = ELOSystem.elo_difference_ci(wins, draws, losses)
    print("\n" + "=" * 52)
    print(f"  vs Stockfish@{args.elo}  ({args.games} games, {sims} sims)")
    print(f"  W{wins} D{draws} L{losses}")
    print(f"  ELO diff vs SF : {diff:+.0f}  (95% CI {lo:+.0f}..{hi:+.0f})")
    print(f"  Estimated ELO  : {args.elo + diff:.0f}  "
          f"(CI {args.elo + lo:.0f}..{args.elo + hi:.0f})")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
