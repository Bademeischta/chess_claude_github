#!/usr/bin/env python3
"""
tools/lichess_bot.py — let the trained AI play on lichess.org.

Zero extra dependencies: talks to the Lichess Bot API directly over stdlib
urllib (ndjson streaming). No `berserk` needed.

Setup (once):
  1. Create a dedicated Lichess account, then upgrade it to a BOT account:
       https://lichess.org/api#tag/Bot  (account must have zero played games)
  2. Generate a personal API token with the `bot:play` scope.
  3. Put the token in the LICHESS_TOKEN env var (or pass --token).

Run:
  set LICHESS_TOKEN=xxxxxxxx
  python tools/lichess_bot.py --resume checkpoints/step_0050000.pt --sims 200

It accepts incoming standard-chess challenges and plays them with MCTS.
Reuses the same engine pieces as `main.py --play`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe
from model.network import build_model
from model.inference import build_inference_engine
from engine.board import Board, STARTPOS_FEN
from engine.rules import get_game_result, GameResult
from engine.movegen import _move_uci
from mcts.tree import MCTSTree

API = "https://lichess.org"


class Lichess:
    def __init__(self, token: str) -> None:
        self._auth = {"Authorization": f"Bearer {token}"}

    def _req(self, method: str, path: str, data: bytes | None = None):
        return urllib.request.Request(
            API + path, data=data, method=method, headers=self._auth)

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self._req("GET", path)) as r:
            return json.loads(r.read())

    def post(self, path: str) -> None:
        try:
            urllib.request.urlopen(self._req("POST", path, data=b"")).read()
        except Exception as e:  # noqa: BLE001
            print(f"[lichess] POST {path} failed: {e}")

    def stream(self, path: str):
        """Yield decoded JSON objects from an ndjson stream."""
        with urllib.request.urlopen(self._req("GET", path)) as r:
            for raw in r:
                raw = raw.strip()
                if raw:
                    yield json.loads(raw)


def _board_from_moves(moves: str) -> Board:
    """Rebuild the position from a space-separated UCI move list."""
    board = Board.from_fen(STARTPOS_FEN)
    if not moves:
        return board
    for uci in moves.split():
        legal = {_move_uci(m): m for m in board.legal_moves()}
        if uci not in legal:  # should not happen — lichess is authoritative
            break
        board = board.apply_move(legal[uci])
    return board


def _play_game(li: Lichess, game_id: str, my_id: str, ctx: dict) -> None:
    print(f"[lichess] game {game_id} started")
    my_white = None
    for ev in li.stream(f"/api/bot/game/stream/{game_id}"):
        if ev["type"] == "gameFull":
            my_white = ev["white"].get("id") == my_id
            state = ev["state"]
        elif ev["type"] == "gameState":
            state = ev
        else:
            continue

        if state.get("status", "started") != "started":
            print(f"[lichess] game {game_id} over: {state.get('status')}")
            return

        board = _board_from_moves(state.get("moves", ""))
        if get_game_result(board) != GameResult.ONGOING:
            return
        our_turn = (board.side_to_move == 0) == my_white
        if not our_turn:
            continue

        tree = MCTSTree(CONFIG, ctx["model"], ctx["device"],
                        engine=ctx["engine"])
        tree.reset(board)
        mv, _ = tree.search(board, ctx["sims"],
                            temperature=CONFIG.temperature_final,
                            use_dca=False, keep_history=False)
        li.post(f"/api/bot/game/{game_id}/move/{_move_uci(mv)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("LICHESS_TOKEN", ""))
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    if not args.token:
        print("[lichess] No token. Set LICHESS_TOKEN or pass --token "
              "(needs the bot:play scope on a BOT account).")
        return 1

    update_config_from_dict(run_probe(write=False))
    device = torch.device(CONFIG.device)

    from utils.ckpt_arch import latest_checkpoint, match_arch
    ckpt = args.resume or latest_checkpoint(CONFIG.checkpoint_dir)
    match_arch(CONFIG, ckpt)
    model = build_model(CONFIG, compile_model=False).eval()

    if ckpt and Path(ckpt).exists():
        st = torch.load(ckpt, map_location=device, weights_only=True)
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        base.load_state_dict(st["model"])
        print(f"[lichess] Loaded {ckpt}")
    else:
        print("[lichess] No checkpoint — UNTRAINED net.")

    ctx = {
        "model": model, "device": device,
        "engine": build_inference_engine(CONFIG, model, device),
        "sims": args.sims if args.sims is not None else CONFIG.mcts_sims,
    }

    li = Lichess(args.token)
    me = li.get("/api/account")
    my_id = me["id"]
    print(f"[lichess] Logged in as {my_id}. Waiting for challenges…")

    for ev in li.stream("/api/stream/event"):
        t = ev.get("type")
        if t == "challenge":
            ch = ev["challenge"]
            if ch["variant"]["key"] == "standard" and not ch.get("rated", False):
                li.post(f"/api/challenge/{ch['id']}/accept")
                print(f"[lichess] accepted challenge {ch['id']}")
            else:
                li.post(f"/api/challenge/{ch['id']}/decline")
        elif t == "gameStart":
            gid = ev["game"]["id"]
            threading.Thread(target=_play_game,
                             args=(li, gid, my_id, ctx),
                             daemon=True, name=f"game-{gid}").start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
