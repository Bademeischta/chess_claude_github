#!/usr/bin/env python3
"""
tools/uci_engine.py — play the trained net in ANY UCI chess GUI.

Speaks the UCI protocol on stdin/stdout, so you can add it as an engine in
Arena, Cute Chess, BanksiaGUI, Nibbler, En Croissant, ScidvsPC, etc. The GUI
then gives you, for free:
  - infinite analysis  (GUI sends `go infinite`; stop with the analysis button)
  - pondering          (GUI sends `go ponder`; the engine thinks on your time)
  - a live read-out of the engine's calculation (the `info` line below:
    score, depth, node count, and the principal variation it expects)

Register it in your GUI as a UCI engine with command:
    python  C:\\...\\chess_ai\\tools\\uci_engine.py
(optionally add  --resume <ckpt>  /  --sims <n>  as arguments)

Search budget priority: `nodes` → `movetime`/clock → `infinite`/`ponder`
(until stop) → the `Sims` option (default = config mcts_sims).
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe
from utils.ckpt_arch import latest_checkpoint, match_arch
from model.network import build_model
from model.inference import build_inference_engine
from engine.board import Board, STARTPOS_FEN
from engine.rules import get_game_result, GameResult
from engine.movegen import _move_uci
from mcts.tree import MCTSTree


def _q_to_cp(q: float) -> int:
    """Root value in [-1,1] → centipawn-ish score for the GUI."""
    w = max(1e-4, min(1.0 - 1e-4, (q + 1.0) / 2.0))
    return int(round(-400.0 * math.log10(1.0 / w - 1.0)))


def _board_from(cmd_tokens: list[str]) -> Board:
    """Parse a UCI `position ...` token list into a Board."""
    # cmd_tokens excludes the leading 'position'
    if cmd_tokens and cmd_tokens[0] == "startpos":
        board = Board.from_fen(STARTPOS_FEN)
        rest = cmd_tokens[1:]
    elif cmd_tokens and cmd_tokens[0] == "fen":
        fen = " ".join(cmd_tokens[1:7])
        board = Board.from_fen(fen)
        rest = cmd_tokens[7:]
    else:
        return Board.from_fen(STARTPOS_FEN)
    if rest and rest[0] == "moves":
        for uci in rest[1:]:
            legal = {_move_uci(m): m for m in board.legal_moves()}
            if uci in legal:
                board = board.apply_move(legal[uci])
    return board


class UCIEngine:
    def __init__(self, model, cfg, device, default_sims: int) -> None:
        self.model, self.cfg, self.device = model, cfg, device
        self.engine = build_inference_engine(cfg, model, device)
        self.default_sims = default_sims
        self.board = Board.from_fen(STARTPOS_FEN)
        self._stop = threading.Event()
        self._cmdq: "queue.Queue[str]" = queue.Queue()

    # ── stdin reader (keeps `stop`/`isready` responsive mid-search) ──────
    def _reader(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            head = line.split()[0]
            if head == "stop":
                self._stop.set()            # interrupt a running search only
            elif head == "quit":
                self._stop.set()            # break any running search …
                self._cmdq.put("quit")      # … then quit in FIFO order
            elif head == "isready":
                print("readyok", flush=True)
            else:
                self._cmdq.put(line)

    # ── search ──────────────────────────────────────────────────────────
    def _go(self, tokens: list[str]) -> None:
        # Parse limits
        sims_target = None
        deadline = None
        infinite = "infinite" in tokens or "ponder" in tokens

        def val(key):
            return int(tokens[tokens.index(key) + 1]) if key in tokens else None

        if (n := val("nodes")) is not None:
            sims_target = n
        mt = val("movetime")
        if mt is not None:
            deadline = time.time() + mt / 1000.0
        else:
            stm = self.board.side_to_move
            clock = val("wtime") if stm == 0 else val("btime")
            inc = (val("winc") if stm == 0 else val("binc")) or 0
            if clock is not None:
                budget = clock / 25.0 + inc * 0.75      # simple, safe
                deadline = time.time() + max(0.05, budget / 1000.0)
        if sims_target is None and deadline is None and not infinite:
            sims_target = self.default_sims

        if get_game_result(self.board) != GameResult.ONGOING:
            print("bestmove 0000", flush=True)
            return

        tree = MCTSTree(self.cfg, self.model, self.device, engine=self.engine)
        tree.begin_search(self.board, add_noise=False)  # no exploration noise

        t0 = time.time()
        n = 0
        last_info = 0.0
        while True:
            if self._stop.is_set():
                break
            if sims_target is not None and n >= sims_target:
                break
            if deadline is not None and time.time() >= deadline:
                break
            tree.run_one_simulation()
            n += 1
            now = time.time()
            if now - last_info >= 0.3:
                self._emit_info(tree, n, now - t0)
                last_info = now

        self._emit_info(tree, n, max(1e-3, time.time() - t0))
        a = tree.root_analysis()
        best = a["top"][0][0] if a["top"] else 0
        if best == 0:  # safety: fall back to a legal move
            mv, _ = tree.select_move(0.01)
            best = mv
        print(f"bestmove {_move_uci(best) if best else '0000'}", flush=True)

    def _emit_info(self, tree: MCTSTree, nodes: int, dt: float) -> None:
        a = tree.root_analysis()
        cp = _q_to_cp(a["value"])
        pv = " ".join(_move_uci(m) for m in a["pv"]) or "0000"
        nps = int(nodes / dt) if dt > 0 else 0
        print(f"info depth {max(1, a['depth'])} score cp {cp} "
              f"nodes {a['nodes'] or nodes} nps {nps} "
              f"time {int(dt * 1000)} pv {pv}", flush=True)

    # ── main loop ───────────────────────────────────────────────────────
    def run(self) -> None:
        threading.Thread(target=self._reader, daemon=True).start()
        while True:
            try:
                line = self._cmdq.get(timeout=0.2)
            except queue.Empty:
                continue
            tok = line.split()
            cmd = tok[0]
            if cmd == "uci":
                print("id name ChessAI", flush=True)
                print("id author chess_ai", flush=True)
                print(f"option name Sims type spin default {self.default_sims} "
                      f"min 1 max 1000000", flush=True)
                print("uciok", flush=True)
            elif cmd == "isready":
                print("readyok", flush=True)
            elif cmd == "setoption":
                if "Sims" in tok and "value" in tok:
                    try:
                        self.default_sims = int(tok[tok.index("value") + 1])
                    except (ValueError, IndexError):
                        pass
            elif cmd == "ucinewgame":
                self.board = Board.from_fen(STARTPOS_FEN)
            elif cmd == "position":
                self.board = _board_from(tok[1:])
            elif cmd == "go":
                self._stop.clear()
                self._go(tok[1:])
            elif cmd == "quit":
                break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None)
    ap.add_argument("--sims", type=int, default=None)
    args = ap.parse_args()

    # CRITICAL: a UCI engine must print ONLY UCI on stdout. All the noisy
    # setup (system probe banner, model build log, ONNX notes) goes to stderr;
    # restore stdout just before the protocol loop.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        update_config_from_dict(run_probe(write=False))
        device = torch.device(CONFIG.device)
        ckpt = args.resume or latest_checkpoint(CONFIG.checkpoint_dir)
        match_arch(CONFIG, ckpt)
        model = build_model(CONFIG, compile_model=False).eval()
        loaded_msg = "no checkpoint - UNTRAINED net"
        if ckpt and Path(ckpt).exists():
            st = torch.load(ckpt, map_location=device, weights_only=True)
            base = model._orig_mod if hasattr(model, "_orig_mod") else model
            base.load_state_dict(st["model"])
            loaded_msg = f"loaded {ckpt}"
        sims = args.sims if args.sims is not None else CONFIG.mcts_sims
        eng = UCIEngine(model, CONFIG, device, sims)
    finally:
        sys.stdout = real_stdout

    print(f"info string {loaded_msg}", flush=True)
    eng.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
