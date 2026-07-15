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

from config import CONFIG
from engine.harness import load_engine, run_search, time_budget_s
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


def _full_strength_elo(cfg) -> float:
    """Best known objective Elo of the CURRENT net at full sims.

    Reads the newest measurement from checkpoints/sf_elo_history.csv (the
    automatic Stockfish-anchor probe, utils/sf_anchor.py). Falls back to a
    rough constant when no measurement exists yet — the UCI_Elo mapping is
    then approximate until the first probe has run."""
    try:
        lines = (Path(cfg.checkpoint_dir) / "sf_elo_history.csv") \
            .read_text("utf-8").strip().splitlines()
        return float(lines[-1].split(",")[6])
    except (OSError, ValueError, IndexError):
        return 1000.0  # pre-measurement ballpark (2026-07 nets)


class _GoControl:
    """stop-Event wrapper implementing real UCI pondering semantics.

    In ponder mode the search runs unbounded until either `stop` (GUI
    aborts: emit bestmove immediately) or `ponderhit` (prediction was
    right: from THAT moment think for the normal clock budget, reusing
    everything searched so far)."""

    def __init__(self, stop_ev: threading.Event,
                 ponderhit_ev: "threading.Event | None",
                 budget_s: "float | None") -> None:
        self._stop = stop_ev
        self._ph = ponderhit_ev
        self._budget = budget_s
        self._deadline: "float | None" = None

    def is_set(self) -> bool:
        if self._stop.is_set():
            return True
        if self._ph is not None and self._ph.is_set():
            if self._deadline is None:
                # Ponderhit just registered: start the clock budget now.
                self._deadline = time.time() + (self._budget or 0.0)
            return time.time() >= self._deadline
        return False


class UCIEngine:
    def __init__(self, handle, default_sims: int) -> None:
        self.model, self.cfg = handle.model, handle.cfg
        self.device, self.engine = handle.device, handle.engine
        self.default_sims = default_sims
        self.board = Board.from_fen(STARTPOS_FEN)
        self._stop = threading.Event()
        self._ponderhit = threading.Event()
        self._cmdq: "queue.Queue[str]" = queue.Queue()
        # UCI options
        self.multipv = 1
        self.limit_strength = False
        self.uci_elo = 1000
        self._full_elo = _full_strength_elo(self.cfg)

    # ── stdin reader (keeps `stop`/`ponderhit`/`isready` responsive) ─────
    def _reader(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            head = line.split()[0]
            if head == "stop":
                self._stop.set()            # interrupt a running search only
            elif head == "ponderhit":
                self._ponderhit.set()       # convert ponder → timed search
            elif head == "quit":
                self._stop.set()            # break any running search …
                self._cmdq.put("quit")      # … then quit in FIFO order
            elif head == "isready":
                print("readyok", flush=True)
            else:
                self._cmdq.put(line)

    # ── strength limiting ────────────────────────────────────────────────
    def _effective_sims(self, sims_target: "int | None") -> "int | None":
        """Map UCI_Elo to a simulation budget when UCI_LimitStrength is on.

        Model: each halving of the search budget costs ~ELO_PER_DOUBLING
        Elo (literature ballpark for AlphaZero-style nets at small budgets).
        Anchored at the newest objective SF-anchor measurement
        (sf_elo_history.csv) = the net's full-sims strength. Approximate by
        construction — the probe curve refines the anchor over time."""
        if not self.limit_strength:
            return sims_target
        ELO_PER_DOUBLING = 90.0
        base = sims_target if sims_target is not None else self.default_sims
        factor = 2.0 ** ((float(self.uci_elo) - self._full_elo)
                         / ELO_PER_DOUBLING)
        return max(2, min(base, int(round(base * factor))))

    # ── search ──────────────────────────────────────────────────────────
    def _go(self, tokens: list[str]) -> None:
        # Parse limits
        sims_target = None
        ponder = "ponder" in tokens
        infinite = "infinite" in tokens

        def val(key):
            return int(tokens[tokens.index(key) + 1]) if key in tokens else None

        if (n := val("nodes")) is not None:
            sims_target = n
        # Shared time-control heuristic (engine.harness): movetime wins,
        # else clock/25 + inc*0.75 — one formula for UCI/web/terminal.
        stm = self.board.side_to_move
        budget_s = time_budget_s(
            val("wtime") if stm == 0 else val("btime"),
            (val("winc") if stm == 0 else val("binc")) or 0,
            movetime_ms=val("movetime"),
        )
        if sims_target is None and budget_s is None \
                and not infinite and not ponder:
            sims_target = self.default_sims
        sims_target = self._effective_sims(sims_target)

        if get_game_result(self.board) != GameResult.ONGOING:
            print("bestmove 0000", flush=True)
            return

        tree = MCTSTree(self.cfg, self.model, self.device, engine=self.engine)
        tree.begin_search(self.board, add_noise=False)  # no exploration noise

        # Batched search (plan C1): the driver collects `batch` leaves per
        # NN forward (virtual loss diversifies the paths) — the same core
        # the web GUI and self-play use. Measured 124 → ~8300 nps here.
        # stop/ponderhit/deadline stay responsive at one-round latency.
        #
        # Pondering (plan C2): `go ponder` searches without a deadline;
        # `ponderhit` starts the normal clock budget from that moment
        # (handled inside _GoControl), `stop` aborts immediately. Either
        # way the bestmove below refers to the pondered position, exactly
        # as the UCI spec wants.
        t0 = time.time()
        batch = int(getattr(self.engine, "batch_size", 0)) or 32
        if ponder:
            control = _GoControl(self._stop, self._ponderhit, budget_s)
            deadline = None
        else:
            control = self._stop
            deadline = time.time() + budget_s if budget_s is not None else None
        n = run_search(
            tree,
            sims=sims_target if not ponder else None,
            deadline=deadline,
            stop=control,
            batch_size=batch,
            info_cb=lambda done: self._emit_info(
                tree, done, max(1e-3, time.time() - t0)),
        )

        self._emit_info(tree, n, max(1e-3, time.time() - t0))
        a = tree.root_analysis(top_k=max(1, self.multipv), pv_len=8)
        best = a["top"][0][0] if a["top"] else 0
        if best == 0:  # safety: fall back to a legal move
            mv, _ = tree.select_move(0.01)
            best = mv
        # `ponder <move>` = our expected reply — the GUI uses it to issue
        # the next `go ponder`. Without it pondering never activates.
        ponder_mv = a["pv"][1] if len(a.get("pv", [])) >= 2 else 0
        line = f"bestmove {_move_uci(best) if best else '0000'}"
        if best and ponder_mv:
            line += f" ponder {_move_uci(ponder_mv)}"
        print(line, flush=True)

    def _emit_info(self, tree: MCTSTree, nodes: int, dt: float) -> None:
        a = tree.root_analysis(top_k=max(1, self.multipv), pv_len=8)
        nps = int(nodes / dt) if dt > 0 else 0
        if self.multipv <= 1:
            cp = _q_to_cp(a["value"])
            pv = " ".join(_move_uci(m) for m in a["pv"]) or "0000"
            print(f"info depth {max(1, a['depth'])} score cp {cp} "
                  f"nodes {a['nodes'] or nodes} nps {nps} "
                  f"time {int(dt * 1000)} pv {pv}", flush=True)
            return
        # MultiPV (plan C2): one line per candidate, ranked by visits.
        # Each candidate's `line` is its own expected continuation and its
        # q is already in the side-to-move frame (root_analysis contract).
        for rank, e in enumerate(a.get("ext", [])[:self.multipv], start=1):
            pv = " ".join(_move_uci(m) for m in e["line"]) or "0000"
            print(f"info multipv {rank} depth {max(1, a['depth'])} "
                  f"score cp {_q_to_cp(e['q'])} "
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
                print("option name MultiPV type spin default 1 min 1 max 16",
                      flush=True)
                print("option name Ponder type check default false", flush=True)
                print("option name UCI_LimitStrength type check default false",
                      flush=True)
                print(f"option name UCI_Elo type spin "
                      f"default {int(self._full_elo)} min 200 max 3000",
                      flush=True)
                print("uciok", flush=True)
            elif cmd == "isready":
                print("readyok", flush=True)
            elif cmd == "setoption":
                self._setoption(tok)
            elif cmd == "ucinewgame":
                self.board = Board.from_fen(STARTPOS_FEN)
            elif cmd == "position":
                self.board = _board_from(tok[1:])
            elif cmd == "go":
                self._stop.clear()
                self._ponderhit.clear()
                self._go(tok[1:])
            elif cmd == "quit":
                break

    def _setoption(self, tok: list[str]) -> None:
        if "name" not in tok or "value" not in tok:
            return
        try:
            name = " ".join(tok[tok.index("name") + 1: tok.index("value")])
            value = tok[tok.index("value") + 1]
        except (ValueError, IndexError):
            return
        name_l = name.lower()
        try:
            if name_l == "sims":
                self.default_sims = max(1, int(value))
            elif name_l == "multipv":
                self.multipv = max(1, min(16, int(value)))
            elif name_l == "uci_limitstrength":
                self.limit_strength = value.lower() == "true"
            elif name_l == "uci_elo":
                self.uci_elo = int(value)
            # "Ponder" needs no state: the GUI drives it via `go ponder`.
        except ValueError:
            pass


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
        # Shared bring-up (engine.harness): probe → Syzygy auto-detect →
        # checkpoint discovery/match_arch → model+weights → inference
        # engine. All setup chatter lands on stderr here.
        handle = load_engine(args.resume)
        loaded_msg = (f"loaded {handle.ckpt_path}" if handle.ckpt_path
                      else "no checkpoint - UNTRAINED net")
        sims = args.sims if args.sims is not None else CONFIG.mcts_sims
        eng = UCIEngine(handle, sims)
    finally:
        sys.stdout = real_stdout

    print(f"info string {loaded_msg}", flush=True)
    eng.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
