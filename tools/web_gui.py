#!/usr/bin/env python3
"""
tools/web_gui.py — modern browser UI to play the trained AI.

Single-page app served from a stdlib http.server. Features:

  * Live AI-thinking display via Server-Sent Events: eval bar, top moves
    with visit counts and Q-values, principal variation, sim progress.
  * All AI settings live in the UI (no CLI flags needed mid-game):
      - Thinking mode: fixed sims | time-budget | dynamic (DCA)
      - Pondering (AI keeps thinking on YOUR time, reuses the tree)
      - Temperature, color, new-game
  * AI commentary on every user move ("I agree, this was my top choice" /
    "Y was stronger — losing ~0.4 in eval").
  * Move list with per-move evaluation, takeback, resign, hint.
  * Dark, polished UI; works fully offline once the page is open
    (the CDN-loaded chessboard.js needs internet on first open).

Backend reuses the production engine: Board, MCTSTree, build_inference_engine.

    python tools/web_gui.py
    python tools/web_gui.py --resume checkpoints/step_0050000.pt --port 8000
    # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch  # noqa: E402

from config import CONFIG, update_config_from_dict   # noqa: E402
from utils.system_probe import run_probe              # noqa: E402
from model.network import build_model                 # noqa: E402
from model.inference import build_inference_engine    # noqa: E402
from engine.board import Board, STARTPOS_FEN          # noqa: E402
from engine.rules import get_game_result, GameResult  # noqa: E402
from engine.movegen import _move_uci                  # noqa: E402
from mcts.tree import MCTSTree                        # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Game session — single local game, but cleanly encapsulated state +
# locking so the SSE thread, the ponder thread, and the HTTP handlers
# never step on each other.
# ─────────────────────────────────────────────────────────────────────────

class GameSession:
    """
    All per-game state. One server = one session (this is a local tool
    intended for the user themselves; multi-user would need keyed sessions).
    """

    def __init__(self, model, device, engine, cfg) -> None:
        self.model  = model
        self.device = device
        self.engine = engine
        self.cfg    = cfg

        # Mutable game state — guarded by self._lock.
        self._lock = threading.RLock()
        self.board: Optional[Board] = None
        self.tree:  Optional[MCTSTree] = None
        self.history: list[dict] = []   # full move history (both colours)
        self.human_white: bool = True
        self.game_over: bool = False
        self.result_text: str = ""

        # AI settings — settable from the UI. Defaults are tuned for the
        # batched-sims path: the CUDA-Graph engine fires one forward per
        # ENGINE.batch_size leaves, so 1500 sims is ~1 second of GPU work
        # on RTX 5070 rather than the 30s it would have been with single-
        # leaf inference.
        self.think_mode:   str = "sims"
        self.think_sims:   int = 1500
        self.think_time_s: float = 5.0
        self.dynamic_min:  int = 600
        self.dynamic_max:  int = 6000
        self.ponder_on:    bool = True
        self.temperature:  float = 0.05        # near-greedy for real play
        # Contempt: breaks queen-shuffle / 3-fold-rep mid-game patterns
        # where the value head can't tell several lines apart and MCTS
        # converges on "no progress" repetition. 0.10 means terminal draws
        # are treated as if they were small losses, pushing MCTS toward
        # fighting lines whenever any non-drawn alternative exists. Does
        # NOT affect tablebase draws (those are genuinely drawn and the
        # engine respects them). Pushed into the *shared* cfg below — only
        # this web-UI process reads it; the training process has its own
        # cfg with mcts_contempt=0.0.
        self.contempt:     float = 0.10
        try:
            self.cfg.mcts_contempt = self.contempt  # type: ignore[attr-defined]
        except Exception:
            pass  # legacy cfg without the field — MCTS getattr fallback covers us

        # Ponder thread (runs MCTS on the current root while it's your turn).
        self._ponder_thread: Optional[threading.Thread] = None
        self._ponder_stop = threading.Event()

        # Search activity flag — set during the AI's own thinking.
        self._searching = threading.Event()

        # SSE pub/sub — each connected client gets its own queue.
        self._sse_clients: list[queue.Queue] = []
        self._sse_lock = threading.Lock()

    # ── SSE plumbing ─────────────────────────────────────────────────────

    def sse_subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._sse_lock:
            self._sse_clients.append(q)
        # Replay the current state so a new tab is never stale.
        self._snapshot_into(q)
        return q

    def sse_unsubscribe(self, q: queue.Queue) -> None:
        with self._sse_lock:
            if q in self._sse_clients:
                self._sse_clients.remove(q)

    def emit(self, event: str, data: dict) -> None:
        with self._sse_lock:
            for q in list(self._sse_clients):
                try:
                    q.put_nowait({"event": event, "data": data})
                except queue.Full:
                    # A client that can't keep up is dropped silently.
                    pass

    def _snapshot_into(self, q: queue.Queue) -> None:
        """Push current game state to a freshly-subscribed client."""
        with self._lock:
            board = self.board
            if board is None:
                return
            payload = {
                "fen": board.to_fen(),
                "history": self.history,
                "human_white": self.human_white,
                "game_over": self.game_over,
                "result": self.result_text,
                "side_to_move": int(board.side_to_move),
                "settings": self.settings_dict(),
            }
            try:
                q.put_nowait({"event": "snapshot", "data": payload})
            except queue.Full:
                pass

    # ── Settings ─────────────────────────────────────────────────────────

    def settings_dict(self) -> dict:
        return {
            "think_mode":   self.think_mode,
            "think_sims":   self.think_sims,
            "think_time_s": self.think_time_s,
            "dynamic_min":  self.dynamic_min,
            "dynamic_max":  self.dynamic_max,
            "ponder_on":    self.ponder_on,
            "temperature":  self.temperature,
            "contempt":     self.contempt,
            "human_white":  self.human_white,
        }

    def apply_settings(self, data: dict) -> None:
        with self._lock:
            tm = str(data.get("think_mode", self.think_mode))
            if tm in ("sims", "time", "dynamic"):
                self.think_mode = tm
            self.think_sims   = max(20,  min(10_000, int(data.get("think_sims",   self.think_sims))))
            self.think_time_s = max(0.2, min(120.0,  float(data.get("think_time_s", self.think_time_s))))
            self.dynamic_min  = max(20,  min(self.dynamic_max - 1,
                                            int(data.get("dynamic_min", self.dynamic_min))))
            self.dynamic_max  = max(self.dynamic_min + 1,
                                    min(20_000, int(data.get("dynamic_max", self.dynamic_max))))
            self.ponder_on    = bool(data.get("ponder_on", self.ponder_on))
            self.temperature  = max(0.001, min(2.0, float(data.get("temperature", self.temperature))))
            self.contempt     = max(0.0,   min(0.5, float(data.get("contempt",    self.contempt))))
            try:
                self.cfg.mcts_contempt = self.contempt  # type: ignore[attr-defined]
            except Exception:
                pass
        # If user just turned ponder on/off, react now.
        if self.ponder_on and not self.game_over and self._is_human_turn():
            self._start_ponder()
        else:
            self._stop_ponder()
        self.emit("settings", self.settings_dict())

    # ── Game life-cycle ──────────────────────────────────────────────────

    def new_game(self, human_white: bool) -> None:
        self._stop_ponder()
        with self._lock:
            self.human_white  = bool(human_white)
            self.board        = Board.from_fen(STARTPOS_FEN)
            self.tree         = MCTSTree(self.cfg, self.model, self.device,
                                         engine=self.engine)
            self.tree.reset(self.board)
            self.history      = []
            self.game_over    = False
            self.result_text  = ""
        self.emit("new_game", {
            "fen": STARTPOS_FEN,
            "human_white": self.human_white,
            "settings": self.settings_dict(),
        })
        # If user picked black, AI plays first.
        if not self.human_white:
            self._ai_move_async()
        else:
            self._start_ponder()

    def set_fen(self, fen: str, human_white: Optional[bool] = None) -> dict:
        """Start a fresh game from an arbitrary FEN (analysis / endgame
        practice / 'play from here'). If human_white is omitted, the human
        takes the side to move."""
        self._stop_ponder()
        try:
            nb = Board.from_fen(fen.strip())
        except Exception:
            return {"error": "invalid FEN"}
        with self._lock:
            if human_white is None:
                human_white = (int(nb.side_to_move) == 0)
            self.human_white = bool(human_white)
            self.board       = nb
            self.tree        = MCTSTree(self.cfg, self.model, self.device,
                                        engine=self.engine)
            self.tree.reset(self.board)
            self.history     = []
            self.game_over   = False
            self.result_text = ""
            self._check_game_over()
        self.emit("new_game", {
            "fen": nb.to_fen(),
            "human_white": self.human_white,
            "settings": self.settings_dict(),
        })
        if not self.game_over and not self._is_human_turn():
            self._ai_move_async()
        elif not self.game_over:
            self._start_ponder()
        return {"ok": True, "fen": nb.to_fen(), "human_white": self.human_white}

    def _is_human_turn(self) -> bool:
        if self.board is None:
            return False
        return (int(self.board.side_to_move) == 0) == self.human_white

    # ── User move ────────────────────────────────────────────────────────

    def user_move(self, uci: str) -> dict:
        """Apply a user move + comment on it + kick off AI's reply."""
        self._stop_ponder()  # ponder tree may be stale after the user picks something other than the predicted reply
        with self._lock:
            if self.board is None or self.game_over:
                return {"error": "no active game"}
            if not self._is_human_turn():
                return {"error": "not your turn"}

            board = self.board
            legal = {(_move_uci(m)): m for m in board.legal_moves()}
            if uci not in legal:
                return {"error": f"illegal: {uci}"}

            # Capture AI's pre-move analysis (built up during ponder) so we
            # can rate the user's move against it. If no ponder ran, do a
            # tiny synchronous search just to get a sensible eval ratio.
            ai_view = self._capture_ponder_view(min_sims=80)

            move_int = legal[uci]
            new_board = board.apply_move(move_int)
            self.tree.advance(move_int, new_board)
            self.board = new_board

            commentary = self._comment_on_user_move(uci, ai_view)
            self.history.append({
                "ply":   len(self.history) + 1,
                "uci":   uci,
                "by":    "human",
                "eval":  ai_view.get("value"),
                "rank":  commentary["rank"],
                "verdict": commentary["verdict"],
                "best_uci":  commentary["best_uci"],
                "best_eval": commentary["best_eval"],
            })
            self._check_game_over()

        self.emit("user_move", {
            "uci":     uci,
            "fen":     self.board.to_fen(),
            "commentary": commentary,
            "history": self.history,
            "game_over": self.game_over,
            "result": self.result_text,
        })

        if not self.game_over:
            self._ai_move_async()
        return {"ok": True}

    def _comment_on_user_move(self, uci: str, ai_view: dict) -> dict:
        """Compare user's move to the AI's preferred move; return verdict.

        Important: when the model is still in draw-collapse / uniform-policy
        territory (Q-spread across the top moves is tiny), the "best move"
        claim is essentially noise. Detect that case and produce an honest
        "no strong preference" verdict instead of misleading praise/blame.
        """
        top = ai_view.get("top", [])
        if not top:
            return {"rank": None, "verdict": "—",
                    "best_uci": None, "best_eval": None,
                    "low_confidence": True}
        best_move_int, best_visits, best_q = top[0]
        best_uci = _move_uci(best_move_int)

        # Spread across the top moves: when this is small, the network has
        # no real opinion and the "off my radar" / "best move" labels are
        # statistical noise. Threshold tuned empirically — anything under
        # 0.05 Q-spread is essentially "every move looks equal to me".
        q_values = [q for (_, _, q) in top]
        q_spread = (max(q_values) - min(q_values)) if len(q_values) >= 2 else 0.0
        low_conf = q_spread < 0.05

        # Look up rank of the user's choice in the visit-ordered list.
        rank = None
        user_q = None
        for i, (m, n, q) in enumerate(top):
            if _move_uci(m) == uci:
                rank = i + 1
                user_q = q
                break

        if low_conf:
            # Honest "I don't really know" branch — the model is still
            # undertrained; eval / rank labels would be misleading.
            verdict = (
                "No strong opinion — all my top moves look about equal "
                f"(Q-spread {q_spread:.3f}). The value head is still in "
                "bootstrap; don't trust my eval yet."
            )
            return {
                "rank":      rank,
                "verdict":   verdict,
                "best_uci":  best_uci,
                "best_eval": float(best_q),
                "low_confidence": True,
            }

        # Heuristic verdict (only meaningful with a real Q-spread).
        if rank == 1:
            verdict = "Best move. I agree."
        elif rank is not None and user_q is not None:
            # Both q's are side-to-move POV (root_analysis negates child.Q),
            # and top[0] is the most-visited = highest-q move, so delta ≥ 0.
            delta = best_q - user_q
            if delta < 0.05:
                verdict = f"#{rank} on my list — almost identical (-{delta:.2f})."
            elif delta < 0.15:
                verdict = f"#{rank} — solid but not best ({best_uci} was better, -{delta:.2f})."
            elif delta < 0.40:
                verdict = f"#{rank} — inaccuracy ({best_uci} was sharper, -{delta:.2f})."
            else:
                verdict = f"Mistake ({best_uci} was much stronger, -{delta:.2f})."
        else:
            # User's move wasn't in the top-k I tracked.
            verdict = f"Off my radar — I was eyeing {best_uci}."

        return {
            "rank":      rank,
            "verdict":   verdict,
            "best_uci":  best_uci,
            "best_eval": float(best_q),
            "low_confidence": False,
        }

    def _capture_ponder_view(self, min_sims: int = 0) -> dict:
        """root_analysis() of the CURRENT root. If <min_sims have run,
        top up with a synchronous batch so the verdict is meaningful."""
        if self.tree is None or self.board is None:
            return {}
        # If ponder didn't get enough sims in, do a quick top-up.
        try:
            n_root = int(getattr(self.tree._root_node, "N", 0) or 0)
        except Exception:
            n_root = 0
        if n_root < min_sims:
            self._run_sync_sims(min_sims - n_root)
        try:
            return self.tree.root_analysis(top_k=8, pv_len=10)
        except Exception:
            return {}

    def _run_sync_sims(self, n: int) -> None:
        """Run `n` extra simulations in the calling thread. No SSE updates."""
        if self.tree is None or n <= 0:
            return
        bs = self._engine_batch()
        try:
            self.tree.run_batched_sims(int(n), batch_size=bs)
        except Exception:
            pass

    # ── AI move ──────────────────────────────────────────────────────────

    def _ai_move_async(self) -> None:
        """Run the AI's search in a background thread; user's request returns
        immediately, the UI receives updates via SSE."""
        t = threading.Thread(target=self._ai_move_thread, daemon=True,
                             name="WebAIThink")
        t.start()

    def _ai_move_thread(self) -> None:
        self._stop_ponder()
        self._searching.set()
        try:
            self.emit("ai_thinking_start", {
                "mode":      self.think_mode,
                "budget":    self._budget_label(),
                "fen":       self.board.to_fen() if self.board else "",
            })

            mv = self._run_ai_search()
            if mv is None:
                # No legal move from AI's side → game over already
                with self._lock:
                    self._check_game_over()
                self.emit("ai_done", {
                    "fen":      self.board.to_fen() if self.board else "",
                    "game_over": self.game_over,
                    "result":   self.result_text,
                })
                return

            with self._lock:
                board = self.board
                if board is None:
                    return
                analysis = {}
                try:
                    analysis = self.tree.root_analysis(top_k=8, pv_len=10)
                except Exception:
                    pass
                uci = _move_uci(mv)
                new_board = board.apply_move(mv)
                self.tree.advance(mv, new_board)
                self.board = new_board
                self.history.append({
                    "ply":   len(self.history) + 1,
                    "uci":   uci,
                    "by":    "ai",
                    "eval":  analysis.get("value"),
                    "rank":  1,
                    "verdict": "",
                    "best_uci":  uci,
                    "best_eval": analysis.get("value"),
                    "pv":    [_move_uci(m) for m in analysis.get("pv", [])],
                })
                self._check_game_over()

            self.emit("ai_move", {
                "uci":      uci,
                "fen":      self.board.to_fen(),
                "analysis": _analysis_for_wire(analysis),
                "history":  self.history,
                "game_over": self.game_over,
                "result":   self.result_text,
            })
        finally:
            self._searching.clear()
            # After the AI moves, kick off ponder if the user's turn (and enabled).
            if self.ponder_on and not self.game_over and self._is_human_turn():
                self._start_ponder()

    def _engine_batch(self) -> int:
        """Inference engine's preferred batch width. For the CUDA-Graph
        backend this is the captured width; smaller batches get zero-padded
        anyway, so we always pay for `batch_size` slots even with B=1."""
        return int(getattr(self.engine, "batch_size", 0)) or 32

    def _ensure_root_ready(self) -> None:
        """Idempotent: expand the root only if it isn't already. Preserves
        every visit count + Q value the ponder thread / tree.advance carried
        over from previous turns. Without this, ``begin_search`` would call
        ``reset()`` and throw the whole subtree away — measurably weakening
        play compared to the terminal ``--play`` mode which never resets."""
        if self.tree is None or self.board is None:
            return
        root = self.tree._root_node
        needs_init = (root is None)
        if not needs_init:
            try:
                needs_init = not root.is_expanded()
            except Exception:
                needs_init = True
        if needs_init:
            # First expansion for this position. Use begin_search with
            # add_noise=False — real play, not training. After this call
            # the root is expanded and subsequent AI thinks on the same
            # tree (across user replies and ponder) will skip past this
            # branch and just keep deepening.
            self.tree.begin_search(self.board, keep_history=True,
                                   add_noise=False)

    def _run_ai_search(self) -> Optional[int]:
        """Drive the MCTS search per the configured mode + budget. Streams
        progress to subscribers. Returns the chosen move int, or None if the
        game has ended."""
        if self.tree is None or self.board is None:
            return None
        if self.board.legal_moves() == []:
            return None

        # CRITICAL: do NOT call begin_search unconditionally — it wipes the
        # tree via reset(), throwing away every sim the ponder thread did
        # AND the subtree we inherited from tree.advance() after the user
        # played their move. Only initialise the root if it is actually
        # unexpanded (first AI move of a new game, or after a takeback).
        self._ensure_root_ready()

        bs = self._engine_batch()
        mode = self.think_mode
        if mode == "sims":
            target = int(self.think_sims)
            done = 0
            last_emit = 0.0
            while done < target:
                chunk = min(bs, target - done)
                actual = self.tree.run_batched_sims(chunk, batch_size=bs)
                if actual <= 0:
                    break  # tree saturated — no more useful work
                done += actual
                now = time.time()
                if now - last_emit > 0.15 or done >= target:
                    self._emit_progress(done, target, time_elapsed_s=None)
                    last_emit = now

        elif mode == "time":
            t0 = time.time()
            budget = float(self.think_time_s)
            done = 0
            last_emit = 0.0
            while time.time() - t0 < budget:
                actual = self.tree.run_batched_sims(bs, batch_size=bs)
                if actual <= 0:
                    break
                done += actual
                now = time.time()
                if now - last_emit > 0.15:
                    self._emit_progress(done, None,
                                        time_elapsed_s=now - t0,
                                        time_budget_s=budget)
                    last_emit = now
            self._emit_progress(done, None,
                                time_elapsed_s=time.time() - t0,
                                time_budget_s=budget)

        else:  # "dynamic" — DCA-style: base budget, extend if no clear best
            base = int(self.dynamic_min)
            ceiling = int(self.dynamic_max)
            done = 0
            last_emit = 0.0
            # 1) Run base budget in batched rounds.
            while done < base:
                chunk = min(bs, base - done)
                actual = self.tree.run_batched_sims(chunk, batch_size=bs)
                if actual <= 0:
                    break
                done += actual
                now = time.time()
                if now - last_emit > 0.15:
                    self._emit_progress(done, ceiling, time_elapsed_s=None)
                    last_emit = now
            # 2) Keep extending in batch-sized bursts until either we've hit
            #    the ceiling or the top move is clearly dominant.
            while done < ceiling:
                analysis = self.tree.root_analysis(top_k=2, pv_len=2)
                top = analysis.get("top", [])
                if len(top) >= 2:
                    n1, q1 = top[0][1], top[0][2]
                    n2, q2 = top[1][1], top[1][2]
                    if n1 >= 1.5 * max(1, n2) and abs(q1 - q2) >= 0.05:
                        break
                actual = self.tree.run_batched_sims(
                    min(bs, ceiling - done), batch_size=bs)
                if actual <= 0:
                    break
                done += actual
                now = time.time()
                if now - last_emit > 0.15:
                    self._emit_progress(done, ceiling, time_elapsed_s=None)
                    last_emit = now

        # Choose the move. Match the terminal --play behaviour: at very low
        # temperature we pick the most-visited move directly from the root
        # analysis (strictly greedy), which is what `tree.search` does
        # internally too at τ < 0.01. Above τ ≥ 0.1 we sample via
        # `select_move` for some randomness. The default temp 0.05 lands in
        # the strict-greedy branch — important: softmax over visits at
        # τ=0.05 looks "near-greedy" but with N≈visits in the thousands
        # any small visit-count noise becomes amplified through visits^20.
        try:
            if self.temperature < 0.1:
                analysis = self.tree.root_analysis(top_k=1, pv_len=1)
                top = analysis.get("top", [])
                if top:
                    move = top[0][0]
                else:
                    move, _ = self.tree.select_move(self.temperature)
            else:
                move, _ = self.tree.select_move(self.temperature)
        except Exception:
            # Empty tree / pathological case — fall back to the first legal move.
            legal = self.board.legal_moves()
            move = legal[0] if legal else 0
        return int(move)

    def _position_view(self) -> dict:
        """Static (search-independent) facts about the current root position:
        WDL breakdown + material-aux from the net, material balance, phase,
        and tactical hints. Computed ONCE per position (cached by FEN) — the
        net forward for WDL is cheap but pointless to repeat every 150 ms tick
        while the root is unchanged."""
        if self.board is None or self.tree is None:
            return {}
        fen = _safe(lambda: self.board.to_fen(), "")
        cached = getattr(self, "_pos_view_cache", None)
        if cached is not None and cached.get("fen") == fen:
            return cached
        view = {"fen": fen}
        view.update(_material_phase(self.board))
        view["tactics"] = _tactical_hints(self.board)
        view["wdl"] = _safe(lambda: self.tree.root_wdl(), {}) or {}
        self._pos_view_cache = view
        return view

    def _emit_progress(self, sims_done: int, sims_target: Optional[int],
                       time_elapsed_s: Optional[float] = None,
                       time_budget_s: Optional[float] = None) -> None:
        try:
            a = self.tree.root_analysis(top_k=5, pv_len=8)
        except Exception:
            a = {}
        self.emit("ai_thinking", {
            "sims_done":     int(sims_done),
            "sims_target":   sims_target,
            "time_elapsed":  time_elapsed_s,
            "time_budget":   time_budget_s,
            "analysis":      _analysis_for_wire(a),
            "position":      self._position_view(),
        })

    def _budget_label(self) -> str:
        if self.think_mode == "sims":
            return f"{self.think_sims} sims"
        if self.think_mode == "time":
            return f"{self.think_time_s:.1f}s"
        return f"dynamic ({self.dynamic_min}-{self.dynamic_max} sims)"

    # ── Ponder ───────────────────────────────────────────────────────────

    def _start_ponder(self) -> None:
        if not self.ponder_on or self.game_over:
            return
        if self._searching.is_set():
            return
        if self._ponder_thread is not None and self._ponder_thread.is_alive():
            return
        self._ponder_stop.clear()
        self._ponder_thread = threading.Thread(
            target=self._ponder_loop, daemon=True, name="WebPonder",
        )
        self._ponder_thread.start()
        self.emit("ponder_state", {"running": True})

    def _stop_ponder(self) -> None:
        if self._ponder_thread is None:
            return
        self._ponder_stop.set()
        try:
            self._ponder_thread.join(timeout=2.0)
        except Exception:
            pass
        self._ponder_thread = None
        self.emit("ponder_state", {"running": False})

    def _ponder_loop(self) -> None:
        last_emit = 0.0
        sims_count = 0
        bs = self._engine_batch()
        # Ponder runs batched sims (matched to the engine's preferred batch
        # width) instead of one-at-a-time, so the CUDA-Graph engine isn't
        # zero-padding ~31/32 slots per call. The stop-event is checked
        # between batches; cancellation latency is one batch worth of work
        # (~30-100 ms on RTX 5070), which is well below human-perceptible.
        while not self._ponder_stop.is_set() and not self.game_over:
            try:
                actual = self.tree.run_batched_sims(bs, batch_size=bs)
            except Exception:
                break
            if actual <= 0:
                # Tree saturated — nothing more to learn at this root. Sleep
                # briefly to avoid a hot loop; UI sees a stable analysis.
                time.sleep(0.25)
                continue
            sims_count += actual
            now = time.time()
            if now - last_emit > 0.35:
                try:
                    a = self.tree.root_analysis(top_k=5, pv_len=8)
                except Exception:
                    a = {}
                self.emit("ponder_update", {
                    "sims": sims_count,
                    "analysis": _analysis_for_wire(a),
                    "position": self._position_view(),
                })
                last_emit = now

    # ── Misc actions ─────────────────────────────────────────────────────

    def takeback(self) -> dict:
        """Undo the last full move pair (your move + AI's reply). Resets the
        tree because the search subtree no longer matches the position."""
        with self._lock:
            if not self.history or self.board is None:
                return {"error": "nothing to take back"}
            # Pop pairs until it's the human's turn again (and we're not at
            # the very start). 1 or 2 pops depending on whose move it was.
            popped = 0
            while popped < 2 and self.history:
                self.history.pop()
                popped += 1
                if self._is_human_turn_after_history():
                    break
            # Reconstruct the board from scratch via FEN play-out — slow but
            # safe; the alternative is hooking into the tree's parent pointers
            # which the C++ backend doesn't expose.
            board = Board.from_fen(STARTPOS_FEN)
            for h in self.history:
                legal = {(_move_uci(m)): m for m in board.legal_moves()}
                mv = legal.get(h["uci"])
                if mv is None:
                    break
                board = board.apply_move(mv)
            self.board = board
            self.tree  = MCTSTree(self.cfg, self.model, self.device,
                                  engine=self.engine)
            self.tree.reset(board)
            self.game_over = False
            self.result_text = ""

        self.emit("takeback", {
            "fen":     self.board.to_fen(),
            "history": self.history,
        })
        if self.ponder_on and self._is_human_turn():
            self._start_ponder()
        return {"ok": True}

    def _is_human_turn_after_history(self) -> bool:
        # In the standard start, white moves first → human_white = white-to-move.
        n = len(self.history)
        white_to_move = (n % 2 == 0)
        return white_to_move == self.human_white

    def resign(self) -> dict:
        with self._lock:
            if self.game_over or self.board is None:
                return {"ok": True}
            self.game_over = True
            self.result_text = ("Black wins (resignation)"
                                if self.human_white else "White wins (resignation)")
        self._stop_ponder()
        self.emit("game_over", {"result": self.result_text})
        return {"ok": True}

    def hint(self) -> dict:
        """Run a quick search and tell the user what the AI would play. Does
        NOT actually make the move."""
        if self.game_over or self.board is None:
            return {"error": "no active game"}
        if not self._is_human_turn():
            return {"error": "wait for your turn"}
        # Brief synchronous burst (cap at think_sims so it stays "quick").
        # IMPORTANT: don't call begin_search — it would reset the tree
        # and discard any pondering work. Just ensure the root is expanded.
        self._stop_ponder()
        self._ensure_root_ready()
        burst = min(self.think_sims, 1500)
        bs = self._engine_batch()
        self.tree.run_batched_sims(burst, batch_size=bs)
        try:
            a = self.tree.root_analysis(top_k=5, pv_len=8)
        except Exception:
            a = {}
        a_wire = _analysis_for_wire(a)
        self.emit("hint", a_wire)
        # Restart ponder so the user keeps benefitting from the tree we just
        # built up.
        if self.ponder_on and not self.game_over:
            self._start_ponder()
        return {"ok": True, "analysis": a_wire}

    def _check_game_over(self) -> None:
        if self.board is None:
            return
        res = get_game_result(self.board)
        if res == GameResult.ONGOING:
            return
        self.game_over = True
        self.result_text = {
            GameResult.WHITE_WIN: "White wins",
            GameResult.BLACK_WIN: "Black wins",
            GameResult.DRAW:      "Draw",
        }.get(res, "")


_PIECE_LETTERS = ["P", "N", "B", "R", "Q", "K"]
_PIECE_VALUES  = [1, 3, 3, 5, 9, 0]


def _material_phase(board) -> dict:
    """Material balance (White − Black, in pawns) + game phase label."""
    diff = 0
    try:
        for sq in range(64):
            p = board.piece_at(sq)
            if p is None or p < 0:
                continue
            v = _PIECE_VALUES[p % 6]
            diff += v if (p // 6) == 0 else -v
    except Exception:
        pass
    phase = {0: "opening", 1: "middlegame", 2: "endgame"}.get(
        _safe(lambda: board.get_phase()), "?")
    return {"material": int(diff), "phase": phase}


def _tactical_hints(board) -> dict:
    """Cheap board-level hints (NOT from the net): is the side to move in
    check, which captures are available (with victim value), and which of the
    side-to-move's own pieces are hanging (attacked by the opponent and not
    defended). A heuristic readout to help SEE what the engine might be
    weighing — not a tactics solver (ignores pins / x-rays / overloads)."""
    stm = int(getattr(board, "side_to_move", 0))
    opp = 1 - stm
    out = {"in_check": bool(_safe(lambda: board.in_check())),
           "captures": [], "hanging": []}
    try:
        for m in board.legal_moves():
            to = (m >> 6) & 0x3F
            tgt = board.piece_at(to)
            if tgt is not None and tgt >= 0 and (tgt // 6) == opp:
                out["captures"].append(
                    {"uci": _move_uci(m), "victim": _PIECE_LETTERS[tgt % 6],
                     "value": _PIECE_VALUES[tgt % 6]})
    except Exception:
        pass
    # Highest-value captures first, keep it short.
    out["captures"].sort(key=lambda c: -c["value"])
    out["captures"] = out["captures"][:6]
    try:
        for sq in range(64):
            p = board.piece_at(sq)
            if p is None or p < 0 or (p // 6) != stm:
                continue
            if board.is_attacked(sq, opp) and not board.is_attacked(sq, stm):
                out["hanging"].append(
                    {"sq": int(sq), "piece": _PIECE_LETTERS[p % 6],
                     "value": _PIECE_VALUES[p % 6]})
    except Exception:
        pass
    out["hanging"].sort(key=lambda h: -h["value"])
    return out


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _analysis_for_wire(a: dict) -> dict:
    """Serialise root_analysis() for JSON. Converts move ints → UCI strings."""
    if not a:
        return {}
    return {
        "value":  float(a.get("value", 0.0)),
        "nodes":  int(a.get("nodes", 0)),
        "depth":  int(a.get("depth", 0)),
        "top":    [{"uci": _move_uci(m), "n": int(n), "q": float(q)}
                   for (m, n, q) in a.get("top", [])],
        "pv":     [_move_uci(m) for m in a.get("pv", [])],
        # Per-move decision detail (prior P, exploration U, mini-PV line).
        "ext":    [{"uci": _move_uci(e["move"]), "n": int(e["n"]),
                    "q": float(e["q"]), "p": float(e["p"]), "u": float(e["u"]),
                    "line": [_move_uci(x) for x in e.get("line", [])]}
                   for e in a.get("ext", [])],
    }


def _threat_map(board) -> dict:
    """Squares attacked by each side + hanging pieces for the current board,
    computed with python-chess (probe-only utility use — NOT move generation).
    Powers the web-UI threat overlay. Returns {} if python-chess is missing."""
    try:
        import chess as _c
    except Exception:
        return {}
    try:
        b = _c.Board(board.to_fen())
    except Exception:
        return {}
    vals = {_c.PAWN: 1, _c.KNIGHT: 3, _c.BISHOP: 3,
            _c.ROOK: 5, _c.QUEEN: 9, _c.KING: 0}
    white_atk, black_atk = [], []
    for sq in _c.SQUARES:
        if b.is_attacked_by(_c.WHITE, sq):
            white_atk.append(_c.square_name(sq))
        if b.is_attacked_by(_c.BLACK, sq):
            black_atk.append(_c.square_name(sq))
    hanging = []
    for sq, piece in b.piece_map().items():
        if piece.piece_type == _c.KING:
            continue
        attackers = b.attackers(not piece.color, sq)
        if not attackers:
            continue
        defenders = b.attackers(piece.color, sq)
        pv = vals[piece.piece_type]
        cheapest = min(vals[b.piece_at(a).piece_type] for a in attackers)
        # Loose = undefended while attacked, OR a cheaper piece can capture it
        # (a winning trade for the attacker). Simplified static eval, no x-rays.
        if (not defenders) or (cheapest < pv):
            hanging.append({"sq": _c.square_name(sq),
                            "sym": piece.symbol(),
                            "white": bool(piece.color)})
    return {
        "white_attacks": white_atk,
        "black_attacks": black_atk,
        "hanging": hanging,
        "stm": "w" if b.turn else "b",
        "in_check": b.is_check(),
    }


_EA_CACHE: dict = {"path": None, "ea": None}
_EA_LOCK = threading.Lock()


def _training_snapshot() -> dict:
    """LIVE training stats for the web monitor. Loss / lr / entropy /
    throughput / buffer sizes / VRAM come straight from the newest TensorBoard
    event file (written every step by the trainer); ELO comes from
    pool_state.json. The EventAccumulator is cached and reloaded incrementally
    so the 4-second poll stays cheap. Missing sources just yield nulls so the
    panel degrades gracefully."""
    out = {k: None for k in (
        "step", "loss_total", "p", "v", "kl", "aux", "lr", "entropy",
        "pos_s", "games_hr", "replay", "teacher", "vram",
        "elo", "best_elo")}
    out["ts"] = time.time()
    root = Path(_root)

    # ELO + best ELO from the opponent pool.
    try:
        ps = root / str(CONFIG.checkpoint_dir) / "pool_state.json"
        if ps.exists():
            d = json.loads(ps.read_text(encoding="utf-8", errors="ignore"))
            ratings = {k: float(v) for k, v in d.get("elo_ratings", {}).items()}
            cur = d.get("current_id")
            if cur in ratings:
                out["elo"] = round(ratings[cur], 1)
            if ratings:
                out["best_elo"] = round(max(ratings.values()), 1)
    except Exception:
        pass

    # Live scalars from the newest tfevents file (incrementally reloaded).
    try:
        import glob
        import os
        from tensorboard.backend.event_processing.event_accumulator \
            import EventAccumulator
        evs = glob.glob(str(root / "runs" / "events.out.tfevents.*"))
        if evs:
            newest = max(evs, key=os.path.getmtime)
            with _EA_LOCK:
                ea = _EA_CACHE.get("ea")
                if _EA_CACHE.get("path") != newest or ea is None:
                    ea = EventAccumulator(newest, size_guidance={"scalars": 64})
                    _EA_CACHE["path"] = newest
                    _EA_CACHE["ea"] = ea
                ea.Reload()
                avail = set(ea.Tags().get("scalars", []))

                def last(tag):
                    if tag not in avail:
                        return None
                    try:
                        pts = ea.Scalars(tag)
                        return max(pts, key=lambda s: s.step).value if pts else None
                    except Exception:
                        return None

                tagmap = {
                    "step": "global_step", "loss_total": "loss/total",
                    "p": "loss/policy", "v": "loss/value",
                    "kl": "loss/teacher_kl", "aux": "loss/aux", "lr": "lr",
                    "entropy": "policy_entropy",
                    "pos_s": "throughput/positions_per_sec",
                    "games_hr": "throughput/games_per_hour",
                    "replay": "replay_buf_size", "teacher": "teacher_buf_size",
                    "vram": "memory/vram_used_mb",
                }
                for key, tag in tagmap.items():
                    v = last(tag)
                    if v is not None:
                        out[key] = v
            if out["step"] is not None:
                out["step"] = int(out["step"])
    except Exception:
        pass

    # Fallback: derive the step from the newest checkpoint filename.
    if out["step"] is None:
        try:
            import glob
            import os
            import re
            cks = glob.glob(str(root / str(CONFIG.checkpoint_dir) / "step_*.pt"))
            if cks:
                m = re.search(r"step_(\d+)\.pt",
                              max(cks, key=os.path.getmtime))
                if m:
                    out["step"] = int(m.group(1))
        except Exception:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────
# HTTP / SSE plumbing
# ─────────────────────────────────────────────────────────────────────────

SESSION: Optional[GameSession] = None  # set in main()


# Piece glyphs — kept as-is from the original (CDN doesn't ship them).
_GLYPH = {"K": "♚", "Q": "♛", "R": "♜",
          "B": "♝", "N": "♞", "P": "♟"}
_PIECE_CACHE: dict = {}


def _piece_svg(code: str):
    if code in _PIECE_CACHE:
        return _PIECE_CACHE[code]
    if len(code) != 2 or code[0] not in "wb" or code[1] not in _GLYPH:
        return None
    glyph = _GLYPH[code[1]]
    if code[0] == "w":
        fill, stroke = "#fafafa", "#202020"
    else:
        fill, stroke = "#1a1a1a", "#e8e8e8"
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 45 45'>"
        f"<text x='22.5' y='37' font-size='40' text-anchor='middle' "
        f"fill='{fill}' stroke='{stroke}' stroke-width='0.9' "
        f"font-family='\"Segoe UI Symbol\",\"DejaVu Sans\",sans-serif'>"
        f"{glyph}</text></svg>"
    ).encode("utf-8")
    _PIECE_CACHE[code] = svg
    return svg


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def log_error(self, *a):    # also quiet (default prints to stderr)
        pass

    def handle(self):
        """Swallow the noisy 'browser closed the SSE connection' exceptions
        Windows' stack raises (WinError 10053 / 10054). These are expected
        whenever a tab is closed or the page is reloaded — they say nothing
        useful about server health, but the default BaseHTTPRequestHandler
        prints a multi-line traceback to stderr for each one."""
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError,
                BrokenPipeError, TimeoutError):
            pass
        except OSError as e:
            # 10053 / 10054 are the WSAECONNABORTED / WSAECONNRESET errnos;
            # anything else from OSError is genuinely unexpected.
            if getattr(e, "winerror", None) in (10053, 10054):
                return
            raise

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(b)
        except (ConnectionError, BrokenPipeError):
            pass

    def _send_json(self, code, obj):
        self._send(code, json.dumps(obj))

    # ── Routes ───────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
            return
        if self.path.startswith("/piece/"):
            code = self.path[len("/piece/"):].split(".")[0]
            svg = _piece_svg(code)
            if svg is None:
                self._send(404, b"no piece", "text/plain")
            else:
                self._send(200, svg, "image/svg+xml")
            return
        if self.path == "/api/state":
            with SESSION._lock:
                board = SESSION.board
                fen = board.to_fen() if board else STARTPOS_FEN
                self._send_json(200, {
                    "fen": fen,
                    "history": SESSION.history,
                    "human_white": SESSION.human_white,
                    "game_over": SESSION.game_over,
                    "result": SESSION.result_text,
                    "settings": SESSION.settings_dict(),
                })
            return
        if self.path == "/api/stream":
            self._stream_sse()
            return
        if self.path == "/api/training":
            self._send_json(200, _training_snapshot())
            return
        if self.path == "/api/threats":
            with SESSION._lock:
                board = SESSION.board
            self._send_json(200, _threat_map(board) if board else {})
            return
        self._send(404, b"not found", "text/plain")

    def _stream_sse(self):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
        except (ConnectionError, BrokenPipeError):
            return

        q = SESSION.sse_subscribe()
        try:
            while True:
                try:
                    msg = q.get(timeout=15.0)
                    payload = (
                        f"event: {msg['event']}\n"
                        f"data: {json.dumps(msg['data'])}\n\n"
                    ).encode("utf-8")
                    self.wfile.write(payload)
                    self.wfile.flush()
                except queue.Empty:
                    # Heartbeat — keeps proxies / browsers from timing out.
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (ConnectionError, BrokenPipeError, OSError):
            pass
        finally:
            SESSION.sse_unsubscribe(q)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            data = {}

        if self.path == "/api/new":
            SESSION.new_game(human_white=bool(data.get("human_white", True)))
            self._send_json(200, {"ok": True})
            return
        if self.path == "/api/move":
            res = SESSION.user_move(str(data.get("uci", "")))
            self._send_json(200, res)
            return
        if self.path == "/api/settings":
            SESSION.apply_settings(data or {})
            self._send_json(200, {"ok": True,
                                  "settings": SESSION.settings_dict()})
            return
        if self.path == "/api/takeback":
            self._send_json(200, SESSION.takeback())
            return
        if self.path == "/api/resign":
            self._send_json(200, SESSION.resign())
            return
        if self.path == "/api/hint":
            self._send_json(200, SESSION.hint())
            return
        if self.path == "/api/setfen":
            self._send_json(200, SESSION.set_fen(
                str(data.get("fen", "")), data.get("human_white")))
            return
        self._send_json(404, {"error": "unknown endpoint"})


# ─────────────────────────────────────────────────────────────────────────
# Frontend (single-page app inline)
# ─────────────────────────────────────────────────────────────────────────

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chess AI — play & analyse</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css">
<style>
  :root{
    --bg-0:#0b0d10; --bg-1:#13171c; --bg-2:#1a1f26; --bg-3:#232a33;
    --ink:#e6edf3; --ink-dim:#9ba6b2; --ink-fade:#5d6a78;
    --accent:#5aa9ff; --good:#3ddc97; --warn:#ffb454; --bad:#ff6f6f;
    --border:#262d36; --shadow:0 6px 24px rgba(0,0,0,.45);
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;background:var(--bg-0);color:var(--ink);
    font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  a{color:var(--accent);text-decoration:none}
  .wrap{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:18px;
        padding:18px;max-width:1380px;margin:0 auto}
  .left{display:flex;flex-direction:column;gap:14px;min-width:0}
  .right{display:flex;flex-direction:column;gap:14px;min-width:0}
  header{display:flex;align-items:center;justify-content:space-between;gap:10px;
         padding:6px 4px}
  header h1{margin:0;font-size:18px;letter-spacing:.4px;
            background:linear-gradient(135deg,#9bd1ff 0%,#5aa9ff 50%,#a2f3c8 100%);
            -webkit-background-clip:text;background-clip:text;color:transparent}
  .pill{font-size:11px;color:var(--ink-dim);padding:2px 8px;
        border:1px solid var(--border);border-radius:999px}
  .card{background:linear-gradient(180deg,var(--bg-1),var(--bg-2));
        border:1px solid var(--border);border-radius:12px;
        box-shadow:var(--shadow);padding:14px}
  .card h3{margin:0 0 10px 0;font-size:11px;letter-spacing:.18em;
           text-transform:uppercase;color:var(--ink-dim);font-weight:600}
  .board-wrap{display:flex;flex-direction:column;align-items:center;gap:12px}
  #board{width:min(72vh,560px);max-width:100%}
  /* chessboard.js overrides — keep their structure but recolor */
  .white-1e1d7{background:#e7ecf3 !important}
  .black-3c85d{background:#7a90aa !important}
  .highlight-last{box-shadow:inset 0 0 0 4px rgba(90,169,255,.55) !important}
  .highlight-hint{box-shadow:inset 0 0 0 4px rgba(61,220,151,.7) !important}
  .move-square{box-shadow:inset 0 0 0 4px rgba(255,180,84,.55) !important}

  /* Eval bar */
  .evalrow{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .evalbar{flex:1;height:18px;background:#0b0e12;border:1px solid var(--border);
           border-radius:10px;overflow:hidden;position:relative;min-width:200px}
  .evalfill{height:100%;background:linear-gradient(90deg,var(--good),var(--accent));
            width:50%;transition:width .35s ease}
  .evalfill.bad{background:linear-gradient(90deg,var(--bad),var(--warn))}
  .evalnum{width:70px;text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
  .matchip{font-size:11px;padding:3px 8px;border-radius:999px;
           border:1px solid var(--border);background:var(--bg-3);
           font-variant-numeric:tabular-nums;color:var(--ink-dim);
           white-space:nowrap}
  .matchip.up{color:var(--good);border-color:rgba(61,220,151,.35)}
  .matchip.down{color:var(--bad);border-color:rgba(255,111,111,.35)}
  .confchip{font-size:10px;padding:2px 8px;border-radius:999px;
            border:1px solid var(--border);background:transparent;
            text-transform:uppercase;letter-spacing:.12em;color:var(--ink-dim);
            white-space:nowrap}
  .confchip.low{color:var(--warn);border-color:rgba(255,180,84,.4)}
  .confchip.med{color:var(--accent);border-color:rgba(90,169,255,.4)}
  .confchip.high{color:var(--good);border-color:rgba(61,220,151,.4)}

  /* Arrow overlay on the board */
  .board-stage{position:relative;display:inline-block;width:min(72vh,560px);max-width:100%}
  .arrows{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;z-index:5}
  .arrows polygon,.arrows line{opacity:.88;mix-blend-mode:screen}

  /* Sims progress */
  .progress{height:8px;background:#0b0e12;border-radius:6px;
            overflow:hidden;border:1px solid var(--border)}
  .progress-fill{height:100%;background:linear-gradient(90deg,#5aa9ff,#7ee0d2);
                 width:0%;transition:width .15s linear}
  .progress-label{font-size:11px;color:var(--ink-dim);margin-top:6px;
                  display:flex;justify-content:space-between}

  /* Top moves table */
  table.moves{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  table.moves th,table.moves td{padding:4px 6px;text-align:left;font-size:13px}
  table.moves th{color:var(--ink-dim);font-weight:500;font-size:11px;
                 text-transform:uppercase;letter-spacing:.1em}
  table.moves tr.top1 td{color:var(--good);font-weight:600}
  table.moves .num{text-align:right}
  table.moves .bar{height:6px;background:#0b0e12;border-radius:3px;overflow:hidden;
                   border:1px solid var(--border);min-width:60px}
  table.moves .bar > span{display:block;height:100%;
                          background:linear-gradient(90deg,#5aa9ff,#9bd1ff)}

  /* PV badge */
  .pv{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
  .pv span{padding:2px 8px;background:var(--bg-3);border:1px solid var(--border);
           border-radius:6px;font-size:12px;color:var(--ink);
           font-family:"SFMono-Regular",Consolas,monospace}

  /* History list */
  .history{max-height:240px;overflow-y:auto;font-family:"SFMono-Regular",Consolas,monospace;
           font-size:13px}
  .history .row{display:grid;grid-template-columns:36px 1fr 1fr;gap:8px;
                padding:4px 6px;border-bottom:1px dashed rgba(255,255,255,.04)}
  .history .row .num{color:var(--ink-dim)}
  .history .row .ai{color:var(--accent)}
  .history .row .verdict{font-family:-apple-system,sans-serif;font-size:11px;
                         color:var(--ink-dim);grid-column:1/-1;padding-left:36px}
  .history .row .verdict.bad{color:var(--bad)}
  .history .row .verdict.warn{color:var(--warn)}
  .history .row .verdict.good{color:var(--good)}

  /* AI commentary */
  #commentary{min-height:42px;padding:10px;border-radius:8px;
              background:rgba(90,169,255,.07);border:1px solid rgba(90,169,255,.18);
              line-height:1.5}
  #commentary.bad{background:rgba(255,111,111,.08);border-color:rgba(255,111,111,.25)}
  #commentary.good{background:rgba(61,220,151,.07);border-color:rgba(61,220,151,.22)}
  #commentary .lead{color:var(--ink-dim);font-size:11px;text-transform:uppercase;
                    letter-spacing:.14em;margin-bottom:4px}

  /* Settings */
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  label.field{display:flex;flex-direction:column;gap:4px}
  label.field > span{color:var(--ink-dim);font-size:11px;letter-spacing:.1em;
                     text-transform:uppercase}
  select,input[type=number],input[type=range]{
    background:var(--bg-3);border:1px solid var(--border);color:var(--ink);
    border-radius:6px;padding:6px 8px;font-size:13px;outline:none;
    font-family:inherit}
  input[type=range]{padding:0}
  input:focus,select:focus{border-color:var(--accent)}
  .toggle{display:inline-flex;align-items:center;gap:8px;cursor:pointer}
  .toggle input{appearance:none;width:36px;height:20px;background:var(--bg-3);
                border:1px solid var(--border);border-radius:999px;position:relative;
                cursor:pointer;transition:background .2s}
  .toggle input::after{content:"";position:absolute;width:14px;height:14px;
                       border-radius:50%;background:var(--ink-dim);top:2px;left:2px;
                       transition:left .2s,background .2s}
  .toggle input:checked{background:rgba(90,169,255,.25);border-color:var(--accent)}
  .toggle input:checked::after{left:18px;background:var(--accent)}

  /* Buttons */
  .btnrow{display:flex;gap:8px;flex-wrap:wrap}
  button{background:var(--bg-3);border:1px solid var(--border);color:var(--ink);
         border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer;
         font-family:inherit;transition:all .15s}
  button:hover:not(:disabled){background:var(--accent);color:#0b0d10;border-color:var(--accent)}
  button.primary{background:var(--accent);color:#0b0d10;border-color:var(--accent);font-weight:600}
  button.primary:hover{filter:brightness(1.1)}
  button.danger{border-color:rgba(255,111,111,.4)}
  button.danger:hover{background:var(--bad);color:#0b0d10}
  button:disabled{opacity:.4;cursor:not-allowed}

  /* Status footer */
  .status{font-size:12px;color:var(--ink-dim);display:flex;justify-content:space-between;
          padding:6px 8px}
  .status .dot{display:inline-block;width:8px;height:8px;border-radius:50%;
               background:var(--ink-fade);margin-right:6px;vertical-align:middle}
  .status .dot.live{background:var(--good);box-shadow:0 0 8px var(--good)}
  .status .dot.thinking{background:var(--warn);box-shadow:0 0 8px var(--warn);
                        animation:pulse 1s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

  @media (max-width: 980px){
    .wrap{grid-template-columns:1fr}
    #board{width:min(90vw,560px)}
  }

  /* ── Observability additions ─────────────────────────────────────── */
  /* WDL bar (win/draw/loss) */
  .wdlbar{display:flex;height:14px;border-radius:6px;overflow:hidden;
          border:1px solid var(--border);background:#0b0e12;width:100%}
  .wdlbar > div{height:100%;transition:width .25s ease}
  .wdl-w{background:#3ddc97}.wdl-d{background:#5a6675}.wdl-l{background:#ff6f6f}
  .wdllbl{display:flex;justify-content:space-between;font-size:10px;
          color:var(--ink-fade);margin-top:2px}
  .wdllbl b{color:var(--ink-dim);font-weight:600}
  /* Top-moves: prior column + expandable mini-PV */
  table.moves td.prior{color:var(--ink-fade);font-size:11px}
  .moverow{cursor:pointer}
  .moverow:hover{background:rgba(255,255,255,.03)}
  .miniline{color:var(--ink-dim);font-size:11px;padding:2px 6px 6px 28px;
            background:rgba(255,255,255,.02)}
  .miniline .puct{color:var(--ink-fade);margin-right:10px}
  .miniline .sanline span{padding:1px 5px;margin-right:3px;border-radius:4px;
            background:var(--bg-3);border:1px solid var(--border)}
  /* Position insight card */
  .pi-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;
          margin-bottom:8px;font-size:12px}
  .pi-chip{padding:3px 9px;border-radius:999px;border:1px solid var(--border);
           background:var(--bg-2);color:var(--ink-dim)}
  .pi-chip b{color:var(--ink)}
  .pi-sec{font-size:11px;color:var(--ink-fade);margin:8px 0 3px;
          text-transform:uppercase;letter-spacing:.04em}
  .tac{display:flex;flex-wrap:wrap;gap:5px}
  .tac span{font-size:11px;padding:2px 7px;border-radius:4px;
            border:1px solid var(--border);background:var(--bg-2)}
  .tac .cap{color:var(--good);border-color:rgba(61,220,151,.3)}
  .tac .hang{color:var(--bad);border-color:rgba(255,111,111,.3)}
  .tac .chk{color:var(--warn);border-color:rgba(255,180,84,.4)}
  /* Convergence line in the thinking card */
  .converge{font-size:11px;color:var(--ink-fade);margin-top:6px}
  .converge b{color:var(--ink-dim)}
  /* Eval-over-game chart */
  #evalchart{width:100%;height:90px;display:block}
  #evalchart .axis{stroke:var(--border);stroke-width:.5}
  #evalchart .zero{stroke:var(--ink-fade);stroke-width:.5;stroke-dasharray:3 3}
  #evalchart .line{fill:none;stroke:var(--accent);stroke-width:1.5}
  /* Board heatmap squares (destination focus of the search) */
  .heat{position:absolute;border-radius:3px;pointer-events:none;
        mix-blend-mode:screen}

  /* ── Pro feature pack ─────────────────────────────────────────────── */
  .protools{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
  .protools button{padding:6px 10px;font-size:12px}
  .protools .tbtn.on{background:var(--accent);color:#0b0d10;border-color:var(--accent)}
  .opening-chip{font-size:11px;color:var(--ink-dim);padding:3px 9px;border-radius:999px;
                border:1px solid var(--border);background:var(--bg-2);white-space:nowrap}
  /* Threat / attack overlay on the board */
  .threats{position:absolute;left:0;top:0;width:100%;height:100%;
           pointer-events:none;z-index:4}
  .threats .tsq{position:absolute;width:12.5%;height:12.5%}
  .threats .atk-w{box-shadow:inset 0 0 0 3px rgba(61,220,151,.5)}
  .threats .atk-b{box-shadow:inset 0 0 0 3px rgba(255,111,111,.5)}
  .threats .atk-both{box-shadow:inset 0 0 0 3px rgba(255,180,84,.6)}
  .threats .hang{background:radial-gradient(circle,rgba(255,111,111,.55),
                 rgba(255,111,111,0) 70%)}
  /* Training monitor */
  .tm-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 14px}
  .tm-cell{display:flex;flex-direction:column}
  .tm-cell .k{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-fade)}
  .tm-cell .v{font-size:16px;font-weight:600;font-variant-numeric:tabular-nums}
  .tm-cell .v.good{color:var(--good)}.tm-cell .v.warn{color:var(--warn)}
  .tm-stale{font-size:11px;color:var(--warn)}
  /* Accuracy / review */
  .acc-row{display:flex;gap:10px;align-items:center;margin-bottom:8px}
  .acc-pill{flex:1;text-align:center;padding:8px;border-radius:8px;
            border:1px solid var(--border);background:var(--bg-2)}
  .acc-pill .big{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
  .acc-pill .lbl{font-size:10px;color:var(--ink-fade);text-transform:uppercase;letter-spacing:.1em}
  .acc-counts{display:flex;gap:6px;flex-wrap:wrap;font-size:11px}
  .acc-counts span{padding:2px 7px;border-radius:5px;border:1px solid var(--border);background:var(--bg-2)}
  .acc-counts .bril{color:#7ee0d2;border-color:rgba(126,224,210,.35)}
  .acc-counts .blu{color:var(--bad);border-color:rgba(255,111,111,.35)}
  .acc-counts .mis{color:var(--warn);border-color:rgba(255,180,84,.35)}
  .glyph{font-weight:700;margin-left:4px}
  .glyph.bril{color:#7ee0d2}.glyph.good{color:var(--good)}
  .glyph.inacc{color:var(--warn)}.glyph.mis{color:#ff9d5c}.glyph.blu{color:var(--bad)}
  /* Review banner */
  #reviewbar{position:fixed;left:50%;transform:translateX(-50%);bottom:18px;
             z-index:50;display:none;gap:8px;align-items:center;padding:8px 12px;
             background:var(--bg-2);border:1px solid var(--accent);border-radius:10px;
             box-shadow:var(--shadow)}
  #reviewbar.show{display:flex}
  /* Help overlay */
  #helpov{position:fixed;inset:0;z-index:60;display:none;align-items:center;
          justify-content:center;background:rgba(0,0,0,.6)}
  #helpov.show{display:flex}
  #helpov .box{background:var(--bg-1);border:1px solid var(--border);border-radius:14px;
               padding:22px;max-width:460px;box-shadow:var(--shadow)}
  #helpov kbd{background:var(--bg-3);border:1px solid var(--border);border-radius:5px;
              padding:1px 7px;font-family:Consolas,monospace;font-size:12px}
  #helpov table{width:100%;border-collapse:collapse}
  #helpov td{padding:5px 6px;border-bottom:1px dashed rgba(255,255,255,.06)}
  /* Focus mode hides the right column, enlarges the board */
  body.focusmode .right{display:none}
  body.focusmode .wrap{grid-template-columns:1fr;max-width:820px}
  /* Coordinates toggle (chessboard.js notation) */
  body.no-coords .notation-322f9{display:none !important}
  .moverow.reviewing{outline:1px solid var(--accent);border-radius:4px}
</style>
</head><body>

<div class="wrap">

  <!-- LEFT COLUMN — board + history -->
  <div class="left">

    <header>
      <h1>♚ Chess AI</h1>
      <div>
        <span class="pill" id="enginepill">engine: loading…</span>
        <span class="pill" id="livepill"><span class="dot" id="livedot"></span><span id="livetxt">connecting</span></span>
      </div>
    </header>

    <div class="card board-wrap">
      <div class="board-stage">
        <div id="board"></div>
        <svg class="arrows" id="arrows" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
      </div>

      <div class="evalrow" style="width:min(72vh,560px);max-width:100%">
        <div class="evalbar"><div id="evalfill" class="evalfill"></div></div>
        <div class="evalnum" id="evalnum">+0.00</div>
      </div>
      <div style="width:min(72vh,560px);max-width:100%">
        <div class="wdlbar" id="wdlbar" title="Win / Draw / Loss probability from the net's value head (side-to-move POV)">
          <div class="wdl-w" id="wdl-w" style="width:33%"></div>
          <div class="wdl-d" id="wdl-d" style="width:34%"></div>
          <div class="wdl-l" id="wdl-l" style="width:33%"></div>
        </div>
        <div class="wdllbl">
          <span>Win <b id="wdl-wt">—</b></span>
          <span>Draw <b id="wdl-dt">—</b></span>
          <span>Loss <b id="wdl-lt">—</b></span>
        </div>
      </div>
      <div class="evalrow" style="width:min(72vh,560px);max-width:100%;gap:6px">
        <span class="matchip" id="matchip">Material: even</span>
        <span class="confchip" id="confchip">confidence —</span>
        <span style="flex:1"></span>
        <span style="font-size:11px;color:var(--ink-fade)" id="movecount">move 0</span>
      </div>

      <div class="btnrow" style="margin-top:4px">
        <button class="primary" onclick="newGame(true)">New game (you=White)</button>
        <button onclick="newGame(false)">You=Black</button>
        <button onclick="takeback()">↶ Take back</button>
        <button onclick="hint()">💡 Hint</button>
        <button class="danger" onclick="resign()">Resign</button>
      </div>
    </div>

    <div class="card">
      <h3>Move history</h3>
      <div class="history" id="history"></div>
    </div>

  </div>

  <!-- RIGHT COLUMN — AI brain + settings -->
  <div class="right">

    <div class="card">
      <h3>AI commentary on your last move</h3>
      <div id="commentary"><div class="lead">No moves yet</div><div>Play a move and I'll tell you what I think of it.</div></div>
    </div>

    <div class="card">
      <h3>AI is thinking <span class="pill" id="thinkmode" style="float:right">—</span></h3>
      <div class="progress"><div id="progressfill" class="progress-fill"></div></div>
      <div class="progress-label">
        <span id="progresstxt">idle</span>
        <span id="nodes"></span>
      </div>
      <div class="converge" id="converge"></div>
    </div>

    <div class="card">
      <h3>Top candidate moves <span style="float:right;color:var(--ink-fade);font-size:10px">click a row for its line</span></h3>
      <table class="moves"><thead><tr>
        <th>#</th><th>Move</th><th class="num">Visits</th>
        <th class="num">Prior</th><th class="num">Q</th><th>Share</th>
      </tr></thead><tbody id="topmoves"><tr><td colspan="6" style="color:var(--ink-dim);text-align:center;padding:18px">no analysis yet</td></tr></tbody></table>
    </div>

    <div class="card">
      <h3>Position insight</h3>
      <div class="pi-row" id="pi-meta">
        <span class="pi-chip">Material <b id="pi-mat">—</b></span>
        <span class="pi-chip">Phase <b id="pi-phase">—</b></span>
        <span class="pi-chip" title="The value head's learnable material-balance signal (side-to-move POV)">Material-sense <b id="pi-aux">—</b></span>
      </div>
      <div class="pi-sec">Tactics on the board (heuristic, not from the net)</div>
      <div class="tac" id="pi-tac"><span style="color:var(--ink-fade);border:none;background:none">—</span></div>
    </div>

    <div class="card">
      <h3>Principal variation</h3>
      <div class="pv" id="pv"><span style="color:var(--ink-dim);background:transparent;border:none">—</span></div>
    </div>

    <div class="card">
      <h3>Eval over game <span style="float:right;color:var(--ink-fade);font-size:10px">side-to-move POV per move</span></h3>
      <svg id="evalchart" viewBox="0 0 300 90" preserveAspectRatio="none"></svg>
    </div>

    <div class="card">
      <h3>AI settings <span style="float:right;color:var(--ink-fade);font-size:10px">applied live</span></h3>

      <label class="field" style="margin-bottom:10px">
        <span>Thinking mode</span>
        <select id="thinkMode" onchange="settingChanged()">
          <option value="sims">Fixed sims (precise)</option>
          <option value="time">Time budget (consistent feel)</option>
          <option value="dynamic">Dynamic (AI decides)</option>
        </select>
      </label>

      <div id="sims-row" class="grid2" style="margin-bottom:10px">
        <label class="field">
          <span>Sims per move</span>
          <input type="number" id="thinkSims" min="20" max="10000" step="10" value="400" onchange="settingChanged()">
        </label>
        <label class="field">
          <span>(more = stronger, slower)</span>
          <input type="range" id="thinkSimsR" min="20" max="3000" step="20" value="400"
                 oninput="document.getElementById('thinkSims').value=this.value;settingChanged()">
        </label>
      </div>

      <div id="time-row" class="grid2" style="margin-bottom:10px;display:none">
        <label class="field">
          <span>Time (seconds)</span>
          <input type="number" id="thinkTime" min="0.5" max="120" step="0.5" value="5" onchange="settingChanged()">
        </label>
        <label class="field">
          <span>(real-time cap)</span>
          <input type="range" id="thinkTimeR" min="0.5" max="60" step="0.5" value="5"
                 oninput="document.getElementById('thinkTime').value=this.value;settingChanged()">
        </label>
      </div>

      <div id="dyn-row" class="grid2" style="margin-bottom:10px;display:none">
        <label class="field">
          <span>Dynamic min sims</span>
          <input type="number" id="dynMin" min="20" max="5000" step="10" value="200" onchange="settingChanged()">
        </label>
        <label class="field">
          <span>Dynamic max sims</span>
          <input type="number" id="dynMax" min="50" max="20000" step="50" value="1600" onchange="settingChanged()">
        </label>
      </div>

      <div class="grid2" style="margin-bottom:10px">
        <label class="toggle">
          <input type="checkbox" id="ponderOn" checked onchange="settingChanged()">
          <span>Ponder (keep thinking on your turn)</span>
        </label>
      </div>

      <label class="field" style="margin-bottom:4px">
        <span>Temperature: <span id="tempLbl">0.05</span> (lower = stricter best-move)</span>
        <input type="range" id="temperature" min="0.001" max="1" step="0.001" value="0.05"
               oninput="document.getElementById('tempLbl').textContent=parseFloat(this.value).toFixed(3);settingChanged()">
      </label>

      <label class="field" style="margin-bottom:4px">
        <span>Contempt: <span id="contemptLbl">0.10</span> (anti-draw bias — higher = fights harder, avoids shuffle-to-3fold)</span>
        <input type="range" id="contempt" min="0" max="0.30" step="0.01" value="0.10"
               oninput="document.getElementById('contemptLbl').textContent=parseFloat(this.value).toFixed(2);settingChanged()">
      </label>
    </div>

    <div class="status">
      <span><span class="dot" id="ponderdot"></span><span id="ponderlbl">ponder idle</span></span>
      <span id="gamestatus">—</span>
    </div>

  </div>

</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
<script>
// ─── State ────────────────────────────────────────────────────────────
let game = new Chess(), board = null, busy = false;
let lastMoveSquares = [];
let humanWhite = true;
let gameOver = false;

// ─── Helpers ──────────────────────────────────────────────────────────
async function post(u, d){
  let r = await fetch(u, {method:'POST', headers:{'Content-Type':'application/json'},
                           body:JSON.stringify(d||{})});
  return await r.json();
}
function setLive(state, txt){
  let d = document.getElementById('livedot');
  d.className = 'dot ' + (state||'');
  document.getElementById('livetxt').textContent = txt;
}
function setEval(v){
  // v ∈ [-1, 1] from side-to-move's POV. Always show from White's POV.
  let white_pov = (game.turn() === 'w') ? v : -v;
  let pct = 50 + white_pov * 50;
  pct = Math.max(2, Math.min(98, pct));
  let bar = document.getElementById('evalfill');
  bar.style.width = pct + '%';
  bar.classList.toggle('bad', white_pov < -0.05);
  let label = (white_pov >= 0 ? '+' : '') + white_pov.toFixed(2);
  document.getElementById('evalnum').textContent = label;
}
function updateMaterial(){
  // Count material directly from the chess.js board — independent of the
  // (potentially broken) value head. Standard piece values.
  const v = {p:1, n:3, b:3, r:5, q:9, k:0};
  let w = 0, b = 0;
  for (let row of game.board()){
    for (let sq of row){
      if (!sq) continue;
      let val = v[sq.type] || 0;
      if (sq.color === 'w') w += val; else b += val;
    }
  }
  let diff = w - b;
  let chip = document.getElementById('matchip');
  if (diff === 0){
    chip.textContent = 'Material: even';
    chip.className = 'matchip';
  } else if (diff > 0){
    chip.textContent = 'Material: White +' + diff;
    chip.className = 'matchip up';
  } else {
    chip.textContent = 'Material: Black +' + (-diff);
    chip.className = 'matchip down';
  }
  // Also update the move counter pill.
  let mc = document.getElementById('movecount');
  if (mc) mc.textContent = 'move ' + Math.ceil((game.history().length + 1) / 2);
}
function updateConfidence(a){
  // Compute confidence from the spread of Q-values across the AI's top moves.
  // Tiny spread = the network can't distinguish moves → don't trust eval.
  let chip = document.getElementById('confchip');
  if (!a || !a.top || a.top.length < 2){
    chip.textContent = 'confidence —';
    chip.className = 'confchip';
    return;
  }
  let qs = a.top.map(m => m.q);
  let spread = Math.max(...qs) - Math.min(...qs);
  let label, cls;
  if (spread < 0.05)      { label = 'confidence low (Δ' + spread.toFixed(3) + ')';   cls = 'low'; }
  else if (spread < 0.20) { label = 'confidence medium (Δ' + spread.toFixed(2) + ')'; cls = 'med'; }
  else                    { label = 'confidence high (Δ' + spread.toFixed(2) + ')';   cls = 'high'; }
  chip.textContent = label;
  chip.className = 'confchip ' + cls;
}

// ─── Top-move arrows overlay ───────────────────────────────────────────
function squareCenter(sq){
  // chessboard.js attaches data-square to every square cell. We measure
  // its position relative to the board's bounding rect, then return as
  // a percentage of the board so the SVG viewBox (0..100) lines up.
  let boardEl = document.getElementById('board');
  let cell = boardEl.querySelector('[data-square="' + sq + '"]');
  if (!cell || !boardEl) return null;
  let br = boardEl.getBoundingClientRect();
  let cr = cell.getBoundingClientRect();
  if (br.width <= 0) return null;
  let cx = (cr.left - br.left + cr.width / 2) / br.width * 100;
  let cy = (cr.top - br.top + cr.height / 2) / br.height * 100;
  return {x: cx, y: cy};
}
function drawArrows(top){
  let svg = document.getElementById('arrows');
  if (!svg) return;
  // Clear
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!top || !top.length) return;
  // Total visits for opacity normalisation
  let totalN = top.reduce((s, m) => s + (m.n||0), 0) || 1;
  // Heatmap: shade each candidate's destination square by visit share, so
  // you can SEE where the search concentrated (drawn first, arrows on top).
  const SQ = 12.5;
  for (let m of top){
    if (!m.uci || m.uci.length < 4) continue;
    let to = squareCenter(m.uci.slice(2, 4));
    if (!to) continue;
    let share = (m.n || 0) / totalN;
    let rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', (to.x - SQ/2).toFixed(2));
    rect.setAttribute('y', (to.y - SQ/2).toFixed(2));
    rect.setAttribute('width', SQ); rect.setAttribute('height', SQ);
    rect.setAttribute('rx', '1.5');
    rect.setAttribute('fill', '#3ddc97');
    rect.setAttribute('opacity', Math.max(0.05, Math.min(0.5, share*0.6)).toFixed(3));
    svg.appendChild(rect);
  }
  // Draw at most 3 arrows, in order. Strongest = green-ish, fading.
  const colors = ['#3ddc97', '#5aa9ff', '#ffb454'];
  for (let i = 0; i < Math.min(3, top.length); i++){
    let m = top[i];
    if (!m.uci || m.uci.length < 4) continue;
    let from = squareCenter(m.uci.slice(0, 2));
    let to   = squareCenter(m.uci.slice(2, 4));
    if (!from || !to) continue;
    let share = (m.n || 0) / totalN;
    let opacity = Math.max(0.25, Math.min(0.95, share * 1.6));
    // line + arrowhead
    let dx = to.x - from.x, dy = to.y - from.y;
    let len = Math.sqrt(dx*dx + dy*dy);
    if (len < 0.5) continue;
    // Shorten the line so it doesn't poke into the piece graphic.
    let shrink = 4.5 / len;
    let endX = to.x - dx * shrink;
    let endY = to.y - dy * shrink;
    let strokeW = 1.8 - i * 0.35;
    let line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', from.x); line.setAttribute('y1', from.y);
    line.setAttribute('x2', endX);   line.setAttribute('y2', endY);
    line.setAttribute('stroke', colors[i]);
    line.setAttribute('stroke-width', strokeW);
    line.setAttribute('stroke-linecap', 'round');
    line.setAttribute('opacity', opacity);
    svg.appendChild(line);
    // arrowhead — equilateral triangle pointing in direction (dx,dy)
    let ux = dx / len, uy = dy / len;
    let head = 3.4 - i * 0.6;
    let tipX = to.x - dx * (shrink * 0.5), tipY = to.y - dy * (shrink * 0.5);
    let p1 = (tipX + ux * head) + ',' + (tipY + uy * head);
    let p2 = (tipX - uy * head * 0.7) + ',' + (tipY + ux * head * 0.7);
    let p3 = (tipX + uy * head * 0.7) + ',' + (tipY - ux * head * 0.7);
    let poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', p1 + ' ' + p2 + ' ' + p3);
    poly.setAttribute('fill', colors[i]);
    poly.setAttribute('opacity', opacity);
    svg.appendChild(poly);
  }
}

function fmtMoveSan(uci){
  // Try to render as SAN via chess.js — but rolled back, so we use a clone.
  let g = new Chess(game.fen());
  let mv = g.move({from: uci.slice(0,2), to: uci.slice(2,4),
                   promotion: uci.length > 4 ? uci[4] : 'q'});
  return mv ? mv.san : uci;
}

// ─── Board ─────────────────────────────────────────────────────────────
function highlightLastMove(uci){
  // Remove previous highlights
  document.querySelectorAll('.highlight-last').forEach(el => el.classList.remove('highlight-last'));
  if (!uci || uci.length < 4) return;
  let from = uci.slice(0,2), to = uci.slice(2,4);
  let s1 = document.querySelector('[data-square="'+from+'"]');
  let s2 = document.querySelector('[data-square="'+to+'"]');
  if (s1) s1.classList.add('highlight-last');
  if (s2) s2.classList.add('highlight-last');
}
function highlightHint(uci){
  document.querySelectorAll('.highlight-hint').forEach(el => el.classList.remove('highlight-hint'));
  if (!uci || uci.length < 4) return;
  let from = uci.slice(0,2), to = uci.slice(2,4);
  let s1 = document.querySelector('[data-square="'+from+'"]');
  let s2 = document.querySelector('[data-square="'+to+'"]');
  if (s1) s1.classList.add('highlight-hint');
  if (s2) s2.classList.add('highlight-hint');
  setTimeout(() => {
    document.querySelectorAll('.highlight-hint').forEach(el => el.classList.remove('highlight-hint'));
  }, 4000);
}

function onDrop(src, tgt){
  if (busy || gameOver) return 'snapback';
  // Whose turn from chess.js's view?
  let turn = game.turn();   // 'w' or 'b'
  let ourTurn = (humanWhite && turn === 'w') || (!humanWhite && turn === 'b');
  if (!ourTurn) return 'snapback';
  // Try move locally
  let mv = game.move({from:src, to:tgt, promotion:'q'});
  if (mv === null) return 'snapback';
  let uci = mv.from + mv.to + (mv.promotion || '');
  busy = true;
  highlightLastMove(uci);
  // Send to server — server confirms via SSE event 'user_move'
  post('/api/move', {uci: uci}).then(res => {
    if (res && res.error){
      game.undo();
      board.position(game.fen());
      setLive('', 'illegal: '+res.error);
      busy = false;
    }
  });
}

// ─── SSE ──────────────────────────────────────────────────────────────
let es;
function connectStream(){
  es = new EventSource('/api/stream');
  es.onopen  = () => setLive('live', 'connected');
  es.onerror = () => setLive('', 'disconnected (auto-retry)');
  es.addEventListener('snapshot', e => applySnapshot(JSON.parse(e.data)));
  es.addEventListener('new_game', e => onNewGame(JSON.parse(e.data)));
  es.addEventListener('user_move', e => onUserMove(JSON.parse(e.data)));
  es.addEventListener('ai_thinking_start', e => onAIStart(JSON.parse(e.data)));
  es.addEventListener('ai_thinking', e => onAIProgress(JSON.parse(e.data)));
  es.addEventListener('ai_move', e => onAIMove(JSON.parse(e.data)));
  es.addEventListener('ai_done', e => onAIDone(JSON.parse(e.data)));
  es.addEventListener('ponder_state', e => onPonderState(JSON.parse(e.data)));
  es.addEventListener('ponder_update', e => onPonderUpdate(JSON.parse(e.data)));
  es.addEventListener('takeback', e => onTakeback(JSON.parse(e.data)));
  es.addEventListener('settings', e => applySettings(JSON.parse(e.data)));
  es.addEventListener('hint', e => onHint(JSON.parse(e.data)));
  es.addEventListener('game_over', e => onGameOver(JSON.parse(e.data)));
}

// ─── SSE handlers ─────────────────────────────────────────────────────
function applySnapshot(d){
  humanWhite = d.human_white;
  gameOver = d.game_over;
  game.load(d.fen);
  board.orientation(humanWhite ? 'white' : 'black');
  board.position(d.fen, false);
  renderHistory(d.history);
  applySettings(d.settings);
  updateMaterial();
  document.getElementById('gamestatus').textContent =
    gameOver ? ('GAME OVER — ' + (d.result||'')) : 'in progress';
}
function onNewGame(d){
  humanWhite = d.human_white;
  gameOver = false;
  game.load(d.fen);
  board.orientation(humanWhite ? 'white' : 'black');
  board.position(d.fen, false);
  document.getElementById('commentary').innerHTML =
    '<div class="lead">Game started</div><div>Make your first move.</div>';
  document.getElementById('commentary').className = '';
  document.getElementById('commentary').style.background = '';
  document.getElementById('commentary').style.borderColor = '';
  renderHistory([]);
  applySettings(d.settings);
  setEval(0);
  updateMaterial();
  drawArrows([]);  // clear any leftover arrows from previous game
  document.getElementById('gamestatus').textContent = 'your move';
  busy = false;
  highlightLastMove(null);
}
function onUserMove(d){
  game.load(d.fen);
  board.position(d.fen, true);
  highlightLastMove(d.uci);
  drawArrows([]);  // user just played — clear AI's recommendation arrows
  renderHistory(d.history);
  renderCommentary(d.commentary);
  updateMaterial();
  if (d.game_over) {
    onGameOver({result:d.result});
  }
}
function onAIStart(d){
  busy = true;
  resetConverge();
  document.getElementById('thinkmode').textContent = d.budget || d.mode || '';
  document.getElementById('progresstxt').textContent = 'starting…';
  document.getElementById('progressfill').style.width = '2%';
  document.getElementById('gamestatus').textContent = 'AI is thinking…';
  setLive('thinking', 'AI thinking');
}
function onAIProgress(d){
  let done = d.sims_done || 0;
  let target = d.sims_target;
  let pct = 0;
  let label = done + ' sims';
  if (target){
    pct = Math.min(99, 100 * done / target);
    label = done + ' / ' + target + ' sims';
  } else if (d.time_elapsed != null && d.time_budget){
    pct = Math.min(99, 100 * d.time_elapsed / d.time_budget);
    label = d.time_elapsed.toFixed(1) + 's / ' + d.time_budget.toFixed(1) + 's (' + done + ' sims)';
  }
  document.getElementById('progressfill').style.width = pct + '%';
  document.getElementById('progresstxt').textContent = label;
  if (d.analysis){
    renderAnalysis(d.analysis);
  }
  if (d.position) renderPosition(d.position);
}
function onAIMove(d){
  game.load(d.fen);
  board.position(d.fen, true);
  highlightLastMove(d.uci);
  drawArrows([]);  // AI just moved — clear the "I'm planning to play" arrows
  renderHistory(d.history);
  renderAnalysis(d.analysis);   // updates eval / top moves / pv / confidence
  drawArrows([]);  // renderAnalysis re-drew them; explicitly clear post-move
  updateMaterial();
  document.getElementById('progressfill').style.width = '100%';
  document.getElementById('progresstxt').textContent = 'done';
  busy = false;
  setLive('live', 'connected');
  document.getElementById('gamestatus').textContent =
    d.game_over ? 'GAME OVER — ' + (d.result||'') : 'your move';
  if (d.game_over) onGameOver({result:d.result});
}
function onAIDone(d){
  busy = false;
  setLive('live', 'connected');
  document.getElementById('gamestatus').textContent =
    d.game_over ? 'GAME OVER — ' + (d.result||'') : 'your move';
  if (d.game_over) onGameOver({result:d.result});
}
function onPonderState(d){
  let dot = document.getElementById('ponderdot');
  let lbl = document.getElementById('ponderlbl');
  if (d.running){
    dot.className = 'dot thinking';
    lbl.textContent = 'pondering on your time…';
  } else {
    dot.className = 'dot';
    lbl.textContent = 'ponder idle';
  }
}
function onPonderUpdate(d){
  document.getElementById('ponderlbl').textContent =
    'pondering — ' + d.sims + ' sims so far';
  if (d.analysis) renderAnalysis(d.analysis);
  if (d.position) renderPosition(d.position);
}
function onTakeback(d){
  game.load(d.fen);
  board.position(d.fen, true);
  renderHistory(d.history);
  drawArrows([]);
  updateMaterial();
  document.getElementById('commentary').innerHTML =
    '<div class="lead">Take back</div><div>Restored to before your last move.</div>';
  document.getElementById('commentary').className = '';
  document.getElementById('commentary').style.background = '';
  document.getElementById('commentary').style.borderColor = '';
  gameOver = false;
  document.getElementById('gamestatus').textContent = 'your move';
}
function onHint(d){
  if (d.top && d.top.length){
    highlightHint(d.top[0].uci);
    let c = document.getElementById('commentary');
    c.className = 'good';
    c.innerHTML = '<div class="lead">Hint</div>' +
                  '<div>I would play <b>' + fmtMoveSan(d.top[0].uci) +
                  '</b> (Q=' + d.top[0].q.toFixed(2) + ').</div>';
  }
  renderAnalysis(d);
}
function onGameOver(d){
  gameOver = true;
  document.getElementById('gamestatus').textContent = 'GAME OVER — ' + (d.result||'');
  let c = document.getElementById('commentary');
  c.className = '';
  c.innerHTML = '<div class="lead">Game over</div><div>' + (d.result||'') + '</div>';
}

// ─── Render ───────────────────────────────────────────────────────────
// Convergence tracking: how the engine's preferred move evolves during a
// search (does it keep changing its mind = uncertain, or lock in early?).
let convergeState = {best:null, changes:0};
function resetConverge(){
  convergeState = {best:null, changes:0};
  let el = document.getElementById('converge');
  if (el) el.textContent = '';
}
function trackConverge(top){
  if (!top || !top.length) return;
  let best = top[0].uci;
  if (convergeState.best && best !== convergeState.best) convergeState.changes++;
  convergeState.best = best;
  let el = document.getElementById('converge');
  if (!el) return;
  let san = fmtMoveSan(best);
  let tag = convergeState.changes === 0
    ? '<b>locked in</b> from the start'
    : 'changed its mind <b>' + convergeState.changes + '×</b>';
  el.innerHTML = 'Current pick: <b>' + san + '</b> · ' + tag;
}
// Replay a UCI line into SAN on a clone of the current position.
function sanLine(uciList){
  let g = new Chess(game.fen());
  let sans = [];
  for (let u of (uciList||[])){
    let mv = g.move({from:u.slice(0,2), to:u.slice(2,4),
                     promotion: u.length>4 ? u[4] : 'q'});
    if (!mv) break;
    sans.push(mv.san);
  }
  return sans;
}
function toggleMoveDetail(i){
  let row = document.getElementById('mdetail-'+i);
  if (row) row.style.display = (row.style.display === 'none' ? '' : 'none');
}
function renderAnalysis(a){
  if (!a) return;
  if (typeof a.value === 'number') setEval(a.value);
  if (typeof a.nodes === 'number') {
    document.getElementById('nodes').textContent =
      a.nodes ? (a.nodes.toLocaleString() + ' nodes · d' + (a.depth||0)) : '';
  }
  updateConfidence(a);
  drawArrows(a.top || []);
  // Prefer the richer `ext` (prior P, exploration U, mini-PV) when present;
  // fall back to plain `top` for older payloads.
  let rows = (a.ext && a.ext.length) ? a.ext : (a.top || []);
  trackConverge(rows);
  // Top moves
  let tbody = document.getElementById('topmoves');
  if (rows.length){
    let total = rows.reduce((s,m) => s + (m.n||0), 0) || 1;
    tbody.innerHTML = rows.map((m, i) => {
      let pct = (m.n||0) / total * 100;
      let cls = i === 0 ? 'top1' : '';
      let priorTxt = (typeof m.p === 'number') ? (m.p*100).toFixed(1)+'%' : '—';
      let main = '<tr class="moverow '+cls+'" onclick="toggleMoveDetail('+i+')">'+
             '<td>'+(i+1)+'</td><td>'+fmtMoveSan(m.uci)+'</td>'+
             '<td class="num">'+m.n+'</td>'+
             '<td class="num prior">'+priorTxt+'</td>'+
             '<td class="num">'+(m.q>=0?'+':'')+m.q.toFixed(2)+'</td>'+
             '<td><div class="bar"><span style="width:'+pct.toFixed(0)+'%"></span></div></td></tr>';
      // Detail row: PUCT decomposition + the move's own expected line.
      let detail = '';
      if (typeof m.p === 'number' || (m.line && m.line.length)){
        let puct = '<span class="puct">Q '+(m.q>=0?'+':'')+m.q.toFixed(3)+'</span>'+
                   '<span class="puct">prior '+priorTxt+'</span>'+
                   (typeof m.u==='number' ? '<span class="puct">explore +'+m.u.toFixed(3)+'</span>' : '');
        let sans = sanLine(m.line);
        let lineHtml = sans.length
          ? '<span class="sanline">'+sans.map(s=>'<span>'+s+'</span>').join('')+'</span>'
          : '<span style="color:var(--ink-fade)">no deeper line searched</span>';
        detail = '<tr id="mdetail-'+i+'" style="display:none"><td colspan="6" class="miniline">'+
                 puct+'<br>'+lineHtml+'</td></tr>';
      }
      return main + detail;
    }).join('');
  }
  // PV
  let pv = document.getElementById('pv');
  if (a.pv && a.pv.length){
    let sans = sanLine(a.pv);
    pv.innerHTML = sans.length ? sans.map(s => '<span>'+s+'</span>').join('') :
                                 '<span style="color:var(--ink-dim);background:transparent;border:none">—</span>';
  }
}
function renderPosition(pos){
  if (!pos) return;
  // WDL bar
  let w = pos.wdl || {};
  if (typeof w.win === 'number'){
    let pw = Math.max(0, w.win)*100, pd = Math.max(0, w.draw)*100, pl = Math.max(0, w.loss)*100;
    document.getElementById('wdl-w').style.width = pw.toFixed(1)+'%';
    document.getElementById('wdl-d').style.width = pd.toFixed(1)+'%';
    document.getElementById('wdl-l').style.width = pl.toFixed(1)+'%';
    document.getElementById('wdl-wt').textContent = pw.toFixed(0)+'%';
    document.getElementById('wdl-dt').textContent = pd.toFixed(0)+'%';
    document.getElementById('wdl-lt').textContent = pl.toFixed(0)+'%';
  }
  // Material / phase / aux
  if (typeof pos.material === 'number'){
    let m = pos.material;
    document.getElementById('pi-mat').textContent =
      (m>0?'White +'+m : m<0?'Black +'+(-m) : 'even');
  }
  if (pos.phase) document.getElementById('pi-phase').textContent = pos.phase;
  if (w && typeof w.aux === 'number')
    document.getElementById('pi-aux').textContent = (w.aux>=0?'+':'')+w.aux.toFixed(2);
  // Tactics
  let t = pos.tactics || {};
  let chips = [];
  if (t.in_check) chips.push('<span class="chk">in check</span>');
  (t.captures||[]).forEach(c =>
    chips.push('<span class="cap">×'+c.victim+' '+fmtMoveSan(c.uci)+'</span>'));
  (t.hanging||[]).forEach(h =>
    chips.push('<span class="hang">hanging '+h.piece+'</span>'));
  let tac = document.getElementById('pi-tac');
  tac.innerHTML = chips.length ? chips.join('') :
    '<span style="color:var(--ink-fade);border:none;background:none">nothing obvious</span>';
}
// Eval-over-game line chart from the move history (side-to-move POV evals).
function renderEvalChart(h){
  let svg = document.getElementById('evalchart');
  if (!svg) return;
  let pts = (h||[]).filter(x => typeof x.eval === 'number');
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const W=300, H=90, mid=H/2;
  const mkline = (cls, d) => {
    let p = document.createElementNS('http://www.w3.org/2000/svg','line');
    p.setAttribute('class',cls);
    p.setAttribute('x1',d[0]);p.setAttribute('y1',d[1]);
    p.setAttribute('x2',d[2]);p.setAttribute('y2',d[3]);
    svg.appendChild(p);
  };
  mkline('zero',[0,mid,W,mid]);
  if (pts.length < 2){
    let t=document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x',W/2);t.setAttribute('y',mid-6);
    t.setAttribute('text-anchor','middle');
    t.setAttribute('fill','#7a8590');t.setAttribute('font-size','11');
    t.textContent='not enough moves yet';svg.appendChild(t);
    return;
  }
  // eval in [-1,1] → y; clamp. Top = +1 (good for side to move at that ply).
  let path='';
  pts.forEach((x,i) => {
    let xx = i/(pts.length-1)*W;
    let e = Math.max(-1, Math.min(1, x.eval));
    let yy = mid - e*(mid-4);
    path += (i===0?'M':'L')+xx.toFixed(1)+' '+yy.toFixed(1)+' ';
  });
  let pl=document.createElementNS('http://www.w3.org/2000/svg','path');
  pl.setAttribute('class','line');pl.setAttribute('d',path);
  svg.appendChild(pl);
}
function renderHistory(h){
  let el = document.getElementById('history');
  renderEvalChart(h);
  if (!h || !h.length) {
    el.innerHTML = '<div style="color:var(--ink-dim);text-align:center;padding:20px">no moves yet</div>';
    return;
  }
  let rows = [];
  let g2 = new Chess();
  for (let i = 0; i < h.length; i++){
    let mv = g2.move({from:h[i].uci.slice(0,2), to:h[i].uci.slice(2,4),
                      promotion: h[i].uci.length>4 ? h[i].uci[4] : 'q'});
    let san = mv ? mv.san : h[i].uci;
    let cls = h[i].by === 'ai' ? 'ai' : '';
    let evstr = (h[i].eval != null) ? ((h[i].eval>=0?'+':'')+h[i].eval.toFixed(2)) : '';
    rows.push('<div class="row"><div class="num">'+Math.ceil(h[i].ply/2)+(h[i].ply%2===1?'.':'…')+'</div>'+
              '<div class="'+cls+'">'+san+'</div>'+
              '<div style="color:var(--ink-dim);text-align:right">'+evstr+'</div>'+
              (h[i].verdict ? '<div class="verdict '+verdictClass(h[i].verdict)+'">'+escapeHtml(h[i].verdict)+'</div>':'')+
              '</div>');
  }
  el.innerHTML = rows.join('');
  el.scrollTop = el.scrollHeight;
}
function verdictClass(v){
  if (!v) return '';
  if (v.startsWith('Best')) return 'good';
  if (v.startsWith('Mistake')) return 'bad';
  if (v.includes('inaccuracy') || v.includes('Off my radar')) return 'warn';
  return '';
}
function escapeHtml(s){ return s.replace(/[&<>"']/g, m => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m])); }
function renderCommentary(c){
  if (!c) return;
  let el = document.getElementById('commentary');
  let cls;
  if (c.low_confidence){
    cls = 'warn-conf';
    el.className = '';   // neutral background for the "no opinion" case
    el.style.background = 'rgba(255,180,84,.07)';
    el.style.borderColor = 'rgba(255,180,84,.25)';
  } else {
    el.style.background = '';
    el.style.borderColor = '';
    el.className = verdictClass(c.verdict);
  }
  let head;
  if (c.low_confidence){
    head = 'AI on your move (low confidence)';
  } else {
    head = 'AI on your move' + (c.rank ? ' (ranked #'+c.rank+' on my list)' : '');
  }
  el.innerHTML = '<div class="lead">' + head + '</div>' +
                 '<div>' + escapeHtml(c.verdict || '—') + '</div>';
}

// ─── Settings ─────────────────────────────────────────────────────────
let settingsDebounce;
function settingChanged(){
  // Sync slider <-> number for sims/time
  document.getElementById('thinkSimsR').value = document.getElementById('thinkSims').value;
  document.getElementById('thinkTimeR').value = document.getElementById('thinkTime').value;
  // Show/hide appropriate row
  let mode = document.getElementById('thinkMode').value;
  document.getElementById('sims-row').style.display = (mode==='sims') ? 'grid' : 'none';
  document.getElementById('time-row').style.display = (mode==='time') ? 'grid' : 'none';
  document.getElementById('dyn-row').style.display  = (mode==='dynamic') ? 'grid' : 'none';
  // Debounced push to server
  clearTimeout(settingsDebounce);
  settingsDebounce = setTimeout(pushSettings, 250);
}
function pushSettings(){
  post('/api/settings', {
    think_mode:   document.getElementById('thinkMode').value,
    think_sims:   parseInt(document.getElementById('thinkSims').value),
    think_time_s: parseFloat(document.getElementById('thinkTime').value),
    dynamic_min:  parseInt(document.getElementById('dynMin').value),
    dynamic_max:  parseInt(document.getElementById('dynMax').value),
    ponder_on:    document.getElementById('ponderOn').checked,
    temperature:  parseFloat(document.getElementById('temperature').value),
    contempt:     parseFloat(document.getElementById('contempt').value),
  });
}
function applySettings(s){
  if (!s) return;
  document.getElementById('thinkMode').value = s.think_mode;
  document.getElementById('thinkSims').value = s.think_sims;
  document.getElementById('thinkSimsR').value = s.think_sims;
  document.getElementById('thinkTime').value = s.think_time_s;
  document.getElementById('thinkTimeR').value = s.think_time_s;
  document.getElementById('dynMin').value = s.dynamic_min;
  document.getElementById('dynMax').value = s.dynamic_max;
  document.getElementById('ponderOn').checked = s.ponder_on;
  document.getElementById('temperature').value = s.temperature;
  document.getElementById('tempLbl').textContent = parseFloat(s.temperature).toFixed(3);
  if (s.contempt !== undefined) {
    document.getElementById('contempt').value = s.contempt;
    document.getElementById('contemptLbl').textContent = parseFloat(s.contempt).toFixed(2);
  }
  document.getElementById('sims-row').style.display = (s.think_mode==='sims') ? 'grid' : 'none';
  document.getElementById('time-row').style.display = (s.think_mode==='time') ? 'grid' : 'none';
  document.getElementById('dyn-row').style.display  = (s.think_mode==='dynamic') ? 'grid' : 'none';
}

// ─── Actions ──────────────────────────────────────────────────────────
function newGame(asWhite){
  post('/api/new', {human_white: !!asWhite});
}
function takeback(){ post('/api/takeback', {}); }
function resign(){ post('/api/resign', {}); }
function hint(){ post('/api/hint', {}); }

// ─── Boot ─────────────────────────────────────────────────────────────
board = Chessboard('board', {
  draggable: true,
  position: 'start',
  onDrop: onDrop,
  pieceTheme: '/piece/{piece}.svg',
});

// Fetch engine info for the pill
fetch('/api/state').then(r => r.json()).then(s => {
  document.getElementById('enginepill').textContent = 'engine: ready';
  applySnapshot(s);
});

connectStream();

window.addEventListener('resize', () => { if (board) board.resize(); });
</script>

<!-- ════════════════════ Chess AI — PRO FEATURE PACK ════════════════════ -->
<script>
(function(){
  'use strict';
  const $  = id => document.getElementById(id);
  const elc = (tag, cls, html) => { const e=document.createElement(tag);
        if(cls) e.className=cls; if(html!=null) e.innerHTML=html; return e; };
  const LS = window.localStorage;
  const get = (k,d) => { const v=LS.getItem(k); return v===null?d:v; };

  // ── State ────────────────────────────────────────────────────────────
  let proHist=[], prevFen=null, reviewPly=null;
  let threatsOn = get('pp_threats','0')==='1';
  let soundOn   = get('pp_sound','1')==='1';

  // ── Threat overlay (inside the board stage) ──────────────────────────
  const stage = document.querySelector('.board-stage');
  const tov = elc('div','threats'); tov.id='threats';
  if (stage) stage.appendChild(tov);

  // ── Pro toolbar (prepended to the left column) ───────────────────────
  const bar = elc('div','card');
  bar.innerHTML =
    '<h3>Pro tools</h3>'+
    '<div class="protools">'+
      '<button class="tbtn" data-a="flip" title="Flip board (F)">⟲ Flip</button>'+
      '<button class="tbtn" data-a="coords" title="Toggle coordinates">#</button>'+
      '<button class="tbtn" data-a="theme" title="Cycle theme">🎨 Theme</button>'+
      '<button class="tbtn" data-a="sound" title="Toggle sound">🔊</button>'+
      '<button class="tbtn" data-a="threats" title="Threat overlay">⚠ Threats</button>'+
      '<button class="tbtn" data-a="focus" title="Focus / zen mode">⛶ Focus</button>'+
      '<button class="tbtn" data-a="pgn" title="Download PGN">⭳ PGN</button>'+
      '<button class="tbtn" data-a="cfen" title="Copy FEN">FEN</button>'+
      '<button class="tbtn" data-a="cpgn" title="Copy PGN">⧉ PGN</button>'+
      '<button class="tbtn" data-a="loadfen" title="Play from a FEN">Load FEN</button>'+
      '<button class="tbtn" data-a="help" title="Keyboard shortcuts (?)">?</button>'+
      '<span class="opening-chip" id="openingChip">—</span>'+
    '</div>';
  const leftCol = document.querySelector('.left');
  if (leftCol) leftCol.insertBefore(bar, leftCol.firstChild);

  // ── Training monitor card (right column) ─────────────────────────────
  const tmCard = elc('div','card');
  tmCard.innerHTML = '<h3>Training monitor <span id="tmStale" class="tm-stale"></span></h3>'+
    '<div class="tm-grid" id="tmGrid"></div>';
  // ── Accuracy / review card (right column) ────────────────────────────
  const accCard = elc('div','card');
  accCard.innerHTML = '<h3>Game accuracy &amp; review</h3>'+
    '<div class="acc-row">'+
      '<div class="acc-pill"><div class="big" id="accW">—</div><div class="lbl">White</div></div>'+
      '<div class="acc-pill"><div class="big" id="accB">—</div><div class="lbl">Black</div></div>'+
    '</div>'+
    '<div class="acc-counts" id="accCounts"></div>'+
    '<div style="font-size:11px;color:var(--ink-fade);margin-top:8px">'+
      'Click a move below, or use ← / → to step through the game.</div>';
  const rightCol = document.querySelector('.right');
  if (rightCol){ rightCol.appendChild(tmCard); rightCol.appendChild(accCard); }

  // ── Fixed chrome: review bar + help overlay ──────────────────────────
  const rbar = elc('div'); rbar.id='reviewbar';
  rbar.innerHTML = '<button data-r="first">⏮</button><button data-r="prev">◀</button>'+
    '<span id="reviewlbl" style="font-variant-numeric:tabular-nums">review</span>'+
    '<button data-r="next">▶</button><button data-r="last">⏭</button>'+
    '<button data-r="live" class="primary">● Live</button>';
  document.body.appendChild(rbar);

  const help = elc('div'); help.id='helpov';
  help.innerHTML = '<div class="box"><h3 style="margin-top:0">Keyboard shortcuts</h3>'+
    '<table>'+
    '<tr><td><kbd>N</kbd></td><td>New game (you as White)</td></tr>'+
    '<tr><td><kbd>F</kbd></td><td>Flip board</td></tr>'+
    '<tr><td><kbd>H</kbd></td><td>Hint</td></tr>'+
    '<tr><td><kbd>T</kbd></td><td>Takeback</td></tr>'+
    '<tr><td><kbd>R</kbd></td><td>Resign</td></tr>'+
    '<tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>Step through game (review)</td></tr>'+
    '<tr><td><kbd>Esc</kbd></td><td>Back to live / close</td></tr>'+
    '<tr><td><kbd>?</kbd></td><td>This help</td></tr>'+
    '</table><div style="text-align:right;margin-top:10px">'+
    '<button data-r="closehelp" class="primary">Got it</button></div></div>';
  document.body.appendChild(help);

  // ── Sound (WebAudio, no assets) ──────────────────────────────────────
  let actx=null;
  function beep(freq,dur,type,gain){ if(!soundOn) return;
    try{ actx=actx||new (window.AudioContext||window.webkitAudioContext)();
      const o=actx.createOscillator(), g=actx.createGain();
      o.type=type||'sine'; o.frequency.value=freq; g.gain.value=gain||0.04;
      o.connect(g); g.connect(actx.destination); o.start();
      g.gain.exponentialRampToValueAtTime(0.0001, actx.currentTime+(dur||0.12));
      o.stop(actx.currentTime+(dur||0.13)); }catch(e){}
  }
  const sMove=()=>beep(440,0.07,'triangle',0.05);
  const sCap =()=>beep(190,0.12,'sawtooth',0.06);
  const sChk =()=>beep(720,0.16,'square',0.045);
  const sEnd =()=>{ beep(330,0.28,'sine',0.06); setTimeout(()=>beep(247,0.4,'sine',0.06),150); };

  // ── Opening detection (longest-prefix match) ─────────────────────────
  const OPEN = [
    [['e2e4'],'King’s Pawn'],[['d2d4'],'Queen’s Pawn'],
    [['c2c4'],'English Opening'],[['g1f3'],'Réti Opening'],
    [['e2e4','e7e5'],'Open Game'],[['e2e4','c7c5'],'Sicilian Defense'],
    [['e2e4','e7e6'],'French Defense'],[['e2e4','c7c6'],'Caro-Kann'],
    [['e2e4','d7d5'],'Scandinavian'],[['e2e4','g8f6'],'Alekhine Defense'],
    [['e2e4','d7d6'],'Pirc Defense'],[['e2e4','b8c6'],'Nimzowitsch Def.'],
    [['e2e4','g7g6'],'Modern Defense'],
    [['e2e4','e7e5','g1f3','b8c6','f1b5'],'Ruy López'],
    [['e2e4','e7e5','g1f3','b8c6','f1c4'],'Italian Game'],
    [['e2e4','e7e5','g1f3','b8c6','d2d4'],'Scotch Game'],
    [['e2e4','e7e5','g1f3','g8f6'],'Petrov Defense'],
    [['e2e4','c7c5','g1f3','d7d6'],'Sicilian: Najdorf-ish'],
    [['d2d4','d7d5'],'Closed Game'],[['d2d4','g8f6'],'Indian Defense'],
    [['d2d4','d7d5','c2c4'],'Queen’s Gambit'],
    [['d2d4','d7d5','c2c4','e7e6'],'QGD'],
    [['d2d4','d7d5','c2c4','c7c6'],'Slav Defense'],
    [['d2d4','g8f6','c2c4','g7g6'],'King’s Indian'],
    [['d2d4','g8f6','c2c4','e7e6'],'Nimzo/QID'],
    [['d2d4','f7f5'],'Dutch Defense'],
  ];
  function openingName(ucis){ let best='—', bl=0;
    for(const [seq,name] of OPEN){
      if(seq.length<=ucis.length && seq.length>bl && seq.every((u,i)=>u===ucis[i])){
        best=name; bl=seq.length; } }
    return best;
  }

  // ── Accuracy (from per-move eval vs best_eval) ───────────────────────
  function classify(delta){
    if(delta==null) return ['',''];
    if(delta<0.02) return ['!!','bril'];
    if(delta<0.05) return ['!','good'];
    if(delta<0.15) return ['','' ];
    if(delta<0.40) return ['?!','inacc'];
    if(delta<0.90) return ['?','mis'];
    return ['??','blu'];
  }
  function accFromDelta(delta){ // crude human-move accuracy curve
    if(delta==null) return null;
    return Math.max(0, Math.min(100, 100*Math.exp(-2.6*Math.max(0,delta))));
  }
  function renderAccuracy(){
    let accs={w:[],b:[]}, cnt={bril:0,blu:0,mis:0,inacc:0};
    for(const h of proHist){
      if(h.by!=='human') continue;
      if(h.eval==null||h.best_eval==null||h.low_confidence) continue;
      const delta=Math.max(0,(h.best_eval||0)-(h.eval||0));
      const side=(h.ply%2===1)?'w':'b';
      const a=accFromDelta(delta); if(a!=null) accs[side].push(a);
      const [g,c]=classify(delta);
      if(c==='bril') cnt.bril++; else if(c==='blu') cnt.blu++;
      else if(c==='mis') cnt.mis++; else if(c==='inacc') cnt.inacc++;
    }
    const avg=a=>a.length?Math.round(a.reduce((s,x)=>s+x,0)/a.length):null;
    const aw=avg(accs.w), ab=avg(accs.b);
    $('accW').textContent = aw==null?'—':aw+'%';
    $('accB').textContent = ab==null?'—':ab+'%';
    $('accCounts').innerHTML =
      '<span class="bril">'+cnt.bril+' brilliant</span>'+
      '<span class="inacc">'+cnt.inacc+' inacc.</span>'+
      '<span class="mis">'+cnt.mis+' mistakes</span>'+
      '<span class="blu">'+cnt.blu+' blunders</span>';
    decorateHistory();
  }
  // Make the existing history rows hoverable + clickable for review.
  function decorateHistory(){
    const hist=$('history'); if(!hist) return;
    hist.querySelectorAll('.row').forEach(row=>row.classList.add('moverow'));
  }

  // ── Threat overlay ───────────────────────────────────────────────────
  function squareBox(sq){ const c=(typeof squareCenter==='function')?squareCenter(sq):null;
    if(!c) return null; return {l:c.x-6.25, t:c.y-6.25}; }
  async function refreshThreats(){
    if(!threatsOn || reviewPly!=null){ tov.innerHTML=''; return; }
    try{ const r=await fetch('/api/threats'); const t=await r.json(); drawThreats(t); }
    catch(e){ /* offline / between games */ }
  }
  function drawThreats(t){
    tov.innerHTML=''; if(!t||!t.hanging) return;
    for(const h of t.hanging){
      const b=squareBox(h.sq); if(!b) continue;
      const d=elc('div','tsq hang'); d.style.left=b.l+'%'; d.style.top=b.t+'%';
      d.title=(h.white?'White ':'Black ')+h.sym+' is loose on '+h.sq;
      tov.appendChild(d);
    }
  }

  // ── Themes ───────────────────────────────────────────────────────────
  const THEMES = {
    midnight:{}, // default :root
    light:{'--bg-0':'#eef1f5','--bg-1':'#ffffff','--bg-2':'#f4f6f9','--bg-3':'#e7ebf1',
           '--ink':'#10151b','--ink-dim':'#475263','--ink-fade':'#8b97a6','--border':'#d3dae3',
           '--shadow':'0 6px 24px rgba(40,60,90,.12)'},
    emerald:{'--bg-0':'#06120e','--bg-1':'#0c1d16','--bg-2':'#102a20','--bg-3':'#16362a',
             '--accent':'#37e0a0','--good':'#7ef0c8','--border':'#1c4234'},
    rose:{'--bg-0':'#160a10','--bg-1':'#221019','--bg-2':'#2c1521','--bg-3':'#3a1c2c',
          '--accent':'#ff7eb6','--good':'#ffa9d0','--border':'#43202f'},
  };
  const THORDER=['midnight','light','emerald','rose'];
  let themeStyle=elc('style'); themeStyle.id='protheme'; document.head.appendChild(themeStyle);
  function applyTheme(name){
    const vars=THEMES[name]||{}; let css=':root{';
    for(const k in vars) css+=k+':'+vars[k]+';'; css+='}';
    themeStyle.textContent=css; LS.setItem('pp_theme',name);
    document.body.classList.toggle('lighttheme', name==='light');
  }
  applyTheme(get('pp_theme','midnight'));

  // ── PGN / FEN ────────────────────────────────────────────────────────
  function curPGN(){ try{ return game.pgn({max_width:80,newline_char:'\n'})||''; }catch(e){ return ''; } }
  function dl(name,text){ const b=new Blob([text],{type:'text/plain'});
    const u=URL.createObjectURL(b); const a=elc('a'); a.href=u; a.download=name;
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(u); }
  async function copy(text){ try{ await navigator.clipboard.writeText(text); flash('Copied'); }
    catch(e){ const ta=elc('textarea'); ta.value=text; document.body.appendChild(ta);
      ta.select(); try{document.execCommand('copy');flash('Copied');}catch(_){} ta.remove(); } }
  let flashT=null;
  function flash(msg){ let f=$('ppflash'); if(!f){ f=elc('div'); f.id='ppflash';
      f.style.cssText='position:fixed;bottom:18px;right:18px;z-index:70;padding:8px 14px;'+
        'background:var(--accent);color:#0b0d10;border-radius:8px;font-weight:600;box-shadow:var(--shadow)';
      document.body.appendChild(f); }
    f.textContent=msg; f.style.opacity='1'; clearTimeout(flashT);
    flashT=setTimeout(()=>{ f.style.transition='opacity .4s'; f.style.opacity='0'; },1100);
  }
  function pgnHeaders(){ const d=new Date().toISOString().slice(0,10);
    return '[Event "Chess AI web"]\n[Site "local"]\n[Date "'+d+'"]\n'+
      '[White "'+(humanWhite?'You':'ChessAI')+'"]\n[Black "'+(humanWhite?'ChessAI':'You')+'"]\n\n'; }

  // ── Board controls ───────────────────────────────────────────────────
  function flip(){ if(board) board.flip(); setTimeout(refreshThreats,60); }
  function toggleCoords(){ document.body.classList.toggle('no-coords');
    LS.setItem('pp_coords', document.body.classList.contains('no-coords')?'0':'1'); }
  function toggleFocus(){ document.body.classList.toggle('focusmode');
    setTimeout(()=>{ if(board) board.resize(); refreshThreats(); },60); }
  function cycleTheme(){ const cur=get('pp_theme','midnight');
    applyTheme(THORDER[(THORDER.indexOf(cur)+1)%THORDER.length]); }
  function toggleSound(){ soundOn=!soundOn; LS.setItem('pp_sound',soundOn?'1':'0'); syncBtns();
    if(soundOn) sMove(); }
  function toggleThreats(){ threatsOn=!threatsOn; LS.setItem('pp_threats',threatsOn?'1':'0');
    syncBtns(); refreshThreats(); }
  function loadFen(){ const f=prompt('Paste a FEN to play from:'); if(f&&f.trim())
    post('/api/setfen',{fen:f.trim()}).then(r=>{ if(r&&r.error) alert(r.error); }); }

  // ── Review mode ──────────────────────────────────────────────────────
  function maxPly(){ return proHist.length; }
  function fenAtPly(p){ // reconstruct from startpos by replaying recorded ucis
    try{ const tmp=new Chess();
      for(let i=0;i<p && i<proHist.length;i++){ const u=proHist[i].uci||'';
        if(u.length<4) return null;
        tmp.move({from:u.slice(0,2),to:u.slice(2,4),promotion:(u[4]||'q')}); }
      return tmp.fen();
    }catch(e){ return null; }
  }
  function enterReview(p){ p=Math.max(0,Math.min(maxPly(),p));
    const fen=fenAtPly(p); if(fen==null){ flash('Review N/A (custom start)'); return; }
    reviewPly=p; if(board) board.position(fen,false); tov.innerHTML='';
    rbar.classList.add('show');
    $('reviewlbl').textContent='ply '+p+' / '+maxPly();
  }
  function reviewLive(){ reviewPly=null; rbar.classList.remove('show');
    if(board && typeof game!=='undefined') board.position(game.fen()); refreshThreats(); }
  function reviewStep(d){ if(reviewPly==null) enterReview(maxPly()+d); else enterReview(reviewPly+d); }

  // ── Training monitor poll ────────────────────────────────────────────
  function fmt(v,dp){ return (v==null)?'—':(dp!=null?Number(v).toFixed(dp):v); }
  let lastStep=null, lastStepTs=0;
  async function refreshTraining(){
    let d; try{ d=await (await fetch('/api/training')).json(); }catch(e){ return; }
    const cells=[
      ['Step', d.step!=null?d.step.toLocaleString():'—',''],
      ['ELO', fmt(d.elo,1), 'good'],
      ['Loss total', fmt(d.loss_total,3),''],
      ['Policy loss', fmt(d.p,3),''],
      ['Value loss', fmt(d.v,3),''],
      ['Teacher KL', fmt(d.kl,3),''],
      ['Entropy', fmt(d.entropy,2),''],
      ['LR', d.lr!=null?Number(d.lr).toExponential(2):'—',''],
      ['Throughput', d.pos_s!=null?(Number(d.pos_s).toFixed(0)+' pos/s'):'—',''],
      ['Games/hr', d.games_hr!=null?Math.round(d.games_hr).toLocaleString():'—',''],
      ['Replay buf', d.replay!=null?Math.round(d.replay).toLocaleString():'—',''],
      ['VRAM', d.vram!=null?(Math.round(d.vram)+' MB'):'—',''],
    ];
    $('tmGrid').innerHTML = cells.map(([k,v,c])=>
      '<div class="tm-cell"><span class="k">'+k+'</span><span class="v '+(c||'')+'">'+v+'</span></div>'
    ).join('');
    // "alive" detector: did the step advance recently?
    const now=Date.now()/1000;
    if(d.step!=null && d.step!==lastStep){ lastStep=d.step; lastStepTs=now; }
    let st='';
    if(d.step==null) st='(no run data yet)';
    else if(now-lastStepTs>120) st='● idle (no new steps)';
    else st='● live';
    $('tmStale').textContent=st;
    $('tmStale').style.color = st==='● live' ? 'var(--good)' : 'var(--warn)';
  }

  // ── Wire toolbar + review + help buttons ─────────────────────────────
  function syncBtns(){
    document.querySelectorAll('.tbtn').forEach(b=>{
      const a=b.dataset.a;
      if(a==='sound'){ b.classList.toggle('on',soundOn); b.textContent=soundOn?'🔊':'🔇'; }
      if(a==='threats') b.classList.toggle('on',threatsOn);
      if(a==='focus') b.classList.toggle('on',document.body.classList.contains('focusmode'));
    });
  }
  bar.addEventListener('click',e=>{ const b=e.target.closest('.tbtn'); if(!b) return;
    const a=b.dataset.a;
    if(a==='flip') flip();
    else if(a==='coords') toggleCoords();
    else if(a==='theme') cycleTheme();
    else if(a==='sound') toggleSound();
    else if(a==='threats') toggleThreats();
    else if(a==='focus') toggleFocus();
    else if(a==='pgn') dl('chessai_game.pgn', pgnHeaders()+curPGN());
    else if(a==='cfen'){ try{ copy(game.fen()); }catch(_){ } }
    else if(a==='cpgn') copy(pgnHeaders()+curPGN());
    else if(a==='loadfen') loadFen();
    else if(a==='help') help.classList.add('show');
  });
  rbar.addEventListener('click',e=>{ const b=e.target.closest('button'); if(!b) return;
    const r=b.dataset.r;
    if(r==='first') enterReview(0); else if(r==='prev') reviewStep(-1);
    else if(r==='next') reviewStep(1); else if(r==='last') enterReview(maxPly());
    else if(r==='live') reviewLive(); });
  help.addEventListener('click',e=>{ if(e.target===help||e.target.dataset.r==='closehelp')
    help.classList.remove('show'); });
  // Click a move in the existing history list → review that ply.
  const histEl=$('history');
  if(histEl) histEl.addEventListener('click',e=>{ const row=e.target.closest('.row');
    if(!row) return; const rows=[...histEl.querySelectorAll('.row')];
    const idx=rows.indexOf(row); if(idx<0) return;
    // each row holds a move PAIR (white+black) → jump to end of that pair
    enterReview(Math.min(maxPly(),(idx+1)*2)); });

  // ── Keyboard shortcuts ───────────────────────────────────────────────
  document.addEventListener('keydown',e=>{
    if(e.target.matches('input,textarea,select')) return;
    const k=e.key;
    if(k==='ArrowLeft'){ reviewStep(-1); e.preventDefault(); }
    else if(k==='ArrowRight'){ reviewStep(1); e.preventDefault(); }
    else if(k==='Escape'){ if(help.classList.contains('show')) help.classList.remove('show');
      else if(reviewPly!=null) reviewLive(); }
    else if(k==='?'||(k==='/'&&e.shiftKey)) help.classList.toggle('show');
    else if(k==='f'||k==='F') flip();
    else if(k==='n'||k==='N'){ if(typeof newGame==='function') newGame(true); }
    else if(k==='h'||k==='H'){ if(typeof hint==='function') hint(); }
    else if(k==='t'||k==='T'){ if(typeof takeback==='function') takeback(); }
    else if(k==='r'||k==='R'){ if(typeof resign==='function') resign(); }
  });

  // ── Own SSE stream (additive; server supports many clients) ──────────
  function pieceCount(fen){ return (fen.split(' ')[0].match(/[a-zA-Z]/g)||[]).length; }
  function reactMove(newFen){
    let cap=false, chk=false;
    if(prevFen && newFen) cap = pieceCount(newFen) < pieceCount(prevFen);
    try{ chk = new Chess(newFen).in_check(); }catch(_){}
    if(chk) sChk(); else if(cap) sCap(); else sMove();
    prevFen=newFen;
  }
  function ingest(data, isMove){
    if(data.history) proHist=data.history;
    if(data.fen!=null) prevFen=prevFen||data.fen;
    const ucis=proHist.map(h=>h.uci).filter(Boolean);
    const oc=$('openingChip'); if(oc) oc.textContent=openingName(ucis);
    renderAccuracy();
    if(reviewPly==null) refreshThreats();
    if(isMove && data.fen) reactMove(data.fen);
    if(data.game_over) sEnd();
  }
  try{
    const es=new EventSource('/api/stream');
    const on=(name,fn)=>es.addEventListener(name,e=>{ try{ fn(JSON.parse(e.data)); }catch(_){} });
    on('snapshot', d=>{ proHist=d.history||[]; prevFen=d.fen||null; reviewPly=null;
                        rbar.classList.remove('show'); ingest(d,false); });
    on('new_game', d=>{ proHist=[]; prevFen=d.fen||null; reviewPly=null;
                        rbar.classList.remove('show');
                        const oc=$('openingChip'); if(oc) oc.textContent='—';
                        renderAccuracy(); refreshThreats(); });
    on('user_move', d=>ingest(d,true));
    on('ai_move',   d=>ingest(d,true));
    on('takeback',  d=>{ if(d.history) proHist=d.history; reviewLive(); ingest(d,false); });
    on('game_over', d=>{ sEnd(); ingest(d,false); });
  }catch(e){ /* EventSource unsupported */ }

  // ── Boot ─────────────────────────────────────────────────────────────
  if(get('pp_coords','1')==='0') document.body.classList.add('no-coords');
  syncBtns();
  refreshTraining(); setInterval(refreshTraining, 4000);
  setTimeout(refreshThreats, 800);
  console.log('[ChessAI] Pro feature pack loaded.');
})();
</script>
</body></html>
"""


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Play the trained chess AI in your browser.",
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resume", default=None,
                    help="Checkpoint to load (default: newest in checkpoint dir).")
    ap.add_argument("--sims", type=int, default=None,
                    help="Initial sims-per-move (UI can change this).")
    ap.add_argument("--side", choices=["white", "black"], default="white",
                    help="Your initial colour. UI can flip on 'New game'.")
    args = ap.parse_args()

    update_config_from_dict(run_probe(write=False))
    device = torch.device(CONFIG.device)

    from utils.ckpt_arch import latest_checkpoint, match_arch
    ckpt = args.resume or latest_checkpoint(CONFIG.checkpoint_dir)
    match_arch(CONFIG, ckpt)
    model = build_model(CONFIG, compile_model=False).eval()

    if ckpt and Path(ckpt).exists():
        st = torch.load(ckpt, map_location=device, weights_only=True)
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        base.load_state_dict(st["model"], strict=False)
        print(f"[web] Loaded {ckpt} (step {st.get('step', '?')})")
    else:
        print("[web] No checkpoint — UNTRAINED net.")

    engine = build_inference_engine(CONFIG, model, device)

    global SESSION
    SESSION = GameSession(model, device, engine, CONFIG)
    if args.sims is not None:
        SESSION.think_sims = int(args.sims)
    SESSION.new_game(human_white=(args.side == "white"))

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[web] http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped.")
        try:
            SESSION._stop_ponder()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
