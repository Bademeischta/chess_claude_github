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
            delta = best_q - user_q   # both from side-to-move's POV; ≥0
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
    }


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
    </div>

    <div class="card">
      <h3>Top candidate moves</h3>
      <table class="moves"><thead><tr>
        <th>#</th><th>Move</th><th class="num">Visits</th>
        <th class="num">Q</th><th>Share</th>
      </tr></thead><tbody id="topmoves"><tr><td colspan="5" style="color:var(--ink-dim);text-align:center;padding:18px">no analysis yet</td></tr></tbody></table>
    </div>

    <div class="card">
      <h3>Principal variation</h3>
      <div class="pv" id="pv"><span style="color:var(--ink-dim);background:transparent;border:none">—</span></div>
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
function renderAnalysis(a){
  if (!a) return;
  if (typeof a.value === 'number') setEval(a.value);
  if (typeof a.nodes === 'number') {
    document.getElementById('nodes').textContent =
      a.nodes ? (a.nodes.toLocaleString() + ' nodes · d' + (a.depth||0)) : '';
  }
  updateConfidence(a);
  drawArrows(a.top || []);
  // Top moves
  let tbody = document.getElementById('topmoves');
  if (a.top && a.top.length){
    let total = a.top.reduce((s,m) => s + (m.n||0), 0) || 1;
    tbody.innerHTML = a.top.map((m, i) => {
      let pct = (m.n||0) / total * 100;
      let cls = i === 0 ? 'top1' : '';
      return '<tr class="'+cls+'"><td>'+(i+1)+'</td><td>'+fmtMoveSan(m.uci)+
             '</td><td class="num">'+m.n+'</td>'+
             '<td class="num">'+(m.q>=0?'+':'')+m.q.toFixed(2)+'</td>'+
             '<td><div class="bar"><span style="width:'+pct.toFixed(0)+'%"></span></div></td></tr>';
    }).join('');
  }
  // PV
  let pv = document.getElementById('pv');
  if (a.pv && a.pv.length){
    // Render PV as SAN by replaying on a clone.
    let g = new Chess(game.fen());
    let sans = [];
    for (let u of a.pv){
      let mv = g.move({from:u.slice(0,2), to:u.slice(2,4),
                       promotion: u.length>4 ? u[4] : 'q'});
      if (!mv) break;
      sans.push(mv.san);
    }
    pv.innerHTML = sans.length ? sans.map(s => '<span>'+s+'</span>').join('') :
                                 '<span style="color:var(--ink-dim);background:transparent;border:none">—</span>';
  }
}
function renderHistory(h){
  let el = document.getElementById('history');
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
