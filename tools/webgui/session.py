"""
tools/webgui/session.py — per-game state, search, ponder, SSE pub/sub.

Moved verbatim out of the tools/web_gui.py monolith (audit 2026-07-15,
plan B1 step 2). One GameSession = one game; the SessionManager in
webgui.manager keys several of them by browser cookie (plan B2).
"""

from __future__ import annotations

import json  # noqa: F401  (kept: helpers serialise wire payloads)
import queue
import threading
import time
from typing import Optional

from config import CONFIG  # noqa: F401
from engine.board import Board, STARTPOS_FEN
from engine.rules import get_game_result, GameResult
from engine.movegen import _move_uci
from mcts.tree import MCTSTree

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

    def tree_snapshot(self, max_depth: int = 3, top_k: int = 3) -> dict:
        """A nested view of the search tree (root → top-k children → …) for the
        live tree visualiser. MUST be called from the SEARCH thread between sim
        batches (ponder is stopped during the AI's own search), where the GIL
        is held and no other thread mutates the tree — reading the C++ node
        tree from a different thread mid-search could segfault. SAN labels via
        python-chess by pushing/popping a scratch board down each branch."""
        tree, board = self.tree, self.board
        if tree is None or board is None:
            return {}
        root = getattr(tree, "_root_node", None)
        if root is None:
            return {}
        cb = None
        try:
            import chess as _c
            cb = _c.Board(board.to_fen())
        except Exception:
            cb = None

        def walk(node, depth):
            try:
                kids = list(tree._children(node))
            except Exception:
                return []
            kids.sort(key=lambda mc: getattr(mc[1], "N", 0), reverse=True)
            out = []
            for m, c in kids[:top_k]:
                n_c = int(getattr(c, "N", 0))
                if n_c <= 0:
                    continue
                uci = _move_uci(m)
                san, pushed = uci, False
                if cb is not None:
                    try:
                        mv = cb.parse_uci(uci)
                        san = cb.san(mv)
                        cb.push(mv)
                        pushed = True
                    except Exception:
                        san = uci
                node_d = {
                    "san": san, "uci": uci, "n": n_c,
                    "q": -float(getattr(c, "Q", 0.0)),
                    "p": float(getattr(c, "P", 0.0)),
                    "kids": walk(c, depth + 1) if depth + 1 < max_depth else [],
                }
                if pushed:
                    cb.pop()
                out.append(node_d)
            return out

        try:
            return {
                "n": int(getattr(root, "N", 0)),
                "q": float(_safe(lambda: self.tree.root_value(), 0.0) or 0.0),
                "kids": walk(root, 0),
            }
        except Exception:
            return {}

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
        # Live search-tree snapshot (throttled). Built HERE, in the search
        # thread between batches, so reading the C++ node tree is race-free.
        now = time.time()
        if now - getattr(self, "_last_tree_emit", 0.0) > 0.35:
            self._last_tree_emit = now
            snap = self.tree_snapshot()
            if snap:
                self.emit("tree", snap)

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


def _policy_for_fen(session: "GameSession", fen: str) -> dict:
    """Return neural-network policy priors for all legal moves in the
    current position (no MCTS — just the root forward pass).

    The root is already expanded while it is the human's turn (ponder runs
    continuously), so this is usually a zero-cost read of cached priors.
    Falls back to a single expansion pass if the tree hasn't been touched."""
    with session._lock:
        if session.board is None or session.tree is None:
            return {"error": "no game"}
    # Idempotent: only triggers a forward pass when root not yet expanded.
    session._ensure_root_ready()
    try:
        a = session.tree.root_analysis(top_k=500, pv_len=1)
    except Exception:
        a = {}
    ext = a.get("ext", [])
    if not ext:
        top = a.get("top", [])
        ext = [{"move": m, "p": 1.0 / max(1, len(top))} for m, _n, _q in top]
    total = sum(float(e.get("p", 0)) for e in ext)
    policy: dict = {}
    for e in ext:
        uci = _move_uci(e["move"])
        p = float(e.get("p", 0))
        policy[uci] = p / total if total > 0 else 0.0
    return {"policy": policy, "value": float(a.get("value", 0.0))}

