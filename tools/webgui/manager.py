"""
tools/webgui/manager.py — multi-session support for the web GUI (plan B2).

One server process holds ONE model + ONE inference engine (VRAM); several
browser sessions each get their own GameSession (board, tree, ponder,
SSE subscribers), keyed by a cookie. Two safety properties:

* GPU serialisation: the CUDA-graph engine replays into static buffers —
  concurrent infer() calls from two sessions' search/ponder threads would
  corrupt each other's outputs. LockedEngine serialises infer() AND
  materialises the results (clone) inside the lock, because the raw
  engine returns *views* of the static buffers that the next replay
  overwrites.
* Bounded resources: at most `max_sessions` live sessions; the least-
  recently-used one is evicted (ponder stopped) when a new one needs a
  slot, and sessions idle beyond `idle_timeout_s` are reaped
  opportunistically on every access.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Optional, Tuple

from tools.webgui.session import GameSession

COOKIE_NAME = "chessai_sid"


class LockedEngine:
    """Serialises inference across sessions sharing one engine.

    clone() inside the lock: the CUDA-graph engine returns narrow views
    of its static output buffers; without materialisation a second
    session's replay would overwrite them between this session's infer()
    return and its D2H copy."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._gpu_lock = threading.Lock()

    def infer(self, boards, hist):
        with self._gpu_lock:
            policy, value = self._inner.infer(boards, hist)
            return policy.clone(), value.clone()

    def __getattr__(self, name):
        # batch_size and any other engine attributes pass through.
        return getattr(self._inner, name)


class SessionManager:
    def __init__(self, model, device, engine, cfg, *,
                 max_sessions: int = 4,
                 idle_timeout_s: float = 1800.0,
                 default_sims: Optional[int] = None,
                 default_side_white: bool = True) -> None:
        self.model, self.device, self.cfg = model, device, cfg
        self.engine = LockedEngine(engine) if engine is not None else None
        self.max_sessions = int(max_sessions)
        self.idle_timeout_s = float(idle_timeout_s)
        self.default_sims = default_sims
        self.default_side_white = bool(default_side_white)
        self._lock = threading.Lock()
        self._sessions: dict[str, GameSession] = {}
        self._last_seen: dict[str, float] = {}

    # ── Resolution ─────────────────────────────────────────────────────

    def get_or_create(self, sid: Optional[str]) -> Tuple[str, GameSession, bool]:
        """Return (sid, session, created). Reaps idle sessions; evicts the
        LRU session when the cap is reached. Thread-safe."""
        now = time.time()
        evicted: list[GameSession] = []
        with self._lock:
            self._reap_idle_locked(now, evicted)
            if sid and sid in self._sessions:
                self._last_seen[sid] = now
                sess = self._sessions[sid]
                created = False
            else:
                while len(self._sessions) >= self.max_sessions:
                    lru = min(self._last_seen, key=self._last_seen.get)
                    evicted.append(self._sessions.pop(lru))
                    self._last_seen.pop(lru, None)
                # Adopt a client-supplied id (the frontend's per-TAB
                # sessionStorage id) so the tab keeps hitting the session
                # it created; only mint one when none was sent.
                sid = sid or uuid.uuid4().hex
                sess = GameSession(self.model, self.device,
                                   self.engine, self.cfg)
                if self.default_sims is not None:
                    sess.think_sims = int(self.default_sims)
                self._sessions[sid] = sess
                self._last_seen[sid] = now
                created = True
        # Outside the lock: ponder threads join on stop; game setup runs a
        # root expansion (GPU forward) — neither belongs under the manager
        # lock.
        for old in evicted:
            self._safe_stop(old)
        if created:
            sess.new_game(human_white=self.default_side_white)
        return sid, sess, created

    def peek(self, sid: Optional[str]) -> Optional[GameSession]:
        """Existing session or None — never creates (for read-only routes)."""
        with self._lock:
            if sid and sid in self._sessions:
                self._last_seen[sid] = time.time()
                return self._sessions[sid]
        return None

    def _reap_idle_locked(self, now: float, evicted: list) -> None:
        for k in [k for k, t in self._last_seen.items()
                  if now - t > self.idle_timeout_s]:
            evicted.append(self._sessions.pop(k))
            self._last_seen.pop(k, None)

    @staticmethod
    def _safe_stop(sess: GameSession) -> None:
        try:
            sess._stop_ponder()
        except Exception:
            pass

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._last_seen.clear()
        for s in sessions:
            self._safe_stop(s)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)
