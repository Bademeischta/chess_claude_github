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


# ONE lock for the whole GPU, shared by every engine AND the model
# loader. Verified failure mode without it (2026-07-15): building a
# second model (cudaMalloc + weight H2D) while another session's ponder
# thread replayed the CUDA graph produced `cudaErrorIllegalAddress` and
# poisoned the context for the whole process.
_GPU_LOCK = threading.Lock()


class LockedEngine:
    """Serialises inference across sessions sharing the GPU.

    clone() inside the lock: the CUDA-graph engine returns narrow views
    of its static output buffers; without materialisation a second
    session's replay would overwrite them between this session's infer()
    return and its D2H copy."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def infer(self, boards, hist):
        with _GPU_LOCK:
            policy, value = self._inner.infer(boards, hist)
            return policy.clone(), value.clone()

    def __getattr__(self, name):
        # batch_size and any other engine attributes pass through.
        return getattr(self._inner, name)


class ModelCache:
    """Checkpoint path → (model, LockedEngine) for the UI's engine
    selector (plan B3). The PRIMARY model keeps its production inference
    engine (usually CUDA-graph); additionally loaded checkpoints get the
    eager Torch engine — a second CUDA-graph capture would cost extra
    static VRAM per model for a casual-play feature. At most
    `max_models` models stay resident (LRU eviction; ~93 MB fp32 each).

    NOTE: loading a checkpoint runs match_arch against the shared CONFIG
    (old checkpoints may carry a different head/tower shape) — fine for
    this play-only server process, which builds models exclusively here.
    """

    def __init__(self, primary_key: str, primary_model, primary_engine,
                 device, cfg, max_models: int = 3) -> None:
        self.device, self.cfg = device, cfg
        self.max_models = int(max_models)
        self._lock = threading.Lock()
        self._items: dict[str, tuple] = {primary_key: (primary_model,
                                                       primary_engine)}
        self._order: list[str] = [primary_key]  # LRU, front = oldest
        self._primary_key = primary_key

    def get(self, path: str):
        key = str(path)
        with self._lock:
            if key in self._items:
                self._order.remove(key)
                self._order.append(key)
                return self._items[key]
        # Load under the GLOBAL GPU lock: allocating + uploading a new
        # model while another session's CUDA-graph replay is in flight
        # crashed the CUDA context (illegal memory access — see the lock's
        # comment above). Disk read + cudaMalloc serialised is a few
        # seconds once per checkpoint; the cache makes repeats free.
        from utils.ckpt_arch import match_arch
        from model.network import build_model
        from model.inference import TorchInferenceEngine
        from engine.harness import load_state_into
        with _GPU_LOCK:
            match_arch(self.cfg, key)
            model = build_model(self.cfg, compile_model=False).eval()
            load_state_into(model, key, self.device)
        pair = (model, LockedEngine(TorchInferenceEngine(model)))
        with self._lock:
            self._items[key] = pair
            self._order.append(key)
            while len(self._order) > self.max_models:
                victim = next((k for k in self._order
                               if k != self._primary_key), None)
                if victim is None:
                    break
                self._order.remove(victim)
                self._items.pop(victim, None)
        return pair


class SessionManager:
    def __init__(self, model, device, engine, cfg, *,
                 max_sessions: int = 4,
                 idle_timeout_s: float = 1800.0,
                 default_sims: Optional[int] = None,
                 default_side_white: bool = True,
                 primary_ckpt: Optional[str] = None) -> None:
        self.model, self.device, self.cfg = model, device, cfg
        self.engine = LockedEngine(engine) if engine is not None else None
        self.max_sessions = int(max_sessions)
        self.idle_timeout_s = float(idle_timeout_s)
        self.default_sims = default_sims
        self.default_side_white = bool(default_side_white)
        self.primary_ckpt = primary_ckpt
        self.models = ModelCache(primary_ckpt or "__untrained__",
                                 model, self.engine, device, cfg)
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
                sess.ckpt_path = self.primary_ckpt  # engine-selector label
                sess.load_persisted_settings()      # user prefs (plan B4)
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

    # ── Engine selection (plan B3) ──────────────────────────────────────

    def checkpoint_inventory(self) -> list[dict]:
        """Playable checkpoints with strength labels (engine.harness)."""
        from engine.harness import list_checkpoints
        return list_checkpoints(self.cfg)

    def switch_model(self, sess: GameSession, path: str) -> dict:
        """Point `sess` at a different checkpoint and start a fresh game.

        `path` must come from checkpoint_inventory() — the client cannot
        load arbitrary filesystem paths."""
        allowed = {e["path"] for e in self.checkpoint_inventory()}
        if path not in allowed:
            return {"error": "unknown checkpoint"}
        # Quiet THIS session's GPU work before the load; other sessions'
        # searches are held off by the global GPU lock inside models.get.
        self._safe_stop(sess)
        try:
            model, engine = self.models.get(path)
        except Exception as e:  # load failure must not kill the server
            return {"error": f"load failed: {type(e).__name__}: {e}"}
        with sess._lock:
            sess.model = model
            sess.engine = engine
            sess.ckpt_path = path
            human_white = sess.human_white
        sess.new_game(human_white=human_white)
        return {"ok": True, "current": path}
