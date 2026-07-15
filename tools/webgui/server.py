"""
tools/webgui/server.py — HTTP/SSE server for the web GUI.

Routing-table dispatch (plan B1 step 2) instead of the old if-cascade,
cookie-keyed multi-session support via webgui.manager (plan B2), static
assets from webgui/static/. tools/web_gui.py is the thin CLI launcher.

Threading model (documented per plan B2): ThreadingHTTPServer gives one
thread per request; long-lived SSE streams each hold one. Per-session
mutable state is guarded by that session's RLock (as before); the ONE
shared GPU engine is serialised by manager.LockedEngine at the infer()
boundary, so any number of sessions' search/ponder threads interleave
safely at one-forward granularity. The SessionManager's own dict is
guarded by its private lock and never held across GPU work.

Security note: binds 127.0.0.1 and has NO authentication by design —
user decision 2026-07-15: the GUI is a local tool. Do not expose it via
ngrok/port-forwarding without adding auth first.
"""

from __future__ import annotations

import json
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote_plus

from engine.board import STARTPOS_FEN

from tools.webgui.manager import COOKIE_NAME, SessionManager
from tools.webgui.monitor import _training_snapshot
from tools.webgui.session import GameSession, _policy_for_fen, _threat_map

MANAGER: Optional[SessionManager] = None  # set by serve()


# ── Static assets ──────────────────────────────────────────────────────────

_STATIC_DIR = (Path(__file__).resolve().parent / "static").resolve()
_STATIC_MIME = {".html": "text/html; charset=utf-8",
                ".css":  "text/css; charset=utf-8",
                ".js":   "application/javascript; charset=utf-8",
                ".svg":  "image/svg+xml"}


def _static_body(name: str):
    """(bytes, mime) for a static asset, or (None, None). Resolves and
    prefix-checks the path so ../ traversal cannot escape the asset dir."""
    try:
        f = (_STATIC_DIR / name).resolve()
    except (OSError, ValueError):
        return None, None
    if not f.is_file() or _STATIC_DIR not in f.parents:
        return None, None
    return f.read_bytes(), _STATIC_MIME.get(f.suffix, "application/octet-stream")


# ── Piece SVGs ─────────────────────────────────────────────────────────────
# White pieces: outlined/hollow glyphs U+2654–2659 (♔♕♖♗♘♙)
# Black pieces: solid/filled glyphs  U+265A–265F (♚♛♜♝♞♟)
_GLYPH_W = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙"}
_GLYPH_B = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟"}
_PIECE_CACHE: dict = {}


def _piece_svg(code: str):
    if code in _PIECE_CACHE:
        return _PIECE_CACHE[code]
    if len(code) != 2 or code[0] not in "wb" or code[1] not in _GLYPH_W:
        return None
    if code[0] == "w":
        glyph = _GLYPH_W[code[1]]
        fill, stroke = "#fafafa", "#202020"
    else:
        glyph = _GLYPH_B[code[1]]
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


# ── Route handlers ────────────────────────────────────────────────────────
# Signature: fn(session, data) -> JSON-serialisable dict. `data` is the
# parsed POST body ({} for GET routes). Session resolution/creation and
# cookie plumbing live in the Handler, not here.

def _r_state(session: GameSession, data: dict) -> dict:
    with session._lock:
        board = session.board
        fen = board.to_fen() if board else STARTPOS_FEN
        return {
            "fen": fen,
            "history": session.history,
            "human_white": session.human_white,
            "game_over": session.game_over,
            "result": session.result_text,
            "settings": session.settings_dict(),
        }


def _r_threats(session: GameSession, data: dict) -> dict:
    with session._lock:
        board = session.board
    return _threat_map(board) if board else {}


def _r_new(session: GameSession, data: dict) -> dict:
    session.new_game(human_white=bool(data.get("human_white", True)))
    return {"ok": True}


def _r_move(session: GameSession, data: dict) -> dict:
    return session.user_move(str(data.get("uci", "")))


def _r_settings(session: GameSession, data: dict) -> dict:
    session.apply_settings(data or {})
    return {"ok": True, "settings": session.settings_dict()}


def _r_setfen(session: GameSession, data: dict) -> dict:
    return session.set_fen(str(data.get("fen", "")), data.get("human_white"))


GET_ROUTES = {
    "/api/state":    _r_state,
    "/api/threats":  _r_threats,
}

POST_ROUTES = {
    "/api/new":      _r_new,
    "/api/move":     _r_move,
    "/api/settings": _r_settings,
    "/api/takeback": lambda s, d: s.takeback(),
    "/api/resign":   lambda s, d: s.resign(),
    "/api/hint":     lambda s, d: s.hint(),
    "/api/setfen":   _r_setfen,
}


# ── HTTP handler ──────────────────────────────────────────────────────────

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

    # ── Session/cookie plumbing (plan B2) ────────────────────────────────

    def _cookie_sid(self) -> Optional[str]:
        # Per-TAB id first: the frontend keeps one in sessionStorage and
        # appends ?sid=... to every /api call — two tabs of the SAME
        # browser then play two independent games (cookies are shared
        # across tabs, so they only separate browsers/profiles).
        if "?" in self.path:
            for part in self.path.split("?", 1)[1].split("&"):
                if part.startswith("sid=") and len(part) > 4:
                    return unquote_plus(part[4:])[:64]
        raw = self.headers.get("Cookie", "") or ""
        for part in raw.split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE_NAME and v:
                return v
        return None

    def _session(self, create: bool = True) -> Optional[GameSession]:
        """Resolve this request's session. When a new one is created, the
        response will carry the Set-Cookie header (self._set_cookie)."""
        self._set_cookie: Optional[str] = None
        sid = self._cookie_sid()
        if create:
            sid, sess, created = MANAGER.get_or_create(sid)
            if created:
                self._set_cookie = (f"{COOKIE_NAME}={sid}; Path=/; "
                                    f"HttpOnly; SameSite=Lax")
            return sess
        return MANAGER.peek(sid)

    # ── Response helpers ─────────────────────────────────────────────────

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        if getattr(self, "_set_cookie", None):
            self.send_header("Set-Cookie", self._set_cookie)
        self.end_headers()
        try:
            self.wfile.write(b)
        except (ConnectionError, BrokenPipeError):
            pass

    def _send_json(self, code, obj):
        self._send(code, json.dumps(obj))

    # ── Routes ───────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._session(create=True)  # sets the cookie on first load
            body, mime = _static_body("index.html")
            if body is None:
                self._send(500, b"index.html missing under tools/webgui/static/",
                           "text/plain")
            else:
                self._send(200, body, mime)
            return
        if path.startswith("/static/"):
            body, mime = _static_body(path[len("/static/"):])
            if body is None:
                self._send(404, b"no such asset", "text/plain")
            else:
                self._send(200, body, mime)
            return
        if path.startswith("/piece/"):
            code = path[len("/piece/"):].split(".")[0]
            svg = _piece_svg(code)
            if svg is None:
                self._send(404, b"no piece", "text/plain")
            else:
                self._send(200, svg, "image/svg+xml")
            return
        if path == "/api/training":
            # Session-independent (reads training artifacts from disk).
            self._send_json(200, _training_snapshot())
            return
        if path == "/api/stream":
            self._stream_sse()
            return
        if path.startswith("/api/policy"):
            fen = ""
            if "?" in self.path:
                for part in self.path.split("?", 1)[1].split("&"):
                    if part.startswith("fen="):
                        fen = unquote_plus(part[4:])
            self._send_json(200, _policy_for_fen(self._session(), fen))
            return
        fn = GET_ROUTES.get(path)
        if fn is not None:
            self._send_json(200, fn(self._session(), {}))
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            data = {}
        fn = POST_ROUTES.get(self.path.split("?", 1)[0])
        if fn is None:
            self._send_json(404, {"error": "unknown endpoint"})
            return
        self._send_json(200, fn(self._session(), data))

    # ── SSE ──────────────────────────────────────────────────────────────

    def _stream_sse(self):
        session = self._session(create=True)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            if getattr(self, "_set_cookie", None):
                self.send_header("Set-Cookie", self._set_cookie)
            self.end_headers()
        except (ConnectionError, BrokenPipeError):
            return

        q = session.sse_subscribe()
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
            session.sse_unsubscribe(q)


# ── Entry point (called by tools/web_gui.py) ──────────────────────────────

def serve(port: int, resume: Optional[str], sims: Optional[int],
          side_white: bool, max_sessions: int = 4) -> int:
    from engine.harness import load_engine

    handle = load_engine(resume,
                         log=lambda m: print(m.replace("[harness]", "[web]")))

    global MANAGER
    MANAGER = SessionManager(
        handle.model, handle.device, handle.engine, handle.cfg,
        max_sessions=max_sessions,
        default_sims=sims,
        default_side_white=side_white,
    )

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[web] http://127.0.0.1:{port}  (Ctrl+C to stop)  "
          f"[multi-session: up to {max_sessions} browser sessions]")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped.")
        MANAGER.stop_all()
    return 0
