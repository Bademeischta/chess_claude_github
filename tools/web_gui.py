#!/usr/bin/env python3
"""
tools/web_gui.py — modern browser UI to play the trained AI.

Thin CLI launcher; the implementation lives in the tools/webgui/ package
(plan B1 step 2 — the former 3.3k-line monolith):

  webgui/session.py — GameSession: per-game state, search, ponder, SSE
  webgui/manager.py — cookie-keyed multi-session support + GPU lock (B2)
  webgui/monitor.py — live training monitor (TensorBoard + pool state)
  webgui/server.py  — HTTP routing table, SSE streaming, static assets
  webgui/static/    — index.html, style.css, app.js, app2.js

Features: live AI-thinking via Server-Sent Events (eval bar, top moves,
PV, search-tree visualizer), pondering, AI commentary on your moves,
click-to-move with policy heatmap, takeback/resign/hint, live training
monitor, several independent browser sessions (each its own game).

    python tools/web_gui.py
    python tools/web_gui.py --resume checkpoints/step_0050000.pt --port 8000
    # then open http://127.0.0.1:8000

SECURITY: binds 127.0.0.1, no authentication (user decision 2026-07-15:
local tool). Do NOT expose it via ngrok / port-forwarding as-is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if sys.platform == "win32":  # cp1252 console guard (project convention)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


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
    ap.add_argument("--max-sessions", type=int, default=4,
                    help="Concurrent browser sessions (default 4; each holds "
                         "its own game/tree, all share one GPU engine).")
    args = ap.parse_args()

    from tools.webgui.server import serve
    return serve(port=args.port, resume=args.resume, sims=args.sims,
                 side_white=(args.side == "white"),
                 max_sessions=args.max_sessions)


if __name__ == "__main__":
    raise SystemExit(main())
