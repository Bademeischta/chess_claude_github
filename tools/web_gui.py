#!/usr/bin/env python3
"""
tools/web_gui.py — play the trained AI in the browser (drag & drop).

Zero extra dependencies: built on Python's stdlib http.server. The page
loads chessboard.js / chess.js / jQuery from CDNs (needs internet in the
browser; the server itself is fully local and authoritative).

    python tools/web_gui.py --resume checkpoints/step_0050000.pt --sims 200
    # then open http://127.0.0.1:8000

Backend reuses the exact engine pieces used by `main.py --play`
(Board, MCTSTree, _move_uci, build_inference_engine).
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

STATE: dict = {}  # single local game: board, tree, engine, sims, human_white

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Chess AI</title>
<link rel="stylesheet"
 href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css">
<style>body{font-family:sans-serif;text-align:center}#board{width:480px;margin:20px auto}
#msg{font-size:18px;min-height:24px}</style></head><body>
<h2>Chess AI</h2><div id="board"></div><div id="msg">Loading…</div>
<button onclick="newGame()">New game</button>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
<script>
let game=new Chess(), board=null, busy=false;
function setMsg(t){document.getElementById('msg').innerText=t;}
async function post(u,d){let r=await fetch(u,{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify(d||{})});
 return await r.json();}
function onDrop(src,tgt){
 if(busy) return 'snapback';
 let mv=game.move({from:src,to:tgt,promotion:'q'});
 if(mv===null) return 'snapback';
 busy=true; setMsg('AI thinking…');
 post('/api/move',{uci:mv.from+mv.to+(mv.promotion||'')}).then(res=>{
   if(res.error){game.undo();board.position(game.fen());setMsg('Illegal: '+res.error);busy=false;return;}
   game.load(res.fen); board.position(res.fen);
   setMsg(res.game_over?('Game over: '+res.result):'Your move');
   busy=false;});
}
function newGame(){post('/api/new',{}).then(res=>{game.load(res.fen);
 board.position(res.fen);setMsg(res.msg);});}
board=Chessboard('board',{draggable:true,position:'start',onDrop:onDrop,
 pieceTheme:'/piece/{piece}.svg'});
newGame();
</script></body></html>"""


# Self-hosted piece sprites (no CDN — the chessboard-js cdnjs mirror does NOT
# ship the image folder, hence the broken icons). Filled Unicode chess glyphs
# rendered as SVG: white = white fill + dark outline, black = dark fill +
# light outline, so both read on light AND dark squares.
_GLYPH = {"K": "♚", "Q": "♛", "R": "♜",
          "B": "♝", "N": "♞", "P": "♟"}
_PIECE_CACHE: dict = {}


def _piece_svg(code: str):
    """`code` like 'wK' / 'bP' → SVG bytes, or None if invalid."""
    if code in _PIECE_CACHE:
        return _PIECE_CACHE[code]
    if len(code) != 2 or code[0] not in "wb" or code[1] not in _GLYPH:
        return None
    glyph = _GLYPH[code[1]]
    if code[0] == "w":
        fill, stroke = "#f8f8f8", "#202020"
    else:
        fill, stroke = "#202020", "#d8d8d8"
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 45 45'>"
        f"<text x='22.5' y='37' font-size='40' text-anchor='middle' "
        f"fill='{fill}' stroke='{stroke}' stroke-width='0.8' "
        f"font-family='\"Segoe UI Symbol\",\"DejaVu Sans\",sans-serif'>"
        f"{glyph}</text></svg>"
    ).encode("utf-8")
    _PIECE_CACHE[code] = svg
    return svg


def _ai_reply():
    """Advance: if it's the AI's turn, search+play; return (fen, over, result)."""
    board = STATE["board"]
    tree = STATE["tree"]
    if (get_game_result(board) == GameResult.ONGOING
            and (board.side_to_move == 0) != STATE["human_white"]):
        mv, _ = tree.search(board, STATE["sims"],
                            temperature=CONFIG.temperature_final,
                            use_dca=False, keep_history=True)
        board = board.apply_move(mv)
        tree.advance(mv, board)
        STATE["board"] = board
    res = get_game_result(board)
    over = res != GameResult.ONGOING
    txt = {GameResult.WHITE_WIN: "White wins",
           GameResult.BLACK_WIN: "Black wins",
           GameResult.DRAW: "Draw"}.get(res, "")
    return board.to_fen(), over, txt


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path.startswith("/piece/"):
            code = self.path[len("/piece/"):].split(".")[0]  # e.g. "wK"
            svg = _piece_svg(code)
            if svg is None:
                self._send(404, b"no piece", "text/plain")
            else:
                self._send(200, svg, "image/svg+xml")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            data = {}

        if self.path == "/api/new":
            board = Board.from_fen(STARTPOS_FEN)
            tree = MCTSTree(CONFIG, STATE["model"], STATE["device"],
                            engine=STATE["engine"])
            tree.reset(board)
            STATE["board"], STATE["tree"] = board, tree
            fen, over, txt = _ai_reply()  # AI moves first if human is black
            self._send(200, json.dumps(
                {"fen": fen, "msg": "Your move" if not over else txt}))
            return

        if self.path == "/api/move":
            board = STATE["board"]
            uci = str(data.get("uci", ""))
            legal = {(_move_uci(m)): m for m in board.legal_moves()}
            if uci not in legal:
                self._send(200, json.dumps({"error": uci or "no move"}))
                return
            board = board.apply_move(legal[uci])
            STATE["tree"].advance(legal[uci], board)
            STATE["board"] = board
            fen, over, txt = _ai_reply()
            self._send(200, json.dumps(
                {"fen": fen, "game_over": over, "result": txt}))
            return

        self._send(404, json.dumps({"error": "unknown endpoint"}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--side", choices=["white", "black"], default="white")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    update_config_from_dict(run_probe(write=False))
    device = torch.device(CONFIG.device)

    from utils.ckpt_arch import latest_checkpoint, match_arch
    ckpt = args.resume or latest_checkpoint(CONFIG.checkpoint_dir)
    match_arch(CONFIG, ckpt)  # align block count before building
    model = build_model(CONFIG, compile_model=False).eval()

    if ckpt and Path(ckpt).exists():
        st = torch.load(ckpt, map_location=device, weights_only=True)
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        base.load_state_dict(st["model"])
        print(f"[web] Loaded {ckpt}")
    else:
        print("[web] No checkpoint — UNTRAINED net.")

    STATE.update(
        model=model, device=device,
        engine=build_inference_engine(CONFIG, model, device),
        sims=args.sims if args.sims is not None else CONFIG.mcts_sims,
        human_white=(args.side == "white"),
    )
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[web] http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
