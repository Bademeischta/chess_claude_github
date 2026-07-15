"""Smoke tests for the web GUI's HTTP API (plan B4).

End-to-end over real HTTP against a tiny CPU model: state, move + AI
reply, static assets, per-sid session separation. Fast enough for CI
(CPU-only, ~8-sim searches on a 1-block net).
"""
import dataclasses
import json
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG  # noqa: E402


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from model.network import build_model
    from model.inference import TorchInferenceEngine
    from tools.webgui import server as srv_mod
    from tools.webgui.manager import SessionManager
    from tools.webgui.session import GameSession

    cfg = dataclasses.replace(
        CONFIG, device="cpu", precision="fp32", inference_backend="torch",
        num_res_blocks=1, num_channels=16, mcts_sims=8,
    )
    # Keep test artifacts out of the repo: settings + PGN paths.
    tmp = tmp_path_factory.mktemp("webgui")
    GameSession._SETTINGS_PATH = tmp / "settings.json"
    GameSession._SETTINGS_PATH.write_text(
        json.dumps({"ponder_on": False, "think_sims": 24}), encoding="utf-8")

    model = build_model(cfg, compile_model=False).eval()
    engine = TorchInferenceEngine(model)
    srv_mod.MANAGER = SessionManager(model, torch.device("cpu"), engine, cfg,
                                     max_sessions=3)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv_mod.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    srv_mod.MANAGER.stop_all()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        body = r.read()
        return r.status, r.headers.get_content_type(), body


def _post(base, path, obj):
    req = urllib.request.Request(
        base + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read())


def test_state_returns_startpos(server):
    code, ctype, body = _get(server, "/api/state?sid=smoke1")
    assert code == 200 and ctype == "application/json"
    st = json.loads(body)
    assert st["fen"].startswith("rnbqkbnr/pppppppp/")
    assert st["history"] == []
    assert "settings" in st and st["settings"]["ponder_on"] is False


def test_move_gets_ai_reply(server):
    code, res = _post(server, "/api/move?sid=smoke1", {"uci": "e2e4"})
    assert code == 200 and res.get("ok")
    deadline = time.time() + 60
    moves = []
    while time.time() < deadline:
        _, _, body = _get(server, "/api/state?sid=smoke1")
        moves = [h["uci"] for h in json.loads(body)["history"]]
        if len(moves) >= 2:
            break
        time.sleep(0.5)
    assert moves[:1] == ["e2e4"] and len(moves) >= 2, moves


def test_sessions_are_independent(server):
    _, _, body = _get(server, "/api/state?sid=smoke2")
    st = json.loads(body)
    assert st["history"] == []  # smoke1's game must not leak in


def test_new_game_resets(server):
    code, res = _post(server, "/api/new?sid=smoke1", {"human_white": True})
    assert code == 200 and res.get("ok")
    _, _, body = _get(server, "/api/state?sid=smoke1")
    assert json.loads(body)["history"] == []


def test_index_and_static_assets(server):
    code, ctype, body = _get(server, "/")
    assert code == 200 and ctype == "text/html" and b"<!doctype html>" in body
    code, ctype, _ = _get(server, "/static/style.css")
    assert code == 200 and ctype == "text/css"
    code, ctype, _ = _get(server, "/static/app.js")
    assert code == 200


def test_traversal_guard(server):
    try:
        code, _, _ = _get(server, "/static/../../config.py")
    except urllib.error.HTTPError as e:
        code = e.code
    assert code == 404


def test_unknown_endpoint_404(server):
    try:
        code, _ = _post(server, "/api/nope?sid=smoke1", {})
    except urllib.error.HTTPError as e:
        code = e.code
    assert code == 404
