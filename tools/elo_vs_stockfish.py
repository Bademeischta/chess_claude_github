#!/usr/bin/env python3
"""
tools/elo_vs_stockfish.py — objective ELO measurement against Stockfish.

Plays a series against Stockfish pinned to a known strength
(UCI_LimitStrength + UCI_Elo, which is a real ELO anchor), alternating
colours, and reports our estimated ELO with a 95 % confidence interval
(point estimate = Stockfish_Elo + elo_diff_from_series).

    python tools/elo_vs_stockfish.py --stockfish C:\\path\\stockfish.exe \\
        --games 30 --elo 1500 --resume checkpoints/step_0050000.pt

Read-only w.r.t. training (loads a checkpoint, plays games, prints a number).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import chess
import chess.engine

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe
from utils.elo import ELOSystem
from model.network import build_model
from model.inference import build_inference_engine
from engine.board import Board
from engine.movegen import _move_uci
from mcts.tree import MCTSTree


from utils.ckpt_arch import latest_checkpoint, match_arch


def _our_move(tree: MCTSTree, board: Board, sims: int) -> int:
    move, _ = tree.search(board, sims, temperature=CONFIG.temperature_final,
                          use_dca=False, keep_history=True)
    return move


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", default=CONFIG.stockfish_path,
                    help="Path to Stockfish UCI binary.")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--elo", type=int, default=1500,
                    help="Stockfish UCI_Elo anchor (≈1320–3190).")
    ap.add_argument("--sims", type=int, default=None,
                    help="Our MCTS sims/move (default: config).")
    ap.add_argument("--movetime", type=float, default=None,
                    help="Stockfish thinking time per move (s). Default: "
                         "auto-derived from our MCTS sims so SF gets a fair "
                         "comparable budget (~mcts_sims * 0.025 s).")
    ap.add_argument("--depth", type=int, default=None,
                    help="Use a fixed search depth for Stockfish instead of "
                         "--movetime (more reproducible, but ignores our "
                         "actual thinking time).")
    ap.add_argument("--sf-hash-mb", type=int, default=256,
                    help="Stockfish hash-table size (MB).")
    ap.add_argument("--sf-threads", type=int, default=4,
                    help="Stockfish worker threads.")
    ap.add_argument("--sf-skill", type=int, default=None,
                    help="Stockfish Skill Level 0-20 (rough ELO: 0≈800, "
                         "5≈1500, 10≈2000, 20=full). Use this instead of "
                         "--elo when you need to test below SF's UCI_Elo "
                         "floor of 1320 (e.g. very early-training nets). "
                         "Disables UCI_LimitStrength.")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint (default: newest in checkpoint dir).")
    ap.add_argument("--auto", action="store_true",
                    help="Adaptive ELO search: short mini-matches at varying "
                         "Stockfish strengths, narrowed via bisection until "
                         "we score ~50 %%. Outputs a single ELO estimate.")
    ap.add_argument("--auto-budget", type=int, default=60,
                    help="Total game budget for --auto (default: 60, split "
                         "across ~3-5 rounds).")
    ap.add_argument("--auto-low", type=int, default=1320,
                    help="Lower bound for the auto-search anchor.")
    ap.add_argument("--auto-high", type=int, default=3190,
                    help="Upper bound for the auto-search anchor.")
    args = ap.parse_args()

    if not args.stockfish or not Path(args.stockfish).exists():
        # One last try: pick up a fresh install via the sentinel.
        sentinel = Path(__file__).resolve().parent.parent / "runs" / ".stockfish_installed"
        if sentinel.exists():
            cand = sentinel.read_text(encoding="utf-8").strip()
            if cand and cand != "FAILED" and Path(cand).exists():
                args.stockfish = cand
        if not args.stockfish or not Path(args.stockfish).exists():
            print("[elo] No Stockfish binary found.")
            print("[elo] Run once:  python tools/install_stockfish.py")
            print("[elo] Or pass --stockfish <path-to-binary>.")
            return 1

    # Resolve to an absolute path. python-chess uses asyncio.subprocess on
    # Windows, which fails to locate a relative path passed verbatim (the
    # internal CreateProcess call doesn't apply CWD the way Popen does).
    args.stockfish = str(Path(args.stockfish).resolve())

    update_config_from_dict(run_probe(write=False))
    device = torch.device(CONFIG.device)

    ckpt = args.resume or latest_checkpoint(CONFIG.checkpoint_dir)
    match_arch(CONFIG, ckpt)
    model = build_model(CONFIG, compile_model=False).eval()

    if ckpt and Path(ckpt).exists():
        st = torch.load(ckpt, map_location=device, weights_only=True)
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        # Strict-load with one tolerated absence: `_material_scale` is a
        # buffer added after the first refactor and not present in older
        # checkpoints. Falling back to strict=False would silently swallow
        # real architectural mismatches; explicitly whitelist only this key.
        missing, unexpected = base.load_state_dict(st["model"], strict=False)
        allowed_missing = {"value_head._material_scale"}
        real_missing = [k for k in missing if k not in allowed_missing]
        if real_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint/model mismatch.\n  missing : {real_missing}"
                f"\n  unexpected: {list(unexpected)}"
            )
        print(f"[elo] Loaded {ckpt} (step {st.get('step', '?')})")
    else:
        print("[elo] No checkpoint — measuring an UNTRAINED net.")

    engine_inf = build_inference_engine(CONFIG, model, device)
    sims = args.sims if args.sims is not None else CONFIG.mcts_sims

    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    # Stockfish resource config — applied once. Hash + Threads matter for a
    # fair comparison: a Stockfish on the default 16 MB hash / 1 thread is
    # ~150-300 ELO weaker than the same binary tuned for the machine.
    try:
        sf.configure({"Hash": int(args.sf_hash_mb), "Threads": int(args.sf_threads)})
    except chess.engine.EngineError as e:
        print(f"[elo] Could not set SF Hash/Threads ({e}); using defaults.")

    # Fair-play time budget: if --movetime is not given, match Stockfish's
    # think time to our MCTS budget. Rough calibration on RTX 5070 with the
    # 14M-param net: ~25 ms / sim (mostly NN forward). Avoids the historic
    # 100 ms-vs-multi-second asymmetry that systematically over-scored us.
    if args.movetime is None and args.depth is None:
        args.movetime = max(0.5, sims_estimate := (
            (args.sims if args.sims is not None else CONFIG.mcts_sims) * 0.025
        ))
        print(f"[elo] Auto movetime: {args.movetime:.2f}s "
              f"(matches our MCTS sims ≈ {sims_estimate / 0.025:.0f}).")

    def _sf_limit() -> chess.engine.Limit:
        if args.depth is not None:
            return chess.engine.Limit(depth=int(args.depth))
        return chess.engine.Limit(time=float(args.movetime))

    def _set_sf_elo(elo: int) -> None:
        if args.sf_skill is not None:
            # Skill Level mode — covers the regime below the UCI_Elo floor.
            # Stockfish must NOT be in UCI_LimitStrength mode when Skill Level
            # drives the strength, otherwise it silently ignores the skill
            # setting and plays at the floor's strength instead.
            try:
                sf.configure({"UCI_LimitStrength": False,
                              "Skill Level": int(args.sf_skill)})
                print(f"[elo] Stockfish Skill Level = {int(args.sf_skill)} "
                      f"(approx. {[800,950,1100,1250,1400,1500,1600,1700,1800,1900,2000,2050,2100,2150,2200,2300,2400,2500,2700,2850,3000][max(0,min(20,int(args.sf_skill)))]} ELO).")
            except chess.engine.EngineError as e:
                print(f"[elo] Failed to set Skill Level: {e}")
        else:
            try:
                sf.configure({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
            except chess.engine.EngineError:
                print(f"[elo] Stockfish rejected UCI_Elo={elo} "
                      f"(SF floor is 1320 — use --sf-skill 0-20 for weaker "
                      f"opponents). Using default strength.")

    def _play_batch(n: int, label: str = "") -> tuple[int, int, int]:
        """Play `n` games alternating colours; return (W, D, L)."""
        bw = bd = bl = 0
        for gi in range(n):
            we_white = (gi % 2 == 0)
            pyb = chess.Board()
            ob = Board.from_fen(pyb.fen())
            tree = MCTSTree(CONFIG, model, device, engine=engine_inf)
            tree.reset(ob)
            while not pyb.is_game_over(claim_draw=True):
                our_turn = (pyb.turn == chess.WHITE) == we_white
                if our_turn:
                    mv = _our_move(tree, ob, sims)
                    pyb.push(chess.Move.from_uci(_move_uci(mv)))
                else:
                    res = sf.play(pyb, _sf_limit())
                    pyb.push(res.move)
                ob = Board.from_fen(pyb.fen())
                tree.reset(ob)
            res = pyb.result(claim_draw=True)
            if res == "1/2-1/2":      bd += 1
            elif (res == "1-0") == we_white: bw += 1
            else:                      bl += 1
            print(f"  {label}  game {gi+1}/{n}  "
                  f"{'W' if we_white else 'B'}  result={res}  "
                  f"(W{bw} D{bd} L{bl})", flush=True)
        return bw, bd, bl

    # ── Adaptive bisection search over Stockfish ELO ─────────────────
    if args.auto:
        lo, hi = args.auto_low, args.auto_high
        # 4 rounds × budget/4 games. Each round picks the midpoint, plays
        # a mini-match, and narrows [lo, hi] based on the score.
        rounds = 4
        per_round = max(8, args.auto_budget // rounds)
        history: list[tuple[int, int, int, int]] = []  # (anchor, W, D, L)
        for r in range(rounds):
            anchor = (lo + hi) // 2
            print(f"\n[auto] round {r+1}/{rounds}  "
                  f"window=[{lo},{hi}]  anchor={anchor}  "
                  f"games={per_round}")
            _set_sf_elo(anchor)
            bw, bd, bl = _play_batch(per_round, label=f"[A{anchor}]")
            history.append((anchor, bw, bd, bl))
            total = bw + bd + bl
            score = (bw + 0.5 * bd) / total if total > 0 else 0.5
            print(f"[auto] round {r+1} → score={score:.2f} at anchor {anchor}")
            # Bisection rule: if we're losing (score<0.4), lower anchor;
            # if we're winning (score>0.6), raise anchor; if ~50%, narrow.
            if score < 0.4:
                hi = anchor - 1
            elif score > 0.6:
                lo = anchor + 1
            else:
                # Already balanced — narrow further around this anchor
                spread = max(50, (hi - lo) // 3)
                lo, hi = anchor - spread, anchor + spread
            if hi <= lo + 50:
                break

        sf.quit()

        total_games = sum(h[1] + h[2] + h[3] for h in history)
        total_w = sum(h[1] for h in history)
        total_d = sum(h[2] for h in history)
        total_l = sum(h[3] for h in history)

        def _score(h: tuple[int, int, int, int]) -> float:
            _, w, d, l = h
            return (w + 0.5 * d) / max(1, w + d + l)

        # Find the anchor whose score lands inside a "meaningful" band —
        # there we have signal for an ELO estimate. Outside that band the
        # ELO model degenerates (loss-only gives diff ≈ −∞, win-only gives
        # +∞; the Wald CI gets meaninglessly wide).
        BAND_LO, BAND_HI = 0.15, 0.85
        in_band = [h for h in history if BAND_LO <= _score(h) <= BAND_HI]

        print("\n" + "=" * 58)
        print(f"  AUTO ELO SEARCH  ({total_games} games total, {sims} sims)")
        for a, w, d, l in history:
            print(f"    SF@{a:>4d}: W{w} D{d} L{l}  score={_score((a,w,d,l)):.2f}")
        print(f"  Overall record         : W{total_w} D{total_d} L{total_l}")
        print("-" * 58)

        if in_band:
            # Use the anchor closest to 50 %.
            best = min(in_band, key=lambda h: abs(_score(h) - 0.5))
            anchor, bw, bd, bl = best
            diff, lo_ci, hi_ci = ELOSystem.elo_difference_ci(bw, bd, bl)
            print(f"  Best balanced anchor   : SF@{anchor}")
            print(f"  ELO diff vs SF@{anchor}: {diff:+.0f}  "
                  f"(95 % CI {lo_ci:+.0f}..{hi_ci:+.0f})")
            print(f"  Estimated ELO          : {anchor + diff:.0f}  "
                  f"(CI {anchor + lo_ci:.0f}..{anchor + hi_ci:.0f})")
        else:
            # All anchors lie outside the meaningful band → only a bound.
            best_score_lo = min(history, key=_score)   # closest to losing all
            best_score_hi = max(history, key=_score)   # closest to winning all
            if _score(best_score_hi) < BAND_LO:
                # Lost at every anchor — model is weaker than the lowest one.
                floor = best_score_lo[0]
                print(f"  No anchor in [{int(BAND_LO*100)}–"
                      f"{int(BAND_HI*100)} %] score band.")
                print(f"  Model scored ≤ {_score(best_score_hi)*100:.0f} %% "
                      f"even at the lowest tested SF anchor ({floor}).")
                print(f"  → Estimated ELO is BELOW ~{floor}  "
                      f"(Stockfish's UCI floor is 1320, so we cannot probe "
                      f"lower without external rating systems).")
                print(f"  Hints: increase --sims, lower --movetime, or train "
                      f"the net further before re-measuring.")
            elif _score(best_score_lo) > BAND_HI:
                # Won at every anchor — model is stronger than the highest.
                ceiling = best_score_hi[0]
                print(f"  Won at every tested anchor (top {ceiling}).")
                print(f"  → Estimated ELO is ABOVE ~{ceiling}. Re-run with "
                      f"--auto-low {ceiling + 100} to refine.")
            else:
                # Crossed the band but never landed in it — degenerate.
                print("  Search did not converge — anchors flipped between "
                      "≤15% and ≥85% with no balanced match. Try a denser "
                      "search by raising --auto-budget.")
        print("=" * 58)
        return 0

    # ── Single fixed-anchor match (legacy path) ──────────────────────
    _set_sf_elo(args.elo)

    wins = draws = losses = 0
    for g in range(args.games):
        we_white = (g % 2 == 0)
        pyb = chess.Board()
        ob = Board.from_fen(pyb.fen())
        tree = MCTSTree(CONFIG, model, device, engine=engine_inf)
        tree.reset(ob)

        while not pyb.is_game_over(claim_draw=True):
            our_turn = (pyb.turn == chess.WHITE) == we_white
            if our_turn:
                mv = _our_move(tree, ob, sims)
                uci = _move_uci(mv)
                pyb.push(chess.Move.from_uci(uci))
            else:
                res = sf.play(pyb, chess.engine.Limit(time=args.movetime))
                pyb.push(res.move)
                uci = res.move.uci()
            ob = Board.from_fen(pyb.fen())
            tree.reset(ob)  # re-sync tree to the authoritative position

        res = pyb.result(claim_draw=True)  # "1-0","0-1","1/2-1/2"
        if res == "1/2-1/2":
            draws += 1
        elif (res == "1-0") == we_white:
            wins += 1
        else:
            losses += 1
        print(f"  game {g+1}/{args.games}  "
              f"{'W' if we_white else 'B'}  result={res}  "
              f"(W{wins} D{draws} L{losses})", flush=True)

    sf.quit()

    diff, lo, hi = ELOSystem.elo_difference_ci(wins, draws, losses)
    print("\n" + "=" * 52)
    print(f"  vs Stockfish@{args.elo}  ({args.games} games, {sims} sims)")
    print(f"  W{wins} D{draws} L{losses}")
    print(f"  ELO diff vs SF : {diff:+.0f}  (95% CI {lo:+.0f}..{hi:+.0f})")
    print(f"  Estimated ELO  : {args.elo + diff:.0f}  "
          f"(CI {args.elo + lo:.0f}..{args.elo + hi:.0f})")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
