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

The match protocol itself lives in utils/sf_anchor.py (shared with the
automatic in-training probe wired into main.py); this tool adds the CLI,
the adaptive --auto bisection search, and result presentation.
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

from config import CONFIG                                    # noqa: E402
from engine.harness import load_engine                       # noqa: E402
from utils.elo import ELOSystem                              # noqa: E402
from utils.sf_anchor import (SKILL_ELO, play_anchor_match,   # noqa: E402
                             resolve_stockfish_path)


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
                         "comparable budget (~0.125 ms/sim, measured).")
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
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Leaves per shared NN forward in our search "
                         "(default 32; 0 = slow single-leaf loop).")
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

    sf_path = None
    if args.stockfish and Path(args.stockfish).exists():
        # python-chess needs an absolute path on Windows (asyncio subprocess
        # does not resolve relative paths against CWD).
        sf_path = str(Path(args.stockfish).resolve())
    else:
        sf_path = resolve_stockfish_path(CONFIG)
    if not sf_path:
        print("[elo] No Stockfish binary found.")
        print("[elo] Run once:  python tools/install_stockfish.py")
        print("[elo] Or pass --stockfish <path-to-binary>.")
        return 1

    # Shared bring-up: probe → Syzygy → checkpoint → model → engine.
    handle = load_engine(args.resume,
                         log=lambda m: print(m.replace("[harness]", "[elo]")))
    sims = args.sims if args.sims is not None else CONFIG.mcts_sims

    def _match(n_games: int, *, sf_elo=None, sf_skill=None,
               label: str = "") -> tuple[int, int, int]:
        res = play_anchor_match(
            handle.model, handle.device, CONFIG,
            games=n_games, sf_elo=sf_elo, sf_skill=sf_skill,
            sims=sims,
            movetime=(args.movetime or 0.0),
            sf_depth=args.depth,
            stockfish_path=sf_path,
            batched_bs=int(args.batch_size),
            engine_inf=handle.engine,
            sf_threads=int(args.sf_threads),
            sf_hash_mb=int(args.sf_hash_mb),
            progress=lambda d, t, w, dr, l: print(
                f"  {label}  game {d}/{t}  (W{w} D{dr} L{l})", flush=True),
        )
        if res is None:
            raise RuntimeError("Stockfish match could not start "
                               f"(binary: {sf_path})")
        return res["wins"], res["draws"], res["losses"]

    # ── Adaptive bisection search over Stockfish ELO ─────────────────
    if args.auto:
        lo, hi = args.auto_low, args.auto_high
        rounds = 4
        per_round = max(8, args.auto_budget // rounds)
        history: list[tuple[int, int, int, int]] = []  # (anchor, W, D, L)
        for r in range(rounds):
            anchor = (lo + hi) // 2
            print(f"\n[auto] round {r+1}/{rounds}  "
                  f"window=[{lo},{hi}]  anchor={anchor}  "
                  f"games={per_round}")
            bw, bd, bl = _match(per_round, sf_elo=anchor, label=f"[A{anchor}]")
            history.append((anchor, bw, bd, bl))
            total = bw + bd + bl
            score = (bw + 0.5 * bd) / total if total > 0 else 0.5
            print(f"[auto] round {r+1} → score={score:.2f} at anchor {anchor}")
            if score < 0.4:
                hi = anchor - 1
            elif score > 0.6:
                lo = anchor + 1
            else:
                spread = max(50, (hi - lo) // 3)
                lo, hi = anchor - spread, anchor + spread
            if hi <= lo + 50:
                break

        total_games = sum(h[1] + h[2] + h[3] for h in history)
        total_w = sum(h[1] for h in history)
        total_d = sum(h[2] for h in history)
        total_l = sum(h[3] for h in history)

        def _score(h: tuple[int, int, int, int]) -> float:
            _, w, d, l = h
            return (w + 0.5 * d) / max(1, w + d + l)

        BAND_LO, BAND_HI = 0.15, 0.85
        in_band = [h for h in history if BAND_LO <= _score(h) <= BAND_HI]

        print("\n" + "=" * 58)
        print(f"  AUTO ELO SEARCH  ({total_games} games total, {sims} sims)")
        for a, w, d, l in history:
            print(f"    SF@{a:>4d}: W{w} D{d} L{l}  score={_score((a,w,d,l)):.2f}")
        print(f"  Overall record         : W{total_w} D{total_d} L{total_l}")
        print("-" * 58)

        if in_band:
            best = min(in_band, key=lambda h: abs(_score(h) - 0.5))
            anchor, bw, bd, bl = best
            diff, lo_ci, hi_ci = ELOSystem.elo_difference_ci(bw, bd, bl)
            print(f"  Best balanced anchor   : SF@{anchor}")
            print(f"  ELO diff vs SF@{anchor}: {diff:+.0f}  "
                  f"(95 % CI {lo_ci:+.0f}..{hi_ci:+.0f})")
            print(f"  Estimated ELO          : {anchor + diff:.0f}  "
                  f"(CI {anchor + lo_ci:.0f}..{anchor + hi_ci:.0f})")
        else:
            best_score_lo = min(history, key=_score)
            best_score_hi = max(history, key=_score)
            if _score(best_score_hi) < BAND_LO:
                floor = best_score_lo[0]
                print(f"  No anchor in [{int(BAND_LO*100)}–"
                      f"{int(BAND_HI*100)} %] score band.")
                print(f"  Model scored ≤ {_score(best_score_hi)*100:.0f} %% "
                      f"even at the lowest tested SF anchor ({floor}).")
                print(f"  → Estimated ELO is BELOW ~{floor}  "
                      f"(Stockfish's UCI floor is 1320 — probe lower with "
                      f"--sf-skill 0-3 fixed matches).")
                print(f"  Hints: increase --sims, lower --movetime, or train "
                      f"the net further before re-measuring.")
            elif _score(best_score_lo) > BAND_HI:
                ceiling = best_score_hi[0]
                print(f"  Won at every tested anchor (top {ceiling}).")
                print(f"  → Estimated ELO is ABOVE ~{ceiling}. Re-run with "
                      f"--auto-low {ceiling + 100} to refine.")
            else:
                print("  Search did not converge — anchors flipped between "
                      "≤15% and ≥85% with no balanced match. Try a denser "
                      "search by raising --auto-budget.")
        print("=" * 58)
        return 0

    # ── Single fixed-anchor match ─────────────────────────────────────
    if args.sf_skill is not None:
        wins, draws, losses = _match(args.games, sf_skill=int(args.sf_skill))
        anchor = SKILL_ELO[max(0, min(20, int(args.sf_skill)))]
        label = f"Stockfish skill {int(args.sf_skill)} (≈{anchor})"
    else:
        wins, draws, losses = _match(args.games, sf_elo=int(args.elo))
        anchor = args.elo
        label = f"Stockfish@{args.elo}"

    diff, lo, hi = ELOSystem.elo_difference_ci(wins, draws, losses)
    print("\n" + "=" * 52)
    print(f"  vs {label}  ({args.games} games, {sims} sims)")
    print(f"  W{wins} D{draws} L{losses}")
    print(f"  ELO diff vs SF : {diff:+.0f}  (95% CI {lo:+.0f}..{hi:+.0f})")
    print(f"  Estimated ELO  : {anchor + diff:.0f}  "
          f"(CI {anchor + lo:.0f}..{anchor + hi:.0f})")
    if args.sf_skill is not None:
        print("  NOTE: skill-level ELO anchors are rough — use --auto or "
              "UCI_Elo anchors (≥1320) for calibrated numbers.")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
