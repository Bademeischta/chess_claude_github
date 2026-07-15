#!/usr/bin/env python3
"""
tools/run_arena.py — run an arena match on demand (without waiting for the
in-training gate at every arena_every_n_steps).

It plays the SAME match the trainer plays internally (trainer.run_arena:
greedy, no exploration noise, random openings from both colours), between a
"current" checkpoint and an "opponent" checkpoint, and prints W/D/L, the draw
rate, the score, and the implied ELO difference.

READ-ONLY: it never writes pool_state.json or any checkpoint, so it is safe to
run while training is live (they only share the GPU — see the note below).

Usage:
    python tools/run_arena.py                  # latest net vs best pool member
    python tools/run_arena.py --games 100
    python tools/run_arena.py --opponent checkpoints/net_step5030.pt
    python tools/run_arena.py --current checkpoints/step_0040018.pt \\
                              --opponent checkpoints/net_step35001.pt --sims 128

Notes:
  * Default opponent is the strongest pool member from pool_state.json. To
    measure PROGRESS over a known gap, pass --opponent <an older net_step*.pt>.
  * BATCHED by default: each move's search shares one NN forward across
    --batch-size leaves (≈ one forward per batch instead of per sim), so the
    whole run finishes in minutes even while sharing the GPU with training.
    Pass --no-batched for the old single-leaf path.
  * If training is running, both processes share the GPU: the load-governor
    throttles training to ~25% while this runs. Because the batched path is
    short, that throttle window is brief — fine to run alongside training.
    For an absolutely clean number you can still pause training first.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe
from utils.ckpt_arch import latest_checkpoint, match_arch
from model.network import build_model
from engine.harness import load_state_into
from training.pbt import OpponentPool
from training.trainer import Trainer


def _implied_elo(wins: int, draws: int, losses: int) -> float | None:
    n = wins + draws + losses
    if n == 0:
        return None
    score = (wins + 0.5 * draws) / n
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default=None,
                    help="checkpoint to evaluate (default: latest in checkpoints/)")
    ap.add_argument("--opponent", default=None,
                    help="opponent checkpoint (default: best pool member)")
    ap.add_argument("--games", type=int, default=40,
                    help="number of arena games (default 40; trainer uses 100)")
    ap.add_argument("--sims", type=int, default=64,
                    help="MCTS sims per move (default 64 = fast diagnostic; "
                         "raise toward 128 for a closer-to-training comparison)")
    ap.add_argument("--max-moves", type=int, default=160,
                    help="per-game move cap (default 160). Arena has no resign, "
                         "so weak conversions otherwise drag games to 300 moves "
                         "and the run takes hours; unconverted wins count as draws.")
    ap.add_argument("--batched", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="share one NN forward across --batch-size leaves per "
                         "search round (default on = fast, training-friendly). "
                         "--no-batched uses the old single-leaf path.")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="leaves per batched NN forward (default 32). Higher = "
                         "fewer GPU launches but more wasted sims if it "
                         "overshoots the remaining count.")
    args = ap.parse_args()

    update_config_from_dict(run_probe(write=False))
    device = torch.device(CONFIG.device)
    CONFIG.mcts_sims = int(args.sims)

    # ── Resolve checkpoints ──────────────────────────────────────────────
    current_path = args.current or latest_checkpoint(CONFIG.checkpoint_dir)
    if not current_path or not Path(current_path).exists():
        print(f"[arena] no current checkpoint found ({current_path})")
        return 1

    if args.opponent:
        opp_path = args.opponent
    else:
        pool = OpponentPool(checkpoint_dir=CONFIG.checkpoint_dir)
        pool.load_state()                       # READ-ONLY: never save_state()
        opp_path = pool.best_opponent_path()
        if not opp_path:
            print("[arena] pool is empty — pass --opponent <checkpoint>")
            return 1
    if not Path(opp_path).exists():
        print(f"[arena] opponent checkpoint missing: {opp_path}")
        return 1

    # ── Build the two nets ───────────────────────────────────────────────
    match_arch(CONFIG, current_path)
    current = build_model(CONFIG, compile_model=False).eval()
    load_state_into(current, current_path, device)   # shared allowlist policy
    opponent = build_model(CONFIG, compile_model=False).eval()
    load_state_into(opponent, opp_path, device)

    # ── Minimal Trainer shell (run_arena only needs cfg + device) ────────
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = CONFIG
    trainer.device = device

    batched_bs = int(args.batch_size) if args.batched else 0
    mode = f"batched x{batched_bs}" if batched_bs else "single-leaf"

    print(f"[arena] current : {current_path}")
    print(f"[arena] opponent: {opp_path}")
    print(f"[arena] {args.games} games @ {CONFIG.mcts_sims} sims/move, "
          f"cap {args.max_moves} moves, "
          f"openings={getattr(CONFIG, 'arena_opening_plies', 8)} plies, "
          f"greedy, {mode}")

    def _progress(done, total, w, d, l):
        print(f"\r[arena] {done}/{total} games  W{w}/D{d}/L{l}",
              end="", flush=True)

    print("[arena] playing...")
    wins, draws, losses = trainer.run_arena(
        current, opponent, args.games,
        progress=_progress, max_moves=args.max_moves,
        batched_bs=batched_bs)
    print()  # newline after the live tally

    n = wins + draws + losses
    draw_rate = draws / n if n else 0.0
    score = (wins + 0.5 * draws) / n if n else 0.0
    elo = _implied_elo(wins, draws, losses)
    elo_str = f"{elo:+.0f}" if elo is not None else "n/a (shutout)"

    print("\n" + "=" * 56)
    print(f"  RESULT (current's POV):  W{wins} / D{draws} / L{losses}")
    print(f"  Score        : {score*100:.1f}%")
    print(f"  Draw rate    : {draw_rate*100:.1f}%   "
          f"(was ~93% at step ~30k -- watch this fall)")
    print(f"  Implied ELO  : {elo_str}  vs the opponent")
    print("=" * 56)
    print("  (read-only: pool_state.json was NOT modified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
