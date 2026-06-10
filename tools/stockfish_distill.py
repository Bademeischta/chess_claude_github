#!/usr/bin/env python3
"""
tools/stockfish_distill.py — generate Stockfish-vs-Stockfish games and write
them to the replay buffer in PositionRecord format for supervised learning.

Why this exists:
    AlphaZero-style self-play with a randomly-initialised network on a
    single GPU does not bootstrap to interesting chess in any human-scale
    amount of compute. The net gets stuck in a "draw-collapse" attractor:
    its value head learns that the safest prediction is always 0 and its
    policy stays near-uniform, so MCTS produces shuffle-to-draw games,
    which are then used as training data, which reinforces the collapse.

    Distilling from a strong reference engine (Stockfish) sidesteps the
    bootstrap entirely: the network is shown legitimate winning games at
    a fixed strength level and trained to imitate them. Once the policy
    has real preferences and the value head has real signal, self-play
    can take over and refine further.

Usage:
    # Generate 1000 games at SF Skill 8 (~1700 ELO) → ./replay_buffer/sf_distill.pkl
    python tools/stockfish_distill.py --games 1000 --skill 8

    # Strong-teacher mode (skill 12 ≈ 2050 ELO), longer think time per move
    python tools/stockfish_distill.py --games 500 --skill 12 --movetime 0.3

    # Mixed-skill: 1/3 weak, 1/3 medium, 1/3 strong (broader feature coverage)
    python tools/stockfish_distill.py --games 1500 --skill-mix 4,8,12

After it runs, fold the generated buffer into the existing replay buffer
via `--phase=sf_distill` in main.py (set up separately in main.py).

Output PositionRecord encoding:
    * board_tensor   = the (21,8,8) state at SF's decision point
    * history_tensor = zeros (training will use the game-final outcome,
                       not temporal context — keeps it fast)
    * policy_target  = one-hot on SF's chosen action (prob = 1.0)
    * wdl_label      = game-final outcome from this side's POV
                       (1.0 = won, 0.5 = drew, 0.0 = lost)
    * is_teacher     = True (TeacherBuffer in trainer.py picks this up)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Project setup — must precede project imports.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# UTF-8 console on Windows (matches main.py — print() can hit Unicode in
# the progress lines without this on cp1252 terminals).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

import chess
import chess.engine
import numpy as np

from config import CONFIG
from engine.board import Board
from engine.movegen import move_to_action_index, _move_uci
from training.replay_buffer import PositionRecord, PrioritizedReplayBuffer


# Approximate ELO per Stockfish Skill Level (from the SF developer table).
# Used only for the human-readable progress print.
SKILL_TO_ELO = [800, 950, 1100, 1250, 1400, 1500, 1600, 1700,
                1800, 1900, 2000, 2050, 2100, 2150, 2200, 2300,
                2400, 2500, 2700, 2850, 3000]


def uci_to_move_int(board: Board, uci: str) -> Optional[int]:
    """Map a UCI move string to the engine's internal move int.

    Slow path (linear scan over legal moves) but cached by the LRU on
    move_to_action_index downstream, so the cost is amortised. Returns
    None if the UCI doesn't match any legal move on this board — that
    can happen if Stockfish's promotion notation differs (e.g. "e7e8q"
    vs "e7e8Q") or if our move generator disagrees with python-chess
    on legality of a position (a known C++ parity bug for very few
    edge-case FENs).
    """
    target = uci.lower()
    for m in board.legal_moves():
        if _move_uci(m).lower() == target:
            return m
    return None


def play_one_game(
    sf: chess.engine.SimpleEngine,
    skill: int,
    movetime_s: float,
    max_plies: int,
    opening_random_plies: int,
) -> tuple[list[tuple[str, str, int]], float]:
    """Play one SF-vs-SF game.

    Args:
        opening_random_plies: number of starting plies where we play a
            uniform-random LEGAL move instead of asking SF. This forces
            opening diversity — SF at low skill levels tends to play the
            same first few moves every game, which would give the
            distillation set zero opening variety.

    Returns:
        (history, wdl_white) where history is a list of
        (fen_before_move, played_move_uci, side_to_move_int) tuples and
        wdl_white is the final game result from White's POV
        (1.0 = win, 0.5 = draw, 0.0 = loss).
    """
    # Skill setup. UCI_LimitStrength must be False — otherwise SF silently
    # ignores "Skill Level" and plays at its UCI_Elo floor (1320) instead.
    sf.configure({"UCI_LimitStrength": False, "Skill Level": int(skill)})

    board = chess.Board()
    history: list[tuple[str, str, int]] = []

    # Random opening plies for diversity.
    for _ in range(opening_random_plies):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = random.choice(legal)
        # Don't record random opening moves as teacher signal — they're noise.
        board.push(mv)

    # SF-driven main game.
    for _ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        try:
            result = sf.play(board, chess.engine.Limit(time=movetime_s))
        except chess.engine.EngineTerminatedError:
            print("[sf-distill] WARNING: Stockfish crashed mid-game, "
                  "abandoning.", file=sys.stderr)
            return [], 0.5
        except Exception as e:
            print(f"[sf-distill] WARNING: SF play failed ({type(e).__name__}): "
                  f"{e}", file=sys.stderr)
            break
        if result.move is None:
            break
        history.append((
            board.fen(),
            result.move.uci(),
            0 if board.turn == chess.WHITE else 1,
        ))
        board.push(result.move)

    # Resolve game outcome.
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        # Hit max_plies without a real result — treat as draw.
        wdl_white = 0.5
    elif outcome.winner is None:
        wdl_white = 0.5
    elif outcome.winner == chess.WHITE:
        wdl_white = 1.0
    else:
        wdl_white = 0.0

    return history, wdl_white


def build_position_records(
    history: list[tuple[str, str, int]],
    wdl_white: float,
    cfg,
) -> list[PositionRecord]:
    """Turn a finished game into PositionRecord objects.

    The policy target is a one-hot spike on SF's chosen move (probability
    1.0). That's a deliberately strong signal — distillation cares more
    about "this move was good enough" than about exploring distributions.
    The value target is the game-final result from each side's POV, which
    is the same MC return AlphaZero uses (no per-move eval).
    """
    records: list[PositionRecord] = []
    for fen, played_uci, side_to_move in history:
        try:
            board_obj = Board.from_fen(fen)
        except Exception:
            # Some FENs trigger a C++ parity issue — just skip those.
            continue

        move_int = uci_to_move_int(board_obj, played_uci)
        if move_int is None:
            # SF picked a move our engine doesn't enumerate (rare promotion
            # corner cases). Better to skip than emit a garbage target.
            continue

        action_idx = move_to_action_index(move_int, board_obj.side_to_move)
        if action_idx < 0 or action_idx >= cfg.num_actions:
            continue

        policy = np.zeros(cfg.num_actions, dtype=np.float32)
        policy[action_idx] = 1.0

        wdl = wdl_white if side_to_move == 0 else (1.0 - wdl_white)

        try:
            records.append(PositionRecord(
                board_tensor   = board_obj.to_tensor(),
                history_tensor = np.zeros(
                    (cfg.gru_history_len, cfg.input_planes, 8, 8),
                    dtype=np.float32,
                ),
                policy_target  = policy,
                wdl_label      = float(wdl),
                phase          = int(board_obj.get_phase()),
                piece_count    = int(board_obj.piece_count()),
                move_number    = 0,
                is_teacher     = True,
            ))
        except Exception as e:
            # Defensive: a single bad FEN shouldn't tank the whole batch.
            print(f"[sf-distill] WARNING: PositionRecord build failed for "
                  f"FEN {fen}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

    return records


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stockfish", default=CONFIG.stockfish_path,
                    help="Path to Stockfish UCI binary.")
    ap.add_argument("--games", type=int, default=1000,
                    help="Number of SF-vs-SF games (default: 1000).")
    ap.add_argument("--skill", type=int, default=8,
                    help="Stockfish Skill Level 0-20 (default: 8 ~ 1700 ELO).")
    ap.add_argument("--skill-mix", type=str, default=None,
                    help="Override --skill with a comma-separated mix, e.g. "
                         "'4,8,12' for 1/3 weak, 1/3 medium, 1/3 strong.")
    ap.add_argument("--movetime", type=float, default=0.1,
                    help="Seconds per move for SF (default: 0.1).")
    ap.add_argument("--max-plies", type=int, default=200,
                    help="Hard cap on game length in half-moves (default: 200).")
    ap.add_argument("--opening-random-plies", type=int, default=4,
                    help="Random plies at game start for opening variety "
                         "(default: 4).")
    ap.add_argument("--out", default="replay_buffer/sf_distill.pkl",
                    help="Output buffer file relative to project root "
                         "(default: replay_buffer/sf_distill.pkl).")
    ap.add_argument("--threads", type=int, default=2,
                    help="Stockfish worker threads (default: 2).")
    ap.add_argument("--hash-mb", type=int, default=128,
                    help="Stockfish hash MB (default: 128).")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducibility (default: 42).")
    ap.add_argument("--print-every", type=int, default=10,
                    help="Print progress every N games (default: 10).")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.stockfish or not os.path.isfile(args.stockfish):
        print(f"[sf-distill] FATAL: Stockfish binary not found at "
              f"'{args.stockfish}'. Pass --stockfish or set "
              f"CONFIG.stockfish_path.", file=sys.stderr)
        return 1

    # Parse --skill-mix (overrides --skill if present).
    skill_pool: list[int]
    if args.skill_mix:
        try:
            skill_pool = [int(x.strip()) for x in args.skill_mix.split(",")
                          if x.strip()]
        except ValueError:
            print(f"[sf-distill] FATAL: --skill-mix must be a comma-separated "
                  f"list of integers, got '{args.skill_mix}'.",
                  file=sys.stderr)
            return 1
        if not skill_pool or any(s < 0 or s > 20 for s in skill_pool):
            print(f"[sf-distill] FATAL: --skill-mix values must be in 0..20.",
                  file=sys.stderr)
            return 1
    else:
        skill_pool = [args.skill]

    # Spawn Stockfish.
    try:
        sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        sf.configure({"Hash": int(args.hash_mb), "Threads": int(args.threads)})
    except Exception as e:
        print(f"[sf-distill] FATAL: could not start Stockfish: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    # Capacity estimate: avg game length 80 plies, but allow generous slack.
    cap = max(args.games * 200, 50_000)
    buf = PrioritizedReplayBuffer(capacity=cap)

    print(f"[sf-distill] Stockfish ready.")
    print(f"[sf-distill] skill={skill_pool}  "
          f"(~{[SKILL_TO_ELO[s] for s in skill_pool]} ELO)")
    print(f"[sf-distill] movetime={args.movetime}s  games={args.games}  "
          f"max_plies={args.max_plies}  opening_random={args.opening_random_plies}")
    print()

    t0 = time.time()
    games_valid = 0
    games_skipped = 0
    positions_added = 0
    results_counter = {1.0: 0, 0.5: 0, 0.0: 0}

    try:
        for g in range(args.games):
            skill = random.choice(skill_pool)
            history, wdl_white = play_one_game(
                sf,
                skill=skill,
                movetime_s=args.movetime,
                max_plies=args.max_plies,
                opening_random_plies=args.opening_random_plies,
            )
            if not history:
                games_skipped += 1
                continue
            records = build_position_records(history, wdl_white, CONFIG)
            if not records:
                games_skipped += 1
                continue
            for r in records:
                buf.add(r)
            games_valid += 1
            positions_added += len(records)
            results_counter[wdl_white] = results_counter.get(wdl_white, 0) + 1

            if (g + 1) % args.print_every == 0 or g == 0:
                elapsed = time.time() - t0
                g_per_h = (g + 1) / max(elapsed, 1e-3) * 3600
                p_per_s = positions_added / max(elapsed, 1e-3)
                w, d, l = (results_counter.get(1.0, 0),
                           results_counter.get(0.5, 0),
                           results_counter.get(0.0, 0))
                print(f"[sf-distill] {g+1:>5d}/{args.games} games  "
                      f"valid={games_valid} skipped={games_skipped}  "
                      f"pos={positions_added:>7d}  "
                      f"W/D/L={w}/{d}/{l}  "
                      f"({g_per_h:.0f} g/h, {p_per_s:.0f} pos/s)",
                      flush=True)
    except KeyboardInterrupt:
        print("\n[sf-distill] Interrupted — saving what we have.")
    finally:
        try:
            sf.quit()
        except Exception:
            pass

    # Save to disk.
    out_path = _root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf.save(str(out_path))

    dt = time.time() - t0
    print()
    print("=" * 60)
    print(f"  Stockfish distillation done in {dt/60:.1f} min")
    print(f"  games:     valid={games_valid}  skipped={games_skipped}")
    print(f"  positions: {positions_added:,}")
    print(f"  results:   W={results_counter.get(1.0, 0)}  "
          f"D={results_counter.get(0.5, 0)}  "
          f"L={results_counter.get(0.0, 0)}")
    print(f"  saved to:  {out_path}")
    print("=" * 60)
    print()
    print("Next: load this buffer into training. The easiest way is:")
    print()
    print(f"  python main.py --phase=sf_distill "
          f"--sf-buffer=\"{out_path}\" --resume=checkpoints/latest.pt")
    print()
    print("…which appends these records to the active replay buffer and runs")
    print("the normal trainer over them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
