#!/usr/bin/env python3
"""
tools/lichess_puzzles.py — train policy on Lichess's open puzzle database.

Lichess publishes ~4 million human-curated tactical puzzles under CC0.
Each puzzle is a position where one side has a forcing winning sequence
(mate, decisive material gain, etc.). This script downloads the DB,
decodes the puzzles, and emits PositionRecord objects with the puzzle's
correct move as the policy target — pure supervised tactics training.

Why this matters for *this* AI:
    The model has shown it doesn't recognise basic tactical patterns
    (mate-in-1 threats, hanging pieces, forks). Self-play won't teach
    these — the model can't generate examples of patterns it can't
    see. The Lichess DB IS a curated source of exactly these patterns,
    pre-labelled with the correct response.

Database format (CSV, Zstandard-compressed — Lichess migrated from BZIP2
in early 2024, so older docs that mention .bz2 are now stale):
    PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity,
    NbPlays, Themes, GameUrl, OpeningTags

    FEN:    the position BEFORE the opponent makes the "setup move"
            that triggers the puzzle.
    Moves:  UCI move sequence. Move [0] is the opponent's setup; moves
            [1], [3], [5]... are the player's solution moves (what we
            want to train on); moves [2], [4]... are opponent's forced
            responses.

Usage:
    # First run — auto-downloads the ~270 MB DB to ./data/lichess_puzzles.csv.bz2
    python tools/lichess_puzzles.py --max-rating 1600 --n-puzzles 200000

    # Reuse the existing DB, narrower rating band
    python tools/lichess_puzzles.py --max-rating 1400 --min-rating 600 \\
        --n-puzzles 500000 --themes mate,mateIn2,mateIn3

    # Only mate puzzles (high-signal patterns for a weak network)
    python tools/lichess_puzzles.py --themes "mate,mateIn1,mateIn2" \\
        --n-puzzles 100000
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

# Lichess switched the puzzle DB from BZIP2 to Zstandard in early 2024;
# the .bz2 mirror was retired and now 404s. zstandard is a pure-Python
# install (`pip install zstandard`) — we import lazily and surface a
# clear message if it's missing so the failure isn't cryptic.
try:
    import zstandard as _zstd  # type: ignore[import-not-found]
    _HAVE_ZSTD = True
except ImportError:
    _zstd = None
    _HAVE_ZSTD = False

# Project setup.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# UTF-8 console on Windows.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

import chess
import numpy as np

from config import CONFIG
from engine.board import Board
from engine.movegen import move_to_action_index, _move_uci
from training.replay_buffer import PositionRecord, PrioritizedReplayBuffer


PUZZLE_DB_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
DEFAULT_DB_PATH = _root / "data" / "lichess_puzzles.csv.zst"


def _download_puzzle_db(dest: Path) -> bool:
    """Stream-download the puzzle DB. ~300 MB so we show a progress meter."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[puzzles] Downloading puzzle DB from {PUZZLE_DB_URL}")
    print(f"[puzzles] -> {dest}")
    print(f"[puzzles] (this is ~300 MB compressed; one-time download)")

    req = urllib.request.Request(
        PUZZLE_DB_URL,
        headers={"User-Agent": "chess_ai-lichess-puzzles/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            block = 256 * 1024
            written = 0
            t0 = time.time()
            with dest.open("wb") as fh:
                while True:
                    chunk = resp.read(block)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = written / total * 100
                        rate = written / max(time.time() - t0, 1e-3) / 1024 / 1024
                        print(f"\r[puzzles]   {written/1024/1024:6.1f} / "
                              f"{total/1024/1024:6.1f} MB  ({pct:5.1f}%)  "
                              f"{rate:.2f} MB/s",
                              end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"\n[puzzles] FATAL: download failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        # Don't leave a half-downloaded file masquerading as a complete one.
        if dest.exists():
            try:
                dest.unlink()
            except Exception:
                pass
        return False


def _uci_to_move_int(board: Board, uci: str) -> Optional[int]:
    """Map UCI string → engine move int (linear scan over legal moves)."""
    target = uci.lower()
    for m in board.legal_moves():
        if _move_uci(m).lower() == target:
            return m
    return None


def _parse_puzzle_row(row: dict) -> Optional[tuple]:
    """Pull the bits we care about out of a CSV row."""
    fen = row.get("FEN", "").strip()
    moves_str = row.get("Moves", "").strip()
    if not fen or not moves_str:
        return None
    moves = moves_str.split()
    if len(moves) < 2:
        return None
    try:
        rating = int(row.get("Rating", "0") or "0")
    except ValueError:
        rating = 0
    themes = (row.get("Themes", "") or "").lower().split()
    return fen, moves, rating, themes


def _build_records_for_puzzle(
    fen: str,
    moves: list[str],
    cfg,
) -> list[PositionRecord]:
    """Walk a puzzle and emit PositionRecord at each player-decision point.

    Convention (Lichess):
        * The FEN is the position BEFORE the opponent's setup move.
        * moves[0] is the opponent's setup move (not a training target —
          it's just "what happened").
        * After applying moves[0], the puzzle SOLVER is to move.
        * moves[1], moves[3], ... are the solver's responses (training
          targets — policy = one-hot on this move).
        * moves[2], moves[4], ... are the opponent's forced responses
          (not training targets; just played to advance the board).

    The value target for every solver-decision point is wdl = 1.0 from
    the solver's POV. This matches the semantics of a tactical puzzle:
    the solver wins by definition once they're in the position. (We
    ignore the rare "save the draw" puzzles — those are <0.5% of the
    DB and we filter them out via the `equality` theme.)
    """
    # Apply opponent's setup move on a python-chess board (which is more
    # robust to weird FENs than our C++ engine for the setup phase).
    try:
        pcb = chess.Board(fen)
    except Exception:
        return []
    try:
        setup = chess.Move.from_uci(moves[0])
    except Exception:
        return []
    if setup not in pcb.legal_moves:
        return []
    pcb.push(setup)

    # Solver's side is whoever is to move AFTER the setup.
    solver_side = 0 if pcb.turn == chess.WHITE else 1
    records: list[PositionRecord] = []

    # Iterate over the solution moves.
    for i in range(1, len(moves)):
        uci = moves[i]
        # Is it the solver's turn? The solver moves at odd i (i=1,3,5...).
        is_solver_move = ((i - 1) % 2 == 0)

        if is_solver_move:
            # Build a PositionRecord for this decision point.
            try:
                board_obj = Board.from_fen(pcb.fen())
            except Exception:
                break
            move_int = _uci_to_move_int(board_obj, uci)
            if move_int is None:
                # Engine doesn't enumerate this move — skip remaining.
                break
            action_idx = move_to_action_index(move_int, board_obj.side_to_move)
            if action_idx < 0 or action_idx >= cfg.num_actions:
                break

            policy = np.zeros(cfg.num_actions, dtype=np.float32)
            policy[action_idx] = 1.0

            try:
                records.append(PositionRecord(
                    board_tensor   = board_obj.to_tensor(),
                    history_tensor = np.zeros(
                        (cfg.gru_history_len, cfg.input_planes, 8, 8),
                        dtype=np.float32,
                    ),
                    policy_target  = policy,
                    wdl_label      = 1.0,  # Solver wins by definition.
                    phase          = int(board_obj.get_phase()),
                    piece_count    = int(board_obj.piece_count()),
                    move_number    = 0,
                    is_teacher     = True,
                ))
            except Exception:
                break

        # Advance the board (whether solver or opponent move).
        try:
            mv = chess.Move.from_uci(uci)
            if mv not in pcb.legal_moves:
                break
            pcb.push(mv)
        except Exception:
            break

    return records


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH),
                    help=f"Path to lichess_db_puzzle.csv.bz2 "
                         f"(default: {DEFAULT_DB_PATH.relative_to(_root)}). "
                         f"Auto-downloads if missing.")
    ap.add_argument("--n-puzzles", type=int, default=200_000,
                    help="Max number of puzzles to convert "
                         "(default: 200,000).")
    ap.add_argument("--min-rating", type=int, default=600,
                    help="Skip puzzles below this rating (default: 600).")
    ap.add_argument("--max-rating", type=int, default=1800,
                    help="Skip puzzles above this rating (default: 1800). "
                         "Lower = easier patterns. For a weak network start "
                         "with 1400-1600.")
    ap.add_argument("--themes", default=None,
                    help="Comma-separated list of themes to require "
                         "(any-of). E.g. 'mate,mateIn2,mateIn3,fork,pin'. "
                         "Default: no theme filter.")
    ap.add_argument("--out", default="replay_buffer/lichess_puzzles.pkl",
                    help="Output buffer file relative to project root "
                         "(default: replay_buffer/lichess_puzzles.pkl).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-every", type=int, default=5000,
                    help="Print progress every N puzzles read "
                         "(default: 5000).")
    args = ap.parse_args()

    random.seed(args.seed)

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        if not _download_puzzle_db(db_path):
            return 1
        if not db_path.exists():
            print(f"[puzzles] FATAL: download claimed success but file "
                  f"missing at {db_path}", file=sys.stderr)
            return 1

    required_themes = None
    if args.themes:
        required_themes = {t.strip().lower() for t in args.themes.split(",")
                           if t.strip()}
        print(f"[puzzles] Filtering for themes (any-of): "
              f"{sorted(required_themes)}")

    print(f"[puzzles] Reading puzzles from {db_path}")
    print(f"[puzzles] Rating filter: [{args.min_rating}, {args.max_rating}]")
    print(f"[puzzles] Target: {args.n_puzzles:,} puzzles")
    print()

    # Generous capacity — average puzzle ~3 solver moves.
    cap = max(args.n_puzzles * 10, 100_000)
    buf = PrioritizedReplayBuffer(capacity=cap)

    t0 = time.time()
    rows_read = 0
    puzzles_kept = 0
    positions_added = 0

    # The DB has a header row. csv.DictReader handles it. The file is
    # Zstandard-compressed — zstandard's stream_reader gives us a bytes
    # stream that io.TextIOWrapper + csv.DictReader can chew through
    # without holding the full ~3 GB uncompressed CSV in memory.
    if not _HAVE_ZSTD:
        print("[puzzles] FATAL: zstandard library required to decompress "
              ".csv.zst. Install with:  pip install zstandard",
              file=sys.stderr)
        return 1

    raw = None
    try:
        dctx = _zstd.ZstdDecompressor()  # type: ignore[union-attr]
        raw = open(db_path, "rb")
        # read_across_frames: Lichess may write the .zst as multiple
        # concatenated frames; without this flag the reader stops after
        # the first one and we'd silently get only ~10% of the puzzles.
        stream = dctx.stream_reader(raw, read_across_frames=True)
        with io.TextIOWrapper(stream, encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows_read += 1
                if puzzles_kept >= args.n_puzzles:
                    break

                parsed = _parse_puzzle_row(row)
                if parsed is None:
                    continue
                fen, moves, rating, themes = parsed

                if rating < args.min_rating or rating > args.max_rating:
                    continue
                # Skip "save the draw" / "save the loss" puzzles — their
                # solver doesn't actually win, so wdl=1.0 would be wrong.
                if "equality" in themes:
                    continue

                if required_themes is not None:
                    if not any(t in required_themes for t in themes):
                        continue

                records = _build_records_for_puzzle(fen, moves, CONFIG)
                if not records:
                    continue

                for r in records:
                    buf.add(r)
                puzzles_kept += 1
                positions_added += len(records)

                if rows_read % args.print_every == 0:
                    elapsed = time.time() - t0
                    print(f"[puzzles] read={rows_read:>8,d}  "
                          f"kept={puzzles_kept:>7,d}  "
                          f"pos={positions_added:>8,d}  "
                          f"({rows_read/max(elapsed,1e-3):.0f} rows/s)",
                          flush=True)
    except KeyboardInterrupt:
        print("\n[puzzles] Interrupted — saving what we have.")
    except Exception as e:
        print(f"\n[puzzles] WARNING: error during read "
              f"({type(e).__name__}: {e}) — saving partial buffer.",
              file=sys.stderr)
    finally:
        # zstandard's stream_reader does NOT close the underlying file
        # when its TextIOWrapper context exits — we have to do it ourselves.
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass

    # Save.
    out_path = _root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf.save(str(out_path))

    dt = time.time() - t0
    print()
    print("=" * 60)
    print(f"  Lichess puzzle conversion done in {dt/60:.1f} min")
    print(f"  rows read:    {rows_read:,}")
    print(f"  puzzles kept: {puzzles_kept:,}")
    print(f"  positions:    {positions_added:,}")
    print(f"  saved to:     {out_path}")
    print("=" * 60)
    print()
    print("Next: fold this buffer into training via:")
    print()
    print(f"  python main.py --phase=puzzles "
          f"--puzzle-buffer=\"{out_path}\" "
          f"--resume=checkpoints/latest.pt")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
