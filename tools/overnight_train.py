#!/usr/bin/env python3
"""
tools/overnight_train.py — unattended training pipeline.

Runs the full bootstrap-recovery sequence in order:

    1. Stockfish-vs-Stockfish game generation (~2-3 h)
       -> replay_buffer/sf_distill.pkl
    2. Supervised training on SF games (3 epochs, ~1.5 h)
    3. Lichess puzzle conversion (~30 min, ~30 if DB cached)
       -> replay_buffer/lichess_puzzles.pkl
    4. Supervised training on puzzles (3 epochs, ~1 h)
    5. Self-play with crash-resilient supervisor (open-ended,
       runs until you Ctrl+C in the morning)

Total before the open-ended phase: ~5-6 hours, then self-play runs as
long as you let it.

Behaviour:
    * Tees every step's output to BOTH the terminal (so you can glance
      at progress) AND `runs/overnight_<timestamp>.log` (so you can
      diagnose anything that went wrong overnight after the fact).
    * Skips a step if its expected output already exists, unless you
      pass --force. Useful when the script gets interrupted partway —
      re-run it and it picks up where it left off.
    * On step failure, logs the error, prints the offending command,
      and STOPS — it does not attempt to continue, because each step
      depends on the previous one's checkpoint.
    * Step 5 (self-play) runs forever. When you want to stop, Ctrl+C
      *once* — the supervisor catches it and saves a final checkpoint
      cleanly before exiting.

Usage:
    python tools/overnight_train.py                  # standard run
    python tools/overnight_train.py --skip-self-play # stop after step 4
    python tools/overnight_train.py --force          # re-do all steps
    python tools/overnight_train.py --games 800      # smaller SF batch
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

_root = Path(__file__).resolve().parent.parent

# Force UTF-8 console on Windows so progress lines don't crash on cp1252.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _ts_now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log_to_both(line: str, log_fh) -> None:
    """Write to terminal AND log file (with timestamp on the file copy)."""
    print(line, flush=True)
    log_fh.write(f"[{_ts_now()}] {line}\n")
    log_fh.flush()


def _run_step(
    title: str,
    cmd: list[str],
    log_fh,
    expected_output: Path | None = None,
    force: bool = False,
) -> bool:
    """Run a step, teeing output. Returns True if step ran or was skipped.

    If expected_output is set and that file exists and --force was not
    passed, the step is skipped (useful for re-running after a partial
    interruption).
    """
    banner = "=" * 70
    _log_to_both("", log_fh)
    _log_to_both(banner, log_fh)
    _log_to_both(f"  STEP: {title}", log_fh)
    _log_to_both(f"  cmd:  {' '.join(cmd)}", log_fh)
    if expected_output:
        _log_to_both(f"  out:  {expected_output}", log_fh)
    _log_to_both(banner, log_fh)

    if expected_output and expected_output.exists() and not force:
        size_mb = expected_output.stat().st_size / 1024 / 1024
        _log_to_both(f"[skip] {expected_output} already exists "
                     f"({size_mb:.1f} MB) — pass --force to re-do.", log_fh)
        return True

    t0 = time.time()

    # Tee through subprocess: read stdout line-by-line, write to both.
    # Python is launched with -u so its output isn't buffered.
    proc = subprocess.Popen(
        cmd,
        cwd=str(_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    interrupted = False
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            # Terminal: live, no timestamp clutter.
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            # Log file: timestamped so you can reconstruct the morning after.
            log_fh.write(f"[{_ts_now()}]   {line}\n")
            log_fh.flush()
    except KeyboardInterrupt:
        interrupted = True
        _log_to_both("\n[overnight] Ctrl+C received — forwarding to child "
                     "and waiting up to 120s for clean shutdown...", log_fh)
        try:
            proc.send_signal(2)  # SIGINT cross-platform
        except Exception:
            pass
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            _log_to_both("[overnight] Child didn't exit — killing.", log_fh)
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()

    proc.wait()
    dt = time.time() - t0

    if interrupted:
        _log_to_both(f"[overnight] Step interrupted after {dt/60:.1f} min.",
                     log_fh)
        return False
    if proc.returncode != 0:
        _log_to_both(f"[overnight] Step FAILED with exit code "
                     f"{proc.returncode} after {dt/60:.1f} min.", log_fh)
        return False
    _log_to_both(f"[overnight] Step OK in {dt/60:.1f} min.", log_fh)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--games", type=int, default=1500,
                    help="Number of Stockfish games (default: 1500). "
                         "Step 1 cost is roughly 6-8 s/game.")
    ap.add_argument("--skill-mix", default="4,8,12",
                    help="Stockfish skill mix for step 1 (default: "
                         "'4,8,12' ~ 1400/1700/2050 ELO).")
    ap.add_argument("--movetime", type=float, default=0.15,
                    help="Stockfish seconds per move in step 1 (default: 0.15).")
    ap.add_argument("--puzzles", type=int, default=200_000,
                    help="Number of Lichess puzzles to convert in step 3 "
                         "(default: 200,000).")
    ap.add_argument("--puzzle-max-rating", type=int, default=1600,
                    help="Lichess puzzles max rating in step 3 (default: "
                         "1600 — accessible patterns for a weak net).")
    ap.add_argument("--sup-epochs", type=int, default=3,
                    help="Supervised training epochs for steps 2 & 4 "
                         "(default: 3).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run all steps even if expected outputs exist.")
    ap.add_argument("--skip-self-play", action="store_true",
                    help="Stop after step 4 instead of running open-ended "
                         "self-play. Use this if you only want the "
                         "supervised bootstrap and not the indefinite "
                         "follow-on.")
    args = ap.parse_args()

    # Log file.
    runs_dir = _root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / f"overnight_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_fh = log_path.open("w", encoding="utf-8")

    started = datetime.now()
    eta = started + timedelta(hours=6)
    _log_to_both(f"[overnight] Pipeline started at "
                 f"{started:%Y-%m-%d %H:%M:%S}", log_fh)
    _log_to_both(f"[overnight] Estimated step-4 finish: "
                 f"~{eta:%H:%M} (then self-play runs until you Ctrl+C)",
                 log_fh)
    _log_to_both(f"[overnight] Log file: {log_path}", log_fh)

    # Paths.
    sf_buf  = _root / "replay_buffer" / "sf_distill.pkl"
    pz_buf  = _root / "replay_buffer" / "lichess_puzzles.pkl"

    py = sys.executable  # current interpreter — keeps virtual-env consistent

    steps = [
        # (title, cmd, expected_output_path)
        (
            "1/5  Generate Stockfish-vs-Stockfish games",
            [py, "-u", "tools/stockfish_distill.py",
             "--games", str(args.games),
             "--skill-mix", args.skill_mix,
             "--movetime", str(args.movetime)],
            sf_buf,
        ),
        (
            "2/5  Train on Stockfish games",
            [py, "-u", "main.py",
             "--phase=sf_distill",
             f"--sup-epochs={args.sup_epochs}",
             "--resume=checkpoints/latest.pt"],
            None,  # No fixed output path — checkpoint update is the artifact
        ),
        (
            "3/5  Download + convert Lichess puzzles",
            [py, "-u", "tools/lichess_puzzles.py",
             "--n-puzzles", str(args.puzzles),
             "--max-rating", str(args.puzzle_max_rating)],
            pz_buf,
        ),
        (
            "4/5  Train on Lichess puzzles",
            [py, "-u", "main.py",
             "--phase=puzzles",
             f"--sup-epochs={args.sup_epochs}",
             "--resume=checkpoints/latest.pt"],
            None,
        ),
    ]

    if not args.skip_self_play:
        steps.append((
            "5/5  Self-play with crash-resilient supervisor (open-ended)",
            [py, "-u", "tools/train_supervisor.py", "--phase=all"],
            None,
        ))

    # Run.
    for i, (title, cmd, out_path) in enumerate(steps, 1):
        ok = _run_step(title, cmd, log_fh, expected_output=out_path,
                       force=args.force)
        if not ok:
            _log_to_both("", log_fh)
            _log_to_both("=" * 70, log_fh)
            _log_to_both("  [overnight] PIPELINE STOPPED at step "
                         f"{i}/{len(steps)}: {title}", log_fh)
            _log_to_both(f"  [overnight] Log: {log_path}", log_fh)
            _log_to_both("  [overnight] Fix the underlying issue, then "
                         "re-run me — completed steps will be skipped.",
                         log_fh)
            _log_to_both("=" * 70, log_fh)
            log_fh.close()
            return 1

    finished = datetime.now()
    _log_to_both("", log_fh)
    _log_to_both("=" * 70, log_fh)
    _log_to_both(f"  [overnight] PIPELINE COMPLETE in "
                 f"{(finished - started).total_seconds() / 3600:.1f} h", log_fh)
    _log_to_both(f"  [overnight] Log: {log_path}", log_fh)
    _log_to_both(f"  [overnight] Test with: python tools/elo_vs_stockfish.py "
                 f"--games 10 --sf-skill 0 --movetime 1.5 --sims 400", log_fh)
    _log_to_both("=" * 70, log_fh)
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
