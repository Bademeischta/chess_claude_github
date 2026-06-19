#!/usr/bin/env python3
"""
tools/overnight.py — fire-and-forget overnight training pipeline.

Run ONE command, go to sleep. It:

  Stage 0  Resume self-play and run it up to --target-step (default 50,000),
           so the step-50k arena fires. AT THE SAME TIME (CPU, in parallel
           with the GPU self-play) it harvests a Stockfish-distillation buffer
           — free parallelism, different hardware.
  Stage 1  Wait for the Stockfish harvest to finish.
  Stage 2  Distil that strong data into the net (--phase=sf_distill) — pure
           GPU, no MCTS bottleneck, the big strength jump.
  Stage 3  (optional) Lichess tactics puzzles -> --phase=puzzles, to attack the
           "hangs pieces / misses 1-move tactics" weakness.
  Stage 4  Resume self-play (crash-resilient supervisor) from the now-stronger
           net, and keep going until you stop it in the morning (Ctrl+C).

Every stage is timestamped to runs/overnight.log AND the console. Non-critical
stage failures are caught so the run always ends back in self-play.

IMPORTANT: stop your CURRENT training first (Ctrl+C in its window) — this
script launches and owns its own training processes; two trainers on one GPU
would collide. It resumes from your latest checkpoint, so no progress is lost.

    python tools/overnight.py
    python tools/overnight.py --target-step 50000 --sf-games 2000 --skip-puzzles
    python tools/overnight.py --dry-run        # just print the plan
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
CKPT = ROOT / "checkpoints"
RB = ROOT / "replay_buffer"
LOG = ROOT / "runs" / "overnight.log"
IS_WIN = os.name == "nt"
NEWGRP = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WIN else 0

# Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def newest_step() -> int:
    best = 0
    for p in CKPT.glob("step_*.pt"):
        m = re.search(r"step_(\d+)\.pt", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def resume_arg() -> list[str]:
    """--resume pointing at the most recently written checkpoint."""
    cands = list(CKPT.glob("*.pt"))
    if not cands:
        return []
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    return [f"--resume={newest.as_posix()}"]


def spawn(args: list[str]):
    log("launch: " + " ".join(str(a) for a in args))
    return subprocess.Popen([str(a) for a in args], cwd=str(ROOT),
                            creationflags=NEWGRP)


def hard_kill(proc) -> None:
    """Kill a process AND its children (the supervisor spawns main.py)."""
    if proc is None or proc.poll() is not None:
        return
    if IS_WIN:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def stop_training(proc, name: str, grace: int = 60) -> None:
    """Try a graceful Ctrl-Break (lets the trainer save), then hard-kill the
    tree. Checkpoints are written every ~200 steps and are atomic, so even a
    hard kill loses at most a couple of minutes of training."""
    if proc is None or proc.poll() is not None:
        return
    log(f"stopping {name}: graceful signal, up to {grace}s to checkpoint...")
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT if IS_WIN else signal.SIGINT)
    except Exception:
        pass
    try:
        proc.wait(timeout=grace)
        log(f"{name} exited gracefully.")
        return
    except Exception:
        pass
    log(f"{name} still alive — hard-killing the process tree.")
    hard_kill(proc)


def run_to_completion(args: list[str], name: str, timeout=None) -> int:
    p = spawn(args)
    try:
        rc = p.wait(timeout=timeout)
        log(f"{name} finished (rc={rc}).")
        return rc
    except subprocess.TimeoutExpired:
        log(f"{name} exceeded {timeout}s — stopping.")
        hard_kill(p)
        return -1
    except Exception as e:
        log(f"{name} error: {type(e).__name__}: {e}")
        hard_kill(p)
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-step", type=int, default=50_000,
                    help="run current self-play up to this step first")
    ap.add_argument("--arena-grace", type=int, default=210,
                    help="seconds to let the step-50k arena finish before stopping")
    ap.add_argument("--sf-games", type=int, default=2000)
    ap.add_argument("--sf-skill-mix", default="4,8,12")
    ap.add_argument("--sf-movetime", type=float, default=0.12)
    ap.add_argument("--sf-threads", type=int, default=4)
    ap.add_argument("--sup-epochs", type=int, default=6)
    ap.add_argument("--skip-puzzles", action="store_true")
    ap.add_argument("--puzzle-count", type=int, default=60_000)
    ap.add_argument("--poll", type=int, default=60, help="step-poll interval (s)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sf_buf = RB / "sf_distill.pkl"
    pz_buf = RB / "lichess_puzzles.pkl"

    plan = [
        f"Stage 0  self-play -> step {args.target_step:,}  +  Stockfish harvest "
        f"({args.sf_games} games, skill-mix {args.sf_skill_mix}) IN PARALLEL",
        "Stage 1  wait for the Stockfish harvest to finish",
        f"Stage 2  distil into the net (--phase=sf_distill, {args.sup_epochs} epochs)",
        ("Stage 3  SKIPPED (puzzles)" if args.skip_puzzles else
         f"Stage 3  Lichess puzzles ({args.puzzle_count:,}) -> --phase=puzzles"),
        "Stage 4  resume self-play (supervisor) until you stop it in the morning",
    ]
    log("=" * 64)
    log("OVERNIGHT PIPELINE")
    for s in plan:
        log("  " + s)
    log("=" * 64)
    if args.dry_run:
        log("dry-run: nothing executed.")
        return 0

    # ── Stage 0: self-play to target + Stockfish harvest in parallel ─────
    start = newest_step()
    log(f"current newest checkpoint: step {start:,}")

    sf_proc = None
    if sf_buf.exists():
        log(f"{sf_buf.name} already exists — skipping harvest.")
    else:
        sf_proc = spawn([PY, "-u", "tools/stockfish_distill.py",
                         "--games", args.sf_games,
                         "--skill-mix", args.sf_skill_mix,
                         "--movetime", args.sf_movetime,
                         "--threads", args.sf_threads,
                         "--out", "replay_buffer/sf_distill.pkl"])

    sp = None
    if start < args.target_step:
        sp = spawn([PY, "-u", "tools/train_supervisor.py", "--phase=all"])
        log(f"self-play running; waiting for step {args.target_step:,}...")
        while True:
            time.sleep(args.poll)
            s = newest_step()
            if s > start:
                log(f"  ...step {s:,}")
                start = s
            if s >= args.target_step:
                log(f"target reached (step {s:,}); "
                    f"letting the arena finish ({args.arena_grace}s).")
                time.sleep(args.arena_grace)
                break
            if sp.poll() is not None:
                log("self-play process exited early — moving on.")
                sp = None
                break
        stop_training(sp, "self-play(stage0)")
    else:
        log("already at/past target step — skipping stage-0 self-play.")

    # ── Stage 1: wait for the Stockfish harvest ──────────────────────────
    if sf_proc is not None:
        log("waiting for the Stockfish harvest to finish...")
        try:
            sf_proc.wait()
        except Exception as e:
            log(f"harvest wait error: {e}")
        log("Stockfish harvest done." if sf_buf.exists()
            else "harvest produced no buffer.")

    # ── Stage 2: distil ──────────────────────────────────────────────────
    if sf_buf.exists():
        run_to_completion(
            [PY, "-u", "main.py", "--phase=sf_distill",
             "--sf-buffer=replay_buffer/sf_distill.pkl",
             f"--sup-epochs={args.sup_epochs}", *resume_arg()],
            "sf_distill")
    else:
        log("no Stockfish buffer — skipping distillation.")

    # ── Stage 3: puzzles (optional, tolerant) ────────────────────────────
    if not args.skip_puzzles:
        if not pz_buf.exists():
            run_to_completion(
                [PY, "-u", "tools/lichess_puzzles.py",
                 "--n-puzzles", args.puzzle_count,
                 "--out", "replay_buffer/lichess_puzzles.pkl"],
                "puzzle-gen", timeout=5400)
        if pz_buf.exists():
            run_to_completion(
                [PY, "-u", "main.py", "--phase=puzzles",
                 "--puzzle-buffer=replay_buffer/lichess_puzzles.pkl",
                 f"--sup-epochs={args.sup_epochs}", *resume_arg()],
                "puzzles")
        else:
            log("no puzzle buffer (download/gen failed) — skipping puzzle train.")

    # ── Stage 4: resume self-play (resilient) until morning ──────────────
    log("Stage 4: resuming self-play (supervisor). Stop with Ctrl+C when you wake.")
    final = spawn([PY, "-u", "tools/train_supervisor.py", "--phase=all"])
    try:
        final.wait()
    except KeyboardInterrupt:
        log("Ctrl+C — stopping final self-play.")
        stop_training(final, "self-play(final)", grace=130)
    log("OVERNIGHT PIPELINE END.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
