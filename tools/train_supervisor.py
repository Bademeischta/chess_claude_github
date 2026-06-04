#!/usr/bin/env python3
"""
tools/train_supervisor.py — crash-resilient training wrapper.

Runs `main.py` as a subprocess and restarts it automatically whenever it
dies from a native (non-Python-catchable) crash. The chess_ext C++
extension has at least one known memory bug that triggers a Windows
access violation on certain mate-in-2 positions; that crash takes the
whole Python process down with no chance to handle it in-process. This
wrapper catches the non-zero exit, finds the newest checkpoint, and
relaunches main.py from there.

Behaviour:
    * Picks the newest checkpoint (latest.pt OR step_*.pt, by mtime) on
      every (re)start. First start with no checkpoints = no --resume flag.
    * Exit code 0 from main.py = normal completion → supervisor stops.
    * Exit code != 0 = crash → wait `--cooldown` seconds, restart.
    * Hard cap via `--max-restarts` so a deterministic crash loop bails
      out instead of burning CPU forever.
    * Crash log is appended to `runs/supervisor.log` with timestamps and
      the last ~10 lines of output (handy for diagnosing the trigger FEN).

Usage:
    python tools/train_supervisor.py --phase=all
    python tools/train_supervisor.py --phase=all --max-restarts=20 --cooldown=10
    python tools/train_supervisor.py -- --syzygy=C:\\path\\to\\syzygy

Any argument that does NOT start with `--max-restarts`, `--cooldown`,
`--phase`, or `--no-resume` is forwarded to main.py verbatim. (The `--`
sentinel form also works for full pass-through.)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_CKPT_DIR = _ROOT / "checkpoints"
_LOG_DIR = _ROOT / "runs"
_LOG_FILE = _LOG_DIR / "supervisor.log"


def _newest_checkpoint() -> Path | None:
    """Find the newest checkpoint to resume from.

    Preference order:
      1. `latest.pt` — emergency snapshot written between batches (≤ 3
         batches of work since the last successful checkpoint).
      2. The highest-step `step_*.pt` file.

    Both must exist + be readable; falls back transparently if not.
    """
    if not _CKPT_DIR.exists():
        return None

    latest = _CKPT_DIR / "latest.pt"
    step_files = sorted(_CKPT_DIR.glob("step_*.pt"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True)

    candidates: list[Path] = []
    if latest.exists():
        candidates.append(latest)
    if step_files:
        candidates.append(step_files[0])

    if not candidates:
        return None

    # Pick the one most recently written.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _append_log(line: str) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run_once(forward_args: list[str], no_resume: bool) -> tuple[int, deque[str]]:
    """Run one main.py instance, return (exit_code, last_output_lines)."""
    cmd = [sys.executable, "-u", "main.py"]
    cmd.extend(forward_args)

    if not no_resume:
        ckpt = _newest_checkpoint()
        if ckpt is not None:
            cmd.append(f"--resume={ckpt.as_posix()}")
            print(f"[supervisor] resuming from {ckpt.name} "
                  f"(mtime={datetime.fromtimestamp(ckpt.stat().st_mtime)})",
                  flush=True)
        else:
            print("[supervisor] no checkpoint found — fresh start.", flush=True)

    print(f"[supervisor] launching: {' '.join(cmd)}", flush=True)

    # Keep a rolling buffer of recent stdout/stderr lines for crash diagnosis.
    tail: deque[str] = deque(maxlen=40)

    # Inherit stdout so the user sees the live training output. We tee
    # through Popen.stdout iteration to also capture the tail.
    proc = subprocess.Popen(
        cmd,
        cwd=str(_ROOT),
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
        for line in proc.stdout:
            tail.append(line.rstrip("\n"))
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[supervisor] Ctrl+C received — forwarding SIGINT to main.py "
              "and waiting up to 120s for it to save a checkpoint...",
              flush=True)
        try:
            proc.send_signal(2)  # SIGINT cross-platform; main.py saves a ckpt
        except Exception:
            pass
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            print("[supervisor] main.py did not exit in time — killing.",
                  flush=True)
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()

    proc.wait()
    if interrupted:
        # Signal the main loop to stop without re-raising — we already
        # handled the interrupt cleanly above, no need for a traceback.
        return -1, tail
    return proc.returncode, tail


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Crash-resilient training wrapper around main.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max-restarts", type=int, default=50,
                    help="Max consecutive crashes before the supervisor "
                         "gives up. Default: 50.")
    ap.add_argument("--cooldown", type=float, default=10.0,
                    help="Seconds to wait between crash and restart. "
                         "Default: 10.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Start fresh — do NOT pass --resume on the first "
                         "launch. Subsequent restarts still resume from the "
                         "newest checkpoint.")
    # Accept (but don't consume) main.py arguments. Anything we don't know
    # gets forwarded verbatim.
    args, forward_args = ap.parse_known_args()

    # If the user used the `--` sentinel, argparse already stripped it.
    # forward_args is the rest.

    no_resume_first = args.no_resume

    start_time = time.time()
    restarts = 0
    consecutive_crashes = 0

    _append_log(f"\n=== supervisor start {datetime.now().isoformat()} === "
                f"args={forward_args!r}")

    while True:
        try:
            exit_code, tail = _run_once(forward_args, no_resume=no_resume_first)
        except KeyboardInterrupt:
            print("\n[supervisor] interrupted before subprocess finished — exiting.",
                  flush=True)
            return 0
        # After the first launch, ALWAYS resume on subsequent restarts.
        no_resume_first = False

        if exit_code == 0:
            print("[supervisor] main.py exited cleanly. Done.", flush=True)
            _append_log(f"[{datetime.now().isoformat()}] clean exit "
                        f"after {restarts} restart(s)")
            return 0

        if exit_code == -1:
            # Marker from _run_once: user pressed Ctrl+C and we already
            # forwarded SIGINT + waited for main.py to checkpoint. Don't
            # treat that as a crash; just exit cleanly.
            print("[supervisor] clean shutdown after user interrupt.",
                  flush=True)
            _append_log(f"[{datetime.now().isoformat()}] user interrupt "
                        f"after {restarts} restart(s)")
            return 0

        consecutive_crashes += 1
        restarts += 1

        # Pull the last few lines as a crash hint.
        tail_str = "\n  | ".join(list(tail)[-12:])
        elapsed_h = (time.time() - start_time) / 3600.0
        crash_msg = (
            f"[{datetime.now().isoformat()}] CRASH exit={exit_code} "
            f"restart={restarts} consecutive={consecutive_crashes} "
            f"uptime_h={elapsed_h:.2f}\n"
            f"  tail:\n  | {tail_str}"
        )
        print("\n" + "=" * 60, flush=True)
        print(f"[supervisor] main.py died with exit code {exit_code} "
              f"(restart #{restarts}).", flush=True)
        print("=" * 60, flush=True)
        _append_log(crash_msg)

        if restarts >= args.max_restarts:
            print(f"[supervisor] hit --max-restarts={args.max_restarts}. "
                  f"Bailing out — this looks like a deterministic crash, "
                  f"not a transient one. Investigate runs/supervisor.log.",
                  flush=True)
            return 2

        # Progressive cooldown: if we crashed twice within 60 seconds, the
        # restart loop itself might be aggravating something — slow down.
        cooldown = args.cooldown
        if consecutive_crashes >= 5:
            cooldown = max(cooldown, 60.0)
        print(f"[supervisor] cooling down {cooldown:.0f}s before restart...",
              flush=True)
        try:
            time.sleep(cooldown)
        except KeyboardInterrupt:
            print("[supervisor] Ctrl+C during cooldown — exiting.", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
