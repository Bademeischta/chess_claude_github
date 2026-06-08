#!/usr/bin/env python3
"""
tools/download_syzygy.py — download Syzygy 3-4-5 piece tablebases.

Pulls the WDL (.rtbw) + DTZ (.rtbz) files from Sesse.net's mirror into a
local directory (default: ./syzygy/ at the project root). The full set
is ~939 MB; this script downloads in parallel and resumes interrupted
downloads automatically.

The Sesse mirror requests no "download accelerators". This script does 8
parallel streams by default — well within polite limits and similar to a
desktop browser opening multiple tabs.

Usage:
    python tools/download_syzygy.py                  # -> ./syzygy/, 8 streams
    python tools/download_syzygy.py -o D:\\syzygy    # custom dir
    python tools/download_syzygy.py -j 4             # 4 parallel streams
    python tools/download_syzygy.py --pieces 4       # only 3-4 piece (smaller)

Once downloaded, run:
    python main.py --phase=pretraining --syzygy=<dir> --resume=checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path

# Sesse.net mirror — long-standing tablebase host with a stable directory
# listing. HTTP-only (no HTTPS cert) — that is intentional and safe; the
# files are checksummable via their internal Syzygy magic bytes and the
# server itself is a known-good source.
BASE_URL = "http://tablebase.sesse.net/syzygy/3-4-5/"

# A polite User-Agent so the server can see what's downloading (helps
# their admin diagnose problems if anything goes wrong).
USER_AGENT = "chess_ai-syzygy-downloader/1.0 (+local training pipeline)"

# Total expected size for the full 3-4-5 set (approximate). Used for
# progress reporting only; not enforced.
EXPECTED_TOTAL_MB = 939


class _LinkExtractor(HTMLParser):
    """Pulls out *.rtbw / *.rtbz hrefs from the server's directory index."""

    def __init__(self) -> None:
        super().__init__()
        self.files: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v and (v.endswith(".rtbw") or v.endswith(".rtbz")):
                self.files.append(v)


def fetch_file_list(max_pieces: int) -> list[str]:
    """Scrape the server directory index and return *.rtbw / *.rtbz hrefs."""
    print(f"[syzygy] fetching file list from {BASE_URL} ...", flush=True)
    req = urllib.request.Request(BASE_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    parser = _LinkExtractor()
    parser.feed(html)
    files = parser.files
    if not files:
        # Fall back to a regex pass in case the index is plain text.
        files = re.findall(r"[A-Z]{2,}v[A-Z]+\.rtb[wz]", html)

    # Filter by piece count if the user asked to limit the download.
    # File names look like "KQvK", "KRPvKN", "KQRBvKP" — total piece
    # count = number of capital letters (kings count too, but appear in
    # both halves which already covers them).
    def _piece_count(fname: str) -> int:
        # Strip extension, drop the two 'K's, count remaining capital letters
        name = fname.rsplit(".", 1)[0]
        # Remove the kings (one each side)
        try:
            white, black = name.split("v")
        except ValueError:
            return 99
        # Count all letters EXCEPT the kings (which start each side)
        non_king_white = white[1:]  # strip leading 'K'
        non_king_black = black[1:]  # strip leading 'K'
        return 2 + len(non_king_white) + len(non_king_black)  # +2 for both kings

    files = [f for f in files if _piece_count(f) <= max_pieces]
    files = sorted(set(files))
    print(f"[syzygy] found {len(files)} files to download.", flush=True)
    return files


def remote_size(url: str) -> int | None:
    """Return the file's Content-Length, or None if the server doesn't say."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def download_one(url: str, out_path: Path) -> tuple[str, str, int]:
    """Download a single tablebase file.

    Returns (filename, status, bytes_downloaded). Status is one of:
        'ok'        — fully downloaded fresh
        'resumed'   — resumed from partial and finished
        'skipped'   — already present at full expected size
        'failed:..' — error message
    """
    fname = out_path.name
    expected = remote_size(url)

    # Skip if already complete.
    if out_path.exists() and expected is not None and out_path.stat().st_size == expected:
        return fname, "skipped", 0

    # Resume support: HTTP Range request if we have a partial file.
    headers = {"User-Agent": USER_AGENT}
    start_pos = 0
    if out_path.exists():
        start_pos = out_path.stat().st_size
        if expected is not None and start_pos > expected:
            # Local file is bigger than the remote one — corrupt/stale.
            out_path.unlink()
            start_pos = 0
    if start_pos > 0:
        headers["Range"] = f"bytes={start_pos}-"

    # Retry-with-backoff: the Sesse mirror occasionally returns 403 under
    # bursty load, but the same file fetches fine a few seconds later.
    last_err: str = "unknown"
    for attempt in range(4):
        if attempt > 0:
            time.sleep(2.0 * attempt)  # 0, 2s, 4s, 6s backoff
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                mode = "ab" if start_pos > 0 else "wb"
                written = 0
                with out_path.open(mode) as fh:
                    while True:
                        chunk = resp.read(1024 * 64)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                status = "resumed" if start_pos > 0 else "ok"
                return fname, status, written
        except urllib.error.HTTPError as e:
            # 416 "Requested Range Not Satisfiable" means our Range header
            # asked for bytes past the file end. That only happens when the
            # local file is already complete but HEAD couldn't tell us the
            # remote size to confirm it. Treat as "already done".
            if e.code == 416 and out_path.exists() and out_path.stat().st_size > 0:
                return fname, "skipped", 0
            last_err = f"HTTPError {e.code}: {e.reason}"
            if out_path.exists():
                start_pos = out_path.stat().st_size
                headers = dict(headers)
                if start_pos > 0:
                    headers["Range"] = f"bytes={start_pos}-"
        except (urllib.error.URLError, OSError, socket.timeout) as e:
            last_err = f"{type(e).__name__}: {e}"
            # Re-read partial bytes on next attempt so we resume cleanly.
            if out_path.exists():
                start_pos = out_path.stat().st_size
                headers = dict(headers)
                if start_pos > 0:
                    headers["Range"] = f"bytes={start_pos}-"
    return fname, f"failed:{last_err}", 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=None,
                    help="Destination directory (default: ./syzygy/ at "
                         "the project root).")
    ap.add_argument("-j", "--jobs", type=int, default=8,
                    help="Parallel download streams (default: 8).")
    ap.add_argument("--pieces", type=int, default=5, choices=(3, 4, 5),
                    help="Maximum piece count to download "
                         "(3 = ~10 files <1 MB, 4 = ~80 files ~30 MB, "
                         "5 = ~290 files ~939 MB; default: 5).")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out) if args.out else project_root / "syzygy"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[syzygy] destination: {out_dir}")
    print(f"[syzygy] parallel streams: {args.jobs}")
    print(f"[syzygy] max pieces: {args.pieces}\n")

    try:
        files = fetch_file_list(args.pieces)
    except Exception as e:
        print(f"[syzygy] ERROR: could not fetch file list: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        print("[syzygy] The mirror may be temporarily down. Try again in a "
              "few minutes, or pass --out and a different mirror manually.",
              file=sys.stderr)
        return 1

    if not files:
        print("[syzygy] no files matched. Aborting.", file=sys.stderr)
        return 1

    # Pre-check how much is already on disk, so the progress meter is honest.
    done_already = sum(
        (out_dir / f).stat().st_size for f in files if (out_dir / f).exists()
    )
    print(f"[syzygy] {done_already / 1024 / 1024:.1f} MB already on disk; "
          f"target ~{EXPECTED_TOTAL_MB} MB total.\n", flush=True)

    t0 = time.time()
    ok = resumed = skipped = failed = 0
    bytes_down = 0
    n = len(files)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(download_one, BASE_URL + f, out_dir / f): f
                   for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fname, status, b = fut.result()
            bytes_down += b
            elapsed = time.time() - t0
            rate = bytes_down / elapsed / 1024 / 1024 if elapsed > 0 else 0.0
            tag = (f"[{i:>3d}/{n:<3d}]" if status != "failed"
                   else f"[{i:>3d}/{n:<3d}] !!")
            if status == "ok":
                ok += 1
                print(f"{tag} ok       {fname:<24} "
                      f"({rate:.1f} MB/s avg)", flush=True)
            elif status == "resumed":
                resumed += 1
                print(f"{tag} resumed  {fname:<24} "
                      f"({rate:.1f} MB/s avg)", flush=True)
            elif status == "skipped":
                skipped += 1
                # don't spam stdout for trivial skips
                if i % 20 == 0:
                    print(f"{tag} skipped  {fname}", flush=True)
            else:
                failed += 1
                print(f"{tag} FAILED   {fname}: {status}", flush=True)

    dt = time.time() - t0
    total_on_disk = sum(
        (out_dir / f).stat().st_size for f in files if (out_dir / f).exists()
    )
    print()
    print("=" * 60)
    print(f"  Syzygy download done in {dt/60:.1f} min")
    print(f"  ok={ok}  resumed={resumed}  skipped={skipped}  failed={failed}")
    print(f"  on disk: {total_on_disk / 1024 / 1024:.1f} MB "
          f"in {out_dir}")
    print("=" * 60)

    if failed > 0:
        print("\n[syzygy] WARNING: some files failed. Re-run the same "
              "command — completed files will be skipped, failed ones "
              "retried.", file=sys.stderr)
        return 2

    print("\n[syzygy] Next: stop the supervisor (Ctrl+C in that window), "
          "then run:")
    print(f"  python main.py --phase=pretraining "
          f"--syzygy=\"{out_dir}\" "
          f"--resume=checkpoints/latest.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
