#!/usr/bin/env python3
"""
tools/install_stockfish.py — Download and install Stockfish for ELO measurement.

Downloads the official Stockfish binary from the GitHub release page, verifies
it with a UCI handshake, and patches `config.stockfish_path` so the training
pipeline and `tools/elo_vs_stockfish.py` find it without a manual --stockfish
argument.

Usage:
    python tools/install_stockfish.py             # idempotent — skips if installed
    python tools/install_stockfish.py --force     # re-download even if present
    python tools/install_stockfish.py --quiet     # only error output

Stockfish is GPL-3.0. We download (not redistribute) directly from
github.com/official-stockfish/Stockfish releases.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# Pin to a specific Stockfish release so we can verify SHA256 hashes against
# a known-good table. /releases/latest/download was convenient but defeats
# any integrity check (the asset bytes can change underneath us). Bump
# _SF_RELEASE_TAG + _TRUSTED_SHA256 together to upgrade.
_SF_RELEASE_TAG = "sf_17.1"
_RELEASE_BASE = (
    "https://github.com/official-stockfish/Stockfish/releases/download/"
    f"{_SF_RELEASE_TAG}"
)

# SHA-256 of the official Stockfish release assets for _SF_RELEASE_TAG.
# Generated from the upstream release page on 2026-05-27. If you bump the
# release tag you MUST refresh these — `sha256sum <file>` on each archive.
# An empty/missing entry triggers an interactive prompt instead of failing
# closed, so a new platform can still bootstrap (after the user confirms
# the printed hash matches the upstream release page out-of-band).
_TRUSTED_SHA256: dict[str, str] = {
    # Verified on 2026-05-27 via PowerShell `Get-FileHash … -Algorithm SHA256`
    # against the upstream release asset.
    "stockfish-windows-x86-64-avx2.zip":
        "92a77f8d8116b4331696eeb7b232bd03db30d6641a6ce1be1759478b8931d28b",
    # Linux/macOS hashes — add by running:
    #   curl -L https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/<asset> | sha256sum
    # then pasting the hex digest below.
    # "stockfish-ubuntu-x86-64-avx2.tar": "...",
    # "stockfish-macos-x86-64-avx2.tar": "...",
    # "stockfish-macos-m1-apple-silicon.tar": "...",
}


def _asset_for_platform() -> tuple[str, list[str]]:
    """Return (asset filename, list of candidate inner binary names).

    Returns multiple candidates because the layout inside the archive differs
    between releases.
    """
    sysname = platform.system()
    machine = platform.machine().lower()
    if sysname == "Windows":
        # AVX2 covers everything since Haswell (2013); fallback to plain x86-64
        # if the AVX2 build is unavailable for some release.
        return "stockfish-windows-x86-64-avx2.zip", [
            "stockfish-windows-x86-64-avx2.exe",
            "stockfish-windows-x86-64.exe",
            "stockfish.exe",
        ]
    if sysname == "Linux":
        return "stockfish-ubuntu-x86-64-avx2.tar", [
            "stockfish-ubuntu-x86-64-avx2",
            "stockfish-ubuntu-x86-64",
            "stockfish",
        ]
    if sysname == "Darwin":
        if "arm" in machine or "aarch" in machine:
            return "stockfish-macos-m1-apple-silicon.tar", [
                "stockfish-macos-m1-apple-silicon",
                "stockfish",
            ]
        return "stockfish-macos-x86-64-avx2.tar", [
            "stockfish-macos-x86-64-avx2",
            "stockfish",
        ]
    raise RuntimeError(f"Unsupported platform: {sysname} / {machine}")


def _download(url: str, dest: Path, quiet: bool = False, retries: int = 3) -> None:
    """Download `url` to `dest` with bounded retries on transient errors.

    Transient errors (URLError, socket timeout, partial download) trigger
    exponential backoff up to `retries` times. HTTPError is re-raised
    immediately because the caller switches assets on 404.
    """
    if not quiet:
        print(f"[install_stockfish] Downloading {url}")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                with dest.open("wb") as f:
                    shutil.copyfileobj(resp, f)
            if not quiet:
                size_mb = dest.stat().st_size / 1024 / 1024
                print(f"[install_stockfish] Downloaded {size_mb:.1f} MB → {dest}")
            return
        except urllib.error.HTTPError:
            # 4xx/5xx — semantic failure, do not retry blindly
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
            if attempt < retries:
                wait = 2 ** (attempt - 1)
                if not quiet:
                    print(f"[install_stockfish] Network error "
                          f"({type(e).__name__}: {e}); retry {attempt}/{retries} "
                          f"in {wait}s…")
                time.sleep(wait)
            else:
                raise
    # Unreachable, but keeps the type checker happy.
    if last_err is not None:
        raise last_err


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 of `path` streaming `chunk` bytes at a time."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _verify_hash(
    archive: Path,
    asset: str,
    *,
    allow_unverified: bool,
    quiet: bool = False,
) -> bool:
    """Verify the SHA-256 of `archive` against the trusted table.

    Returns True on success (hash matched OR hash table empty + allow_unverified).
    Returns False if the hash mismatches, OR is missing and the user did NOT
    pass --allow-unverified. The caller MUST treat False as a fatal error
    and delete the archive without extracting it.
    """
    actual = _sha256_file(archive)
    expected = _TRUSTED_SHA256.get(asset)
    if expected is None:
        msg = (f"[install_stockfish] No trusted SHA-256 known for asset "
               f"{asset!r} (release {_SF_RELEASE_TAG}). "
               f"Computed: sha256={actual}.")
        if allow_unverified:
            if not quiet:
                print(msg + "  ALLOWED (--allow-unverified).")
            return True
        print(msg, file=sys.stderr)
        print("[install_stockfish] Refusing to install an unverified binary. "
              "Verify the hash against the upstream release page out-of-band, "
              "then add it to _TRUSTED_SHA256 (or re-run with "
              "--allow-unverified at your own risk).", file=sys.stderr)
        return False
    if actual.lower() != expected.lower():
        print(f"[install_stockfish] *** SHA-256 MISMATCH for {asset} ***",
              file=sys.stderr)
        print(f"  expected: {expected}", file=sys.stderr)
        print(f"  actual  : {actual}", file=sys.stderr)
        print("[install_stockfish] Refusing to install — the downloaded file "
              "does NOT match the pinned release. Possible causes: "
              "release re-publish, network interception, mirror corruption.",
              file=sys.stderr)
        return False
    if not quiet:
        print(f"[install_stockfish] SHA-256 verified for {asset}")
    return True


def _extract(archive: Path, out_dir: Path, quiet: bool = False) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    elif archive.suffix in (".tar", ".gz", ".tgz"):
        with tarfile.open(archive) as t:
            t.extractall(out_dir)
    else:
        raise RuntimeError(f"Unknown archive type: {archive.suffix}")
    if not quiet:
        print(f"[install_stockfish] Extracted to {out_dir}")


def _locate_binary(search_root: Path, candidates: list[str]) -> Path | None:
    """Search recursively for the binary by candidate names."""
    for name in candidates:
        hits = list(search_root.rglob(name))
        if hits:
            return hits[0]
    # Last resort: any file containing "stockfish" in the name, executable.
    for p in search_root.rglob("*stockfish*"):
        if p.is_file():
            return p
    return None


def _verify_uci(binary: Path, quiet: bool = False) -> str | None:
    """Run a UCI handshake. Return engine ID string on success, None on failure."""
    try:
        proc = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as e:
        if not quiet:
            print(f"[install_stockfish] Cannot run binary: {e}")
        return None

    try:
        assert proc.stdin and proc.stdout
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        engine_id = None
        # Read up to ~50 lines waiting for 'uciok'
        for _ in range(80):
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("id name"):
                engine_id = line[len("id name "):]
            if line == "uciok":
                proc.stdin.write("quit\n")
                proc.stdin.flush()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return engine_id or "Stockfish"
        # No uciok reached
        proc.kill()
        return None
    except (OSError, BrokenPipeError) as e:
        if not quiet:
            print(f"[install_stockfish] UCI handshake error: {e}")
        try:
            proc.kill()
        except Exception:
            pass
        return None


def _patch_config_path(stockfish_path: Path, quiet: bool = False) -> None:
    """Persist stockfish_path into config.py (line-based patch, no parser)."""
    config_path = _root / "config.py"
    if not config_path.exists():
        if not quiet:
            print(f"[install_stockfish] config.py not found at {config_path}")
        return

    src = config_path.read_text(encoding="utf-8").splitlines()
    patched = list(src)
    # Use forward slashes — works on Windows for Python str literals too.
    rel = stockfish_path.as_posix()
    for idx, line in enumerate(src):
        stripped = line.lstrip()
        if stripped.startswith("stockfish_path:") or stripped.startswith("stockfish_path "):
            indent = " " * (len(line) - len(stripped))
            patched[idx] = f'{indent}stockfish_path: str = "{rel}"'
            break
    config_path.write_text("\n".join(patched) + "\n", encoding="utf-8")
    if not quiet:
        print(f"[install_stockfish] config.stockfish_path → {rel}")


def install(
    force: bool = False,
    quiet: bool = False,
    allow_unverified: bool = False,
) -> Path | None:
    """Main entry. Returns path to installed binary, or None on failure.

    `allow_unverified` lets the caller proceed even when no trusted SHA-256
    is known for the platform's asset (e.g. when bootstrapping a new
    release tag). The downloaded hash is still printed so the operator can
    cross-check it against the upstream release page out-of-band.
    """
    install_dir = _root / "tools" / "stockfish"
    install_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent check: existing binary that passes UCI → done.
    existing_candidates = list(install_dir.rglob("stockfish*"))
    existing_bin = next(
        (p for p in existing_candidates if p.is_file() and os.access(p, os.X_OK)),
        None,
    )
    # On Windows there's no x-bit; fall back to .exe suffix.
    if existing_bin is None:
        existing_bin = next(
            (p for p in existing_candidates if p.is_file() and p.suffix == ".exe"),
            None,
        )

    if existing_bin and not force:
        engine_id = _verify_uci(existing_bin, quiet=quiet)
        if engine_id:
            if not quiet:
                print(f"[install_stockfish] Already installed: {engine_id}  ({existing_bin})")
            _patch_config_path(existing_bin, quiet=quiet)
            return existing_bin
        if not quiet:
            print("[install_stockfish] Existing binary failed UCI handshake — re-downloading.")

    asset, candidates = _asset_for_platform()
    url = f"{_RELEASE_BASE}/{asset}"
    archive = install_dir / asset

    try:
        _download(url, archive, quiet=quiet)
    except urllib.error.HTTPError as e:
        if not quiet:
            print(f"[install_stockfish] {url} → HTTP {e.code}; trying non-AVX2 fallback.")
        # Fallback: strip "-avx2" if present
        if "-avx2" in asset:
            asset = asset.replace("-avx2", "")
            url_fb = f"{_RELEASE_BASE}/{asset}"
            archive = install_dir / asset
            try:
                _download(url_fb, archive, quiet=quiet)
            except urllib.error.HTTPError as e2:
                print(f"[install_stockfish] Fallback also failed: HTTP {e2.code}")
                return None
        else:
            return None
    except urllib.error.URLError as e:
        # Offline / DNS failure / TLS issue. Non-fatal for caller — they'll
        # retry next run.
        if not quiet:
            print(f"[install_stockfish] Network error: {e.reason}")
        return None

    # SHA-256 gate: refuse to extract anything that doesn't match the pinned
    # release. This is the integrity check between "GitHub served us bytes"
    # and "we trust those bytes enough to run them as a subprocess".
    if not _verify_hash(archive, asset,
                         allow_unverified=allow_unverified, quiet=quiet):
        try:
            archive.unlink()
        except OSError:
            pass
        return None

    _extract(archive, install_dir, quiet=quiet)

    try:
        archive.unlink()
    except OSError:
        pass

    binary = _locate_binary(install_dir, candidates)
    if binary is None:
        print(f"[install_stockfish] Could not locate binary inside {install_dir}")
        return None

    # Linux/macOS: ensure executable bit set
    if platform.system() != "Windows":
        st = binary.stat()
        binary.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    engine_id = _verify_uci(binary, quiet=quiet)
    if engine_id is None:
        print(f"[install_stockfish] Downloaded binary failed UCI handshake: {binary}")
        return None

    if not quiet:
        print(f"[install_stockfish] Installed: {engine_id}")
        print(f"[install_stockfish] Binary    : {binary}")
        print("[install_stockfish] License   : GPL-3.0  "
              "(https://github.com/official-stockfish/Stockfish)")

    _patch_config_path(binary, quiet=quiet)

    # Update the live CONFIG too (no-op if config module not yet loaded)
    try:
        from config import CONFIG  # type: ignore
        CONFIG.stockfish_path = binary.as_posix()
    except Exception:
        pass

    # Write sentinel so main.py knows the auto-trigger is done
    sentinel = _root / "runs" / ".stockfish_installed"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(binary.as_posix(), encoding="utf-8")

    return binary


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Stockfish for ELO measurement.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if Stockfish is already installed.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress non-error output.")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="Allow install when no trusted SHA-256 is pinned for "
                         "this platform's asset. The actual hash is still "
                         "printed for out-of-band verification. Use only when "
                         "bootstrapping a new release tag.")
    args = ap.parse_args()

    binary = install(
        force=args.force,
        quiet=args.quiet,
        allow_unverified=args.allow_unverified,
    )
    return 0 if binary is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
