"""Run `main.py --test` with checkpoint/tensorboard dirs redirected into
audit/tmp so the integration suite cannot touch production state."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TMP = ROOT / "audit" / "tmp"
(TMP / "checkpoints").mkdir(parents=True, exist_ok=True)
(TMP / "runs").mkdir(parents=True, exist_ok=True)

from config import CONFIG  # noqa: E402

CONFIG.checkpoint_dir = str(TMP / "checkpoints")
CONFIG.tensorboard_dir = str(TMP / "runs")

import main  # noqa: E402

sys.argv = ["main.py", "--test"]
main.main()
