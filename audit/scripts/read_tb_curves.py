"""Audit T0.2/T5.1: reconstruct training curves from committed TB event files.

Read-only on runs/. Writes JSON summary next to itself.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

runs = ROOT / "runs"
files = sorted(runs.rglob("events.out.tfevents.*"))
print(f"{len(files)} event files")

series: dict[str, dict[int, float]] = defaultdict(dict)
for f in files:
    try:
        acc = EventAccumulator(str(f), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            for ev in acc.Scalars(tag):
                series[tag][ev.step] = ev.value  # later files override dups
    except Exception as e:  # noqa: BLE001
        print(f"skip {f.name}: {type(e).__name__}: {e}")

summary = {}
for tag, pts in sorted(series.items()):
    steps = sorted(pts)
    n = len(steps)
    if n == 0:
        continue
    # downsample to ~80 points for the report
    sel = steps[:: max(1, n // 80)]
    summary[tag] = {
        "n_points": n,
        "step_range": [steps[0], steps[-1]],
        "samples": [[s, round(pts[s], 5)] for s in sel],
    }

# focused extraction around the documented 30k collapse for loss/value
for tag in ("loss/value", "loss/total", "loss/aux", "loss/policy",
            "policy_entropy"):
    if tag in series:
        pts = series[tag]
        window = {s: round(v, 5) for s, v in sorted(pts.items())
                  if 25000 <= s <= 40000}
        sel = list(window.items())[:: max(1, len(window) // 40)]
        summary.setdefault("_collapse_window_25k_40k", {})[tag] = sel

out = Path(__file__).with_name("tb_curves.json")
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"tags: {list(series.keys())}")
print(f"-> {out}")
