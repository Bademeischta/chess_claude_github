import json
from pathlib import Path

d = json.load(open(Path(__file__).with_name("tb_curves.json")))
tags = ["loss/total", "loss/policy", "loss/value", "loss/aux",
        "loss/teacher_kl", "policy_entropy", "throughput/positions_per_sec",
        "throughput/games_per_hour", "replay_buf_size", "lr",
        "dirichlet_eps", "alpha_v_global"]
for tag in tags:
    if tag not in d:
        print(f"{tag}: MISSING")
        continue
    s = d[tag]["samples"]
    pick = [s[0], s[len(s) // 4], s[len(s) // 2], s[3 * len(s) // 4], s[-1]]
    n, rng = d[tag]["n_points"], d[tag]["step_range"]
    print(f"{tag:38s} n={n:6d} steps={rng}")
    print("    " + "  ".join(f"@{p[0]}:{p[1]:.4g}" for p in pick))
w = d.get("_collapse_window_25k_40k", {})
for tag, sel in w.items():
    line = "  ".join(f"@{s}:{v:.4g}" for s, v in sel[:25])
    print(f"window25-40k {tag}: {line}")
