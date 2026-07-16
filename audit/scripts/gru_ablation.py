"""
audit/scripts/gru_ablation.py — the S13 experiment the audits kept asking
for: does the GRU history encoder earn its cost?

Two SEQUENTIAL from-scratch runs (identical seeds, identical config
except `use_gru_history`), each with a fixed WALLCLOCK budget, then:
  * throughput:   self-play positions/s, median train-step ms
  * strength (1): objective SF-anchor match per variant (skill 0 ≈ 800)
  * strength (2): 60-game head-to-head arena nogru vs gru (the sharpest
                  comparator — same data budget, direct opposition)

ISOLATION: everything under audit/tmp/gru_ablation/<variant>/ — never
touches checkpoints/, runs/, replay_buffer/. Results land in
audit/scripts/gru_ablation.json AND audit/findings/gru_ablation_2026-07.md
(written by THIS script so the numbers exist even if nobody babysits it).

Run (overnight):  python audit/scripts/gru_ablation.py --hours 4
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe

SEED = 20260715


def _seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    try:
        import chess_ext
        chess_ext.set_rng_seed(SEED)  # audit S9 — C++ sampling/noise too
    except (ImportError, AttributeError):
        pass


def run_variant(name: str, use_gru: bool, hours: float, sims: int) -> dict:
    from model.network import build_model
    from training.replay_buffer import PrioritizedReplayBuffer, TeacherBuffer
    from training.self_play import SelfPlayWorker
    from training.trainer import Trainer
    from training.pbt import OpponentPool
    from utils.logger import TrainingLogger

    base = ROOT / "audit" / "tmp" / "gru_ablation" / name
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base / "runs").mkdir(parents=True, exist_ok=True)

    CONFIG.checkpoint_dir = str(base / "checkpoints")
    CONFIG.tensorboard_dir = str(base / "runs")
    CONFIG.use_gru_history = use_gru
    CONFIG.mcts_sims = sims                 # bootstrap regime, both variants
    CONFIG.mcts_sims_teacher = sims
    CONFIG.replay_start_training = 5_000
    CONFIG.checkpoint_every = 10_000_000    # manual saves only
    CONFIG.teacher_game_ratio = 0.0
    CONFIG.sf_elo_every_n_steps = 0         # no probes inside the loop

    _seed_all()
    device = torch.device(CONFIG.device)
    model = build_model(CONFIG, compile_model=False)
    replay = PrioritizedReplayBuffer(capacity=150_000, alpha=CONFIG.per_alpha,
                                     beta_start=CONFIG.per_beta_start,
                                     beta_end=CONFIG.per_beta_end,
                                     total_steps=CONFIG.total_steps)
    teacher = TeacherBuffer(capacity=1_000)
    worker = SelfPlayWorker(CONFIG, model, replay, teacher, device)
    pool = OpponentPool(checkpoint_dir=CONFIG.checkpoint_dir)
    logger = TrainingLogger(log_dir=CONFIG.tensorboard_dir, log_every=200)
    trainer = Trainer(CONFIG, model, replay, teacher, pool, logger, device)

    deadline = time.time() + hours * 3600.0
    t0 = time.time()
    gen_secs = 0.0
    positions = 0
    games = 0
    step_times: list[float] = []
    last_report = t0

    print(f"[{name}] start: use_gru={use_gru} sims={sims} "
          f"budget={hours:.1f}h parallel={CONFIG.parallel_games}", flush=True)
    while time.time() < deadline:
        tg = time.time()
        n_new = worker.generate_batch(n_games=CONFIG.parallel_games)
        gen_secs += time.time() - tg
        positions += n_new
        games += CONFIG.parallel_games
        if len(replay) >= CONFIG.replay_start_training:
            n_steps = max(10, int(n_new * 4 / CONFIG.batch_size))
            for _ in range(n_steps):
                ts = time.perf_counter()
                trainer.train_step()
                step_times.append(time.perf_counter() - ts)
        if time.time() - last_report > 300:
            print(f"[{name}] t={time.time()-t0:6.0f}s step={trainer.global_step} "
                  f"pos={positions} ({positions/max(1e-9,gen_secs):.1f} pos/s gen)",
                  flush=True)
            last_report = time.time()

    ckpt = str(base / "checkpoints" / f"{name}_final.pt")
    trainer.save_checkpoint(ckpt)
    res = {
        "variant": name, "use_gru": use_gru,
        "wallclock_h": round((time.time() - t0) / 3600.0, 2),
        "steps": trainer.global_step,
        "games": games, "positions": positions,
        "selfplay_pos_per_s": round(positions / max(1e-9, gen_secs), 1),
        "median_step_ms": round(statistics.median(step_times) * 1e3, 1)
        if step_times else None,
        "final_losses": {k: round(v, 4)
                         for k, v in (trainer.train_step() or {}).items()
                         if k.startswith("loss/")},
        "checkpoint": ckpt,
        "cfg_variant": dataclasses.asdict(CONFIG) | {},
    }
    res["cfg_variant"] = {k: res["cfg_variant"][k] for k in
                          ("use_gru_history", "mcts_sims", "batch_size",
                           "parallel_games", "num_res_blocks", "num_channels")}
    logger.close()
    del trainer, worker, replay, teacher, pool
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{name}] done: {json.dumps(res, default=str)[:400]}", flush=True)
    return res, model


def sf_probe(model, cfg, games: int, sims: int) -> dict:
    from utils.sf_anchor import play_anchor_match
    r = play_anchor_match(model, torch.device(cfg.device), cfg,
                          games=games, sf_skill=0, sims=sims, batched_bs=32)
    return r or {"error": "no stockfish"}


def head_to_head(model_a, model_b, cfg, games: int) -> dict:
    """60-game arena: A = nogru (current), B = gru (opponent)."""
    from training.trainer import Trainer
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.device = torch.device(cfg.device)
    w, d, l = trainer.run_arena(model_a, model_b, games,
                                max_moves=160, batched_bs=32)
    return {"nogru_wins": w, "draws": d, "gru_wins": l, "games": w + d + l}


def _load_finished_variant(name: str, use_gru: bool, stats_json: str):
    """Resume support: rebuild a completed variant's model from its saved
    checkpoint + its recorded stats (recovered from the run log), instead
    of burning another wallclock budget. Added after the 2026-07-15 night
    run was killed between variants by a session teardown."""
    from model.network import build_model
    from engine.harness import load_state_into

    res = json.loads(Path(stats_json).read_text(encoding="utf-8"))
    assert res.get("variant") == name, f"stats file is not for '{name}'"
    CONFIG.use_gru_history = use_gru
    _seed_all()
    device = torch.device(CONFIG.device)
    model = build_model(CONFIG, compile_model=False).eval()
    load_state_into(model, res["checkpoint"], device)
    print(f"[{name}] resumed from {res['checkpoint']} "
          f"(steps={res.get('steps')})", flush=True)
    return res, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=4.0,
                    help="wallclock budget PER VARIANT (default 4h)")
    ap.add_argument("--sims", type=int, default=50,
                    help="self-play sims/move (bootstrap regime default 50)")
    ap.add_argument("--probe-games", type=int, default=40)
    ap.add_argument("--h2h-games", type=int, default=60)
    ap.add_argument("--gru-done", default=None, metavar="STATS_JSON",
                    help="skip the GRU variant: path to its recovered "
                         "result-dict JSON (checkpoint is loaded from the "
                         "'checkpoint' field for the measurements)")
    args = ap.parse_args()

    update_config_from_dict(run_probe(write=False))

    out: dict = {"seed": SEED, "hours_per_variant": args.hours,
                 "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    if args.gru_done:
        res_gru, model_gru = _load_finished_variant("gru", True, args.gru_done)
    else:
        res_gru, model_gru = run_variant("gru", True, args.hours, args.sims)
    out["gru"] = res_gru
    res_nog, model_nog = run_variant("nogru", False, args.hours, args.sims)
    out["nogru"] = res_nog

    print("[ablation] SF anchor probes...", flush=True)
    # cfg currently holds the nogru variant; flip per probe.
    CONFIG.use_gru_history = True
    out["gru"]["sf_probe"] = sf_probe(model_gru, CONFIG, args.probe_games, 128)
    CONFIG.use_gru_history = False
    out["nogru"]["sf_probe"] = sf_probe(model_nog, CONFIG, args.probe_games, 128)

    print("[ablation] head-to-head arena...", flush=True)
    out["head_to_head"] = head_to_head(model_nog, model_gru, CONFIG,
                                       args.h2h_games)

    jpath = Path(__file__).with_name("gru_ablation.json")
    jpath.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    # ── findings doc (written by the experiment itself) ──────────────────
    g, n, hh = out["gru"], out["nogru"], out["head_to_head"]

    def _elo(p):
        return (f"{p.get('estimated_elo', float('nan')):.0f} "
                f"(CI {p.get('est_lo', 0):.0f}..{p.get('est_hi', 0):.0f}, "
                f"W{p.get('wins', '?')}/D{p.get('draws', '?')}/L{p.get('losses', '?')})"
                if "estimated_elo" in p else str(p))

    md = f"""# GRU-Ablation (audit S13) — {time.strftime('%Y-%m-%d')}

Zwei from-scratch-Läufe, identische Seeds ({SEED}) und Config bis auf
`use_gru_history`, je {args.hours:.1f} h Wallclock, {args.sims} Sims/Zug
(Bootstrap-Regime), danach SF-Anker (Skill 0, {args.probe_games} Partien,
128 Sims) und {args.h2h_games}-Partien-Direktvergleich. Treiber:
audit/scripts/gru_ablation.py; Rohdaten: gru_ablation.json.

| Metrik | GRU an | GRU aus |
|---|---|---|
| Trainingssteps | {g['steps']} | {n['steps']} |
| Self-Play-Positionen | {g['positions']} | {n['positions']} |
| Self-Play pos/s | {g['selfplay_pos_per_s']} | {n['selfplay_pos_per_s']} |
| Median train-step (ms) | {g['median_step_ms']} | {n['median_step_ms']} |
| SF-Anker-Elo | {_elo(g.get('sf_probe', {}))} | {_elo(n.get('sf_probe', {}))} |

**Direktvergleich (nogru vs gru, {hh.get('games', '?')} Partien):**
nogru W{hh.get('nogru_wins', '?')} / D{hh.get('draws', '?')} / gru W{hh.get('gru_wins', '?')}

## Interpretation (bitte nach Sichtung ergänzen/bestätigen)

- Durchsatz: pos/s-Verhältnis nogru/gru = {n['selfplay_pos_per_s'] / max(0.001, g['selfplay_pos_per_s']):.2f}×.
- Entscheidung GRU behalten/entfernen anhand: (a) Durchsatzgewinn,
  (b) Direktvergleich, (c) SF-Anker. Gleichstand bei (b)/(c) + Gewinn bei
  (a) ⇒ GRU entfernen (audit-S13-Empfehlung wäre damit bestätigt).
"""
    fpath = ROOT / "audit" / "findings" / "gru_ablation_2026-07.md"
    fpath.write_text(md, encoding="utf-8")
    print(f"[ablation] -> {jpath}\n[ablation] -> {fpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
