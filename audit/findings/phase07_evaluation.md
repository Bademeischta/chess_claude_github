# Phase 7 — Evaluation & ELO Measurement System

Files: trainer.run_arena, main.py arena wiring, training/pbt.py, utils/elo.py, tools/elo_vs_stockfish.py, checkpoints/pool_state.json, checkpoints/elo_history.csv, TB tags.

## 7.1 Verdict: strength measurement is non-functional end-to-end

### Internal arena (self-rating) — FINDING P7-1 (HIGH, = P5-6, S2/T0.3 RESOLVED)
Five stacked defects:
1. main.py:663 `if trainer.global_step % CONFIG.arena_every_n_steps == 0` — global_step advances in chunks of ~10–25 per batch; exact multiples of 10,000 are almost never hit. Same for pool registration (% 5,000). Evidence: TB contains NO elo/* tags at all; elo_history.csv has header only; pool holds just net_step0 + net_step35000 after 63,400 steps.
2. main.py:1081 `pool.register_current(model, step=0, elo=initial)` runs at EVERY process start BEFORE checkpoint resume → checkpoints/net_step0.pt is overwritten with freshly-initialized random weights on every (re)start — including every supervisor crash-restart. The pool's reference opponent is not what it claims to be. (Also: `main.py --test` does this too — running the integration tests corrupts the pool entry.)
3. main.py:669 `pool.update_elo(pool._current_id, "best_pool", ...)` rates a phantom player literally named "best_pool"; the actual best entry's rating never updates.
4. utils/elo.update_from_series collapses an entire 100-game series into ONE K=32 update → max ±32 ELO per arena even on a 100–0 sweep; phase targets (1600/2200/2400) could never be reached from 1000 even if arena ran every time. Phase transitions therefore only occur via max_games (50k/200k/500k games ≈ months) — the phase ladder documented in the README effectively does not gate on strength.
5. run_arena evaluates WITH root Dirichlet noise (add_noise defaults True, not overridden — trainer.py:815) and τ=0.1 sampling instead of argmax (P5-5).

### External ELO (tools/elo_vs_stockfish.py) — FINDING P7-2 (HIGH)
- `--auto` mode (the recommended path, used by overnight pipeline docs) calls `_our_move(tree, ob, sims)` at line 194 — **the function does not exist anywhere** (ruff F821). NameError on the first own move → auto-ELO has never produced a result.
- `_play_batch` also calls `tree.reset(ob)` after EVERY ply (line 200) → discards the subtree AND truncates GRU history to 1 frame — the net would be evaluated on out-of-distribution inputs (zero-padded history).
- The fixed-anchor path (--elo N, lines 307-369) is intact and correctly: alternates colors, claim_draw=True, advances tree for both sides, greedy selection, no noise, Wald CI. **This is the one working strength probe in the repo.**
- Fair-time calibration assumes 25 ms/sim; measured: 16 ms per 128-sim move (CUDA-graph) = 0.125 ms/sim → the tool grants Stockfish ~200× the model's actual think time at default settings (sims*0.025 s = 3.2 s vs our 16 ms). ELO estimates from defaults would be strongly PESSIMISTIC. Use --movetime ~0.05 or --depth for fairness.

## 7.2 ELO estimation framework recommendation
Short-term: fix P7-2 (--auto NameError + reset-per-ply + movetime calibration), then calibrate vs SF Skill 0–5 with ≥100 games at fixed anchor; SPRT optional later. Internal arena: gate on `global_step // arena_every != last_arena_step // arena_every` (crossing-based trigger), evaluate noise-free/argmax, per-game ELO updates or proper series math (e.g. BayesElo-style), stable pool ids.

## 7.3 Evaluation game quality
No PGN export/game logging exists for arena or self-play (web GUI shows live games only) → recommend dumping arena PGNs (cheap: python-chess already a dependency). Cannot check historic eval games — none were recorded.

## Status of measured strength
Objective strength of step-63400 weights is UNKNOWN. Given TB evidence (policy flat at CE≈3.2 since step 4k; value head predicting ≈0 everywhere), expectation is well below SF UCI floor (1320), i.e. only measurable via --sf-skill 0-3. A short fixed-anchor probe is possible post-audit if desired ([SKIPPING: running a 30-game SF match now — ~30+ min GPU/CPU for a number the curves already bound qualitatively; deferred to roadmap item with the fixed tool]).
