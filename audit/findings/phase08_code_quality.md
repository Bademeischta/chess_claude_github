# Phase 8 — Code Quality, Maintainability & Correctness

## 8.1 Dead code & dead features (the dominant quality problem)
Confirmed dead (no call sites, verified by grep):
1. `PrioritizedReplayBuffer._current_draw_mult` + 3 config knobs (draw_priority_*) — P4-3, deployed-fix placebo.
2. `Trainer._save_buffers` (the rolling replay.pkl.gz writer) — nothing calls it; committed .gz snapshots stem from an older revision.
3. config `buffer_save_every` — consumed by nothing.
4. `OpponentPool.get_opponent_path` / `should_play_pool` / config `self_play_vs_pool_ratio` — "30% vs pool" feature unwired (P4-5).
5. `SelfPlayWorker._trees` — `parallel_games` MCTSTrees + engines allocated at init, never used (self_play.py:60-62).
6. config `dataloader_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory` + ReplayDataset/collate_records — no DataLoader path is wired; trainer samples buffers directly.
7. config `aux_steps` (admitted legacy), `mcts_sims_critical` is used only via DCA budget (ok), `initial_elo/elo_k_factor` feed the dead arena.
8. tools/elo_vs_stockfish.py `_our_move` — worse than dead: referenced but UNDEFINED (F821) — --auto mode crashes (P7-2).
9. mcts_core.cpp backup() TD-λ params accepted and ignored (documented).
10. tree.py:857 `values_np` in _batch_begin_search — computed (D2H sync) then discarded.
11. heads.PolicyHead.masked_softmax — third masking implementation, used only by tests.

## 8.2 Magic numbers
Hyperparameters are exemplary-centralized in config.py with rationale comments. Remaining magic literals: resign 20-move gate (tree.py:1078), reuse=4 (main._train_steps_for), teacher pool divisor 6 (self_play.py:221), VL=1.0 (mcts_core), 25 ms/sim calibration (elo tool — now measured wrong, see P7-2), ONNX `parallel_games` fallback default 122 (inference.py:188).

## 8.3 Error handling
53 broad/bare excepts across 13 files — most deliberate ("self-play must never die") and commented; the risky ones: rules.get_game_result swallows engine exceptions into legal=[] (could mask C++ faults as terminal positions); web_gui/lichess tools blanket-catch. Missing-checkpoint: handled (FileNotFoundError raise + play-mode fallback). No-CUDA: probe falls back to CPU/fp32 ✓. Empty-buffer training: is_ready gate + sample() raises ✓. CUDA OOM: not specifically caught (process dies; supervisor restarts) — acceptable given supervisor, note for ops. Disk-full during the 1000-step buffer dump (trainer.py:412 unguarded) would crash training (P5-4).

## 8.4 Reproducibility — FINDING P8-1 (MEDIUM)
No torch/numpy/python seeding anywhere in the training pipeline (grep: only tools/lichess_puzzles.py and tools/stockfish_distill.py seed). C++ MCTS RNG seeded from std::random_device (mcts_core.cpp:34-38). Checkpoints store no RNG state. cudnn.benchmark=True (non-deterministic kernel selection). ⇒ no run, game, or training step is reproducible; debugging data-dependent bugs (e.g. the mate-in-2 crash) is needlessly hard. Search itself is deterministic absent noise (measured, bench_mcts.json determinism=true) — the nondeterminism enters only via Dirichlet noise, temperature sampling and PER sampling, so seeding those is sufficient and cheap.

## 8.5 Logging & observability
TensorBoard logging is solid (16 tags, file-backed, committed). Gaps: no gradient-norm logging, no draw-rate/resign-rate/game-length histograms (the exact metrics needed to see the draw-collapse!), no stale_move_count surfacing in the training loop, elo tags wired but never fire. Console logs are flushed and timestamped via supervisor tee ✓.

## 8.6 Test coverage
26 tests, all green, runtime 4.9 s. Real coverage: perft (good), C++ parity differential (good), buffer round-trip/sparse encode (good), TD-λ math (good), network shapes (good).
**Blind spots — every CRITICAL/HIGH finding of this audit lives outside test coverage:** no test for (a) value-label assignment incl. move-cap games (P4-1 would be caught by a 5-line test), (b) draw down-weighting actually changing sampling (test only checks the interpolation arithmetic — false sense of coverage), (c) aux-loss reaching the tower (P2-1), (d) scheduler preserving group lrs (P5-1), (e) arena trigger firing (P5-6), (f) encode/decode bijectivity (added in audit scripts), (g) GRU history conventions, (h) plane-20 parity. Estimated share of audit findings that existing tests could catch: ~0%.

## 8.7 Python quality (ruff, 205 findings)
68×E402 (path-bootstrap pattern, acceptable), 38×E701, 36×I001, 31×F401 unused imports, 14×F541, 10×F841 unused vars (incl. the tree.py:857 waste and rules.py tb_board), 3×E741, 2×F811, **1×F821 (real bug, P7-2)**. Type hints: good coverage on signatures; docstrings: extensive and unusually honest. main.py at ~1,160 lines is a god-module (orchestration + tests + play UI) — split candidate. No circular imports observed (late imports used deliberately). Global mutable singletons: CONFIG (by design), rules.py _tb_reader (fine).

## SELF-EXPAND
- T8.1 Add the 8 missing regression tests above to roadmap Section H (they're cheap and each pins a real found bug).
- T8.2 Seed-everything helper + checkpoint RNG state (pairs with P10.4 result).
- T8.3 Surface stale_move_count + draw-rate + game-length in TB.
