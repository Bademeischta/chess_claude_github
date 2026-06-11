# Phase 4 — Self-Play Pipeline Audit (static; throughput measurement pending)

Files: mcts/tree.py (game_stream/ParallelMCTS), training/self_play.py, training/replay_buffer.py, training/pbt.py, main.py phases, utils/system_probe.py (all fully read).

## 4.1 Game loop (ParallelMCTS.game_stream — the production path)
Per move, for all (≤32) trees: one batched root expansion → Dirichlet noise (once per move ✓ root-only ✓) → 128 rounds of [select 1 leaf/tree → ONE batched forward (width = #active trees) → expand+backup] → DCA extra rounds → per tree: temperature-sample move (τ=1 first 30 plies, then 0.1), record (board fp16, history fp16, full visit distribution, root Q, phase, piece count), advance root (subtree reuse ✓), check termination, recycle slot (fresh board+record, `fresh` set re-seeds GRU history). Pool never drains ✓. Board and MCTS tree verified reset between games (reset() on fresh slots; advance() otherwise) ✓.

### FINDING P4-1 (CRITICAL, label corruption — S1 CONFIRMED): move-cap games labeled as draws
tree.py:1099-1101: `if (result != GameResult.ONGOING or move_nums[i] >= self.cfg.max_game_moves): self._finalize_record(rec, result)` — at the cap, `result` is ONGOING(=2); `_finalize_record` (tree.py:985-991) maps anything ≠ WHITE_WIN/BLACK_WIN to 0.5. **With max_game_moves=80 (40 full moves), every game lasting >40 moves — including completely winning positions — produces value targets of "draw" for all of its ≤80 positions.** Deliberate per config.py:165-175, but: (a) it caps the strength ceiling — the net can never learn that material advantage converts, because conversions take >40 moves; (b) endgames are nearly absent from data (alpha_v_endgame=1.8 weighting acts on a starved class); (c) resign (Q<-0.55×6) only rescues positions the (collapsed) value head already understands. Recommended: raise cap (≥160 half-moves) once value-head fixes (P2-1/P2-2) land, and/or label cap-hit games via Q-sign or adjudication (tablebase/material) instead of 0.5.

### FINDING P4-2 (LOW frequency / MEDIUM impact): terminal fresh board → sentinel move 0 applied
`_fresh_board()` plays 6 uniform-random plies; a game can be OVER within ≤6 random plies (fool's-mate patterns ≥4 plies). Then: `_batch_begin_search` skips expansion (no legal moves) → `select_move` fallback (tree.py:581-610) finds 0 children AND 0 legal moves → returns sentinel 0 with all-zero policy → recorded as a training row (zero policy target!) → `board.apply_move(0)` (tree.py:1083). C++ apply_move(0)=a1a1: with a rook on a1 it acts as a **pass move** (side flips, no stale-counter hit); without a piece it increments stale_move_count. Garbage rows + corrupted game flow, rare (random-mate probability small but nonzero across millions of games). Fix: check `get_game_result(board) != ONGOING` right after `_fresh_board()` and re-deal.
Related: opening_book_path FENs with Black to move would flip EVERY value label of the game (parity convention `side = mn % 2` assumes first recorded position is White-to-move, tree.py:993-995). Book unused today (path="") — landmine, document.

## 4.2 Replay buffer
- PER SumTree: textbook implementation, O(log N) ✓; stratified sampling per batch; IS weights (N·P)^-β/max ✓; β annealed 0.4→1.0 over total_steps via _step ✓ (persisted in checkpoint ✓).
- Storage: fp16 board+history, sparse policy (int16 idx + fp32 val, validated <32768 ✓), ≈25-40 KB/pos dominated by the dense (8,21,8,8) fp16 history (21.5 KB).
- FIFO ring overwrite, min-fill gate replay_start_training=20k ✓ (bootstrap lowers to 5k).
- CPU-resident ✓. No DataLoader (trainer samples directly; dataloader_workers/prefetch_factor/persistent_workers config knobs are **dead** — P8).

### FINDING P4-3 (HIGH — placebo #2): draw down-weighting is dead code
replay_buffer.py:219-226 `_current_draw_mult()` has **zero call sites** (verified by grep across repo). `add()` (line 228-238) assigns max-priority to every new record; `sample()` never consults wdl_label. config draw_priority_mult=0.1 (+ramp), main.py:1070-1072 passes it in, tests/test_replay_buffer.py:46-72 test only the interpolation arithmetic — never that sampling changes. **The documented "almost completely exclude draws" fix (config.py:227-236) does nothing.** Fix: multiply priority by draw-mult in add()/update_priorities for records with wdl_label==0.5, or filter at sampling.

### FINDING P4-4 (MEDIUM — placebo #3): ERED dirichlet_eps regulation is dead
trainer._regulate_entropy adjusts `self._dirichlet_eps` (trainer.py:504-513); grep shows **nothing writes it back** to CONFIG or any MCTSTree — trees snapshot `cfg.dirichlet_eps` once at __init__ (tree.py:143). The eps half of ERED only affects logging + checkpoint contents. (`_alpha_v_global` half works ✓.) Fix: trainer writes CONFIG.dirichlet_eps and trees read live, or push to worker trees on update.

### FINDING P4-5 (MEDIUM — dead feature): "30% of games vs opponent pool" never wired
config.self_play_vs_pool_ratio and pbt.OpponentPool.get_opponent_path/should_play_pool have **no callers** (grep). game_stream has no opponent-model logic — self-play is 100% latest-vs-latest. Loss of opponent diversity vs design intent.

## 4.3 Parallelization
Single process, single thread; ≤32 trees batched into one inference stream (hw-derived parallel_games=min(32, vram//256)=**32 on this machine**, NOT config default 6). No multiprocessing — Python select/expand serializes between GPU calls (system_probe comment estimates GPU idle 30-40%). Teacher games: pool width `parallel_games//6`=5. SelfPlayWorker._trees (cfg.parallel_games MCTSTrees built at __init__, self_play.py:60-62) is **never used** — dead allocation incl. inference engines.
run_single_game (non-pool path, used by tests): records PRE-search history (zeros at move 0) — convention differs from game_stream (post-advance); only affects test data, note for P8.

## 4.4 Training/self-play coordination
Sequential alternation in main.py phases: generate parallel_games games → `_train_steps_for` = max(floor, n_new·4/batch) gradient steps (~4× reuse per position) → repeat. Same model object; CUDA-graph engine sees in-place weight updates automatically ✓. No staleness mechanism beyond alternation (acceptable for AZ-style).

## 4.5 Opening diversity
random_opening_plies=6 uniform-random (NOT recorded ✓) + per-move root Dirichlet. Reasonable. Opening book unused. NOTE: uniform-random openings give 6-ply positions of wildly varying quality — value labels learn "position after random plies", acceptable for diversity but biases the early-opening prior toward noise; AlphaZero used pure starts + noise. Comparative note for P9.

## 4.6 Throughput estimate (to be measured)
README claims 150-300k pos/h @128 sims; system_probe estimates 30k evals/s ceiling × 0.6 eff / 128 ≈ 140 pos/s ≈ 500k pos/h @ bootstrap-50 sims. Measured numbers → audit/benchmarks (runtime phase). 63,400 steps × 512 batch ÷ 4 reuse ≈ 8.1M positions consumed → at ~40 pos/game ≈ ~200k games over ~14 effective training days.

## SELF-EXPAND tasks
- T4.2 Measure: pos/s, games/h, GPU util during a bounded self-play window (audit/tmp); record stale_move_count before/after.
- T4.3 Verify hw-derived values live (parallel_games=32? batch=1024? replay cap from RAM?) — affects every Section B number.
- T4.4 Check trainer batch_size override interaction: probe caps batch at 1024 — VRAM check for training step at 1024.
- T4.5 Verify replay.pkl.gz/teacher.pkl.gz provenance: trainer._save_buffers has NO callers; train_step writes step_xxx_replay.pkl (uncompressed) every 1000 steps but no such files exist in checkpoints/ — reconcile (did buffer persistence silently never run? are mtimes git-LFS artifacts?). Buffer-persistence audit.
- T4.6 buffer_save_every config knob dead (grep: no consumer) — P8 list.
