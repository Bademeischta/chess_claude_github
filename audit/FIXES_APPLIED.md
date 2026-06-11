# Audit Fixes Applied — 2026-06-11

Implements Sections C, D, E of `audit/AUDIT_REPORT.md` plus bug-class F items
(F3 overflow, F4 dead code, F9 regression tests). Skipped by design — restart-
class *experiments*, not defects: F1 board-flip canonicalization, F2 pool-width
scaling, F6 FP8, F7 WSL2/compile; E7 pybind ownership refactor (hardened via
runtime size check instead; unreproduced crash class). D2 needed no code change
(config already defaults to the 32-channel policy head; `match_arch` keeps old
checkpoints loadable).

## CRITICAL (Section C)

**C1 — PUCT perspective inversion (P3-5)** · `engine/mcts_core.cpp` ucb(),
`mcts/node.py` ucb(): selection now uses `-Q - virtual_loss + U` (parent
perspective). `.pyd` rebuilt. Verified: stub-net mate-in-1 visits 1/300 → 281/300;
real-weights mate-in-1 found from 10 sims (was: never at 1000); promotion
position converted (f7f8q, was king shuffle); `main.py --test` mate test now
asserts the actual mating move.

**C2 — Move-cap draw labels (P4-1)** · `config.py` max_game_moves 80→160 +
new `adjudicate_q_threshold=0.30`; `mcts/tree.py` `_finalize_record` adjudicates
cap-hits from the final search Q (White-perspective sign) instead of blanket 0.5.

**C3 — Disconnected aux head (P2-1)** · `model/heads.py`: new `fc_aux` Linear
consumes the value-tower hidden features; learnable piece-values remain as an
additive material prior inside the tanh. `config.py` aux_loss_weight 2.0→0.5.
`trainer.py` allows the new keys as fresh-init on old checkpoints; `net2net`
already strict=False. Verified by regression test (tower gradient nonzero).

**C4 — Dead evaluation subsystem (P5-6/P7-1/P7-2)** ·
- `main.py`: interval-crossing gates (`_interval_crossed`) replace `% N == 0`
  for arena + pool registration in all three phases; shared `_run_arena_and_update`.
- `main.py`: baseline pool registration only on first-ever run and never from
  `--test` (previously overwrote net_step0.pt with current/random weights on
  every start).
- `main.py`/`training/pbt.py`: ELO updates now rate the real best-pool entry
  (`best_opponent_id()`), not a phantom "best_pool" id.
- `utils/elo.py`: series ELO applied per game (interleaved), not one K=32 update.
- `tools/elo_vs_stockfish.py`: missing `_our_move` implemented (batched sims +
  greedy); `--auto` no longer resets tree+GRU history every ply (advance-based);
  auto movetime recalibrated 25 ms/sim → measured 0.125 ms/sim.

## HIGH (Section D)

- **D1 (P2-2)** `model/heads.py` ValueHead emits raw WDL **logits**
  (+`wdl_probs()`); `trainer.py` value loss = soft 3-class cross-entropy against
  the outcome (handles 0.25/0.75 tablebase labels) + 0.5 × TD(λ)-blended scalar
  MSE. `scalar_value()` softmaxes internally → infer()/ONNX unchanged.
- **D3 (P2-3)** `model/network.py` build_model keeps **fp32 master weights**
  (no more `.to(bfloat16)`); `infer()` autocasts via `net._amp_dtype`;
  `model/attention.py` GRU block explicitly disables autocast (cuDNN fast path).
  Trainer's existing autocast becomes true mixed precision; AdamW moments fp32.
- **D4 (P5-1)** `trainer.py` OneCycleLR gets per-group `max_lr` list — head/GRU
  multipliers survive (regression-tested).
- **D5 (P4-3)** `training/replay_buffer.py` draw multiplier wired into `add()`
  AND preserved through `update_priorities()` via per-slot `_prio_mult`;
  `config.py` draw_priority_mult 0.1→0.5 (0.1 was tuned while the knob was dead).
- **D6 (P11-1)** `main.py` phase_pretraining writes/honours
  `checkpoints/.pretraining_done` — crash-restarts no longer flood the replay
  buffer with 500k Syzygy rows.
- **D7 (P4-4)** ERED writes `cfg.dirichlet_eps` back; `mcts/tree.py` reads eps
  live at search time; restored from checkpoint into cfg on resume.
- **D8 (P3-4)** History convention unified: GRU history = positions strictly
  BEFORE the evaluated board, everywhere (recording in game_stream +
  run_single_game [also fixed its stale pre-search capture], root evals via
  `exclude_last=True`, leaf evals keep root-inclusive past). attention.py
  docstring updated.
- **D9 (P5-5/S10)** Arena (`trainer.run_arena`) and `tools/lichess_bot.py`
  play noise-free + greedy.

## MEDIUM (Section E)

- **E1 (P1-2)** plane-20 repetition marker: C++ scans all prior hashes (both
  parities); Python fallback now writes plane 20 too.
- **E2 (P4-2)** `_fresh_board` re-deals terminal random openings;
  `game_stream` guards sentinel move 0 (finalize, never apply the pass-move).
- **E3** NaN guard every step (hoists the already-paid loss/total sync before
  backward); poisoned batches can no longer reach AdamW.
- **E4 (P5-4)** ONE buffer-persistence path: `_save_buffers` (gzip, atomic
  tmp+rename) on the `checkpoint_every × buffer_save_every` cadence; the
  unguarded uncompressed step-pkl dump removed.
- **E5** `save_checkpoint` writes atomically (tmp + os.replace).
- **E6 (P8-1)** `config.seed` seeds torch/cuda/numpy/random at startup;
  checkpoints store + restore all RNG states (weights_only-safe encoding).
- **E7 (P3-2)** C++ `expand()` throws on legal/priors size mismatch (release
  builds previously read out of bounds).
- **E8 (P1-1)** EP square recorded only when the capture is fully LEGAL
  (pin-aware `ep_capture_legal` helper) — matches python-chess semantics;
  parity + perft suites stay green.
- **E9 (P4-5)** dead "30% vs pool" feature removed (config knob + 3 methods).
- **E10 (P11-2)** `PYTORCH_CUDA_ALLOC_CONF` set before torch import.
- **E11** elo-tool movetime calibration (see C4).
- **E12 (P1-6)** fallback `Board.hash` via `chess.polyglot.zobrist_hash`;
  logger U+2192 → ASCII.
- **P1-3** C++ stale-move recovery returns a clean board copy (hash intact).
- **F3 (P3-3)** temperature sampling in log-space (C++ + Python) — no overflow
  for τ ∈ (0.01, 0.05].
- **F4** dead code removed: `SelfPlayWorker._trees`, discarded root-value D2H,
  `tb_board`, step-pkl dump path.

## Tests

`tests/test_audit_regressions.py` (11 tests) pins: PUCT sign (stub-net search +
node-level), cap adjudication (3 cases), draw-mult priorities (+ survival
through updates), OneCycle group LRs, aux→tower gradient, WDL logits contract,
per-game ELO series, plane-20 both parities, pinned-EP gate, encoding
bijectivity. `tests/test_network.py` updated to the logits contract.
`main.py --test` mate-in-1 strengthened to assert the mating move.

## Verification status (all measured post-fix)

- pytest: **37/37** green (26 original + 11 new regression tests).
- `main.py --test` (isolated dirs): **8/8** green — incl. the strengthened
  mate-in-1 test (asserts the actual mating move, found by an UNTRAINED net).
- verify_chess_logic.py: **16/16** on the rebuilt .pyd (perft, 200-game
  parity with the new legal-EP gate, plane-20 both parities, pinned-EP
  non-divergence).
- PUCT fix, search behavior (step-63400 weights): mate-in-1 found from
  10 sims (pre-fix: never, even at 1000); K+P promotion converted (f7f8q,
  root Q rising with sims); stub-net visit concentration 281/300 on the mate
  (pre-fix: 1/300).
- Post-fix from-scratch stress window (`audit/scripts/stress_train.json`):
  labels **4,230 W / 14,720 D / 4,176 L** (pre-fix: 0/23,040/0 — 100% draws);
  avg game 115.6 plies (pre-fix: pinned at the 80-ply cap); training
  17.43 → 1.50 over 1000 steps, 0 NaN; **per-group LRs after run
  [5.2e-4, 2.6e-4, 1.6e-4] = exact 1.0/0.5/0.3 ratios**; 70.8 pos/s
  (255k pos/h) self-play.
- RNG save/restore round-trip: exact for numpy + python + torch (direct
  mechanism test). NOTE: the stress run's resumed losses still differ from
  the original because that test shares a LIVE buffer that evolved past the
  checkpoint (different PER priorities ⇒ different samples) and the one-step
  deferred priority update is not checkpointed — bit-exact resume requires
  restoring the buffer snapshot too (production does, via the rewired
  `_save_buffers` cadence; the stress harness deliberately disabled it).
