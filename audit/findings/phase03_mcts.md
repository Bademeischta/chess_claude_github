# Phase 3 — MCTS Forensic Audit

## ⚠ FINDING P3-5 (CRITICAL — root cause of the training failure): PUCT selection uses the child's own-perspective Q without negation

Discovered via runtime stress test (mate-in-1 never chosen at 1000 sims), proven mechanically with a stub network (audit/scripts/test_puct_sign.json): in a back-rank mate-in-1 with uniform priors and value≡0, the mating move received **1 visit out of 300** (its single visit returns terminal Q=-1 in the child's perspective and the move is then maximally avoided); all 15 shuffle moves got 19 visits each. Correct PUCT concentrates ~all visits on the mate.

Mechanism: backup (mcts_core.cpp:251-259 / tree.py:454-461) stores every node's Q from the perspective of the side to move AT THAT NODE (leaf gets +v_leaf, parent −v_leaf, …) — internally consistent, and root_value()/resign/TD-bootstrap read the root's own-perspective Q correctly. But selection (mcts_core.cpp:49-53, 71-87; mcts/node.py:56-63, 84-95) computes
```cpp
float q_adj = Q - (float)virtual_loss;   // child's own perspective!
return q_adj + u;                        // argmax over children
```
For argmax-selection the child Q must be the PARENT's action value, i.e. **−child.Q**. As written, the search maximizes the opponent's expected outcome: winning lines are suppressed after one visit, losing lines are reinforced. Both backends share the inversion (consistent duplicates), so C++/Python parity tests could never catch it; main.py --test's mate-in-1 only asserts `move != 0`.

Downstream effects (ties the whole audit together): games cannot converge on wins → cap-draws dominate (measured: 100% draws in the from-scratch window) → value targets all 0.5 → value head collapse (TB: loss/value≈0 for 50k steps) → Q≈0 everywhere → search is prior/noise-driven → policy targets ≈ noise → policy CE flat at ~3.2 from step 4k → ERED/aux/draw-weight/cap/resign countermeasures were symptom-fights (several themselves dead code). Also explains config.py:129-133's observation that 800-sim teacher rollouts played no better than 50-sim ones.

Exact fix (both backends + comment):
```cpp
float q_parent = -(Q) - (float)virtual_loss;   // parent perspective + VL penalty
return q_parent + u;
```
```python
q_parent = -self.Q - float(self.virtual_loss)
return q_parent + u
```
Verify with: audit/scripts/test_puct_sign.py (expects visits to concentrate on a1a8), then re-run mate-in-1 with real weights, then 1000-step from-scratch (expect decisive games to appear within the window).
Note on FPU after the fix: unvisited children have Q=0 while a winning parent line pushes sibling q_parent up — consider standard FPU-reduction later; not blocking.

(Original static-analysis notes below; the selection-formula section is superseded by P3-5.)

Files audited line-by-line: engine/mcts_core.h/.cpp (312 ln), engine/bindings.cpp (230 ln), mcts/tree.py (1130 ln, full read). Pending: mcts/node.py, mcts/dca.py.

## 3.1 Tree data structure

- C++ MCTSNode: N, W, Q, P, virtual_loss, move, parent*, children map<Move, unique_ptr>, terminal, terminal_value. Tree owns nodes via unique_ptr; root replaced on advance_root/new_root.
- Board NOT stored per node — reconstructed by replaying moves from root during selection (apply_move chain). Memory-lean; CPU cost = path-length × board copy (incl. hash_history vector).
- Tree reuse: advance_root moves child subtree to root (mcts_core.cpp:164-176); un-searched move → fresh root. Python wrapper adds `_needs_cpp_reset` wipe after degenerate fallback (tree.py:200-209). Correct.

### FINDING P3-1 (HIGH, stability): pybind lifetime hazard around node references
bindings.cpp exposes MCTSNode with `py::nodelete` + `return_value_policy::reference` (lines 143, 154, 167, 173, 183, 190). Any Python-held node/SelectionResult.path reference becomes a **dangling pointer** the moment advance_root/new_root frees that subtree. Current drivers (tree.py/ParallelMCTS) consume pendings before advance — no live violation found in the audited paths — but the API offers no protection; root_analysis results, stored `_root_node` of the OLD root after advance, or any future caller error → use-after-free → the class of access violations documented in config.py:278-282. The select_move zero-children fallback + `_needs_cpp_reset` is symptomatic defense. Recommendation: shared_ptr ownership or copy-out APIs.

### FINDING P3-2 (MEDIUM, latent UB): expand() size-mismatch unchecked in release
mcts_core.cpp:211 `assert(legal_moves.size() == priors.size())` is compiled out (NDEBUG); bindings numpy_to_vector takes whatever size Python sends. A too-short priors array → OOB vector read (UB). Current callers always size priors == legal (policy_array_to_legal_priors returns len(legal)); add a runtime check + throw.

## 3.2 Selection (PUCT)

mcts_core.cpp:49-53:
```cpp
float u = c_puct * P * sqrt((float)parent_n) / (1.0f + (float)(N + virtual_loss));
float q_adj = Q - (float)virtual_loss * 1.0f;
return q_adj + u;
```
- Matches AlphaZero U(s,a)=Q+c·P·√N(s)/(1+N(s,a)) with parent_n = parent's own N (standard variant). c_puct=2.0 (config; AlphaZero ~1.0–2.5 range; Lc0 ≈2.5+ but with different value scale — acceptable, tuning task).
- N=0 / parent_n=0 safe: division guarded by 1+N; sqrt(0)=0 → first child pick driven by Q init 0 + tie order.
- **No random tie-breaking**: best_child takes first strictly-greater score; map<Move,…> iterates in move-int order → deterministic low-square bias among exact ties (mostly harmless after priors differ; note for determinism tests — C++ sample_move RNG is seeded from random_device → games non-reproducible. Reproducibility finding for P8).
- Terminal nodes: selection stops (`!node->terminal`), value re-derived from board each visit via rules.get_game_result — correct for checkmate AND all draw types; avoids relying on stored terminal_value=0 default (the C++ expand() empty-legal path sets terminal w/o value, but Python _finish_leaf always calls mark_terminal(value) for terminals — verified tree.py:405-407, so the default-0 hazard is closed in current drivers).
- Virtual loss: subtracts FULL 1.0 per pending visit from Q in [-1,1] space (equiv. to assuming a full loss; multiple VLs push Q below -1). Aggressive but bounded; purpose = batch diversification. Removal verified: backup undoes VL on every path node (root's no-op guarded by `if (virtual_loss>0)`).

## 3.3 Expansion

- Priors: zero-policy → `policy_array_to_legal_priors` = softmax over LEGAL logits only (mathematically identical to mask-to--1e9-then-softmax, renormalized). Illegal moves never become children. Correct.
- try_emplace double-expansion guard (mcts_core.cpp:219-223) prevents subtree leak on re-expansion. Correct.
- Tablebase leaves (tree.py:490-500): value=2·wdl-1 from side-to-move (incl. cursed 0.75/blessed 0.25 mapping), node expanded with UNIFORM priors (zero logits) — net is bypassed; acceptable design, documented.
- Newly expanded node immediately backed up (single backup call covers expand+backup paths). Correct.

## 3.4 Backup

mcts_core.cpp:251-259: walks path leaf→root, undo VL, W+=value, N++, Q=W/N, value=-value each level. Leaf value from leaf side-to-move perspective → parent (opponent) gets -v. **Sign convention verified consistent** with terminal values from _resolve_non_nn (side-to-move perspective: mated side at leaf → -1). td_lambda/move_number params accepted but UNUSED (pure MC; TD(λ) applied in trainer — documented in both files; dead params noted).

## 3.5 Dirichlet noise

- C++ add_dirichlet_noise: proper Gamma(α,1) draws normalized (comment documents fixing an earlier exponential-noise bug — verify the .pyd binary actually contains the fix, ties to T0.1/T1.6!).
- Root-only: called only on root nodes (tree.py begin_search / _batch_begin_search step 1). Once per MOVE (not per simulation). Reused-subtree roots: their children carry clean NN priors (noise was applied to the old root's children = the new root itself, not its children) → exactly one noise application per move. Correct.
- α=0.3, ε=0.25 (AlphaZero values; α≈10/avg_legal≈0.33 → 0.3 appropriate).
- Eval paths: tree.search(add_noise=False) exists for arena/play; consumers verified in P7 (S10 still open for web GUI/UCI/arena callers).

## 3.6 Temperature

- τ=1.0 first 30 moves then τ=0.1 (config temperature_moves/final); applied to VISIT COUNTS: sample ∝ N^(1/τ) (mcts_core.cpp:108-125). Correct (not Q-based).
- τ≤0.01 → greedy argmax. **FINDING P3-3 (LOW, fragility)**: for τ in (0.01, ~0.05], N^(1/τ) overflows float (e.g. 128^50≈1e105 → inf → sampling degenerates). Config τ=0.1 → 128^10≈1.2e21 safe; 800 sims at τ=0.1 ≈ 1e29 still safe; document boundary, compute in log-space ideally.

## 3.7 Virtual loss / parallelism

- Single Python thread drives all trees; **bindings release no GIL** → C++ calls serialize anyway; thread_local RNG comment is moot. No race conditions possible in current architecture (verified: ParallelMCTS is sequential rounds of [select all → one batched infer → apply all]).
- Batch width = number of ACTIVE TREES (one leaf per tree per round, tree.py:932-946) = parallel_games (config default 6, hw-derived). NOT hundreds of leaves. → GPU batch 6..32; throughput audit in P6. Single-tree run_batched_sims (web UI path) gathers up to engine.batch_size leaves within one tree via VL.

## 3.8 Training-data generation

- Policy target = FULL visit distribution (get_policy_target normalizes child N), mapped via move_to_action_index(stm) — same convention as priors. Correct soft target.
- Value labels: _finalize_record assigns per-position WDL from side parity (mn%2). Game result from White's perspective ↔ per-side label = game_val or 1-game_val. **Both players' positions stored.** Correct GIVEN first recorded position is White-to-move — see P4-2 hazard (odd random-opening break / opening-book FEN with Black to move flips ALL labels of a game).
- mcts_q recorded from root Q (side-to-move perspective) pre-advance → TD(λ) bootstrap target. Consistent.
- No augmentation (no horizontal flip) — fine (correctness-safe; potential 2× data via mirror flip noted for P9 as opportunity since castling breaks naive mirroring; AlphaZero also skips augmentation for chess).

### FINDING P3-4 (HIGH, train/inference skew): GRU history last-frame convention differs between training data and leaf evaluation
- Recording (tree.py:1062-1069): history = last T boards INCLUDING the current position as final frame (board_history's last entry == the position being recorded; appended by reset/advance).
- MCTS leaf eval (tree.py:368-374, 896-897): board_t = LEAF board, history = ROOT-relative history whose final frame is the ROOT board (intermediate search moves absent).
- Net is trained with history[-1] == current board; evaluated at leaves with history[-1] == root ≠ leaf. If the GRU learns to rely on the final frame (the easiest signal — it duplicates the main input), every leaf evaluation receives root-flavored context → systematically wrong context planes during search. Root evals (begin_search) match the training convention; leaf evals (the vast majority) do not.
- Measure in P10: compare value outputs for leaf positions with (a) root-relative history vs (b) correct path history vs (c) zero history; if (a)≈(c) deviation is small the 4 context planes are being ignored (wasted capacity); if large, search values are skewed.

## 3.9 Benchmarks: pending (bench_mcts.py) — sims/s batched/unbatched, C++ vs CHESS_AI_NO_CPP=1, time-split selection/inference/movegen, determinism check (expected NON-deterministic: random_device seeds, np RNG unseeded).

## SELF-EXPAND tasks
- T3.1 Verify Dirichlet-fix & apply_move-guard actually present in the SHIPPED .pyd (behavioral test, ties T0.1).
- T3.2 Measure GRU history-skew effect (P3-4).
- T3.3 dca.py audit: get_sim_budget/compute_extra_sims correctness (called per move, |Q1-Q2| gate).
- T3.4 Verify mcts/node.py Python fallback parity with C++ (UCB identical? sample_move? noise via np.random.dirichlet unseeded).
- T3.5 VL leak audit on exception paths (run_batched_sims catch blocks count `completed` without undoing VL of failed pendings — tree permanently penalized; LOW since exceptions are rare).
- T3.6 Tie-break determinism + RNG seeding policy → P8 reproducibility.
