# LIVING TASK QUEUE — Total Systems Audit

Legend: `[ ]` not started · `[~]` in progress · `[✓]` completed · `[!]` completed with critical finding · `[➕]` self-generated task added mid-audit

Updated: 2026-06-10 evening (runtime phase in progress)

## Phases

- [✓] P0 Orientation & initial scan — findings/phase00_orientation.md (hardware fully measured: Ryzen 8C/16T, 31.7 GB RAM, RTX 5070 12 GB, driver 596.49)
- [!] P1 Chess logic — findings/phase01_chess_logic.md + verify_chess_logic.json: 16/16 runtime sections OK; perft+200-game parity pass; encoding bijective (2427 positions + underpromos both colors); CONFIRMED P1-1 pinned-EP divergence, P1-2 plane-20 parity bug, P1-6 fallback .hash AttributeError; movegen 1.6 µs/call — not a bottleneck
- [!] P2 Network — findings/phase02_network.md + audit_network.json: 14.2M params DEPLOYED with LEGACY 2-CHANNEL policy head (P2-4 NEW); bf16 master weights + bf16 AdamW moments CONFIRMED; aux-loss isolation PROVEN (gradient reaches zero tower params); WDL-as-scalar confirmed
- [!] P3 MCTS — findings/phase03_mcts.md + bench_mcts.json: PUCT/noise/backup/temperature verified correct; P3-1 nodelete lifetime hazard; P3-4 GRU history skew; benchmarks done (7,965 sims/s single-tree cudagraph; 6,603 sims/s 32-pool; 62% infer/13% select/24% expand)
- [!] P4 Self-play — findings/phase04_selfplay.md: P4-1 move-cap=draw-label CONFIRMED (stress window: ~80 moves/game = 100% cap hits from scratch); P4-3 draw-weight dead; P4-4 ERED-eps dead; P4-5 pool-opponent dead; measured 81 pos/s @50 sims from scratch
- [!] P5 Training — findings/phase05_training.md: loss anatomy verified (no double-softmax); P5-1 OneCycle clobbers group LRs (PROVEN); P5-2 refinement-LR placebo; P5-4 buffer persistence incoherent; P5-6 arena gating dead; TB curves reconstructed: policy CE flat ~3.2 from step 4k→63k, value loss ≈0 (collapse) → 0.72 late (data changed, head didn't)
- [~] P6 GPU perf — findings/phase06_gpu_perf.md: inference sweep + MCTS benches done; pending: training-step time + GPU util (in stress run), profiler top-5 (T6.2)
- [!] P7 Evaluation — findings/phase07_evaluation.md: arena 5-defect chain (incl. net_step0 overwritten on EVERY start incl. --test); elo_vs_stockfish --auto crashes on undefined _our_move (F821); per-ply tree+history reset in --auto path; movetime calibration 200× off
- [✓] P8 Code quality — findings/phase08_code_quality.md: ruff 205 (1 real bug F821); 53 broad excepts; 11 dead features/knobs; zero seeding in training pipeline; test suite green but covers ~0% of found bugs
- [✓] P9 Comparative — findings/phase09_comparative.md: AZ/Lc0/KataGo table; adopt WDL-CE, board-flip, trunk-fed aux; keep CUDA-graph engine, C++ core, DCA, TD(λ)
- [~] P10 Stress tests — stress_train.py running (1000-step from-scratch + resume test); stress_positions.py written, runs next
- [!] P11 Security/ops — findings/phase11_security_ops.md: P11-1 crash-restart Syzygy buffer-flood (HIGH); P11-2 alloc-conf set after CUDA init (likely no-op); checkpoint saves not atomic; SIGINT chain solid
- [ ] P12 Queue exhaustion & final report (AUDIT_REPORT.md Sections A–I)

## Resolved seed tasks
- [!] S1 → P4-1 confirmed (cap-draws; 100% of from-scratch games hit cap)
- [!] S2 → P5-6/P7-1 root cause: modulo gating + 4 stacked defects
- [✓] S3 → 10 blocks deployed; ckpt_arch.match_arch aligns; BUT P2-4: 2-channel policy head deployed
- [!] S4 → P2-2 confirmed (P(draw) unsupervised)
- [✓] S5 → consistency verified across MCTS priors/targets/trainer/tools (same convention everywhere); efficiency concern stands (P1-4)
- [✓] S6 → VL = full-loss penalty, removal correct; aggressive but valid
- [✓] S7 → original UB fixed by stale-guard; remaining suspect = P3-1 nodelete; not reproduced (stale counter 0 everywhere)
- [✓] S8 → Triton absent; cudagraph backend real and 2.5×; compile permanently off on Windows
- [✓] S9 → TB curves: value loss ≈0 by step ~2k (immediate collapse), brief blips, 0.72 at 59k (data fixed, head still flat); aux fix was placebo (P2-1)
- [✓] S10 → web GUI ✓ noise-free, UCI ✓, elo-tool ✓ (but broken --auto), arena ✗ noisy (P5-5), lichess bot ✗ noisy
- [✓] T0.1/T3.1/T1.6 → .pyd current (built same day as last C++ commit; gated-EP + stale-counter behaviors present)
- [✓] T0.2/T5.1 → tb_curves.json
- [✓] T0.3 → net_step0 overwritten each start; arena modulo near-never fires; net_step35000 = lucky exact-multiple registration
- [✓] T0.4/T4.3 → batch 1024, parallel 32, replay cap ~190-235k (RAM-derived)
- [✓] T1.1-T1.5, T2.1-T2.3, T5.2, T5.3 → done via scripts (see JSONs)
- [✓] T3.3 dca.py audited (sound); T3.4 node.py parity ✓; T3.6 → P8-1

## NEW after stress testing
- [!] **P3-5 CRITICAL (root cause): PUCT selection uses child-perspective Q without negation — search avoids winning moves.** Mechanically proven (test_puct_sign.json: mate move 1/300 visits with stub net). Both backends. One-line fix each. Explains draw-collapse → value-flatline → policy-flatline equilibrium end-to-end.
- [!] P10: from-scratch self-play window = 100.0% draw labels (23,040/0/0); promotion-wins position unconvertible by trained net; resume not bit-exact (no RNG state).
- [➕][✓] logger.py U+2192 crash on cp1252 consoles (LOW, found incidentally).

## Open (work remaining)
- [✓] T10.A stress_train done (loss 17.4→1.93, 0 NaN, 214 ms/step, resume non-bit-exact)
- [✓] T10.B stress_positions done → led to P3-5
- [✓] T6.2 collapsed into time-split + latency sweep + step-time measurements ([SKIPPING kernel-level profiler listing: bottleneck question already answered])
- [✓] T4.5/T5.4 concluded code-side: _save_buffers dead, %1000 step-pkl path is the only writer and writes uncompressed into checkpoint_dir; committed .gz files stem from an older revision (mtimes are git-LFS artifacts) — buffer persistence story in phase05/phase11
- [✓] P12 final report Sections A–I → audit/AUDIT_REPORT.md
- [✓] Final isolation check: git status clean except AUDIT_PROMPT.md + audit/; checkpoints/runs/replay_buffer timestamps unchanged; audit/tmp deleted (0.2 MB of deliverables remain)

## AUDIT COMPLETE — 2026-06-11. Open items: 4, all explicitly deferred with justification (see "Deferred" above).

## FIX PASS — 2026-06-11 (see audit/FIXES_APPLIED.md)

- [✓] All Section C criticals implemented & verified (C1 sign fix → mate-in-1 found from 10 sims, promotion converted; C2 adjudication; C3 aux rewire; C4 evaluation chain incl. _our_move)
- [✓] All Section D items (D1 WDL-CE, D3 fp32 master, D4 LR groups, D5 draw-mult wired, D6 pretraining marker, D7 ERED live, D8 history convention, D9 eval noise) — D2 = config already correct
- [✓] All Section E items (E7 hardened-not-refactored, justified)
- [✓] F3, F4, F9 (11 regression tests); F1/F2/F6/F7 deliberately not done (experiments, not defects)
- [✓] chess_ext.pyd rebuilt from fixed sources (MSVC 14.51)
- [✓] pytest 37/37; isolated `main.py --test` 8/8; verify_chess_logic 16/16
- [✓] post-fix from-scratch stress window: 36% decisive labels (was 0%), avg game 115.6 plies (was 80.0 = cap), loss 17.4→1.50 stable, LR ratios 1.0/0.5/0.3 preserved, RNG round-trip exact (resume bit-exactness additionally needs the buffer snapshot — production cadence saves it)

## FIX PASS COMPLETE — 2026-06-11.

## Deferred (justified)
- T6.1 parallel_games {64,96,128} A/B — roadmap item; needs recapture loop; estimate given in phase06 from latency sweep
- T3.2 GRU history-skew magnitude measurement — roadmap; requires path-history reconstruction harness
- Stockfish calibration match — tool's --auto is broken (P7-2); fixed-anchor run possible post-fix; curves already bound strength qualitatively
- T3.5 VL leak on exception paths — LOW, exception-only
- T8.x additions — folded into report Section H
