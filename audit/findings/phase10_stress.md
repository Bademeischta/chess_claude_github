# Phase 10 — Stress Testing & Edge Case Probing (all runtime, measured)

Artifacts: stress_train.json, stress_positions.json, test_puct_sign.json, verify_chess_logic.json.

## 10.1 Tricky positions (rules layer + step-63400 net @200 sims, noise-free)
| Case | Rules verdict | Search behavior |
|---|---|---|
| startpos | 20 legal ✓ ONGOING | picks g1f3, root Q −0.0 |
| EP available | e5f6 in legal ✓ | ok |
| Fool's-mate position | 0 legal, BLACK_WIN ✓ | terminal — no search |
| Stalemate | 0 legal, DRAW ✓ | search returns sentinel 0 + zero policy via guarded fallback (WARN printed) ✓ |
| Promotion-wins (K+P f7) | ONGOING, 9 legal | **picks e1f1 (king shuffle), root Q 0.005 — cannot find promotion** |
| Underpromotion case | ONGOING | shuffles too |
| Zugzwang / KP endgame / 49-half-move | correct rules verdicts | root Q ≈ 0 everywhere |
| One-legal-move | n_legal=1 ✓ | (probe only) |

Net-level verdict: the trained net evaluates ~everything to 0 and the search cannot convert trivially-won positions — consistent with the TB diagnosis and explained by P3-5.

## 10.2 MCTS stress
- Sim scaling on mate-in-1 (real weights): 1→1000 sims NEVER selects Ra8#; root Q decreases with more sims (0.92→0.28) as the (inverted) search mixes in worse lines. → led to discovery of **P3-5 (CRITICAL, see phase03)**; mechanically proven with stub network: mate move N=1/300 visits.
- Terminal-position search: returns immediately (7 ms), sentinel handled.
- Same-position determinism: identical visit distributions without seeding when noise-free (bench_mcts.json) ✓; nondeterminism enters only via noise/sampling RNG.

## 10.3 Training stability (1000 steps from scratch, isolated audit/tmp)
- Data: 320 games / 25,600 positions @ 50 sims — **100.0% draw labels** (23,040 draw / 0 win / 0 loss): every game hit the 80-move cap; resign never fired (flat value head). The bootstrap regime trains on a constant-0.5 value signal. (P4-1 + P3-5 in action.)
- Loss: 17.40 @1 → 6.79 @100 → 3.15 @500 → 1.93 @1000 — monotone decrease, **0 NaN/Inf**, grad-norm samples 5.01 (at clip), 1.18, 3.5, 3.26.
- Initial policy CE ≈ 8.44 ≈ ln(4672) ✓ matches theory (full-vector softmax) and the historic TB value @step1 (8.441) exactly.
- Median train step **214.3 ms @ batch 1024 → 16,800 steps/h**. Training consumes ~6% of GPU time at the 4×-reuse policy; self-play is the binding constraint.
- Self-play window throughput (from-scratch net, 50 sims, 32-pool, cudagraph): **82.6 pos/s = 297,410 pos/h; 3,718 games/h; avg game = 80.0 plies (= cap)**; stale_move_count 0.
- All-group LR identical after run (0.000521…) — P5-1 confirmed in the production Trainer class.

## 10.4 Recovery & resume
- save@100 → resume → steps 101–110: losses differ run-to-run (6.3–7.5 vs 5.2–6.6) but match in magnitude; `identical: false`. Expected: checkpoints store NO RNG states, PER sampling unseeded (P8-1). Resume is functionally correct (weights/optimizer/scheduler/ERED/β-anneal restored) but not bit-exact — bug-reproduction via resume is impossible today.

## Incidental finding during testing
- utils/logger.py prints U+2192 and crashes under cp1252 consoles (UnicodeEncodeError) when used outside main.py (which reconfigures stdout). LOW; guard the print or ascii-fallback.

## SELF-EXPAND
- T10.1 Post-fix re-run protocol (after P3-5 fix): test_puct_sign expects a1a8 visit-domination; stress window expects nonzero win/loss labels; mate-in-1 found ≤50 sims. → report Section H Day-1 verification.
- T10.2 [SKIPPING: 1000-step bf16-vs-fp32-master A/B (T2.4) — superseded in priority by P3-5; a clean A/B is only meaningful after the search fix. Roadmap item.]
- T10.3 [SKIPPING: torch.profiler top-5 op listing — the measured time split (62% infer / 13% select / 24% expand) + latency sweep + 214 ms step time already answer the bottleneck question the profiler item targets; kernel-level micro-opt is below the impact threshold of current findings.]
