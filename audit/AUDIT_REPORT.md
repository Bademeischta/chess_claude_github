# TOTAL SYSTEMS AUDIT — CHESS AI (AlphaZero / PyTorch / MCTS)

Audited: 2026-06-10/11 · Hardware: RTX 5070 12 GB, Ryzen 8C/16T, 31.7 GB RAM, Win 11, torch 2.11.0+cu128
Method: full static read of all 11.3k Python + 1.7k C++ LOC, 16-section runtime verification suite, GPU benchmarks, 1000-step from-scratch stress training, mechanical bug-proofs. All numbers measured (scripts + JSON in `audit/scripts/`, raw findings in `audit/findings/phase*.md`, complete task log in `audit/TASK_QUEUE.md`).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION A — EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The chess rules engine is excellent — perft-perfect, parity-verified against python-chess across 200 random games, bijective move encoding, 1.6 µs/movegen — and the systems engineering (CUDA-graph inference, crash supervisor, checkpointing) is far above hobby standard. The learning loop, however, has been silently broken end-to-end: **(1) PUCT selection reads the child's Q in the wrong perspective, so the search systematically avoids winning moves** (mechanically proven: in a mate-in-1 with a stub net, the mating move gets 1 of 300 visits) — this single sign error explains the entire observed pathology of the 63,400-step run (100% cap-draw games → value-head collapse → flat policy CE ≈ 3.2 since step 4k). **(2) Every game longer than 40 moves is labeled a draw** (max_game_moves=80), corrupting value targets at scale. **(3) Most deployed countermeasures were placebos**: the aux loss (weight 2.0) sends zero gradient to the value tower; draw down-weighting is dead code; ERED's noise regulation never reaches the trees; the refinement-phase LR cut is overwritten by the scheduler; and the entire ELO/arena/promotion subsystem never executed (modulo-gating + an undefined function in the external ELO tool). Improvement potential: the Day-1 fixes below are small (the core one is one line per backend), after which the measured infrastructure can already sustain ~300k positions/hour — the project's first genuinely learning run is days, not months, away.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION B — PERFORMANCE BASELINE (MEASURED, RTX 5070)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Neural network (deployed step-63400 artifact):
- Parameters: **14,209,946** (10×256 SE-ResNet + 2-layer GRU; **legacy 2-channel policy head** — the intended 32-channel head yields 23.2M and was never trained)
- Model size: 56.8 MB fp32-equiv; checkpoint file 95.5 MB (bf16 weights + bf16 AdamW moments + fp32 GRU)
- Inference latency (eager bf16): batch 1 = 5.26 ms · batch 32 = 4.73 ms · batch 256 = 12.0 ms · batch 512 = 20.9 ms
- CUDA-graph @ batch 32: **1.91 ms** (2.5× vs eager)
- Throughput: 6,760 pos/s @32 eager · 16,716 @32 graph · **24,482 peak @512**
- VRAM during training: **2.63 GB peak / 12 GB** at batch 1024 fwd+bwd (huge headroom)
- Training step: **214.3 ms median @ batch 1024 → 16,802 steps/hour**

MCTS:
- Simulations/second: 165 (unbatched) · 3,873 (batched eager) · **7,965 (batched CUDA-graph, single tree)** · 6,603 (32-tree production pool)
- Time per move: 16.1 ms @128 sims (single tree graph) · 620 ms per 32-game pool round @128 sims
- 800-sim move: 100 ms (vs AlphaZero's ~30 ms on a TPU pod)
- Bottleneck: GPU inference at batch 32 — measured split **62% inference / 13% selection / 24% expand+backup**

Self-play pipeline (measured):
- Average game length: 80.0 plies — **= the cap; 100% of from-scratch games are truncation draws**
- Pure self-play: **82.6 pos/s = 297,410 positions/hour, 3,718 games/hour** @50 sims (from-scratch window)
- Theoretical @128 sims (pool bench): ~186,000 positions/hour
- Production sustained (TB history, incl. training interleave): **17–19 pos/s ≈ 65k positions/hour**, 650–900 games/hour
- Historic run: 63,400 steps over ~24 days; ELO: **never measured** (subsystem dead)

Estimated timeline (post-fix, at measured rates):
- 1M positions: ~3.4 h pure bootstrap self-play (50 sims) · ~10 h at 128 sims with training interleave
- Beating Stockfish level 1 / level 5: not estimable from this run (strength was never measured and learning was broken); first calibration point available ~1–2 days after Day-1 fixes via the repaired ELO tool

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION C — CRITICAL BUGS (fix before any further training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**C1 · PUCT Q-perspective inversion (P3-5)** — `engine/mcts_core.cpp:49-53` + `mcts/node.py:56-63`
Backup stores each node's Q from that node's side-to-move perspective; selection then takes `argmax(child.Q + U)` — which maximizes the OPPONENT's value. Winning moves are visited once and then avoided (proof: `audit/scripts/test_puct_sign.json`, mate move 1/300 visits). Root cause of the draw-collapse / flat-policy equilibrium.
Fix (both backends): `q_parent = -Q - virtual_loss; return q_parent + u;` then rebuild the .pyd. Verify: test_puct_sign.py (mate move dominates visits), mate-in-1 found ≤50 sims with real weights, from-scratch window produces decisive games.

**C2 · Move-cap games labeled as draws (P4-1)** — `mcts/tree.py:1099-1101, 985-991` with `max_game_moves=80`
Truncated games finalize `result=ONGOING → 0.5`. Every >40-move game — including won positions — trains the value head toward "draw". After C1, raise the cap (≥160 plies) and adjudicate cap hits (root-Q sign and/or material/tablebase) instead of labeling 0.5.

**C3 · Aux loss never reaches the value tower (P2-1)** — `model/heads.py:181-191`, `model/network.py:209-214`, weight at `trainer.py:323-324`
`aux = tanh(material_diff @ piece_values / 12)` is computed from input planes + 6 scalars only; `aux_loss_weight=2.0` (the documented anti-collapse fix) trains 6 parameters and inflates loss curves. Re-wire to consume tower features, or drop to probe-weight 1e-3 and fix value learning via C1/C2/H1.

**C4 · Evaluation subsystem non-functional (P5-6/P7-1/P7-2)** — five stacked defects:
(a) arena gate `global_step % 10_000 == 0` virtually never true (step advances in chunks) — use interval-crossing; (b) `pool.register_current(model, step=0)` at every start overwrites `net_step0.pt` with current/random weights (also triggered by `main.py --test`!) — register once behind existence check; (c) ELO updates a phantom id `"best_pool"` (main.py:669) — use the real entry id; (d) one K=32 update per 100-game series caps movement at ±32 ELO — update per game or use proper series math; (e) `tools/elo_vs_stockfish.py --auto` calls undefined `_our_move` (line 194, NameError) and resets tree+GRU history every ply (line 200). Until fixed there is no strength signal of any kind.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION D — HIGH PRIORITY (this week)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| # | Finding | Impact | Effort | Recommendation |
|---|---|---|---|---|
| D1 | WDL head trained only via P(win)−P(loss) MSE; P(draw) unsupervised (P2-2) | collapse-prone by construction; no draw calibration | S | 3-class cross-entropy on WDL logits (drop the in-forward softmax, use CE); keep scalar_value for MCTS |
| D2 | Deployed net still has the legacy 2-channel policy head (P2-4) — 128-dim bottleneck for 4672 logits | likely co-cause of flat policy | S | Since C1 resets learning anyway: restart with the 32-channel head (config already defaults to it) |
| D3 | bf16 master weights + bf16 AdamW moments (P2-3/P5-3) | silent update-rounding, late-training stall | S | fp32 weights + autocast(bf16) compute; +~95 MB VRAM (2.6/12 GB used — free) |
| D4 | OneCycleLR scalar max_lr wipes per-group LR multipliers (P5-1, proven) | head/GRU LR knobs do nothing | XS | pass per-group max_lr list |
| D5 | Draw down-weighting dead code (P4-3) | the documented sampling fix does nothing | XS | apply `_current_draw_mult()` in add/update_priorities, or delete feature + config |
| D6 | Crash-restart re-runs Syzygy pretraining, floods ~200k-cap buffer with 500k endgame rows, evicting all self-play data (P11-1) | repeated data-distribution resets in production | S | `pretraining_done` marker in checkpoint; supervisor passes `--phase=selfplay` after first completion |
| D7 | ERED dirichlet_eps never reaches trees (P4-4) | half the entropy controller is display-only | XS | write back to CONFIG + trees read live (or drop eps control) |
| D8 | GRU history convention skew: training history[-1]=current board, leaf-eval history[-1]=root (P3-4) | systematic train/inference input mismatch | M | exclude current board from recorded history (cheapest), or reconstruct path history at leaves; measure with T3.2 harness |
| D9 | Arena/lichess-bot play WITH Dirichlet noise (P5-5; S10) | noisy eval, weaker bot | XS | pass `add_noise=False` (+ τ=0 argmax in arena) |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION E — MEDIUM PRIORITY (next iteration)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- E1 Plane-20 repetition marker: even-index-only scan misses ~half of repetitions; Python fallback never sets it (P1-2, proven) — fix both, add parity test.
- E2 Terminal fresh-board after random opening plies → sentinel move 0 applied as a "pass move" that also strips castling rights; garbage training row (P4-2 + P1-5) — re-deal terminal openings; make `action_index_to_move` return −1 and assert at call sites.
- E3 NaN guard samples every 50 steps while `loss/total` is already synced every step (P5-4 note) — make the guard per-step for free; also: backward+step still execute on unchecked NaN steps.
- E4 Buffer persistence incoherent: `_save_buffers` (gz) dead, `buffer_save_every` dead, the live %1000-step dump writes uncompressed multi-GB pickles and is unguarded against disk-full (P5-4) — pick ONE path (gz + atomic tmp-rename), call it on the checkpoint cadence.
- E5 Checkpoint saves not atomic (torch.save direct); supervisor resumes from possibly-truncated `latest.pt` (P11) — tmp+replace, keep previous good.
- E6 No RNG state in checkpoints + zero seeding in pipeline + cudnn.benchmark (P8-1; resume proven non-bit-exact) — seed-everything helper, store RNG states, optional deterministic debug mode.
- E7 pybind `nodelete` + reference returns on MCTSNode → dangling-pointer API; the likely residual source of the historical access violations (P3-1); also release-build expand() size-mismatch UB (P3-2) — shared_ptr or copy-out; runtime size check.
- E8 EP-square gate is pseudo-legal vs python-chess "legal" (P1-1, proven divergence on pinned-EP FEN) — align with full legality check (affects Zobrist/threefold/plane-13/FEN parity).
- E9 "30% vs opponent-pool" self-play feature unwired (P4-5) — wire `get_opponent_path` into game scheduling or remove.
- E10 PYTORCH_CUDA_ALLOC_CONF set after CUDA init — likely no-op (P11-2) — move before torch import/launcher.
- E11 elo-tool fair-time calibration assumes 25 ms/sim; measured 0.125 ms/sim → grants SF ~200× the model's budget (P7-2) — recalibrate.
- E12 Fallback `Board.hash` → AttributeError (P1-6, proven); logger U+2192 crash on cp1252 (proven) — small fixes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION F — LOW PRIORITY & EXPERIMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- F1 Board-flip perspective canonicalization instead of 180°-rotated actions for Black (P1-4): standard, halves the policy learning problem; schedule together with D2 (both reset the policy head).
- F2 parallel_games 32 → 96–128: latency sweep projects ~2.2–2.6× self-play throughput (→ ~400–480k pos/h); needs CUDA-graph recapture at new width; A/B before committing.
- F3 sample_move float-overflow for τ∈(0.01,0.05] (N^(1/τ) → inf) — compute in log-space.
- F4 Dead-code sweep: `SelfPlayWorker._trees`, ReplayDataset/DataLoader knobs, `aux_steps`, masked_softmax duplicate, `values_np` discard in `_batch_begin_search`, 31 unused imports (ruff).
- F5 PGN export for arena/eval games; TB logging of draw-rate, game-length, grad-norm, stale_move_count.
- F6 FP8/TensorRT inference on Blackwell (~1.5–2× at batch≥128) — only after strength measurement works.
- F7 WSL2/Linux benchmark for torch.compile (Triton unavailable on Windows — verified).
- F8 Virtual-loss magnitude (full −1.0 per pending visit) and FPU-reduction tuning after C1.
- F9 Regression tests for every Section C/D item (the current suite, 26 green tests, catches ~none of them).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION G — ARCHITECTURE RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

No teardown needed: the C++ engine, the batched CUDA-graph inference path, and the ops tooling are keepers. The architecture-level changes that matter, in dependency order:
1. **Search first** (C1): everything downstream is unmeasurable until selection is minimax-correct.
2. **Value learning** (D1+C3+C2): WDL cross-entropy + trunk-fed (or demoted) aux + honest game labels. This removes the need for the entire draw-suppression apparatus (draw_priority, 80-cap, aggressive resign) — unwind those knobs to neutral and re-tune from evidence.
3. **Policy capacity & representation** (D2+F1): 32-channel head + board-flip canonicalization, one combined breaking change, fresh run.
4. **History encoding**: either fix the GRU convention (D8) and prove its value with the T3.2 ablation, or replace with AZ-style stacked history planes (simpler, no skew, no fp32 island, removes ~12% of forward latency at batch 32).
5. **Precision**: fp32 master + bf16 autocast (D3).
Expected compound effect: a system that learns (vs. one that provably could not), at ≥300k pos/h bootstrap on the existing hardware, with strength measurable weekly via the repaired ELO tool.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION H — PRIORITIZED IMPLEMENTATION ROADMAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Day 1 — make it learn** (C1 sign fix both backends + .pyd rebuild · C2 cap/labeling · D4 LR groups · D5 draw-weight wire-or-delete · D9 eval noise off) + verification: test_puct_sign green, mate-in-1 ≤50 sims, from-scratch window shows decisive games.
**Day 2–3 — make it measurable** (C4 arena/ELO chain + `_our_move` fix + movetime calibration · D6 pretraining marker · E3 NaN guard · E5 atomic saves) + first real ELO datapoint vs SF skill 0–3.
**Week 2 — make it efficient and clean** (D1 WDL-CE · C3 aux rewire · D2+F1 fresh run with 32-ch head + board-flip · D3 fp32 master · D8/history decision · E1/E2/E6/E7 · F9 regression tests).
**Month 2 — scale and polish** (F2 pool-width A/B · F5 observability · F6/F7 backends · c_puct/VL/FPU re-tuning on a working system · Net2Net grow to 20 blocks when curves saturate).

Expected after Day 1: same ~300k pos/h bootstrap, but games with real outcomes (the 100%-draw window becomes mixed within hours) — first meaningful value/policy gradients in the project's history.
Expected after Week 1: strength measurable; honest baseline ELO.
Expected after Month 1: not promised in ELO terms (no honest basis yet) — but every blocker identified by this audit is removed, and per Section B the hardware sustains AlphaZero-Zero-style learning loops at 1 GPU scale.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION I — COMPLETE TASK QUEUE LOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The living queue with every task, status marker, and finding cross-reference is maintained in **`audit/TASK_QUEUE.md`** (phases P0–P12, seed tasks S1–S10, self-generated tasks T0.x–T10.x, all resolved or explicitly deferred with justification: 4 deferred items — pool-width A/B, GRU-skew magnitude harness, post-fix Stockfish calibration, VL-exception-path leak — each gated on the Day-1 fixes or below the impact threshold). Phase detail: `audit/findings/phase00…phase11*.md`. Measurements: `audit/scripts/*.json` (verify_chess_logic, audit_network, bench_mcts, tb_curves, stress_train, stress_positions, test_puct_sign). Existing test suite: 26/26 green before and after the audit; working tree untouched outside `audit/`.
