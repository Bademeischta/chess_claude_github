# TOTAL SYSTEMS AUDIT #2 — POST-FIX VERIFICATION (2026-06-12)

Re-execution of AUDIT_PROMPT.md against the post-fix codebase (HEAD = 2476993 +
seed tweak). The 2026-06-10/11 audit (`audit/AUDIT_REPORT.md`) found 4 critical
+ 9 high bugs; all fixes were committed in c83ddd4/2476993. **This audit
independently re-verified every fix from source and runtime evidence (not the
changelog), audited all code added since (hotfix 2476993), analyzed the first
post-fix production run, re-measured all benchmarks on the current 23.2M-param
network, and found + fixed 3 new defects (NEW-1..3, regression-tested).**

Hardware (probed): RTX 5070 11.9 GB CC 12.0 (48 SM) · Ryzen 8C/16T · 31.7 GB
RAM · WD SN3000 1 TB NVMe · Win 11 26200 · Python 3.11.8 · torch 2.11.0+cu128
· cuDNN 91900 · numpy 2.4.3 · python-chess 1.11.2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION A — EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The system now **demonstrably learns**: the first post-fix production run
(Jun 11, steps 0→9098) shows policy CE 3.67→1.84 and entropy 3.62→1.85 nats —
the pre-fix run was flat at CE ≈ 3.2 for 63k steps. All four critical fixes
from audit #1 were independently re-verified at source level and by runtime
proof (PUCT minimax convention mathematically consistent in both backends;
WDL logits/CE contract clean; cap adjudication sign-correct; eval-chain gates
fired in production at step 5008). Chess logic is perfect (16/16 suite, perft,
200-game parity). Both post-fix sessions ran 0 native crashes (pre-fix: 8 in
12.7h). Three NEW defects were found, fixed and regression-tested in this
session: (1) pool registrations were never persisted — a restart wiped the
pool and **overwrote the random ELO baseline with step-7828 weights** (proven
from file mtimes); (2) WDL labels/resigns derived the mover from move-number
parity — silently inverting every label for games seeded from Black-to-move
or odd-ply openings; (3) the legacy buffer-load path could silently corrupt
SumTree invariants on capacity change. Remaining improvement potential is
throughput (124k pos/h at 128 sims vs ~15.4k pos/s GPU inference ceiling) and
the still-unmeasured absolute strength — the first Stockfish calibration is
now unblocked. Tests: 40/40 pytest + 8/8 integration green.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION B — PERFORMANCE BASELINE (MEASURED 2026-06-12, RTX 5070)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Neural network (current 32-ch policy head, 10×256 SE-ResNet + 2-layer GRU):
- Parameters:                 **23,188,213** (all trainable) · 88.5 MB fp32
- Inference (eager, autocast bf16): b1 = 6.12 ms · b32 = 7.30 ms (4,386 pos/s)
  · b256 = 17.1 ms (14,981 pos/s) · b512 = 33.3 ms (**15,388 pos/s peak**)
- CUDA-graph @ b32 (production): **2.23 ms → 14,368 pos/s (3.3× vs eager)**
- Training step (b512, fwd+bwd+AdamW): **99.4 ms → 36,235 steps/h ceiling**
- VRAM: 2.25 GB bench peak · 4.28 GB observed production peak / 12 GB

MCTS (step-9098 weights):
- Sims/s: 107 unbatched → 2,864 batched eager → **5,676 batched CUDA-graph**
  (53× from batching+graph) · 32-tree pool: 4,412 sims/s
- 800-sim move: 141 ms (graph) · 7,489 ms (unbatched)
- Bottleneck: GPU inference — pool split **68.8% infer / 7.2% select /
  23.0% expand+backup**
- Move generation: 2.2 µs startpos · 5.5 µs middlegame · perft4 13.1M nodes/s
  (not remotely a bottleneck)

Self-play pipeline (production TB, Jun 11 run):
- Bootstrap (50 sims): 65–78 pos/s sustained incl. training interleave
- Post-ramp (128 sims + DCA): 15.5 pos/s ≈ 56k pos/h · ~1,160 games/h
  · ~48 plies/game average (decisive games; pre-fix: 100% pinned at cap)
- Pure 128-sim pool rate: **124k positions/hour**
- Stability: 0 crashes in 8.3 h across 2 sessions (pre-fix: 8 in 12.7 h)

Timeline estimates:
- 1M positions: ~4 h bootstrap · ~8–18 h at 128-sim quality
- Stockfish level 1/5: measurable for the FIRST time — eval chain verified
  live; first arena fires at step 10k (run was at 9,098 when stopped)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION C — CRITICAL BUGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**None open.** All four audit-#1 criticals re-verified fixed at source +
runtime:
- C1 PUCT: backup stores Q from the node's own mover (tree.py:461-468 /
  mcts_core.cpp:278-285); selection negates (`-Q − VL + U`, node.py:68-70 /
  mcts_core.cpp:49-62). Stub-net mate proof green on the shipped .pyd.
- C2 cap-draws: adjudication from final root Q, sign-correct via recorded
  side (improved by NEW-2 below); cap 160.
- C3 aux head: fc_aux consumes value-tower features (heads.py:190); gradient
  reach regression-tested.
- C4 eval chain: interval gates + real-id ELO + per-game series; **proven
  live** — pool registration fired at step 5008 in production.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION D — HIGH PRIORITY (NEW findings — FIXED in this session)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**NEW-1 · Pool state never persisted at registration → ELO baseline
overwritten on restart** — `training/pbt.py register_current()`
`save_state()` was only reachable via arena (first at step 10k) or graceful
shutdown. PROVEN in production: pool_state.json absent after 2 registrations;
net_step0.pt mtime = 13:48 restart ⇒ "random baseline" actually contains
step-7828 weights. Every pre-arena restart silently reset pool identity.
FIX (applied): `save_state()` inside `register_current()`; main.py baseline
guard additionally requires `net_step0.pt` absent on disk.
Test: `test_pool_register_persists_state`.
⚠ Operational note: the CURRENT checkpoints/net_step0.pt is corrupt as a
baseline (holds step-7828 weights). Delete it + net_step5008.pt before the
next fresh run, or accept a mislabeled anchor.

**NEW-2 · WDL labels & resign sides derived from mn%2 parity, not the
board** — `mcts/tree.py _finalize_record/game_stream`,
`training/self_play.py run_single_game`. Parity holds only while games start
White-to-move after an even ply count. An odd `random_opening_plies` (config
comment suggests "6–10"!), any Black-to-move opening-book FEN, or a
stale-move side desync inverts EVERY label in the game (and assigns resign
wins to the wrong side). Latent today (plies=6, no book) — but one config
edit away from silently corrupting all value targets.
FIX (applied): `GameRecord.sides` recorded from `board.side_to_move` at each
position; labels, cap adjudication and resign streaks now use it (parity
kept only as fallback for sides-less records).
Test: `test_wdl_labels_use_recorded_sides`.

**NEW-3 · Legacy buffer load: unchecked SumTree/data rebind on capacity
change** — `training/replay_buffer.py load()` + dead-armed caller in
main.py --resume. `tree.tree = state["tree"]` with a different capacity
breaks the leaf-offset invariant (wrong sampling); `_data` shorter than
capacity → IndexError in add().
FIX (applied): caller removed (Trainer._load_buffers is the single restore
path); load() hardened — data list normalized to capacity, tree rebuilt from
leaves when shapes differ. Test: `test_legacy_buffer_load_capacity_change`.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION E — MEDIUM PRIORITY (open, next iteration)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- E-1 **Snapshot staleness guard**: run-1 (pre-hotfix) lost ~4.7 h of
  self-play data because a stale Jun-10 replay.pkl.gz was silently restored
  (TB: buffer 168,003 → 5,648 at restart). Hotfix fixed writing; add a loud
  warning when the snapshot's age/step is far behind the checkpoint's.
- E-2 **run_single_game lacks cap adjudication** (labels cap-hits 0.5).
  Secondary path (tools/tests only) — port `_finalize_record` or delegate.
- E-3 **_load_buffers shrink path resets `_ptr = old_ptr % cap`** instead of
  0 after tail-truncation — overwrite order mixes ages on capacity shrink.
- E-4 **audit_network.py stale**: feeds bf16 inputs to fp32 weights without
  autocast (script predates the D3 fix) — crashes; product path unaffected.
- E-5 **heads.py module docstring** still says value head outputs "WDL
  probabilities" — it outputs logits since D1. Doc rot on a contract line.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION F — LOW PRIORITY & OBSERVATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- F-1 Deferred PER priority update can alias a ring-wrapped slot (1-step
  window, 2M cap — negligible).
- F-2 `gates={}` per process: every restart fires an arena after the first
  batch post-resume (harmless; arguably a free ELO datapoint).
- F-3 ERED verified working as designed — eps pinned at 0.10 floor is the
  dead-zone [2.0, 3.0] doing its job, not a stuck controller.
- F-4 Ruff: 121 findings (38 E701, 25 unused imports; 60 autofixable). No
  bare excepts. Zero TODO/FIXME outside vendored Stockfish.
- F-5 Stress suite: all terminals correct; the 9k-step net not finding the
  winning promotion is weak-net behaviour, not a logic bug (step-63400
  weights converted it in audit #1).
- F-6 Throughput knobs unchanged from audit #1: pool width 96–128 A/B,
  FP8/TensorRT, WSL2 torch.compile — all gated on strength measurement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION G — ARCHITECTURE RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

No structural changes recommended. Audit #1's architecture verdict stands
and is now runtime-validated: C++ engine + batched CUDA-graph inference +
crash supervisor are keepers; the learning loop works. The single
architecture-level decision still open is the GRU history encoder vs.
AZ-style stacked planes (D8/T3.2 ablation) — defer until the current run
produces its first Stockfish ELO curve, then ablate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION H — ROADMAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Day 1 (DONE in this session): NEW-1/2/3 fixed + 3 regression tests; 40/40.
Day 2: delete the corrupted net_step0.pt/net_step5008.pt pool files; start a
  fresh run (memory note: all old checkpoints predate the fix era anyway);
  let it cross step 10k → first live arena + first honest ELO datapoint;
  run tools/elo_vs_stockfish.py vs skill 0–3.
Week 2: E-1..E-5; draw-rate/game-length/grad-norm TB panels (F5 of audit #1).
Month: pool-width A/B (~2.2–2.6× projected self-play throughput), GRU
  ablation, c_puct/virtual-loss/FPU sweep on the now-learning system,
  Net2Net grow 10→20 blocks at curve saturation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION I — TASK QUEUE LOG (this audit)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[✓] P0   Orientation: 13.6k LOC, deps pinned, HW probed, delta-vs-audit = hotfix+seed
[!] P0.y Post-fix TB analysis → learning confirmed; exposed NEW-1 + snapshot loss (E-1)
[✓] T1   PUCT re-verified mathematically in BOTH backends + runtime stub proof
[✓] T2   Hotfix 2476993 snapshot code audited (layout assumption verified vs SumTree)
[✓] T3   WDL-CE: no double softmax; soft-target mapping exact for {1,.5,0}, TB labels OK
[✓] T4   Cap-adjudication sign verified; hardened by NEW-2
[!] T5   TB curves: policy CE 3.67→1.84 = first real learning; buffer-loss event found
[!] T6→NEW-2  mn%2 parity labels — FIXED + test
[!] T7→NEW-1  pool persistence — FIXED + test (production-proven impact)
[!] T8→NEW-3  legacy load path — FIXED + test
[✓] P1   16/16 chess-logic suite (perft, parity, bijectivity, EP, plane-20)
[✓] P2   Heads/precision contracts verified (logits, fp32 master, GRU fp32 island)
[✓] P3   MCTS: backup/selection/noise(Gamma)/temperature(log-space)/VL verified
[✓] P4   Self-play: game_stream trace, fresh-board re-deal, sentinel guard, resign
[✓] P5   Trainer: step order, NaN guard/step, per-group OneCycle, PER math, AdamW fp32
[✓] P6   GPU: full latency sweep, CUDA-graph, train step, VRAM, time-split (Section B)
[✓] P7   Eval chain re-verified + live-fire evidence (step-5008 registration)
[✓] P8   Ruff 121 (style-tier), no bare excepts, no TODOs, docstring rot noted (E-5)
[✓] P9   Audit #1 comparative analysis stands; no new deviations introduced
[✓] P10  Stress: 11 tricky positions, terminals, determinism, stale_moves=0
[✓] P11  0 crashes post-fix (8 pre-fix); atomic saves verified; weights_only load
[✓] P12  Queue exhausted; deferred (justified): pool-width A/B, GRU ablation,
         FP8 — all gated on first ELO measurement (hardware time, not audit work)

Verification after fixes: pytest **40/40** · main.py --test **8/8** ·
chess-logic **16/16** · PUCT proof verdict_inverted=false.
