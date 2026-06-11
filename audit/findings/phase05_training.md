# Phase 5 — Training Loop & Optimization Audit (static; TB-curve + runtime checks pending)

File: training/trainer.py (841 ln, fully read), main.py phase drivers, utils/elo.py.

## 5.1 Loss anatomy
- **Policy**: `log_probs = F.log_softmax(policy_logits.float()); loss = -(targets * log_probs).sum(-1)` (trainer.py:290-291). Network emits raw logits (verified P2) → **no double-softmax** ✓. Soft targets = full MCTS visit distribution ✓. Illegal moves: targets 0, gradient flows through denominator (pushes illegal logits down) — standard AZ ✓. fp32 upcast before softmax ✓.
- **Value**: q = wdl[:,0]−wdl[:,2] (softmaxed WDL); target = TD(λ)-blend of MC return (mapped [0,1]→[-1,1]) with recorded MCTS root Q, window moves 3..100, λ=0.8 ⇒ target = 0.8·z + 0.2·q_mcts (trainer.py:297-311). Spaces consistent ([-1,1] both, side-to-move perspective both — verified against tree.py recording) ✓. MSE. The bootstrap is the RECORDED search Q (grad-free) ✓ not self-prediction. See P2-2 for the WDL/CE critique.
- **Aux**: 2.0 × MSE(aux, z) — trains only 6 piece_values (P2-1 CRITICAL placebo). Plus 1e-3 anchor MSE(piece_values, classic).
- **Teacher KL**: `F.kl_div(log_softmax(student_logits), teacher_visit_targets, batchmean)` = KL(p_teacher‖p_student) ✓ correct direction; zero-targets safe (xlogy). Teacher data = 256-sim self-play games (self-distillation from deeper search), 30% of every batch from a 200k uniform buffer.
- **Combination**: mean(per-sample (α_p·CE + α_v·MSE) · IS_weight) + 2.0·aux + anchor + 1.0·KL. Phase LUTs α_p∈{1.5,1.0,0.5}, α_v∈{1.0,1.0,1.8}·alpha_v_global(ERED).
- **L2**: AdamW weight_decay=1e-4 on ALL params — GroupNorm scales/biases and piece_values not excluded (common-practice deviation, LOW).

## 5.2 Optimizer / schedule — FINDINGS

### FINDING P5-1 (HIGH — silent config no-op): OneCycleLR erases the per-group LR multipliers
trainer.py:71-77 builds 3 param groups with lrs {1.0×, 0.5×(heads), 0.3×(GRU)}·1e-3; trainer.py:99-107 then constructs `OneCycleLR(optimizer, max_lr=cfg.lr_initial, ...)` with a SCALAR max_lr — PyTorch broadcasts it to every group and overwrites group lrs each step. **lr_head_mult and lr_gru_mult do nothing.** All groups follow the same 4e-5→1e-3→1e-7 cosine. Fix: pass `max_lr=[lr*mult for each group]`.

### FINDING P5-2 (MEDIUM — placebo #5): phase_refinement LR×0.1 is overwritten by the scheduler
main.py:754-755 multiplies `pg["lr"] *= 0.1` once; the next `scheduler.step()` recomputes lr from the OneCycle curve → the reduction survives <1 step. The documented "Refinement: LR ×0.1" phase behavior does not exist.

- AdamW fused on CUDA ✓ (fallback try/except ✓). Defaults β=(0.9,0.999) un-tuned (fine).
- Schedule sanity: total_steps=500k, warmup 2k, div 25/1e4. At step 63,400 (12.7% in) lr ≈ near-peak ~9.7e-4.
- `if self.global_step < cfg.total_steps: scheduler.step()` guards OneCycle overflow ✓.

## 5.3 Step trace (verified order)
zero_grad(set_to_none) → autocast forward (×2: replay + teacher) → losses fp32 → backward → clip_grad_norm(5.0) → step → scheduler.step. FP16 path would correctly scale→unscale→clip→step→update ✓ (unused; bf16 active). model.train() during training ✓ / .eval() in self-play (worker sets eval) ✓ — GroupNorm makes the flag nearly irrelevant except grad-checkpointing branch ✓.

### FINDING P5-3 (carried from P2-3, HIGH): the whole pipeline optimizes bf16 master weights (no fp32 master copy). AdamW moments in bf16. Update-rounding risk at decayed LR.

## 5.4 Numerical stability
- NaN guard (trainer.py:359-371): `torch.isfinite(loss)` only every 50 steps (sync-cost mitigation), tight-mode 10 steps after a hit. **Contradiction with its own rationale** ("a single backward with NaN permanently corrupts AdamW moments"): up to 49 un-checked steps backward+step on non-finite loss before detection. MEDIUM. Cheaper alternative: check the already-synced `loss/total` metric value each step host-side (it's `float(loss.cpu())` anyway at line 428!) — the guard could be free and per-step. (The metric sync at line 428 means the every-step cost argument is moot — flag as easy fix.)
- log(0): log_softmax safe; kl_div xlogy-safe; policy_array softmax guarded; masked_softmax nan_to_num-guarded ✓.
- 0/0: Q=W/N guarded by N increment-before-divide in backup ✓; policy renorm fallback uniform ✓.
- Expected initial losses: policy ≈ ln(~uniform over 4672 with ~30 legal …) — CE vs soft targets starts ≈ -Σπ·log(1/4672) ≈ 8.45 (not ln 30 — full-vector softmax); value MSE ≈ E[z²] given q≈0 → ~0.5 with decisive data, ~0 with draws. Verify in stress test.

## 5.5 Mixed precision
bf16 autocast, no GradScaler (correct for bf16) — but see P5-3/P2-3: weights pre-cast to bf16 makes autocast moot. GRU island fp32 ✓. TF32 enabled globally (main.py:146-158) ✓.

## 5.6 Checkpointing
- Saved: step, model, optimizer, scheduler, scaler(None), ERED state (eps, alpha_v, entropy history), replay_step (β-anneal) — thorough.
- **RNG states NOT saved** (torch/numpy/python/CUDA) → resume is not bit-exact (P10.4 expectation: divergence).
- load: weights_only=True ✓ (anti-RCE), legacy migrations (fc_aux drop, policy-head reinit on shape change) with allowlist; optimizer-restore failure handled loudly ✓; scheduler restored without double-advance ✓.
- emergency_save → latest.pt every 3 self-play batches ✓ bounded crash loss.

### FINDING P5-4 (MEDIUM, ops): buffer persistence is incoherent
(a) `_save_buffers` (gz, rolling replay.pkl.gz) has **no callers**; (b) train_step writes UNCOMPRESSED `step_xxx_replay.pkl` every checkpoint_every·5 = 1000 steps via `replay_buffer.save()` (trainer.py:410-413) — multi-GB per dump at scale, yet none exist on disk (reconcile at runtime — silent failure? exception would crash train_step: unguarded!); (c) config `buffer_save_every=1` consumed by nothing; (d) resume path loads BOTH replay.pkl.gz (inside load_checkpoint) and step_xxx_replay.pkl (main.py:1107-1112) if present. The committed replay.pkl.gz/teacher.pkl.gz were presumably written by an older code revision (mtimes are git-LFS checkout artifacts). Crash-recovery may silently rewind the buffer to a stale snapshot, exactly what config.py:284-291 says it was tuned to avoid.

## 5.7 Monitoring
TensorBoard via utils/logger.py (loss/total each step; policy/value/KL/aux/entropy each 100). No gradient-norm logging (clip happens but the norm isn't recorded — cheap add). NaN events only printed. ELO logging wired but arena dead (see P7). TB event files committed → curve reconstruction task T0.2 still pending (runtime).

## Arena (trainer.run_arena) — feeds P7
### FINDING P5-5 (MEDIUM): evaluation games are noisy
run_arena (trainer.py:794-840) calls `tree.search(board, n_sims, temperature=cfg.temperature_final /*0.1*/, use_dca=False, keep_history=True)` — **add_noise defaults True** → root Dirichlet noise in eval games; τ=0.1 sampling, not argmax. Violates eval-determinism best practice (AUDIT items 7.1); weakens discrimination between nets.
### FINDING P5-6 (HIGH, wiring — with main.py): arena effectively never runs
main.py gates arena on `trainer.global_step % arena_every_n_steps == 0` but global_step advances in chunks of `_train_steps_for(...)` (≈10-25) — the modulo almost never hits exactly. Same for pool registration (`% 5000`). Consequences (all confirmed by artifacts): elo_history.csv empty; phase ELO-targets unreachable (pool.current_elo() stays 1000) so phases only end at max_games; pool stale. ALSO main.py:1081 `pool.register_current(model, step=0, elo=initial)` runs at EVERY startup, overwriting net_step0.pt with the CURRENT weights — the pool's "old" opponent is secretly the newest net. AND main.py:669-671 updates ELO against literal id "best_pool" instead of the actual best entry's player_id. AND elo.update_from_series collapses a whole 100-game series into ONE K=32 update (max ±32 ELO per arena regardless of score). Four stacked defects — the evaluation/promotion subsystem is non-functional end-to-end.

## SELF-EXPAND tasks
- T5.1 Reconstruct TB curves (loss/total, loss/value, policy_entropy, lr) → verify 30k collapse, aux-floor (T2.3), current health at 63.4k.
- T5.2 Runtime: confirm OneCycle group-lr clobbering (read optimizer.param_groups after 2 steps).
- T5.3 Runtime: confirm param/moment dtypes (P5-3).
- T5.4 Buffer-persistence reconcile (P5-4) — also check train_step's unguarded replay_buffer.save would crash on disk-full.
- T5.5 Initial-loss sanity in stress run: policy ≈ 8.45 then quick drop; value path per P2-2.
- T5.6 Per-step NaN guard via existing loss/total sync (free fix) — verify claim that .item() sync already happens every step.
