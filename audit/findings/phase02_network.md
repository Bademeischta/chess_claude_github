# Phase 2 — Neural Network Architecture Audit (static; runtime script pending)

Files: model/network.py, model/heads.py, model/attention.py, model/inference.py (all fully read).

## 2.1 Input representation
21 planes (12 piece + stm + ep + 4 castling + halfmove/100 + fullmove/200 + repetition marker) + 4 GRU context planes = 25 into the tower. Always White-POV (no flip; plane 12 disambiguates). Values in [0,1]. dtype float32 from engine, cast to net dtype (bf16) at H2D. Encode/decode roundtrip + plane-20 issues covered in P1 (P1-2).

## 2.2 Residual tower
- ChessNet: input_proj Conv3×3(25→256)+GN(8)+GELU; N ResBlocks [Conv3×3→GN8→GELU→Conv3×3→GN8→(SE)→GELU(x+res)]; SE every 4 blocks (reduction 8). **Deployed N=10** (config; ChessNet signature default 20 is vestigial — build_model passes cfg). ckpt_arch.match_arch aligns to checkpoint depth at every load entry → S3 RESOLVED mechanism-wise; runtime confirmation from latest.pt pending.
- GroupNorm+GELU instead of BN+ReLU: deliberate, batch-size independent (good for CUDA-graph inference at varying effective batch); no train/eval BN divergence issues. Deviation from AZ noted in P9.
- No DataParallel/DDP ✓ single GPU.
- Gradient checkpointing auto-off at ≤14 blocks (build_model gc_from=num_blocks) ✓ matches comment.

## 2.3 Policy head
Conv1×1 256→32, GN(8), GELU, flatten, Linear 2048→4672. Raw logits, **no softmax in forward** ✓ (heads.py:58-67). masked_softmax helper masks BEFORE softmax with max-subtraction ✓ (heads.py:90-106) — but MCTS uses the numpy path (movegen.policy_array_to_legal_priors), trainer uses log_softmax; masked_softmax is a third, mostly-unused implementation (duplication noted P8).

## 2.4 Value head — FINDINGS

### FINDING P2-1 (CRITICAL, training effectiveness): aux head provides ZERO gradient to the value tower — the deployed anti-collapse fix is a placebo
heads.py:176-191: `wdl` comes from tower features x; `aux = tanh((material_diff @ piece_values)/scale)` where material_diff is computed in network.py:209-214 **directly from the input board planes** and `piece_values` is a standalone (6,) parameter. The tower, conv, fc1 are NOT in the aux computation graph.
config.py:102-113 justifies `aux_loss_weight: float = 2.0` with "2.0 makes the material-balance aux head the dominant gradient source, FORCING the value tower to learn something non-trivial". **False**: `2.0 * F.mse_loss(aux, mc_value)` (trainer.py:323-324) trains exactly 6 scalars (piece_values) and nothing else. The fix deployed against the step-30k value-head collapse cannot have done anything for the value head; it only inflates loss/total by a near-constant term (aux quickly converges to its best material-only fit) and distorts the logged loss curve.
Fix options: (a) feed tower features into the aux head (aux = f(x) supervised by material), (b) drop the term and address collapse at the root (see P2-2/P5), (c) keep piece_values as an interpretability probe with weight ~1e-3.

### FINDING P2-2 (HIGH, design): WDL head supervised only through P(win)−P(loss) ⇒ P(draw) unidentifiable; draw-collapse-prone by construction (S4 CONFIRMED)
heads.py:179 applies softmax → (B,3); trainer.py:298,311 trains `q = wdl[:,0]-wdl[:,2]` with MSE vs blended target. No cross-entropy over the 3 classes anywhere. Consequences: (1) P(draw) receives no direct supervision — (0.5,0,0.5) and (0,1,0) are both "q=0"; (2) the head is functionally a tanh-scalar with 1 wasted DOF; (3) the trivial solution q≡0 for draw-heavy data is exactly the collapse observed at step 30k. Standard fix (Lc0/KataGo): 3-class CE against {win,draw,loss} one-hot (or soft) targets — also gives a calibrated draw probability that MCTS contempt could use.

## 2.5 Parameters / memory (estimates; runtime script to confirm)
Per-layer estimate at 10 blocks/256ch: tower ≈ 10×(2×590k)+input_proj 57.6k ≈ 11.9M; SE×2 ≈ 33k; policy head ≈ conv 8.2k + FC 2048×4672 ≈ 9.57M(!); value head ≈ 17k+0.8k; GRU ≈ 1.63M; proj 65k. **Total ≈ 23.3M params** (README's "~14M" is off; FC policy head dominates — verify exact count at runtime). bf16 weights ≈ 47 MB; fp32 GRU adds. Checkpoint 92 MB ≈ weights + AdamW moments in bf16 — consistent. VRAM headroom analysis in P6.

### FINDING P2-3 (HIGH, training quality): entire network trained with bf16 MASTER weights (not autocast-mixed-precision)
network.py:305-306 `net = net.to(dtype=torch.bfloat16)` at build; trainer then autocasts (no-op on already-bf16 weights) and AdamW steps directly on bf16 parameters (bf16 moments). bf16 has 8 mantissa bits (~0.4% relative precision): weight updates smaller than ~0.2% of weight magnitude are rounded away — exactly the regime of late training (OneCycle LR decays to 1e-7·…). Standard recipe keeps fp32 master weights + autocast compute. Likely contributes to plateau/stall behavior. (GRU is fp32 — only it escapes.) Runtime check: confirm param/optimizer dtypes; recommend switching to fp32 weights + autocast(bf16), which on a 23M-param model costs ~70 MB extra VRAM — trivial on 12 GB.

## 2.6 Initialization
Kaiming-normal(fan_out, relu) for conv (≈right for GELU), Xavier for Linear (keeps 4672-logit field small → near-uniform softmax ✓), GN weight=1/bias=0 ✓, GRU PyTorch default. Value at init: Xavier fc_wdl → wdl≈uniform(1/3) → q≈0 ✓ (the desired near-zero init; runtime confirm).

## 2.7 Forward pass
Shapes verified by reading: (B,21,8,8)+(B,T,21,8,8) → policy (B,4672), wdl (B,3), aux (B,1). Batch-1 path used by _nn_evaluate ✓; history=None pads zeros ✓. GroupNorm → no train/eval divergence except checkpointing branch (training only). Gradient-flow runtime check pending (note: with P2-1, value_head.fc_wdl gets gradient only via WDL-MSE term; piece_values only via aux+anchor+decay).

## GRU encoder (attention.py)
(B,T,21·64) → 2-layer GRU(256) fp32 → Linear → (B,4,8,8) → GN(4). flatten_parameters per call. ~1.6M params. Sequential over T=8 → latency-bound at small batch (CUDA-graph mitigates). **History-skew finding P3-4 applies** (training history[-1]==current board; leaf eval history[-1]==root).

## Inference engines (inference.py)
- TorchInferenceEngine: model.infer ✓.
- CudaGraphInferenceEngine: static buffers, 3 warmup fwd on side stream, capture under inference_mode, pad-to-batch + narrow views; in-place weight updates keep graph valid (documented), `load_state_dict` invalidates (recapture needed — **check**: pool/arena load weights into deepcopies, not the live model ✓; play/web load BEFORE engine build ✓).
- OnnxInferenceEngine: fp32 export, fixed-batch padding (re-plan avoidance) ✓; CPU roundtrip per call (acceptable, optional backend).

## SELF-EXPAND tasks
- T2.1 Runtime: exact param count, per-layer table, dtype audit (weights/grads/optimizer moments), value-at-init, batch1-vs-N, grad-flow.
- T2.2 Inspect latest.pt: block count, policy_mid_channels, optimizer state dtypes, total size accounting.
- T2.3 Quantify aux-loss floor: compute best-achievable MSE of tanh(material/12) vs game outcome on real buffer data (shows what loss/total contains).
- T2.4 Estimate cost of bf16-master-weights: compare 1000-step from-scratch run bf16-master vs fp32-master+autocast (stress-test phase, P10).
- T2.5 SE every 4 with 10 blocks → SE at blocks 4,8 only (README says every 4 → fine, but 20-block claim in network.py docstring stale — P8 doc rot).
