# Phase 9 — Comparative Analysis vs Reference Implementations

## 9.1 AlphaZero (Silver et al. 2018) compliance table

| Dimension | AlphaZero | This project | Verdict |
|---|---|---|---|
| Input planes | 119 (8-step history ×14 + 7 aux) | 21 (no in-plane history) + GRU(8 frames)→4 context planes | Deliberate experiment. Risky: history via fp32 GRU costs latency, and P3-4 shows the train/search history convention is inconsistent — AZ's stacked planes cannot have that bug. |
| Perspective | Board flipped to side-to-move | Board always White-POV + stm plane; ACTIONS 180°-rotated for Black | Deliberate but costly (P1-4): policy must learn two action semantics; halves effective policy sample efficiency. No reference engine does this. |
| Tower | 20 blocks × 256, BN+ReLU | 10 × 256, GroupNorm+GELU, SE every 4 | Reasonable downscale for 1 GPU. GN+GELU is modern and CUDA-graph friendly (no BN batch-stat issues) — keep. SE = Lc0-style, good. |
| Policy head | conv 73-plane spatial | conv 1×1 → 2 ch (deployed!) → FC 128→4672 | The DEPLOYED head is a 128-dim bottleneck for a 4672-dim output — far below both AZ and this repo's own intended 32-ch version (P2-4). Likely a key cause of the flat policy loss. |
| Value head | scalar tanh, MSE | WDL softmax but trained as scalar P(w)−P(l) MSE (+ disconnected aux) | Worst of both worlds (P2-2): no calibrated draw prob, collapse-prone. Lc0/KataGo use real WDL cross-entropy — adopt. |
| MCTS sims | 800 | 50 (bootstrap≤8k) / 128 (+DCA to 256) | Defensible for hardware; DCA is a sensible original idea. |
| c_puct | ~1.0–1.25 grow-with-N | 2.0 fixed | In family; tune after value head works. |
| Dirichlet | α=0.3, ε=0.25, root-only | same, root-only, once per move ✓ | Compliant (and the C++ gamma fix is correct). |
| Temperature | τ=1 first 30 plies → argmax | τ=1 first 30 → τ=0.1 | Near-compliant; 0.1 ≈ argmax in practice. |
| Game cap | 512 half-moves, draw label | **80 half-moves, draw label** | 6.4× tighter than AZ — at 80 the mislabeled fraction is large (P4-1). AZ could afford draws at 512 because games rarely got there. |
| Replay buffer | 500k positions, uniform | ~190–235k (RAM-derived), PER(α=0.6, β anneal) | PER on self-play data is a known double-edged sword (KataGo uses uniform-ish with fresh-data emphasis); combined with max-priority insertion it over-samples fresh noisy games. Acceptable experiment, but it interacts badly with draw-flooded data. |
| LR schedule | step decays 0.2→0.02→0.002 over 700k | OneCycle cos to 500k, peak 1e-3 (group mults broken, P5-1) | OneCycle is fine; fix P5-1. |
| Batch | 4096 (TPU farm) | 1024 | Fine for 1 GPU. |
| Train:data ratio | ~1 sample reuse <1 | reuse ≈ 4× | Higher reuse is normal on small compute; with PER it risks overfit to fresh games. |
| Eval gating | none (AZ) / SPRT (Lc0) | designed but non-functional (P5-6/P7-1) | Fix. |

## 9.2 What references do better / what this repo does better
Adopt from references (top 3):
1. **WDL cross-entropy value head** (Lc0/KataGo) — directly fixes the collapse mechanism; gives draw calibration that MCTS contempt and resign can actually use.
2. **Board-flip perspective canonicalization** (everyone) — halves the policy learning problem; requires re-encoding (breaks checkpoint compat — schedule with the policy-head widening since that also resets the head).
3. **Auxiliary targets that feed the trunk** (KataGo: ownership/score; here even simple "material-in-K-moves" or opponent-policy aux) — the intended spirit of the aux head, implemented so gradients reach the tower (P2-1).

Genuinely good/original here (top 3):
1. CUDA-graph batched-leaf inference engine with padding + in-place weight updates — clean, measured 2.5×, rare in hobby AZ implementations.
2. The C++ engine quality: magic bitboards + honest parity testing vs python-chess + perft suite; stale-move diagnostic counter is a thoughtful invariant probe.
3. Operational hardening: supervisor restart loop, emergency checkpoints, faulthandler, graceful SIGINT — beyond most reference hobby implementations (undermined only by the pretraining-on-restart flaw P11-1).

## 9.3 Novel elements classification
| Element | Class | Verdict |
|---|---|---|
| GRU history encoder | deliberate experiment | Fix P3-4 convention or drop; measure value before keeping (T3.2). Cheaper alternative: AZ-style stacked history planes (no latency, no skew). |
| DCA (dynamic sims) | deliberate, sound | Keep; benign. |
| TD(λ) blend w/ search-Q | deliberate, sound (KataGo-adjacent) | Keep; harmless and likely helpful once labels are clean. |
| PER + draw down-weight + teacher buffer | deliberate; draw-weight dead (P4-3), teacher = self-distill from 256 sims | Keep teacher; implement or remove draw-weight; consider uniform sampling ablation. |
| Resign mechanism | deliberate | Keep, but only meaningful once the value head is informative. |
| ERED entropy controller | deliberate; half dead (P4-4) | Wire eps or remove; with target 2.5 vs measured ~3.5 it fought a losing battle all run. |
| 80-move cap | deliberate desperation | Replace with adjudication/Q-labeling (P4-1). |
| Learnable piece values (aux) | deliberate; disconnected (P2-1) | Re-wire or demote to probe. |
