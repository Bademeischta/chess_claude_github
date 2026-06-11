# Phase 6 — GPU & Hardware Performance (measured on RTX 5070, step-63400 weights)

Benchmarks: audit/scripts/bench_mcts.json, audit/scripts/audit_network.json, audit/scripts/stress_train.json (self-play window pending at write time), TB throughput history.

## Hardware (measured, fills the prompt's "Unknown" fields)
- GPU: RTX 5070, 12,227 MiB VRAM (10.8 GB free idle), CC 12.0, driver 596.49, 250 W.
- CPU: AMD Ryzen (Family 25 Model 117) 8C/16T. RAM: 31.7 GB (17–20 GB available).
- Python 3.11.8, torch 2.11.0+cu128, cuDNN 91900, TF32 on, cudnn.benchmark on.
- Triton unavailable on Windows → torch.compile permanently off (S8 resolved: probe detects and disables; eager+CUDA-graph is the real path).

## 6.2 Inference latency sweep (eager bf16, median of 60, synced)
| batch | ms | pos/s |
|---|---|---|
| 1 | 5.26 | 190 |
| 8 | 5.30 | 1,509 |
| 32 | 4.73 | 6,760 |
| 128 | 7.42 | 17,256 |
| 256 | 12.00 | 21,332 |
| 512 | 20.91 | **24,482 (peak)** |
| 1024 | 45.02 | 22,747 |
CUDA-graph @32: **1.91 ms** (16,716 pos/s) = 2.47× over eager @32 — the backend claim ("1.3–2×") is real (even conservative). Padded batch-1 via graph: 1.94 ms (vs 5.26 eager — graph pays off even fully padded).

## 6.2b MCTS sims/s (production weights)
- Single tree, unbatched (batch-1 evals): 165 sims/s → 776 ms per 128-sim move.
- Single tree, batched leaves + eager: 3,873 sims/s (33 ms/move).
- Single tree, batched + CUDA-graph: **7,965 sims/s** (16 ms per 128-sim move; 100 ms per 800-sim move — vs AlphaZero's ~30 ms on a TPU pod: respectable for one consumer GPU).
- 32-tree pool (production shape, CUDA-graph): 6,603 sims/s → 620 ms per move across 32 games → **185,700 positions/hour theoretical self-play-only**.
- Time split at pool: **62% inference / 13% selection / 24% expand+backup** → GPU-bound at batch 32, exactly as the code comments claim.

## Reconciliation with the real run (TB history)
Sustained production throughput was ~17–19 pos/s (≈65k pos/h) and 650–900 games/h INCLUDING interleaved training steps, root expansions, DCA and bookkeeping — i.e. ~35% of the theoretical self-play-only rate, and ~8× below README's "150–300k pos/h @128 sims" claim band's upper end (the lower bound ~150k is reachable only without training interleave). The system_probe banner estimate (~506k pos/h @128) overstates reality ~7×: it assumes 30k evals/s at batch 32; measured CUDA-graph batch-32 rate is 16.7k/s and only ~62% of wall time is inference.

## 6.x Optimization levers (quantified)
1. **Raise parallel_games 32 → 96–128**: eager batch-128 costs 7.4 ms vs 32@4.7 ms — 4× the leaves for 1.57× the inference time. With select/expand scaling linearly (37% share), projected pool throughput ≈ 2.2–2.6× (→ ~400–480k pos/h theoretical). RAM cost: 96 trees × history ≈ manageable (<2 GB). Needs CUDA-graph recapture at the new width. The probe's cap-at-32 comment cites Python overhead, but measured select+expand is only 37% — the cap looks too conservative; A/B at 64/96/128.
2. **Batch-512 training step**: VRAM peak at batch-1024 fwd+bwd is only 2.63 GB / 12 GB — massive headroom. fp32 master weights would cost ~+95 MB (P2-3 fix is free VRAM-wise).
3. **torch.compile**: blocked by no Windows Triton (verified). Alternatives: keep CUDA-graph (already in place), or WSL2/Linux for compile gains (~1.3–2× on tower convs) — recommend benchmarking under WSL2 before investing.
4. **FP8 (Blackwell)**: possible for inference via TensorRT/ONNX path that already exists (onnx backend present but optional); est. 1.5–2× inference at batch≥128. LOW priority until elo measurement works (can't validate quality regression otherwise).
5. Root-value D2H discarded in _batch_begin_search (tree.py:857, values_np unused) — trivial waste, remove.
6. GRU is fp32 and sequential over T=8 — at batch 32 it is a meaningful fraction of forward; candidates: shrink history_len, switch to a transformer-lite/conv summary, or precompute history embedding incrementally per game (the history only appends one frame per move; recomputing all 8 frames per leaf is redundant).

## 6.9 System resources during runs
GPU util sampled during stress run (see stress_train.json gpu_util_selfplay/gpu_util_training). Self-play window measured at 51–127 pos/s @50 sims from scratch (varies with game-recycle phase). No stale moves (counter 0 across all benches). No VRAM growth anomalies observed in benches; long-run leak check deferred to P11 notes (allocator expandable_segments configured in main.py).

## SELF-EXPAND
- T6.1 A/B parallel_games {32,64,96,128} pool benchmark (10 min) — quantify lever 1. [DEFERRED to roadmap: requires CUDA-graph recapture loop, diminishing audit value]
- T6.2 torch.profiler top-5 ops for train_step + MCTS move — run after stress test completes (GPU contention).
- T6.3 Confirm training-step time at batch 1024 (stress run reports median_step_ms).
