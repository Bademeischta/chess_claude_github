# Phase 11 — Security, Safety & Operational Concerns

## 11.1 Resource management
- SIGINT: main.py installs a handler that checkpoints before exit (main.py:64-72) ✓; supervisor forwards SIGINT and waits 120 s ✓; overnight wrapper same ✓. Good chain.
- CUDA contexts: single process; CUDA-graph static buffers live for process lifetime (bounded). Arena loads opponへents via deepcopy (trainer.run_arena) — opponent model + 2 fresh MCTSTrees per game ×200 games: deepcopies are released per loop iteration (no leak), but engine-less trees built per game are wasteful (arena currently dead anyway).
- VRAM fragmentation: PYTORCH_CUDA_ALLOC_CONF expandable_segments set via setdefault (main.py:175-178) — NOTE: set AFTER `import torch` and after CUDA may be initialized → env var read at allocator init; setting it post-import is only effective if CUDA not yet initialized. torch.cuda.is_available() was called in run_probe BEFORE this line → allocator already initialized → **the alloc-conf line is likely a no-op** (LOW finding P11-2; move env var before torch import / set in launcher).

## 11.2 Checkpoint integrity
- torch.load(weights_only=True) everywhere in the live paths ✓ (blocks pickle RCE from tampered .pt).
- Buffer pickles: plain pickle.load with explicit SECURITY comments (trusted-local-only) — acceptable for a local project; replay_buffer/ and checkpoint dirs should never accept third-party files.
- No checksums; no "last-known-good vs latest" separation — supervisor resumes from newest mtime which may be a truncated file after power loss (torch.save is not atomic; no .tmp+rename for checkpoints — _save_buffers does tmp+rename but is dead code). MEDIUM: make save_checkpoint atomic (tmp+replace) and keep one previous latest.
- Stockfish install: SHA256-pinned per commit message (not re-audited here; sentinel mechanism sound).

## 11.3 Long-run stability
### FINDING P11-1 (HIGH, ops): crash-restart re-runs Syzygy pretraining and floods the replay buffer
Supervisor relaunches `main.py --phase=all --resume=latest.pt`; phase_pretraining has no completed-marker, syzygy/ auto-detects, so each restart adds up to 500k endgame positions into a ~190-235k-cap ring buffer → **all accumulated self-play data evicted on every crash**, then ~1k steps train on endgame-only data before self-play resumes. With the documented native-crash frequency this repeatedly poisoned/reset the data distribution during the real run (visible as loss/value spikes; also explains replay.pkl.gz≪cap and replay_step≠global_step skews). Fix: a `pretraining_done` marker in checkpoint/pool state, or supervisor forwarding `--phase=selfplay` after first completion, or pretraining writing to a separate buffer.
- Counters: global_step int (Python) — no overflow. half-move clocks capped. SumTree float64 accumulates rounding from incremental updates over millions of update() calls (drift in internal nodes) — LOW: rebuild tree on buffer load (already effectively done) and periodically.
- Unbounded growth: hash_history cleared on irreversible moves ✓; _entropy_history capped 20 ✓; ELO _history list grows unbounded but tiny. Tree growth bounded by per-move sims + advance_root pruning ✓.
- File handles: TB writer closed via logger.close() at clean end; supervisor log opened append-per-line ✓; gzip/pickle in context managers ✓. python-chess SimpleEngine quit() called on both paths ✓ (but NOT on exception paths in elo tool — engine process could linger; LOW).
- The known native crash class: stale-move guard now prevents the original pieces[-1] UB; remaining suspects for the historical mate-in-2 crash: pybind nodelete dangling refs (P3-1). stale_move_count stayed 0 across all audit runs; the crash was not reproduced during the audit window.
