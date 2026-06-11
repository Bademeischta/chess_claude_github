# Phase 0 — Orientation & Initial Scan

Status: complete (hardware CPU/RAM/disk probe results appended when background tasks finish)

## 0.x Hardware profile (measured)

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB VRAM, compute capability 12.0 (Blackwell), 250 W limit | nvidia-smi |
| Driver | 596.49 (CUDA 13.2 runtime capability) | nvidia-smi |
| Python | 3.11.8 | python --version |
| PyTorch | 2.11.0+cu128, CUDA 12.8, cuDNN 91900, `torch.cuda.is_available()=True` | torch probe |
| numpy / python-chess | 2.4.3 / 1.11.2 (match requirements.txt pins) | import probe |
| CPU / RAM / OS / disk | pending background probe (README claims Ryzen 7 8700F 8C/16T) | — |

## 0.1–0.3 File inventory & LOC (measured 2026-06-10)

Root files: config.py (334 ln), main.py (~50 KB), setup_ext.py, overnight_train.bat, chess_ext.cp311-win_amd64.pyd (built 2026-05-27), README.md (German), Chess_AI_Training.ipynb.

| Module | Files | LOC |
|---|---|---|
| engine/ (py) | 4 | 824 |
| engine/ (C++) | 5 | 1,656 (chess_core.cpp 38 KB, mcts_core.cpp 11 KB, bindings.cpp 11 KB) |
| mcts/ | 4 | 1,178 |
| model/ | 7 | 940 |
| training/ | 5 | 1,687 |
| utils/ | 5 | 651 |
| tests/ | 6 | 254 |
| tools/ | 12 | 4,378 |
| root (main/config/setup) | 3 | 1,415 |
| **Total Python** | | **~11,327** + 1,656 C++ |

Data directories: checkpoints/ 257 files 23.1 GB (latest step_0063400.pt, 2026-06-10 11:48), replay_buffer/ 9.6 GB (2 pickle.gz), runs/ 158 TB event files 26.8 MB, syzygy/ 290 files 938 MB, data/ 6.2 MB.

## 0.2 Entry points

`python main.py` (--phase/--test/--play/--resume/--grow/--sims/...), `overnight_train.bat` → tools/overnight_train.py → tools/train_supervisor.py (crash-restart wrapper), tools/elo_vs_stockfish.py, tools/web_gui.py, tools/uci_engine.py, tools/bench_infer.py, tools/lichess_bot.py.

## 0.4 Dependencies (requirements.txt, pinned — all verified installed)

torch==2.11.0 (cu128), numpy==2.4.3, chess==1.11.2, pybind11==3.0.4, psutil==7.2.2, tensorboard==2.20.0, onnx==1.21.0 (+optional onnxruntime-gpu).

## 0.5 Documentation

README.md (German, ~446 lines): documents 4-phase training plan (Syzygy pretrain → bootstrap → distillation → refinement), claimed perf RTX 5070: NN ceiling 15–30k evals/s, bootstrap 500–900k pos/h, standard 150–300k pos/h. Claims to verify in Phase 6. README claims "10 residual blocks (256ch), SE every 4, ~14M params" — parameter count verified in Phase 2.

## 0.6 Git history

9 commits, May 16 → Jun 10 2026. Trajectory: initial commit → training-efficiency PR → deep refactor + tests + pinned Stockfish → throughput/stability/web-UI overhaul → Syzygy pretraining + draw-collapse tuning → training state @63400. Branch feat/throughput-stability-webui. Working tree clean except AUDIT_PROMPT.md + audit/.

## 0.7 Config

Single dataclass `ChessAIConfig` (config.py), ~80 fields, global singleton CONFIG, runtime overrides via utils/system_probe.py (`#hw-derived` fields: device, precision, torch_compile, pin_memory, parallel_games, replay_buffer_cap, batch_size, dataloader_workers). Full hyperparameter table read and recorded; config comments double as a tuning diary (value-collapse at ~30k, cap 150→120→80, aux 0.2→2.0, resign 0.85/8→0.55/6, checkpoint 1000→200 due to native crashes).

## 0.8 Tests

pytest in tests/ (test_board, test_network, test_replay_buffer, test_trainer_components) + `python main.py --test` integration suite. Run in Phase 1 (results recorded there).

## 0.9 Evidence of past runs

Training ran to step 63,400 (clean stop 2026-06-10 11:48; checkpoints + buffer snapshots + TB logs share timestamp; no lock files). 257 checkpoints retained. `checkpoints/elo_history.csv` EMPTY (header only) → task S2. `pool_state.json` shows opponent pool at net_step0 @ 1000 ELO (+ net_step35000 present).

## 0.10 Red flags from surface scan

1. TODO/FIXME/HACK scan: **0 hits in project code** (verified by grep; only vendored Stockfish sources + AUDIT_PROMPT itself). Tuning history lives in config comments instead.
2. main.py ~50 KB single file (God-module candidate — Phase 8).
3. tools/ contains vendored full Stockfish source tree (77 MB) — checked into repo.
4. `chess_ext.pyd` build date 2026-05-27 — older than several .cpp edits? → task: verify .pyd is built from current C++ sources (compare behavior, not just dates; sources changed 2026-06-10).
5. Known native memory bug documented in three places (config.py:278-282, board.py:43-47, rules.py docstring).
6. Two backends (C++/python-chess) with duplicated semantics → parity risks (plane 20, see Phase 1).
7. `elo_history.csv` empty despite ELO tracking config fields.

## SELF-EXPAND tasks generated (added to TASK_QUEUE.md)

- T0.1 Verify chess_ext.pyd is built from the CURRENT C++ sources (pyd dated 2026-05-27, sources modified later) — if stale, every C++-source-level fix in repo may not be in the running binary!
- T0.2 Read TB event files to reconstruct loss curves & the 30k collapse (→P5).
- T0.3 Check pool_state.json/arena wiring — why only net_step0 + net_step35000 in pool after 63k steps (arena_every_n_steps=10_000 should have added ~6)?
- T0.4 Verify hw-derived overrides in system_probe.py — what values actually apply on this machine (parallel_games? batch?).
- T0.5 Check .gitignore vs committed 23 GB checkpoints (Git LFS) — operational cost note.
