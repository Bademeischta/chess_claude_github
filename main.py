#!/usr/bin/env python3
"""
main.py — Chess AI entry point.

Usage:
    python main.py --phase=all              # Full training pipeline
    python main.py --phase=pretraining      # Endgame pretraining only
    python main.py --phase=selfplay         # Self-play bootstrap only
    python main.py --phase=distillation     # Teacher-distillation phase
    python main.py --phase=refinement       # Refinement phase
    python main.py --test                   # Run integration tests
    python main.py --resume=checkpoints/step_0050000.pt --phase=all
"""

from __future__ import annotations

import argparse
import os
import sys
import signal
import time
from pathlib import Path

# Force UTF-8 output on Windows so Unicode chars in print() don't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# Make sure the project root is on the Python path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import numpy as np

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe
from utils.logger import TrainingLogger
from utils.elo import ELOSystem
from model.network import build_model
from training.replay_buffer import PrioritizedReplayBuffer, TeacherBuffer
from training.self_play import SelfPlayWorker
from training.trainer import Trainer
from training.pbt import OpponentPool
from engine.rules import init_tablebase


# ── Graceful shutdown ────────────────────────────────────────────────────

_trainer_ref = None

def _sigint_handler(sig, frame):
    print("\n[main] Interrupt received — saving checkpoint before exit...")
    if _trainer_ref is not None:
        try:
            path = _trainer_ref.save_checkpoint()
            print(f"[main] Checkpoint saved to {path}")
        except Exception as e:
            print(f"[main] Checkpoint save failed: {e}")
    sys.exit(0)


# ── System startup ────────────────────────────────────────────────────────

def initialise_system(args) -> torch.device:
    """Probe hardware, update config, return device."""
    print("\n" + "=" * 60)
    print("  Chess AI — Initialising")
    print("=" * 60 + "\n")

    # Hardware probe + in-memory config update. write=False: do NOT rewrite
    # config.py source at runtime (fragile string-patching / race-prone) —
    # the override is applied in-memory via update_config_from_dict.
    derived = run_probe(write=False)
    update_config_from_dict(derived)

    # CUDA setup
    device = torch.device(CONFIG.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark  = True

    # Tablebases
    if CONFIG.syzygy_path:
        ok = init_tablebase(CONFIG.syzygy_path)
        print(f"[main] Syzygy tablebases: {'loaded' if ok else 'not found'}")

    print(f"[main] Device : {device}")
    print(f"[main] Precision: {CONFIG.precision}")
    print(f"[main] Batch size: {CONFIG.batch_size}")
    print(f"[main] MCTS sims (standard): {CONFIG.mcts_sims}")
    return device


# ── Integration tests ─────────────────────────────────────────────────────

def run_tests(device: torch.device) -> bool:
    """Run all integration tests. Returns True if all pass."""
    import traceback

    all_passed = True
    results = []

    def test(name: str, fn):
        nonlocal all_passed
        try:
            fn()
            results.append((name, True, None))
        except Exception as e:
            all_passed = False
            results.append((name, False, str(e)))
            traceback.print_exc()

    # 1. system_probe
    test("system_probe → config.py", lambda: run_probe(write=False))

    # 2. Board / move generator
    def test_movegen():
        from engine.movegen import run_perft_tests
        ok = run_perft_tests(verbose=True)
        assert ok, "Perft tests failed — check move generator"
    test("Perft tests (movegen)", test_movegen)

    # 2b. C++ ↔ python-chess parity (differential random-walk)
    def test_parity():
        from engine.movegen import run_parity_tests
        ok = run_parity_tests(n_games=120, max_plies=60, verbose=True)
        assert ok, "C++ engine diverges from python-chess — see log above"
    test("C++/python-chess parity", test_parity)

    # 3. MCTS mate-in-1 (50 sims — enough for a trivial mate, fast on Python backend)
    def test_mcts_mate():
        model = build_model(CONFIG, compile_model=False).eval()
        from mcts.tree import MCTSTree
        from engine.board import Board
        fen = "1r3rk1/5ppp/p1Rp4/8/8/1P6/P4PPP/4R1K1 w - - 0 1"
        board = Board.from_fen(fen)
        tree  = MCTSTree(CONFIG, model, device)
        move, _ = tree.search(board, 50, temperature=0.01, use_dca=False)
        assert move != 0, "MCTS returned null move for mate-in-1 position"
    test("MCTS mate-in-1", test_mcts_mate)

    # 4. Forward pass without OOM
    def test_forward_pass():
        model = build_model(CONFIG, compile_model=False)
        B = min(CONFIG.batch_size, 128)  # Use smaller batch for quick test
        dtype = (torch.bfloat16 if CONFIG.precision == "bf16" else
                 torch.float16  if CONFIG.precision == "fp16" else torch.float32)
        board   = torch.randn(B, CONFIG.input_planes, 8, 8, device=device, dtype=dtype)
        history = torch.randn(B, CONFIG.gru_history_len, CONFIG.input_planes, 8, 8,
                               device=device, dtype=dtype)
        with torch.no_grad():
            pol, wdl, aux = model(board, history)
        assert pol.shape == (B, CONFIG.num_actions), f"Policy shape wrong: {pol.shape}"
        assert wdl.shape == (B, 3),                  f"WDL shape wrong: {wdl.shape}"
        assert not torch.isnan(pol).any(),  "NaN in policy logits"
        assert not torch.isnan(wdl).any(),  "NaN in WDL"
        if device.type == "cuda":
            vram = torch.cuda.memory_allocated() / 1024**3
            assert vram < 8.0, f"VRAM usage too high: {vram:.2f} GB (limit 8 GB)"
    test("Forward pass (OOM + shape check)", test_forward_pass)

    # 5. 10 training steps
    def test_training_steps():
        model         = build_model(CONFIG, compile_model=False)
        # Use small capacity and low start threshold for test speed
        min_buf = 600
        replay_buffer = PrioritizedReplayBuffer(capacity=min_buf + 200,
                                                total_steps=CONFIG.total_steps)
        teacher_buffer= TeacherBuffer(capacity=200)
        pool          = OpponentPool(checkpoint_dir=CONFIG.checkpoint_dir)
        logger        = TrainingLogger(use_tensorboard=False)
        # Temporarily lower start threshold so training begins immediately
        orig_start = CONFIG.replay_start_training
        CONFIG.replay_start_training = min_buf
        trainer       = Trainer(CONFIG, model, replay_buffer, teacher_buffer,
                                pool, logger, device)

        from training.replay_buffer import PositionRecord
        for _ in range(min_buf + 50):
            pos = PositionRecord(
                board_tensor   = np.random.randn(21, 8, 8).astype(np.float32),
                history_tensor = np.random.randn(8, 21, 8, 8).astype(np.float32),
                policy_target  = np.random.dirichlet(
                    np.ones(CONFIG.num_actions)).astype(np.float32),
                wdl_label      = float(np.random.uniform()),
                mcts_q         = float(np.random.uniform(-1.0, 1.0)),
                phase          = int(np.random.randint(0, 3)),
                piece_count    = int(np.random.randint(2, 32)),
                move_number    = int(np.random.randint(0, 50)),
            )
            replay_buffer.add(pos)

        for step in range(10):
            metrics = trainer.train_step()
            assert metrics, "train_step returned empty metrics"
            loss = metrics.get("loss/total", float("nan"))
            assert not (loss != loss), f"NaN loss at step {step}"
        CONFIG.replay_start_training = orig_start  # restore

    test("10 training steps (loss finite)", test_training_steps)

    # 6. Self-play generates games (minimal sims for speed on Python backend)
    def test_self_play():
        model         = build_model(CONFIG, compile_model=False).eval()
        replay_buffer = PrioritizedReplayBuffer(capacity=50000,
                                                total_steps=CONFIG.total_steps)
        teacher_buffer= TeacherBuffer(capacity=5000)
        # Override sim counts and game length for test speed
        orig_sims         = CONFIG.mcts_sims
        orig_sims_teacher = CONFIG.mcts_sims_teacher
        CONFIG.mcts_sims         = 5
        CONFIG.mcts_sims_teacher = 5
        worker = SelfPlayWorker(CONFIG, model, replay_buffer, teacher_buffer, device)
        n = worker.generate_batch(2)  # game 0→teacher, game 1→replay
        CONFIG.mcts_sims         = orig_sims
        CONFIG.mcts_sims_teacher = orig_sims_teacher
        assert n > 0, "Self-play generated 0 positions"
        total_buf = len(replay_buffer) + len(teacher_buffer)
        assert total_buf > 0, f"Both buffers empty after self-play (n={n})"
    test("Self-play: 2 games without deadlock", test_self_play)

    # 7. Checkpoint save + load
    def test_checkpoint():
        import tempfile
        model         = build_model(CONFIG, compile_model=False)
        replay_buffer = PrioritizedReplayBuffer(capacity=5000,
                                                total_steps=CONFIG.total_steps)
        teacher_buffer= TeacherBuffer(capacity=1000)
        pool          = OpponentPool(checkpoint_dir=CONFIG.checkpoint_dir)
        logger        = TrainingLogger(use_tensorboard=False)
        trainer       = Trainer(CONFIG, model, replay_buffer, teacher_buffer,
                                pool, logger, device)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        trainer.save_checkpoint(path)
        trainer.load_checkpoint(path)
        os.remove(path)
    test("Checkpoint save + load", test_checkpoint)

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Integration Test Results")
    print("=" * 60)
    for name, passed, err in results:
        status = "✓  PASS" if passed else "✗  FAIL"
        print(f"  {status}  {name}")
        if err:
            print(f"         Error: {err[:120]}")
    print("=" * 60)
    if all_passed:
        print("\n  All tests PASSED ✓")
    else:
        n_fail = sum(1 for _, p, _ in results if not p)
        print(f"\n  {n_fail} test(s) FAILED ✗")
    print()
    return all_passed


# ── Model warmup ─────────────────────────────────────────────────────────

def warmup_model(model, cfg, device: torch.device) -> None:
    """Prime cuDNN autotuning with a dummy forward pass."""
    compile_active = cfg.torch_compile and hasattr(model, "_orig_mod")
    if compile_active:
        print("\n[main] Warming up neural network (torch.compile first-call JIT)...")
        print("       This can take 5-20 minutes on first run — please wait.", flush=True)
    else:
        print("\n[main] Warming up neural network (cuDNN autotune)...", flush=True)
    t0 = time.time()
    dtype = (torch.bfloat16 if cfg.precision == "bf16" else
             torch.float16  if cfg.precision == "fp16" else torch.float32)
    dummy_board = torch.zeros(1, cfg.input_planes, 8, 8, device=device, dtype=dtype)
    dummy_hist  = torch.zeros(1, cfg.gru_history_len, cfg.input_planes, 8, 8,
                               device=device, dtype=dtype)
    with torch.no_grad():
        model.infer(dummy_board, dummy_hist)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    if compile_active:
        print(f"[main] Neural network ready. Compilation took {elapsed:.0f}s.\n", flush=True)
    else:
        print(f"[main] Neural network ready ({elapsed:.1f}s).\n", flush=True)


def _train_steps_for(n_new: int, cfg, floor: int) -> int:
    """
    Number of gradient steps to run after a self-play batch.

    The continuous self-play pool now yields far more positions per batch than
    the old 6-game cohort (~1.2 K). Training a fixed 10/15/20 steps would
    drastically under-train. Scale steps so each generated position is reused a
    roughly constant number of times (~4×), with `floor` as the phase minimum.
    """
    reuse = 4
    return max(floor, round(n_new * reuse / max(1, cfg.batch_size)))


# ── Training phases ───────────────────────────────────────────────────────

def phase_pretraining(
    trainer, model, replay_buffer, teacher_buffer, pool, logger, worker, device
) -> None:
    """Phase 0: Endgame pretraining from Syzygy tablebases."""
    print("\n[Phase 0] Endgame Pretraining")
    print("─" * 40)

    if not CONFIG.syzygy_path:
        print("[Phase 0] No SYZYGY_PATH set — skipping pretraining.")
        print("          Set config.syzygy_path to enable Syzygy-based pretraining.")
        return

    n = worker.generate_endgame_pretrain(
        n_positions = CONFIG.pretrain_positions,
        syzygy_path = CONFIG.syzygy_path,
    )
    if n == 0:
        return

    # Train for pretrain_epochs (shared trainer — global_step persists)
    steps_per_epoch = n // CONFIG.batch_size
    for epoch in range(CONFIG.pretrain_epochs):
        print(f"[Phase 0] Epoch {epoch+1}/{CONFIG.pretrain_epochs} "
              f"({steps_per_epoch} steps)")
        for step in range(steps_per_epoch):
            metrics = trainer.train_step()
            if metrics and step % 500 == 0:
                logger.log(metrics, trainer.global_step)

    print(f"[Phase 0] Pretraining complete. Steps: {trainer.global_step}")


def phase_selfplay(
    trainer, model, replay_buffer, teacher_buffer, pool, logger, worker, device,
    target_elo: float = 1600.0,
    max_games: int = 50_000,
) -> None:
    """Phase 1: Self-play bootstrap until ELO target or max games."""
    print(f"\n[Phase 1] Self-Play Bootstrap (target ELO {target_elo:.0f})")
    print("─" * 40)

    # Fast-bootstrap: the model is randomly initialised, so 50 sims gives the
    # same quality as 200/800 and fills the buffer far faster. Teacher games
    # are capped too — an 800-sim rollout on a random net is no better than a
    # 50-sim one, it's just 16x slower. Ramp up once real training has begun.
    _orig_sims         = CONFIG.mcts_sims
    _orig_sims_teacher = CONFIG.mcts_sims_teacher
    _orig_start        = CONFIG.replay_start_training
    CONFIG.mcts_sims             = min(50, CONFIG.mcts_sims)
    CONFIG.mcts_sims_teacher     = min(50, CONFIG.mcts_sims_teacher)
    CONFIG.replay_start_training = min(5_000, CONFIG.replay_start_training)
    _ramp_step = CONFIG.bootstrap_ramp_step
    print(f"[Phase 1] Fast-bootstrap: sims={CONFIG.mcts_sims}/"
          f"{CONFIG.mcts_sims_teacher} (std/teacher, ramp to "
          f"{_orig_sims}/{_orig_sims_teacher} after step {_ramp_step:,}), "
          f"replay_start={CONFIG.replay_start_training:,}", flush=True)

    games = 0
    t0 = time.time()

    while games < max_games:
        # Ramp sims back up once the model has seen real training signal
        if CONFIG.mcts_sims < _orig_sims and trainer.global_step >= _ramp_step:
            CONFIG.mcts_sims         = _orig_sims
            CONFIG.mcts_sims_teacher = _orig_sims_teacher
            print(f"[Phase 1] Ramping MCTS sims to "
                  f"{_orig_sims}/{_orig_sims_teacher} (std/teacher)", flush=True)
        batch_num = games // CONFIG.parallel_games + 1
        rb_size   = len(replay_buffer)
        print(f"\n[Phase 1] Batch {batch_num} — generating {CONFIG.parallel_games} games "
              f"(replay buffer: {rb_size:,}/{CONFIG.replay_start_training:,})", flush=True)

        n_new = worker.generate_batch(n_games=CONFIG.parallel_games)
        games += CONFIG.parallel_games

        # Training steps after each game batch (scaled to positions generated)
        trained = 0
        for _ in range(_train_steps_for(n_new, CONFIG, floor=10)):
            metrics = trainer.train_step()
            if metrics:
                trained += 1
                logger.log({**metrics, **logger.get_vram_metrics()},
                           trainer.global_step)

        elapsed = time.time() - t0
        pos_s   = worker.positions_generated / max(elapsed, 1)
        g_hr    = games / max(elapsed / 3600, 1e-6)
        print(f"[Phase 1] Batch {batch_num} done — +{n_new} pos | "
              f"replay={len(replay_buffer):,} | "
              f"train_steps={trained} | "
              f"{pos_s:.0f} pos/s | {g_hr:.0f} games/hr | "
              f"step={trainer.global_step}", flush=True)

        # Arena evaluation
        if trainer.global_step % CONFIG.arena_every_n_steps == 0 and pool._pool:
            opp_path = pool.best_opponent_path()
            if opp_path:
                opp_model = pool.load_model_from_path(model, opp_path, device)
                wins, draws, losses = trainer.run_arena(model, opp_model, 50)
                elo_diff, _, _ = ELOSystem.elo_difference_ci(wins, draws, losses)
                new_elo, _ = pool.update_elo(
                    pool._current_id, "best_pool",
                    wins, draws, losses, trainer.global_step
                )
                logger.log_elo(new_elo, pool.best_pool_elo(), trainer.global_step)
                print(f"[Arena] Step {trainer.global_step}: "
                      f"W{wins}/D{draws}/L{losses}  "
                      f"ELO diff={elo_diff:+.0f}  new_ELO={new_elo:.0f}")

        # Register checkpoint in pool
        if trainer.global_step % (CONFIG.arena_every_n_steps // 2) == 0:
            pool.register_current(model, trainer.global_step)

        logger.log_throughput(pos_s, g_hr, trainer.global_step)

        if pool.current_elo() >= target_elo:
            print(f"[Phase 1] ELO target {target_elo:.0f} reached!")
            break

    CONFIG.mcts_sims             = _orig_sims
    CONFIG.mcts_sims_teacher     = _orig_sims_teacher
    CONFIG.replay_start_training = _orig_start
    print(f"[Phase 1] Done. Games: {games}  Steps: {trainer.global_step}")


def phase_distillation(
    trainer, model, replay_buffer, teacher_buffer, pool, logger, worker, device,
    target_elo: float = 2200.0,
    max_games: int = 200_000,
) -> None:
    """Phase 2: Teacher distillation + DCA active."""
    print(f"\n[Phase 2] Teacher Distillation (target ELO {target_elo:.0f})")
    print("─" * 40)

    games = 0
    while games < max_games:
        n_new = worker.generate_batch(n_games=CONFIG.parallel_games)
        games += CONFIG.parallel_games

        for _ in range(_train_steps_for(n_new, CONFIG, floor=15)):
            metrics = trainer.train_step()
            if metrics:
                logger.log({**metrics, **logger.get_vram_metrics()},
                           trainer.global_step)

        if trainer.global_step % CONFIG.arena_every_n_steps == 0 and pool._pool:
            opp_path = pool.best_opponent_path()
            if opp_path:
                opp_model = pool.load_model_from_path(model, opp_path, device)
                wins, draws, losses = trainer.run_arena(model, opp_model, 50)
                new_elo, _ = pool.update_elo(
                    pool._current_id, "best_pool",
                    wins, draws, losses, trainer.global_step
                )
                logger.log_elo(new_elo, pool.best_pool_elo(), trainer.global_step)
                print(f"[Arena] Step {trainer.global_step}: "
                      f"W{wins}/D{draws}/L{losses}  ELO={new_elo:.0f}")

        if pool.current_elo() >= target_elo:
            print(f"[Phase 2] ELO target {target_elo:.0f} reached!")
            break

    print(f"[Phase 2] Done. Games: {games}  Steps: {trainer.global_step}")


def phase_refinement(
    trainer, model, replay_buffer, teacher_buffer, pool, logger, worker, device,
    target_elo: float = 2400.0,
    max_games: int = 500_000,
) -> None:
    """Phase 3: Refinement — lower LR, higher teacher ratio."""
    print(f"\n[Phase 3] Refinement (target ELO {target_elo:.0f})")
    print("─" * 40)

    # Increase teacher ratio for refinement
    CONFIG.teacher_game_ratio  = 0.20
    CONFIG.teacher_batch_ratio = 0.35

    # Lower LR for refinement (one-time, on the shared trainer/optimizer)
    for pg in trainer.optimizer.param_groups:
        pg["lr"] *= 0.1

    games = 0
    while games < max_games:
        n_new = worker.generate_batch(n_games=CONFIG.parallel_games)
        games += CONFIG.parallel_games

        for _ in range(_train_steps_for(n_new, CONFIG, floor=20)):
            metrics = trainer.train_step()
            if metrics:
                logger.log({**metrics, **logger.get_vram_metrics()},
                           trainer.global_step)

        if trainer.global_step % CONFIG.arena_every_n_steps == 0 and pool._pool:
            opp_path = pool.best_opponent_path()
            if opp_path:
                opp_model = pool.load_model_from_path(model, opp_path, device)
                wins, draws, losses = trainer.run_arena(model, opp_model, 50)
                new_elo, _ = pool.update_elo(
                    pool._current_id, "best_pool",
                    wins, draws, losses, trainer.global_step
                )
                logger.log_elo(new_elo, pool.best_pool_elo(), trainer.global_step)

        if pool.current_elo() >= target_elo:
            print(f"[Phase 3] ELO target {target_elo:.0f} reached!")
            break

    print(f"[Phase 3] Done. Games: {games}  Steps: {trainer.global_step}")


# ── Play against the trained AI ───────────────────────────────────────────

def play_vs_ai(model, cfg, device, human_white: bool, sims: int,
               resume: str | None, show: bool = False,
               ponder: bool = False) -> None:
    """
    Interactive terminal game: human vs the trained network (MCTS).

    Moves are entered in UCI coordinate notation (e.g. ``e2e4``, ``e7e8q``).
    Type ``quit`` to exit. Loads ``--resume`` if given, else the newest
    checkpoint in the checkpoint dir (untrained net if none exists).

    `sims` is the search budget per AI move (the AI's "thinking depth" — more
    sims = stronger, slower). `show` prints what the AI calculated (eval, the
    moves it considered with visit counts, and the line it expects). `ponder`
    lets the AI keep searching the current position while *you* think; that
    work is reused (subtree kept) once you move.
    """
    import threading
    from engine.board import Board, STARTPOS_FEN
    from engine.rules import get_game_result, GameResult
    from engine.movegen import _move_uci
    from mcts.tree import MCTSTree

    from utils.ckpt_arch import latest_checkpoint
    ckpt = resume or latest_checkpoint(cfg.checkpoint_dir)
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        base.load_state_dict(state["model"])
        print(f"[play] Loaded checkpoint: {ckpt} (step {state.get('step', '?')})")
    else:
        print("[play] No checkpoint found — playing against an UNTRAINED net.")
    model.eval()

    try:
        import chess as _pc
        def render(fen: str) -> str:
            return str(_pc.Board(fen))
    except ImportError:
        def render(fen: str) -> str:
            return fen

    from model.inference import build_inference_engine
    engine = build_inference_engine(cfg, model, device)

    board = Board.from_fen(STARTPOS_FEN)
    # ONE persistent tree, reset once. Thereafter only advance() + sims, so the
    # searched subtree is reused across plies (and across pondering).
    tree  = MCTSTree(cfg, model, device, engine=engine)
    tree.reset(board)
    move_count = 0

    def _show(tag: str) -> None:
        a = tree.root_analysis()
        win = (a["value"] + 1.0) / 2.0 * 100.0
        tops = "  ".join(
            f"{_move_uci(m)}({v}, {q:+.2f})" for m, v, q in a["top"])
        pv = " ".join(_move_uci(m) for m in a["pv"])
        print(f"  [{tag}] eval={a['value']:+.3f} (win~{win:.0f}%)  "
              f"nodes={a['nodes']}\n  top: {tops}\n  pv:  {pv}")

    print(f"\n[play] You are {'White' if human_white else 'Black'}. "
          f"AI thinks at {sims} sims/move"
          f"{' (+pondering on your time)' if ponder else ''}. "
          f"Enter moves like 'e2e4'. 'quit' to exit.\n")

    while True:
        print(render(board.to_fen()))
        result = get_game_result(board)
        if result != GameResult.ONGOING or move_count >= cfg.max_game_moves:
            break

        human_turn = (board.side_to_move == 0) == human_white
        if human_turn:
            # Ponder: keep searching THIS position (your replies) while you
            # think. Stops as soon as you submit a move; the work survives in
            # the tree (advance keeps the chosen child's subtree).
            stop_ev = threading.Event()
            pth = None
            if ponder:
                def _ponder():
                    while not stop_ev.is_set():
                        tree.run_one_simulation()
                pth = threading.Thread(target=_ponder, daemon=True)
                pth.start()

            legal = board.legal_moves()
            uci_map = {_move_uci(m): m for m in legal}
            raw = input("Your move: ").strip().lower()

            if pth is not None:
                stop_ev.set()
                pth.join()
            if raw in ("quit", "exit", "q"):
                print("[play] Goodbye.")
                return
            if raw == "show":
                _show("ponder")
                continue
            if raw not in uci_map:
                print(f"  Illegal/unparseable. Legal: "
                      f"{' '.join(sorted(uci_map)[:30])}"
                      f"{' …' if len(uci_map) > 30 else ''}")
                continue
            move = uci_map[raw]
        else:
            print("AI thinking…", flush=True)
            for _ in range(sims):
                tree.run_one_simulation()
            a = tree.root_analysis()
            move = a["top"][0][0] if a["top"] else tree.select_move(0.01)[0]
            print(f"AI plays: {_move_uci(move)}")
            if show:
                _show("AI")

        board = board.apply_move(move)
        tree.advance(move, board)
        move_count += 1

    result = get_game_result(board)
    if result == GameResult.WHITE_WIN:
        outcome = "White wins"
    elif result == GameResult.BLACK_WIN:
        outcome = "Black wins"
    else:
        outcome = "Draw"
    print(f"\n[play] Game over — {outcome} ({move_count} half-moves).")


# ── Main entry point ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Chess AI Training")
    parser.add_argument(
        "--phase",
        choices=["pretraining", "selfplay", "distillation", "refinement", "all"],
        default="all",
        help="Training phase to run.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run integration tests only.",
    )
    parser.add_argument(
        "--syzygy",
        default=None,
        help="Path to Syzygy tablebase directory.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play an interactive game against the trained AI in the terminal.",
    )
    parser.add_argument(
        "--side",
        choices=["white", "black"],
        default="white",
        help="Your colour when using --play (default: white).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=None,
        help="MCTS simulations per AI move in --play (the AI's thinking "
             "depth; default: config value).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="In --play, print what the AI calculated each move "
             "(eval, candidate moves with visit counts, expected line). "
             "You can also type 'show' on your turn.",
    )
    parser.add_argument(
        "--ponder",
        action="store_true",
        help="In --play, let the AI keep thinking while it is your turn "
             "(the work is reused once you move).",
    )
    parser.add_argument(
        "--grow",
        type=int,
        default=None,
        metavar="N",
        help="Net2Net: grow the --resume checkpoint to N residual blocks "
             "(warm-started weights, fresh optimizer), write *_grownN.pt, exit.",
    )
    args = parser.parse_args()

    # Apply CLI overrides
    if args.syzygy:
        CONFIG.syzygy_path = args.syzygy

    # ── System initialisation ─────────────────────────────────────────
    signal.signal(signal.SIGINT, _sigint_handler)
    device = initialise_system(args)

    # ── Net2Net grow (weights only, then exit) ────────────────────────
    if args.grow is not None:
        if not args.resume or not Path(args.resume).exists():
            print("[grow] --grow requires an existing --resume checkpoint.")
            sys.exit(1)
        from model.net2net import grow_model
        out_path = args.resume.replace(".pt", f"_grown{args.grow}.pt")
        old_b, new_b = grow_model(args.resume, CONFIG, args.grow, out_path)
        print(f"[grow] {old_b} → {new_b} blocks. Written: {out_path}")
        print(f"[grow] Resume training with: "
              f"python main.py --phase=all --resume={out_path}")
        return

    # The checkpoint that will actually be loaded may have a different block
    # count than the config default (older 20-block runs, or a Net2Net-grown
    # net). Align the architecture to it BEFORE build_model so the state_dict
    # load succeeds. For --play with no --resume, that's the newest checkpoint
    # (exactly what play_vs_ai picks).
    from utils.ckpt_arch import latest_checkpoint, match_arch
    _eff_ckpt = args.resume
    if _eff_ckpt is None and args.play:
        _eff_ckpt = latest_checkpoint(CONFIG.checkpoint_dir)
    match_arch(CONFIG, _eff_ckpt)

    # ── Components ────────────────────────────────────────────────────
    model = build_model(CONFIG, compile_model=CONFIG.torch_compile)

    # ── Play against the AI (no training pipeline needed) ─────────────
    if args.play:
        sims = args.sims if args.sims is not None else CONFIG.mcts_sims
        play_vs_ai(model, CONFIG, device,
                   human_white=(args.side == "white"),
                   sims=sims, resume=args.resume,
                   show=args.show, ponder=args.ponder)
        return

    replay_buffer  = PrioritizedReplayBuffer(
        capacity    = CONFIG.replay_buffer_cap,
        alpha       = CONFIG.per_alpha,
        beta_start  = CONFIG.per_beta_start,
        beta_end    = CONFIG.per_beta_end,
        total_steps = CONFIG.total_steps,
    )
    teacher_buffer = TeacherBuffer(capacity=CONFIG.teacher_buffer_cap)
    pool           = OpponentPool(
        pool_size      = CONFIG.opponent_pool_size,
        checkpoint_dir = CONFIG.checkpoint_dir,
        initial_elo    = CONFIG.initial_elo,
    )
    pool.load_state()
    pool.register_current(model, step=0, elo=CONFIG.initial_elo)

    logger = TrainingLogger(
        log_dir  = CONFIG.tensorboard_dir,
        log_every = CONFIG.log_every,
    )
    worker = SelfPlayWorker(
        cfg            = CONFIG,
        model          = model,
        replay_buffer  = replay_buffer,
        teacher_buffer = teacher_buffer,
        device         = device,
    )

    # ── Single trainer shared across all phases ───────────────────────
    # One Trainer instance for the whole run so global_step, optimizer
    # moments and the LR schedule persist across phase boundaries.
    trainer = Trainer(CONFIG, model, replay_buffer, teacher_buffer,
                      pool, logger, device)
    global _trainer_ref
    _trainer_ref = trainer

    # ── Resume from checkpoint ────────────────────────────────────────
    if args.resume:
        trainer.load_checkpoint(args.resume)
        # Also try to load buffers
        buf_path = args.resume.replace(".pt", "_replay.pkl")
        if Path(buf_path).exists():
            replay_buffer.load(buf_path)
        tch_path = args.resume.replace(".pt", "_teacher.pkl")
        if Path(tch_path).exists():
            teacher_buffer.load(tch_path)
        print(f"[main] Resumed from step {trainer.global_step}")

    # ── Integration tests ─────────────────────────────────────────────
    if args.test:
        ok = run_tests(device)
        sys.exit(0 if ok else 1)

    # ── Model warmup (triggers torch.compile JIT, must happen before training) ──
    warmup_model(model, CONFIG, device)

    # ── Training ─────────────────────────────────────────────────────
    phase = args.phase

    kwargs = dict(
        trainer=trainer, model=model, replay_buffer=replay_buffer,
        teacher_buffer=teacher_buffer, pool=pool,
        logger=logger, worker=worker, device=device,
    )

    if phase in ("pretraining", "all"):
        phase_pretraining(**kwargs)

    if phase in ("selfplay", "all"):
        phase_selfplay(**kwargs)

    if phase in ("distillation", "all"):
        phase_distillation(**kwargs)

    if phase in ("refinement", "all"):
        phase_refinement(**kwargs)

    print("\n[main] Training complete.")
    pool.save_state()
    elo_path = Path(CONFIG.checkpoint_dir) / "elo_history.csv"
    pool.elo_system.save_csv(elo_path)
    logger.close()


if __name__ == "__main__":
    main()
