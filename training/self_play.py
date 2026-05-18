"""
training/self_play.py

Asynchronous self-play data generation.

Architecture:
  - A SelfPlayWorker runs in the main process (alongside training).
  - It uses the ParallelMCTS from mcts/tree.py to generate games.
  - Teacher games (10 % of total) are run at 800 simulations.
  - Generated positions are pushed to the replay/teacher buffers.

For truly asynchronous operation, SelfPlayWorker can be run in a
background thread; the trainer picks up positions from a thread-safe queue.
"""

from __future__ import annotations

import time
import queue
import threading
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from engine.board import Board, STARTPOS_FEN
from engine.rules import (
    get_game_result, GameResult, probe_tablebase, is_tablebase_available,
)
from mcts.tree import MCTSTree, ParallelMCTS, GameRecord, fill_history
from training.replay_buffer import (
    PrioritizedReplayBuffer, TeacherBuffer, PositionRecord,
)


class SelfPlayWorker:
    """
    Generates self-play games and pushes them into the replay buffers.

    Can run synchronously (call `generate_batch` directly) or
    asynchronously via `start_background` / `stop_background`.
    """

    def __init__(
        self,
        cfg,
        model: nn.Module,
        replay_buffer: PrioritizedReplayBuffer,
        teacher_buffer: TeacherBuffer,
        device: torch.device,
    ) -> None:
        self.cfg            = cfg
        self.model          = model
        self.replay_buffer  = replay_buffer
        self.teacher_buffer = teacher_buffer
        self.device         = device

        self._trees = [
            MCTSTree(cfg, model, device) for _ in range(cfg.parallel_games)
        ]

        # Persistent self-play pools. Created lazily on first use and kept
        # alive across batches so the game pool never drains between calls
        # (a fresh ParallelMCTS per batch would re-introduce the
        # shrinking-cohort tail at every batch boundary).
        self._std_stream = None
        self._tch_stream = None

        self._thread:       Optional[threading.Thread] = None
        self._stop_event    = threading.Event()
        self._stats_lock    = threading.Lock()

        # Running statistics
        self.games_generated   = 0
        self.positions_generated = 0
        self._last_stats_time  = time.time()

    # ── Single-game generation ────────────────────────────────────────────

    def run_single_game(
        self, tree: MCTSTree, is_teacher: bool = False
    ) -> GameRecord:
        """
        Play one complete game from the starting position.
        Returns a GameRecord with all positions, policy targets, and WDL labels.
        """
        n_sims = self.cfg.mcts_sims_teacher if is_teacher else self.cfg.mcts_sims
        board  = Board.from_fen(STARTPOS_FEN)

        record = GameRecord()
        record.is_teacher = is_teacher

        move_num = 0
        while True:
            temperature = (
                1.0 if move_num < self.cfg.temperature_moves
                else self.cfg.temperature_final
            )

            # Capture board state before the move
            board_t   = board.to_tensor()
            history_t = self._get_history(tree)

            # MCTS search. keep_history after move 0 so the GRU history that
            # advance() maintains is not wiped every move.
            move, policy_target = tree.search(
                board, n_sims, temperature, use_dca=True,
                keep_history=(move_num > 0),
            )

            # Record
            record.board_tensors.append(board_t)
            record.history_tensors.append(history_t)
            record.policy_targets.append(policy_target)
            record.mcts_qs.append(tree.root_value())
            record.move_numbers.append(move_num)
            record.phases.append(board.get_phase())
            record.piece_counts.append(board.piece_count())

            # Apply move
            new_board = board.apply_move(move)
            tree.advance(move, new_board)
            board = new_board
            move_num += 1

            # Game termination
            result = get_game_result(board)
            if result != GameResult.ONGOING or move_num >= self.cfg.max_game_moves:
                if result == GameResult.WHITE_WIN:
                    game_val = 1.0
                elif result == GameResult.BLACK_WIN:
                    game_val = 0.0
                else:
                    game_val = 0.5
                record.game_result = game_val

                # Assign per-position WDL labels
                for i, mn in enumerate(record.move_numbers):
                    side = mn % 2  # 0=white, 1=black
                    # Tablebase override where available
                    tb_board = None
                    wdl = (game_val if side == 0 else 1.0 - game_val)
                    record.wdl_labels.append(wdl)

                break

        return record

    def _get_history(self, tree: MCTSTree) -> np.ndarray:
        T = self.cfg.gru_history_len
        buf = np.zeros((T, self.cfg.input_planes, 8, 8), dtype=np.float32)
        return fill_history(buf, tree._board_history, T)

    # ── Batch generation (synchronous, parallel via ParallelMCTS) ────────────

    def generate_batch(self, n_games: int) -> int:
        """
        Generate `n_games` self-play games in parallel and push positions to buffers.
        Teacher and standard games are batched separately so each group can use
        its own sim budget. Returns total positions generated.
        """
        self.model.eval()
        t_batch = time.time()

        # Determine teacher/standard split based on global game counter
        teacher_every = max(1, int(1.0 / self.cfg.teacher_game_ratio))
        game_base = self.games_generated
        n_teacher = sum(
            1 for i in range(n_games)
            if (game_base + i) % teacher_every == 0
        )
        n_standard = n_games - n_teacher

        parts = []
        if n_standard > 0:
            parts.append(f"{n_standard} standard(sims={self.cfg.mcts_sims})")
        if n_teacher > 0:
            parts.append(f"{n_teacher} teacher(sims={self.cfg.mcts_sims_teacher})")
        print(f"  [{', '.join(parts)}] | "
              f"replay={len(self.replay_buffer):,} teacher={len(self.teacher_buffer):,}",
              flush=True)

        total = 0

        if n_standard > 0:
            if self._std_stream is None:
                # Concurrency = the GPU-sized pool, but never wider than this
                # call asks for (keeps the 2-game integration test cheap).
                pool = max(1, min(self.cfg.parallel_games, n_standard))
                pmcts = ParallelMCTS(self.cfg, self.model, self.device,
                                     n_games=pool)
                self._std_stream = pmcts.game_stream(is_teacher=False)
            for _ in range(n_standard):
                record = next(self._std_stream)
                positions = self._record_to_positions(record)
                self.replay_buffer.add_batch(positions)
                total += len(positions)

        if n_teacher > 0:
            if self._tch_stream is None:
                # Teacher games cost up to 16x more sims each — run a smaller
                # concurrent pool so they don't dominate wall-time.
                pool = max(1, min(self.cfg.parallel_games // 6, n_teacher))
                pmcts = ParallelMCTS(self.cfg, self.model, self.device,
                                     n_games=pool)
                self._tch_stream = pmcts.game_stream(is_teacher=True)
            for _ in range(n_teacher):
                record = next(self._tch_stream)
                positions = self._record_to_positions(record)
                self.teacher_buffer.add_batch(positions)
                total += len(positions)

        elapsed = time.time() - t_batch
        avg_moves = total / max(n_games, 1)
        pos_per_s = total / max(elapsed, 1)
        print(f"  done: {n_games} games | +{total} pos | {elapsed:.1f}s | "
              f"{pos_per_s:.0f} pos/s | ~{avg_moves:.0f} moves/game",
              flush=True)

        with self._stats_lock:
            self.games_generated     += n_games
            self.positions_generated += total

        return total

    def _record_to_positions(self, record: GameRecord) -> List[PositionRecord]:
        """Convert a GameRecord to a list of PositionRecord objects."""
        positions = []
        for i in range(len(record.board_tensors)):
            pos = PositionRecord(
                board_tensor   = record.board_tensors[i],
                history_tensor = record.history_tensors[i],
                policy_target  = record.policy_targets[i],
                wdl_label      = record.wdl_labels[i] if i < len(record.wdl_labels) else 0.5,
                mcts_q         = record.mcts_qs[i] if i < len(record.mcts_qs) else 0.0,
                phase          = record.phases[i],
                piece_count    = record.piece_counts[i],
                move_number    = record.move_numbers[i],
                is_teacher     = record.is_teacher,
            )
            positions.append(pos)
        return positions

    # ── Background thread ─────────────────────────────────────────────────

    def start_background(self, games_per_batch: int = 1) -> None:
        """Start continuous self-play in a background thread.

        WARNING: this shares ``self.model`` with the training loop. The
        background thread calls ``model.eval()`` while the trainer calls
        ``model.train()`` + backward on the same module → data race on the
        training/eval flag and concurrent CUDA use without a lock. Only use
        this if self-play runs on a frozen weight snapshot or behind a lock;
        the synchronous ``generate_batch`` path (used by main.py) is safe.
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._background_loop,
            args=(games_per_batch,),
            daemon=True,
            name="SelfPlayWorker",
        )
        self._thread.start()

    def stop_background(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)

    def _background_loop(self, games_per_batch: int) -> None:
        while not self._stop_event.is_set():
            try:
                self.generate_batch(games_per_batch)
            except Exception as e:
                # Log but don't crash the background thread
                print(f"[SelfPlay] Error in background loop: {e}")
                time.sleep(1.0)

    # ── Endgame pretraining ───────────────────────────────────────────────

    def generate_endgame_pretrain(
        self,
        n_positions: int = 2_000_000,
        syzygy_path: str = "",
    ) -> int:
        """
        Generate supervised endgame positions using Syzygy tablebases.
        If no tablebases are available, generates random endgame positions
        and uses the model for evaluation (warm-start only).

        Returns the number of positions generated.
        """
        from engine.rules import init_tablebase, probe_tablebase

        tb_ok = init_tablebase(syzygy_path) if syzygy_path else False
        if not tb_ok:
            print("[SelfPlay] No Syzygy tablebases — skipping endgame pretraining.")
            return 0

        print(f"[SelfPlay] Generating {n_positions:,} endgame positions from Syzygy TBs...")
        import chess
        import chess.syzygy

        total = 0
        # Sample random endgame FENs and probe them
        endgame_fens = self._sample_endgame_fens(n_positions)
        for fen_str in endgame_fens:
            if total >= n_positions:
                break
            board = Board.from_fen(fen_str)
            tb_val = probe_tablebase(board)
            if tb_val is None:
                continue

            # Policy target: uniform over legal moves (no MCTS for pretraining)
            legal = board.legal_moves()
            if not legal:
                continue
            policy = np.zeros(self.cfg.num_actions, dtype=np.float32)
            from engine.movegen import move_to_action_index
            for m in legal:
                idx = move_to_action_index(m, board.side_to_move)
                if idx >= 0:
                    policy[idx] = 1.0 / len(legal)

            wdl = tb_val  # Already in current-side's perspective
            pos = PositionRecord(
                board_tensor   = board.to_tensor(),
                history_tensor = np.zeros((self.cfg.gru_history_len,
                                            self.cfg.input_planes, 8, 8),
                                           dtype=np.float32),
                policy_target  = policy,
                wdl_label      = wdl,
                phase          = board.get_phase(),
                piece_count    = board.piece_count(),
                move_number    = 0,
                is_teacher     = False,
            )
            self.replay_buffer.add(pos)
            total += 1

        print(f"[SelfPlay] Endgame pretraining: added {total:,} positions.")
        return total

    def _sample_endgame_fens(self, n: int) -> List[str]:
        """
        Generate FEN strings for random endgame positions (2-5 pieces).
        Uses python-chess for FEN generation (allowed for this utility purpose).
        """
        import chess
        import random

        fens = []
        # Well-known basic endgame FENs
        base_fens = [
            "8/8/8/3k4/8/8/8/3K4 w - - 0 1",              # K vs K (draw)
            "8/8/8/3k4/8/8/3R4/3K4 w - - 0 1",            # K+R vs K
            "8/8/8/3k4/8/3Q4/8/3K4 w - - 0 1",            # K+Q vs K
            "8/8/3k4/8/8/8/3KP3/8 w - - 0 1",             # K+P vs K
            "8/3k4/8/8/8/3K4/3PP3/8 w - - 0 1",           # K+2P vs K
            "8/8/8/2k5/8/8/2KBB3/8 w - - 0 1",            # K+2B vs K
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",       # Rook endgame
        ]
        fens.extend(base_fens * (n // (len(base_fens) * 10) + 1))

        # Generate random 3-5 piece positions
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        attempt = 0
        while len(fens) < n and attempt < n * 10:
            attempt += 1
            try:
                b = chess.Board.empty()
                # Kings (mandatory)
                k_sq = random.randint(0, 63)
                b.set_piece_at(k_sq, chess.Piece(chess.KING, chess.WHITE))
                ok_sq = random.randint(0, 63)
                while ok_sq == k_sq:
                    ok_sq = random.randint(0, 63)
                b.set_piece_at(ok_sq, chess.Piece(chess.KING, chess.BLACK))
                # 1-3 extra pieces
                n_extra = random.randint(1, 3)
                for _ in range(n_extra):
                    sq = random.randint(0, 63)
                    while b.piece_at(sq):
                        sq = random.randint(0, 63)
                    pt    = random.choice(piece_types[1:])  # No pawns for simplicity
                    color = random.choice([chess.WHITE, chess.BLACK])
                    b.set_piece_at(sq, chess.Piece(pt, color))
                b.turn = random.choice([chess.WHITE, chess.BLACK])
                b.clear_stack()
                if not b.is_valid():
                    continue
                fens.append(b.fen())
            except Exception:
                continue

        return fens[:n]

    # ── Statistics ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._stats_lock:
            elapsed = time.time() - self._last_stats_time
            return {
                "games_generated":     self.games_generated,
                "positions_generated": self.positions_generated,
                "replay_buffer_size":  len(self.replay_buffer),
                "teacher_buffer_size": len(self.teacher_buffer),
            }
