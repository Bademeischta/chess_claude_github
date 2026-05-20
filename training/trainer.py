"""
training/trainer.py

Main training loop integrating:
  - Phase-aware loss weighting (opening / middlegame / endgame)
  - Self-distillation via KL divergence against teacher buffer
  - Entropy-Regulated Exploration Decay
  - Cosine annealing LR with warmup
  - BF16 autocast
  - PER priority updates
  - Checkpoint saving / loading
  - TensorBoard logging
"""

from __future__ import annotations

import os
import time
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset

from config import ChessAIConfig
from training.replay_buffer import (
    PrioritizedReplayBuffer, TeacherBuffer, PositionRecord, ReplayDataset,
)
from training.pbt import OpponentPool
from utils.logger import TrainingLogger


class Trainer:
    """
    Drives the training loop.

    Typical usage:
        trainer = Trainer(cfg, model, replay_buffer, teacher_buffer,
                          opponent_pool, logger, device)
        trainer.train_step()   # called after each self-play batch
    """

    def __init__(
        self,
        cfg: ChessAIConfig,
        model: nn.Module,
        replay_buffer: PrioritizedReplayBuffer,
        teacher_buffer: TeacherBuffer,
        opponent_pool: OpponentPool,
        logger: TrainingLogger,
        device: torch.device,
    ) -> None:
        self.cfg            = cfg
        self.model          = model
        self.replay_buffer  = replay_buffer
        self.teacher_buffer = teacher_buffer
        self.opponent_pool  = opponent_pool
        self.logger         = logger
        self.device         = device

        self.global_step     = 0
        self.total_positions = 0
        self._start_time     = time.time()

        # ── Optimizer with per-group LR ────────────────────────────────
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        param_groups = base_model.parameter_groups(
            lr            = cfg.lr_initial,
            backbone_mult = cfg.lr_backbone_mult,
            head_mult     = cfg.lr_head_mult,
            gru_mult      = cfg.lr_gru_mult,
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay = cfg.weight_decay,
        )

        # ── LR scheduler (Cosine Annealing with linear warmup) ─────────
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr    = cfg.lr_initial,
            total_steps = cfg.total_steps,
            pct_start = cfg.warmup_steps / cfg.total_steps,
            anneal_strategy = "cos",
            div_factor      = 25.0,
            final_div_factor = 1e4,
        )

        # ── Mixed precision ────────────────────────────────────────────
        if cfg.precision in ("bf16", "fp16"):
            self._amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
            # GradScaler only needed for fp16 (bf16 doesn't need loss scaling)
            self._scaler = GradScaler(enabled=(cfg.precision == "fp16"))
        else:
            self._amp_dtype   = torch.float32
            self._scaler      = GradScaler(enabled=False)

        # ── Entropy-Regulated Exploration Decay state ──────────────────
        self._dirichlet_eps   = cfg.dirichlet_eps
        self._alpha_v_global  = 1.0
        self._entropy_history: list = []

        # ── Auxiliary loss (active for first `aux_steps` steps) ────────
        self._aux_active = True

        # ── PER beta tracking ──────────────────────────────────────────
        self._per_beta  = cfg.per_beta_start

        # ── Phase-weight lookup tables (indexed by phase 0/1/2) ────────
        # phase 0 = opening, 1 = mid, 2 = endgame
        self._alpha_p_lut = torch.tensor(
            [cfg.alpha_p_opening, cfg.alpha_p_mid, cfg.alpha_p_endgame],
            device=self.device, dtype=torch.float32,
        )
        self._alpha_v_lut = torch.tensor(
            [cfg.alpha_v_opening, cfg.alpha_v_mid, cfg.alpha_v_endgame],
            device=self.device, dtype=torch.float32,
        )

    # ── Phase-aware loss weights ──────────────────────────────────────────

    def _phase_weights(
        self, phases: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-sample policy and value loss weights from phase tensor.
        phases: (B,) LongTensor with values 0/1/2
        """
        alpha_p = self._alpha_p_lut[phases]
        alpha_v = self._alpha_v_lut[phases] * self._alpha_v_global
        return alpha_p, alpha_v

    # ── TD(λ) value target computation ───────────────────────────────────

    @staticmethod
    def compute_td_lambda_targets(
        wdl_labels: torch.Tensor,
        q_values:   torch.Tensor,
        move_numbers: torch.Tensor,
        td_lambda: float = 0.8,
        td_start: int = 5,
        td_end: int = 25,
    ) -> torch.Tensor:
        """
        Blend the Monte-Carlo return with the bootstrapped value using TD(λ).
        Both `wdl_labels` (MC return) and `q_values` must already be in the same
        space ([-1, 1]). Blending is only applied to moves [td_start, td_end];
        outside that window the pure MC return is used.

        target = λ × mc_return + (1-λ) × q_value
        """
        mask = (move_numbers >= td_start) & (move_numbers <= td_end)
        mask = mask.float()
        blended = td_lambda * wdl_labels + (1.0 - td_lambda) * q_values
        return mask * blended + (1.0 - mask) * wdl_labels

    # ── Single training step ──────────────────────────────────────────────

    def train_step(self) -> dict:
        """
        Sample one batch from replay + teacher buffers and run one gradient step.
        Returns a dict of scalar metrics.
        """
        if not self.replay_buffer.is_ready(self.cfg.replay_start_training):
            return {}

        # ── Sample from replay buffer ─────────────────────────────────
        n_replay  = int(self.cfg.batch_size * (1.0 - self.cfg.teacher_batch_ratio))
        n_teacher = self.cfg.batch_size - n_replay

        records, indices, is_weights = self.replay_buffer.sample(n_replay)
        teacher_records = (self.teacher_buffer.sample(n_teacher)
                           if len(self.teacher_buffer) >= 10 else [])

        # ── Build batch tensors ───────────────────────────────────────
        def to_batch(recs):
            boards   = np.stack([r.board_tensor   for r in recs])
            histories= np.stack([r.history_tensor for r in recs])
            policies = np.stack([r.policy_target  for r in recs])
            wdls     = np.array([r.wdl_label      for r in recs], dtype=np.float32)
            mctsq    = np.array([r.mcts_q         for r in recs], dtype=np.float32)
            phases   = np.array([r.phase          for r in recs], dtype=np.int64)
            mnums    = np.array([r.move_number    for r in recs], dtype=np.int64)
            return boards, histories, policies, wdls, mctsq, phases, mnums

        boards_r, hist_r, pol_r, wdl_r, mq_r, ph_r, mn_r = to_batch(records)

        # Teacher records may be empty
        if teacher_records:
            boards_t, hist_t, pol_t, wdl_t, mq_t, ph_t, mn_t = to_batch(teacher_records)

        # ── Move to device ────────────────────────────────────────────
        _pin = (self.device.type == "cuda")

        def to_device(arr, dtype=torch.float32):
            t = torch.from_numpy(np.ascontiguousarray(arr))
            if _pin:
                t = t.pin_memory()
            return t.to(device=self.device, dtype=dtype, non_blocking=_pin)

        board_r  = to_device(boards_r, self._amp_dtype)
        hist_r_t = to_device(hist_r,   self._amp_dtype)
        pol_r_t  = to_device(pol_r)
        wdl_r_t  = to_device(wdl_r)
        mq_r_t   = to_device(mq_r)
        ph_r_t   = to_device(ph_r, torch.long)
        mn_r_t   = to_device(mn_r, torch.long)
        w_r_t    = torch.from_numpy(is_weights).to(self.device)

        if teacher_records:
            board_t_  = to_device(boards_t, self._amp_dtype)
            hist_t_t  = to_device(hist_t,   self._amp_dtype)
            pol_t_t   = to_device(pol_t)
            wdl_t_t   = to_device(wdl_t)
            ph_t_t    = to_device(ph_t, torch.long)
            mn_t_t    = to_device(mn_t, torch.long)

        # ── Forward pass ─────────────────────────────────────────────
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=self.device.type, dtype=self._amp_dtype,
                      enabled=(self._amp_dtype != torch.float32)):

            policy_logits_r, wdl_r_pred, aux_r = self.model(board_r, hist_r_t)

            if teacher_records:
                policy_logits_t, wdl_t_pred, _ = self.model(board_t_, hist_t_t)

        # ── Loss computation ──────────────────────────────────────────
        # Replay: cross-entropy policy loss + MSE value loss
        alpha_p_r, alpha_v_r = self._phase_weights(ph_r_t)

        # Policy loss (cross-entropy against MCTS visit distribution)
        log_probs_r = F.log_softmax(policy_logits_r.float(), dim=-1)
        policy_loss_r = -(pol_r_t * log_probs_r).sum(dim=-1)  # (B,)

        # TD(λ) blended value target. wdl_r_t is in [0,1] (MC return from the
        # side's view); q_r = P(win)-P(loss) is in [-1,1]. Both operands MUST be
        # in the same space before blending — convert the MC return to [-1,1]
        # first, so no post-blend rescale is needed.
        wdl_r_f = wdl_r_pred.float()
        q_r = wdl_r_f[:, 0] - wdl_r_f[:, 2]  # scalar value in [-1, 1]
        mc_value_r = wdl_r_t * 2.0 - 1.0     # [0,1] → [-1,1]
        # TD(λ) bootstrap = the MCTS root Q recorded during self-play (already
        # in [-1,1], side-to-move perspective), NOT the network's own current
        # prediction. This is the intended AlphaZero-style target: a fixed
        # search-derived value, so the regression goal is grad-free and the
        # value head can't trivially self-bootstrap.
        value_target_r = self.compute_td_lambda_targets(
            mc_value_r, mq_r_t, mn_r_t,
            self.cfg.td_lambda,
            self.cfg.td_lambda_start_move,
            self.cfg.td_lambda_end_move,
        )
        value_loss_r = F.mse_loss(q_r, value_target_r, reduction="none")  # (B,)

        # PER importance-sampling weights
        total_loss_r = (alpha_p_r * policy_loss_r + alpha_v_r * value_loss_r) * w_r_t
        loss = total_loss_r.mean()

        # Auxiliary loss (first `aux_steps` steps only)
        if self._aux_active and self.global_step < self.cfg.aux_steps:
            aux_target = (wdl_r_t * 2.0 - 1.0)  # Scale to [-1, 1]
            aux_loss   = F.mse_loss(aux_r.squeeze(1).float(), aux_target)
            loss = loss + self.cfg.aux_loss_weight * aux_loss
        else:
            self._aux_active = False
            aux_loss = torch.zeros(1, device=self.device)

        # Teacher loss: KL divergence against teacher policy (soft targets)
        teacher_loss = torch.zeros(1, device=self.device)
        if teacher_records:
            log_probs_t  = F.log_softmax(policy_logits_t.float(), dim=-1)
            kl_loss      = F.kl_div(log_probs_t, pol_t_t, reduction="batchmean")
            teacher_loss = kl_loss
            loss = loss + self.cfg.teacher_loss_weight * teacher_loss

        # PER td-errors: compute on-GPU now (reuses value_target_r, no recompute),
        # defer the host copy until after optimizer.step so the D2H transfer
        # overlaps with the backward/step kernels instead of blocking here.
        td_err_gpu = (q_r.detach() - value_target_r).abs()

        # ── Backward ─────────────────────────────────────────────────
        if self._amp_dtype == torch.float16:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()

        if self.global_step < self.cfg.total_steps:
            self.scheduler.step()

        # ── PER priority update ───────────────────────────────────────
        td_errors = td_err_gpu.cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        self.replay_buffer.step()

        # ── Entropy-Regulated Exploration Decay ──────────────────────
        self.global_step += 1
        if self.global_step % self.cfg.entropy_check_every == 0:
            self._regulate_entropy(policy_logits_r.float().detach())

        # ── Checkpoint ───────────────────────────────────────────────
        if self.global_step % self.cfg.checkpoint_every == 0:
            ckpt_path = self.save_checkpoint()

            # Periodically save the replay and teacher buffers (every 5 checkpoints)
            if self.global_step % (self.cfg.checkpoint_every * 5) == 0:
                print(f"[Trainer] Saving buffers at step {self.global_step}...")
                self.replay_buffer.save(ckpt_path.replace(".pt", "_replay.pkl"))
                self.teacher_buffer.save(ckpt_path.replace(".pt", "_teacher.pkl"))

        # Cheap policy entropy for *logging* every step (the ERED regulation
        # in _regulate_entropy stays on its 5000-step cadence and is unchanged
        # — this is display-only so H is never a stale nan).
        with torch.no_grad():
            _lp  = F.log_softmax(policy_logits_r.float(), dim=-1)
            _ent = float((-(_lp.exp() * _lp).sum(-1)).mean().item())

        # ── Metrics ───────────────────────────────────────────────────
        metrics = {
            "policy_entropy":    _ent,
            "loss/total":        float(loss.item()),
            "loss/policy":       float(policy_loss_r.mean().item()),
            "loss/value":        float(value_loss_r.mean().item()),
            "loss/teacher_kl":   float(teacher_loss.item()),
            "loss/aux":          float(aux_loss.item()),
            "lr":                self.optimizer.param_groups[0]["lr"],
            "dirichlet_eps":     self._dirichlet_eps,
            "alpha_v_global":    self._alpha_v_global,
            "replay_buf_size":   len(self.replay_buffer),
            "teacher_buf_size":  len(self.teacher_buffer),
            "global_step":       self.global_step,
        }
        return metrics

    # ── Entropy regulation ────────────────────────────────────────────────

    def _regulate_entropy(self, policy_logits: torch.Tensor) -> float:
        """
        Compute average policy entropy over recent positions.
        Adjust dirichlet_eps and alpha_v_global accordingly.
        """
        probs   = F.softmax(policy_logits, dim=-1).clamp(min=1e-9)
        entropy = -(probs * probs.log()).sum(dim=-1).mean().item()

        self._entropy_history.append(entropy)
        if len(self._entropy_history) > 20:
            self._entropy_history.pop(0)

        avg_entropy = sum(self._entropy_history) / len(self._entropy_history)
        target = self.cfg.entropy_target

        if avg_entropy < target * 0.8:
            # Policy collapsing — inject more noise
            self._dirichlet_eps = min(
                self._dirichlet_eps * 1.2, self.cfg.dirichlet_eps_max
            )
            self._alpha_v_global = max(
                self._alpha_v_global * 0.95, self.cfg.alpha_v_global_min
            )
        elif avg_entropy > target * 1.2:
            # Policy too diffuse — reduce noise, increase value weight
            self._dirichlet_eps = max(
                self._dirichlet_eps * 0.9, self.cfg.dirichlet_eps_min
            )
            self._alpha_v_global = min(
                self._alpha_v_global * 1.05, self.cfg.alpha_v_global_max
            )

        return avg_entropy

    # ── Checkpoint I/O ────────────────────────────────────────────────────

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if path is None:
            path = str(ckpt_dir / f"step_{self.global_step:07d}.pt")

        base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") \
                     else self.model
        state = {
            "step":        self.global_step,
            "model":       base_model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "scaler":      self._scaler.state_dict(),
            "dirichlet_eps":  self._dirichlet_eps,
            "alpha_v_global": self._alpha_v_global,
        }
        torch.save(state, path)

        # ONNX re-export hook (no-op unless inference_backend == "onnx").
        # Self-play's OnnxInferenceEngine picks the new file up by mtime
        # between games. Never let an export failure break checkpointing.
        if getattr(self.cfg, "inference_backend", "torch") == "onnx":
            try:
                from model.onnx_export import export_onnx
                export_onnx(base_model, self.cfg,
                            str(ckpt_dir / "model.onnx"))
            except Exception as e:  # noqa: BLE001
                print(f"[Trainer] ONNX re-export skipped ({type(e).__name__}: {e})")

        return path

    def load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # weights_only=True: the checkpoint only holds tensors / plain
        # dicts / ints, so this is safe and blocks arbitrary-code execution
        # from a tampered .pt file.
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.global_step = state.get("step", 0)

        base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") \
                     else self.model
        base_model.load_state_dict(state["model"])

        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            self._scaler.load_state_dict(state["scaler"])

        self._dirichlet_eps  = state.get("dirichlet_eps",  self.cfg.dirichlet_eps)
        self._alpha_v_global = state.get("alpha_v_global", 1.0)

        # NOTE: do NOT manually fast-forward the scheduler here. Its
        # state_dict (loaded above) already restores last_epoch / _step_count,
        # so an extra loop would double-advance it (wrong LR, OneCycleLR
        # ValueError past total_steps).

        print(f"[Trainer] Loaded checkpoint at step {self.global_step}")

    # ── Arena match ───────────────────────────────────────────────────────

    def run_arena(
        self,
        current_model: nn.Module,
        opponent_model: nn.Module,
        n_games: int = 200,
    ) -> tuple[int, int, int]:
        """
        Play n_games between current_model and opponent_model.
        Returns (wins, draws, losses) for current_model.
        """
        from mcts.tree import MCTSTree
        from engine.rules import get_game_result, GameResult
        from engine.board import Board, STARTPOS_FEN

        wins = draws = losses = 0
        for game_idx in range(n_games):
            # Alternate colors
            current_is_white = (game_idx % 2 == 0)
            white_model = current_model if current_is_white else opponent_model
            black_model = opponent_model if current_is_white else current_model

            white_tree = MCTSTree(self.cfg, white_model, self.device)
            black_tree = MCTSTree(self.cfg, black_model, self.device)

            board = Board.from_fen(STARTPOS_FEN)
            # Seed both trees' GRU history once; advance() then maintains it for
            # both every move, and each search keeps (does not wipe) it.
            white_tree.reset(board)
            black_tree.reset(board)
            move_count = 0
            result = None

            while True:
                is_white_turn = (board.side_to_move == 0)
                tree = white_tree if is_white_turn else black_tree
                n_sims = self.cfg.mcts_sims
                move, _ = tree.search(board, n_sims,
                                      temperature=self.cfg.temperature_final,
                                      use_dca=False, keep_history=True)
                new_board = board.apply_move(move)

                # Advance both trees
                white_tree.advance(move, new_board)
                black_tree.advance(move, new_board)

                board = new_board
                move_count += 1
                r = get_game_result(board)
                if r != GameResult.ONGOING or move_count >= 300:
                    result = r
                    break

            if result == GameResult.WHITE_WIN:
                if current_is_white: wins += 1
                else: losses += 1
            elif result == GameResult.BLACK_WIN:
                if current_is_white: losses += 1
                else: wins += 1
            else:
                draws += 1

        return wins, draws, losses
