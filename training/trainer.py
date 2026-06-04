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
        # Fused AdamW: a single CUDA kernel for the per-tensor update instead
        # of many element-wise launches. Up to ~15 % step-time win on CUDA
        # and zero behaviour change. Only available on CUDA (no fused path on
        # CPU/MPS) — silently fall back to the standard implementation
        # elsewhere or on older PyTorch (<2.0).
        _fused_ok = (self.device.type == "cuda")
        try:
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay = cfg.weight_decay,
                fused        = _fused_ok,
            )
        except (RuntimeError, TypeError):
            # Older PyTorch without `fused` kwarg, or a CUDA build that
            # rejects fused for this param-group layout.
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
        # BF16 has the same dynamic range as FP32, so it does NOT need loss
        # scaling. Only FP16 needs a GradScaler. Keep `_scaler` as None for
        # bf16/fp32 so the backward path is a clean `loss.backward()` /
        # `optimizer.step()` instead of bouncing through Scaler hooks that
        # would no-op anyway.
        if cfg.precision == "fp16":
            self._amp_dtype = torch.float16
            self._scaler = GradScaler(enabled=True)
        elif cfg.precision == "bf16":
            self._amp_dtype = torch.bfloat16
            self._scaler = None
        else:
            self._amp_dtype = torch.float32
            self._scaler = None

        # ── Entropy-Regulated Exploration Decay state ──────────────────
        self._dirichlet_eps   = cfg.dirichlet_eps
        self._alpha_v_global  = 1.0
        self._entropy_history: list = []

        # ── Auxiliary loss (active for first `aux_steps` steps) ────────
        self._aux_active = True

        # ── Sync-cost mitigation state ─────────────────────────────────
        # NaN guard runs every N steps (+ tight window after a hit) instead
        # of every step — `if not torch.isfinite(loss):` is an implicit
        # cudaDeviceSynchronize and dominated step cost in profiling.
        self._nan_check_every = 50
        self._nan_suspect = 0        # steps remaining of tight-mode after a hit
        # PER priority update can run with one-step latency without harm —
        # defer the device→host copy so it overlaps the next forward.
        self._deferred_td_update: tuple | None = None
        # Anchor tensor for the piece-value L2 pull (kept on device, allocated
        # once instead of every train_step).
        self._piece_value_anchor = torch.tensor(
            [1.0, 3.0, 3.0, 5.0, 9.0, 0.0],
            dtype=torch.float32, device=self.device,
        )

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

        # ── Apply deferred PER priority update from the previous step ─
        # The TD-errors from train_step(t-1) were copied async; their D2H
        # is finished by now (a full forward+backward+step happened since).
        if self._deferred_td_update is not None:
            prev_idx, prev_td = self._deferred_td_update
            self.replay_buffer.update_priorities(prev_idx, prev_td.numpy())
            self._deferred_td_update = None

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
        # NOTE: inline `Tensor.pin_memory()` is a *synchronous* pageable→pinned
        # copy on the main thread and only pays off when the copy is hidden
        # behind a background DataLoader worker. Doing it inline costs more
        # than the resulting async H2D saves, so we drop it. If a real
        # DataLoader path is added (see plan §3.4 Variant B), reintroduce
        # pin_memory=True at the loader level.
        _cuda = (self.device.type == "cuda")

        def to_device(arr, dtype=torch.float32):
            t = torch.from_numpy(np.ascontiguousarray(arr))
            return t.to(device=self.device, dtype=dtype, non_blocking=False)

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

        # Auxiliary loss — learnable-piece-value material head against the
        # actual game outcome (mc_value_r). Active over the FULL run (not
        # just aux_steps) because the aux head is now a structural piece-value
        # regressor, not a one-off phase warm-up. Provides a dense signal even
        # when WDL collapses (draw-heavy early training). The MCTS path uses
        # WDL only, so this term never biases self-play.
        aux_loss = F.mse_loss(aux_r.squeeze(1).float(), mc_value_r)
        loss = loss + self.cfg.aux_loss_weight * aux_loss

        # Soft L2 anchor on the learnable piece_values toward the classical
        # chess valuations. Tiny weight (default 1e-3) — purely a tie-breaker
        # for under-determined components; the data still dominates.
        anchor_w = getattr(self.cfg, "piece_value_anchor_weight", 0.0)
        if anchor_w > 0.0:
            base_model = (self.model._orig_mod
                          if hasattr(self.model, "_orig_mod") else self.model)
            pv = base_model.value_head.piece_values
            # Use the cached anchor (built once in __init__) instead of
            # allocating a fresh tensor every step.
            pv_anchor_loss = F.mse_loss(pv.float(), self._piece_value_anchor)
            loss = loss + anchor_w * pv_anchor_loss

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

        # ── NaN/Inf guard: skip the step if loss is non-finite. A single
        # backward with NaN gradients permanently corrupts AdamW's second
        # moments, so it's better to drop the batch than to poison state.
        # `torch.isfinite(loss)` followed by a Python truthiness check is an
        # implicit cudaDeviceSynchronize — too expensive to do every step.
        # Sample every N steps in normal operation; after any hit, run tight
        # for a window so a burst of NaNs is caught immediately.
        do_nan_check = (
            (self.global_step % self._nan_check_every == 0)
            or self._nan_suspect > 0
        )
        if do_nan_check and not torch.isfinite(loss):
            print(f"[Trainer] WARN: non-finite loss at step {self.global_step} "
                  f"— skipping update", flush=True)
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            self._nan_suspect = 10  # tight-mode for the next few steps
            return {"loss": float("nan"), "skipped": 1.0}
        if self._nan_suspect > 0:
            self._nan_suspect -= 1

        # ── Backward ─────────────────────────────────────────────────
        if self._scaler is not None:
            # FP16 path: loss scaling required.
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            # BF16 / FP32 path: clean backward, no scaler overhead.
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()

        if self.global_step < self.cfg.total_steps:
            self.scheduler.step()

        # ── PER priority update ───────────────────────────────────────
        # Defer the device→host copy by one step. By the time the NEXT
        # train_step starts (after a forward+backward worth of GPU work),
        # the pinned-host destination is guaranteed populated — so this
        # non-blocking copy never adds a sync to the current iteration.
        td_pinned = torch.empty_like(td_err_gpu, device="cpu", pin_memory=True)
        td_pinned.copy_(td_err_gpu, non_blocking=True)
        self._deferred_td_update = (indices, td_pinned)
        self.replay_buffer.step()

        # ── Entropy-Regulated Exploration Decay ──────────────────────
        self.global_step += 1
        if self.global_step % self.cfg.entropy_check_every == 0:
            self._regulate_entropy(policy_logits_r.float().detach())

        # ── Checkpoint ───────────────────────────────────────────────
        if self.global_step % self.cfg.checkpoint_every == 0:
            self.save_checkpoint()
            # Buffer persistence: every Nth checkpoint, dump replay + teacher
            # buffers next to the .pt. Survives a crash without losing hours
            # of self-play data. Save cadence configurable; default 5 × ckpt
            # interval keeps overhead well under 1 %.
            buf_every = getattr(self.cfg, "buffer_save_every", 5) * self.cfg.checkpoint_every
            if buf_every > 0 and self.global_step % buf_every == 0:
                self._save_buffers()

        # ── Metrics ───────────────────────────────────────────────────
        # Scalar .item() calls each force a cudaDeviceSynchronize. Strategy:
        # ALWAYS surface `loss/total` (one D2H — cheap enough, callers and
        # integration tests depend on it), but only collect the rest on the
        # logging cadence (~6 → 1 sync per non-log step).
        metrics: dict = {
            "loss/total":        float(loss.detach().cpu()),
            "lr":                self.optimizer.param_groups[0]["lr"],
            "dirichlet_eps":     self._dirichlet_eps,
            "alpha_v_global":    self._alpha_v_global,
            "replay_buf_size":   len(self.replay_buffer),
            "teacher_buf_size":  len(self.teacher_buffer),
            "global_step":       self.global_step,
        }
        if self.global_step % self.cfg.log_every == 0:
            with torch.no_grad():
                _lp = F.log_softmax(policy_logits_r.float(), dim=-1)
                ent = -(_lp.exp() * _lp).sum(-1).mean()
            # Single D2H roundtrip for the remaining diagnostic scalars.
            scalars = torch.stack([
                policy_loss_r.mean().detach(),
                value_loss_r.mean().detach(),
                teacher_loss.detach().reshape(()),
                aux_loss.detach(),
                ent,
            ]).cpu()
            metrics.update({
                "loss/policy":     float(scalars[0]),
                "loss/value":      float(scalars[1]),
                "loss/teacher_kl": float(scalars[2]),
                "loss/aux":        float(scalars[3]),
                "policy_entropy":  float(scalars[4]),
            })
        return metrics

    # ── Emergency snapshot (called between batches; crash-resilience) ────

    def emergency_save(self) -> str:
        """
        Write a lightweight `latest.pt` next to the regular checkpoints.

        Called from the self-play phase loops between batches so that even
        if the process dies hard (C++ access violation in chess_ext, OOM,
        power loss) the most we lose is one batch's worth of work — not
        the full `checkpoint_every` interval.

        Overwrites a single file so disk usage stays bounded. The replay
        buffer is NOT dumped here (too heavy for per-batch frequency); the
        regular `save_checkpoint` cadence handles that.
        """
        path = str(Path(self.cfg.checkpoint_dir) / "latest.pt")
        try:
            return self.save_checkpoint(path)
        except Exception as e:  # noqa: BLE001 — never let bookkeeping crash training
            print(f"[Trainer] emergency_save skipped: "
                  f"{type(e).__name__}: {e}", flush=True)
            return ""

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

        # Asymmetric step sizes — diffuse-policy correction needs to be more
        # aggressive than collapse-correction. Observed in the 14K-step run:
        # entropy drifted 3.36 → 3.68 (above target 2.5 × 1.2 = 3.0), but the
        # ×0.9 step + 5000-step cadence shrank dirichlet_eps only 0.25 → 0.20
        # — far too slow. ×0.85 and a 500-step cadence give meaningful pull.
        if avg_entropy < target * 0.8:
            # Policy collapsing — inject more noise
            self._dirichlet_eps = min(
                self._dirichlet_eps * 1.15, self.cfg.dirichlet_eps_max
            )
            self._alpha_v_global = max(
                self._alpha_v_global * 0.95, self.cfg.alpha_v_global_min
            )
        elif avg_entropy > target * 1.2:
            # Policy too diffuse — reduce noise, increase value weight
            self._dirichlet_eps = max(
                self._dirichlet_eps * 0.85, self.cfg.dirichlet_eps_min
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
            # `_scaler` is None for bf16/fp32; serialize an empty dict so
            # load_checkpoint can no-op-resolve regardless of run precision.
            "scaler":      self._scaler.state_dict() if self._scaler is not None else None,
            "dirichlet_eps":  self._dirichlet_eps,
            "alpha_v_global": self._alpha_v_global,
            # ERED state — persist so a resumed run doesn't restart the
            # entropy-drift estimator from scratch (which would mis-calibrate
            # dirichlet_eps for the first `entropy_check_every` steps).
            "entropy_history": list(self._entropy_history),
            # PER-beta annealing position. Without this, resumed training
            # jumps to a wrong beta and discontinuously reweights IS samples.
            "replay_step":    int(self.replay_buffer._step),
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

    def _save_buffers(self) -> None:
        """Dump replay + teacher buffers as gzip-pickle next to the latest
        checkpoint. Overwrites a single sliding pair (no per-step copies) to
        keep disk usage bounded. Errors are non-fatal — training continues."""
        try:
            import gzip, pickle
            ckpt_dir = Path(self.cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            def _dump(obj, path: Path):
                tmp = path.with_suffix(path.suffix + ".tmp")
                with gzip.open(tmp, "wb", compresslevel=3) as f:
                    pickle.dump(obj, f, protocol=4)
                tmp.replace(path)  # atomic on POSIX, best-effort on Windows

            replay_state = {
                "data":       self.replay_buffer._data,
                "priorities": self.replay_buffer._priorities,
                "ptr":        self.replay_buffer._ptr,
                "size":       self.replay_buffer._size,
                "step":       self.replay_buffer._step,
                "max_prio":   self.replay_buffer._max_prio,
            }
            _dump(replay_state, ckpt_dir / "replay.pkl.gz")

            teacher_state = {
                "data": self.teacher_buffer._data,
                "ptr":  self.teacher_buffer._ptr,
                "size": self.teacher_buffer._size,
            }
            _dump(teacher_state, ckpt_dir / "teacher.pkl.gz")
            print(f"[Trainer] Buffer snapshot saved "
                  f"(replay={len(self.replay_buffer):,}, "
                  f"teacher={len(self.teacher_buffer):,})")
        except Exception as e:  # noqa: BLE001
            print(f"[Trainer] Buffer save skipped: {type(e).__name__}: {e}")

    def _load_buffers(self) -> None:
        """Counterpart to _save_buffers. Called from load_checkpoint."""
        try:
            import gzip, pickle
            ckpt_dir = Path(self.cfg.checkpoint_dir)
            replay_path  = ckpt_dir / "replay.pkl.gz"
            teacher_path = ckpt_dir / "teacher.pkl.gz"
            if replay_path.exists():
                with gzip.open(replay_path, "rb") as f:
                    state = pickle.load(f)
                # SECURITY: only ever load buffer dumps this process wrote
                # itself / from a trusted local checkpoint dir.
                #
                # Capacity reconciliation: the system probe re-derives
                # `replay_buffer_cap` per run (depending on available RAM),
                # so the on-disk snapshot can be shorter OR longer than the
                # freshly constructed buffer's `_data` / `_priorities`. We
                # resize both to the current capacity instead of replacing
                # them outright, otherwise `_ptr % capacity` walks past
                # the loaded list's end and IndexError's on the next add().
                import numpy as _np
                cap   = self.replay_buffer.capacity
                d     = list(state["data"])
                p     = state["priorities"]
                old_n = len(d)
                if old_n < cap:
                    # Pad with None / 0.0 so indices in [old_n, cap) are valid.
                    d.extend([None] * (cap - old_n))
                    p_full = _np.zeros(cap, dtype=_np.float32)
                    p_full[:old_n] = p
                    p = p_full
                elif old_n > cap:
                    # New buffer is smaller — keep the most recent `cap`
                    # entries (the ring buffer's tail) and reset _ptr.
                    keep_from = old_n - cap
                    d = d[keep_from:]
                    p = p[keep_from:]
                self.replay_buffer._data       = d
                self.replay_buffer._priorities = p
                # Clamp restored bookkeeping to the new capacity.
                self.replay_buffer._size = min(int(state["size"]), cap)
                self.replay_buffer._ptr  = int(state["ptr"]) % cap
                self.replay_buffer._step       = state["step"]
                self.replay_buffer._max_prio   = state["max_prio"]
                if old_n != cap:
                    print(f"[Trainer] Replay buffer capacity changed "
                          f"({old_n:,} → {cap:,}); resized snapshot.")
                print(f"[Trainer] Replay buffer restored "
                      f"({len(self.replay_buffer):,} positions)")
            if teacher_path.exists():
                with gzip.open(teacher_path, "rb") as f:
                    state = pickle.load(f)
                # Same capacity-reconciliation as the main replay buffer.
                cap = self.teacher_buffer.capacity
                d = list(state["data"])
                if len(d) < cap:
                    d.extend([None] * (cap - len(d)))
                elif len(d) > cap:
                    d = d[len(d) - cap:]
                self.teacher_buffer._data = d
                self.teacher_buffer._size = min(int(state["size"]), cap)
                self.teacher_buffer._ptr  = int(state["ptr"]) % cap
                print(f"[Trainer] Teacher buffer restored "
                      f"({len(self.teacher_buffer):,} positions)")
        except Exception as e:  # noqa: BLE001
            print(f"[Trainer] Buffer load skipped: {type(e).__name__}: {e}")

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
        # Migration: older checkpoints had ValueHead.fc_aux (Linear 256→1) where
        # the new arch uses ValueHead.piece_values (6,). Drop the obsolete keys
        # so load_state_dict still succeeds — piece_values keeps its default
        # init [1,3,3,5,9,0] which is exactly the warm-start we want.
        sd = state["model"]
        legacy_aux_keys = [k for k in sd
                           if k.startswith(("value_head.fc_aux.",
                                            "_orig_mod.value_head.fc_aux."))]
        for k in legacy_aux_keys:
            sd.pop(k, None)

        # Migration: the policy head was widened from a 2-channel bottleneck
        # to `policy_mid_channels` (default 32). Old checkpoints carry conv /
        # norm / fc weights with shape (2, …); they will not align with the
        # new arch. Drop them and let `_init_weights` reseed — losing the
        # old policy head is preferable to a load_state_dict shape error,
        # and the policy will recover quickly once the network sees data.
        # (Detect by inspecting the saved shape against the current shape.)
        try:
            cur_conv_shape = base_model.policy_head.conv.weight.shape
        except AttributeError:
            cur_conv_shape = None
        if cur_conv_shape is not None:
            ph_prefixes = ("policy_head.", "_orig_mod.policy_head.")
            policy_shape_mismatch = False
            for k in list(sd.keys()):
                if not k.startswith(ph_prefixes):
                    continue
                if not k.endswith("conv.weight"):
                    continue
                if tuple(sd[k].shape) != tuple(cur_conv_shape):
                    policy_shape_mismatch = True
                    break
            if policy_shape_mismatch:
                stripped = [k for k in sd if k.startswith(ph_prefixes)]
                for k in stripped:
                    sd.pop(k, None)
                print(f"[Trainer] policy_head shape changed — dropping "
                      f"{len(stripped)} legacy policy_head keys; head will "
                      f"reinit from `_init_weights`.")

        missing, unexpected = base_model.load_state_dict(sd, strict=False)
        # Allowed-missing keys: parameters that the new arch introduces and
        # that have a sensible default initialisation. Everything else is
        # a real arch mismatch worth reporting.
        _allowed_missing = ("piece_values",
                            "_material_scale",
                            "policy_head.conv.weight",
                            "policy_head.conv.bias",
                            "policy_head.norm.weight",
                            "policy_head.norm.bias",
                            "policy_head.fc.weight",
                            "policy_head.fc.bias")
        for m in missing:
            if not any(m.endswith(s) for s in _allowed_missing):
                print(f"[Trainer] WARNING: missing key on load: {m}")
        for u in unexpected:
            print(f"[Trainer] WARNING: unexpected key on load: {u}")

        if "optimizer" in state:
            # The optimizer state may reference the old fc_aux param tensors;
            # if so, AdamW will silently mismatch param-group order. Wrap in
            # try/except so a transition from old → new arch resets the
            # optimizer state instead of crashing.
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (ValueError, KeyError, RuntimeError) as e:
                # Losing AdamW moments means the first few hundred steps
                # after resume will spike — log loudly so a regression in
                # loss isn't blamed on the data instead of the resume.
                print(f"[Trainer] *** WARNING: optimizer state could NOT be "
                      f"restored ({type(e).__name__}: {e}). AdamW moments "
                      f"reset to zero — expect a transient loss spike. ***",
                      flush=True)
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state and state["scaler"] is not None and self._scaler is not None:
            self._scaler.load_state_dict(state["scaler"])

        self._dirichlet_eps  = state.get("dirichlet_eps",  self.cfg.dirichlet_eps)
        self._alpha_v_global = state.get("alpha_v_global", 1.0)
        self._entropy_history = list(state.get("entropy_history", []))
        if "replay_step" in state:
            self.replay_buffer._step = int(state["replay_step"])

        # NOTE: do NOT manually fast-forward the scheduler here. Its
        # state_dict (loaded above) already restores last_epoch / _step_count,
        # so an extra loop would double-advance it (wrong LR, OneCycleLR
        # ValueError past total_steps).

        print(f"[Trainer] Loaded checkpoint at step {self.global_step}")

        # Restore replay + teacher buffer snapshots if present (saved every
        # buffer_save_every ckpts). Skipping is fine — they refill quickly.
        self._load_buffers()

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
