"""
Central training configuration.
Default values assume RTX 5070 (12 GB VRAM) + Ryzen 7 8700F (8P/16T).
Values tagged #hw-derived are overwritten at runtime by utils/system_probe.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ChessAIConfig:
    # ── Hardware / runtime ──────────────────────────────────────────────
    device: str = "cuda"  # hw-derived
    precision: str = "bf16"  # hw-derived
    torch_compile: bool = False  # hw-derived
    pin_memory: bool = True  # hw-derived

    # ── Inference backend (self-play / arena / --play only; training always
    #    uses eager PyTorch) ───────────────────────────────────────────────
    # "cudagraph" (default — CUDA Graph capture of the inference forward at a
    #              fixed batch=parallel_games; eliminates per-call kernel
    #              launch overhead; ~1.3-2× faster than eager on small-spatial
    #              chess inputs. Silently falls back to eager torch if capture
    #              fails for any reason — never blocks self-play.)
    # "torch"     (eager PyTorch, identical to model.infer)
    # "onnx"      (onnxruntime with GPU execution provider; needs
    #              `onnxruntime-gpu` and a supported provider)
    inference_backend: str = "cudagraph"
    onnx_provider: str = "cuda"      # "cuda" | "trt" | "cpu"
    onnx_reexport_every: int = 5_000  # re-export .onnx every N train steps

    # ── Network architecture ─────────────────────────────────────────────
    # Board encoding: 21 planes from C++ engine
    input_planes: int = 21
    # GRU history encoder adds 4 planes → total 25 input to ResNet
    gru_context_planes: int = 4
    num_channels: int = 256        # Residual channels
    # 10 blocks (was 20): ~2x faster forward, far faster early learning. The
    # net can be grown back to 20 later without losing progress via the
    # Net2Net deepen tool (model/net2net.py, `python main.py --grow`).
    num_res_blocks: int = 10       # Number of residual blocks
    se_every_n: int = 4            # Insert SE block every N res-blocks
    # Gradient checkpointing: trade 20-30% backward speed for activation
    # memory. With 10 blocks × 256 ch + BF16 + B=512 the activations are
    # ~0.3 GB and easily fit on a 12 GB RTX 5070, so checkpointing wastes
    # compute. `None` = auto: disable when num_res_blocks ≤ 14, enable from
    # block 4 onwards otherwise. Set an int explicitly to override.
    grad_ckpt_from: int | None = None
    gru_hidden: int = 256          # GRU hidden state dimension
    gru_layers: int = 2            # GRU depth
    gru_history_len: int = 8       # How many past boards to feed the GRU
    # Policy head: AlphaZero encoding 64×73 = 4672 source×action planes
    num_actions: int = 4672
    # Policy-head bottleneck width (Conv2d output channels). 32 is a memory-
    # cheaper compromise vs. AlphaZero's 73 and gives the FC layer 16× more
    # information than the legacy 2-channel bottleneck. Set to 2 to load
    # pre-fix checkpoints. Bumping this widens the FC weight matrix
    # (~mid_channels × 64 × 4672 params).
    policy_mid_channels: int = 32
    # Tanh denominator for the auxiliary material-balance head. Lower → more
    # decisive aux signal early; higher → finer granularity. 12.0 matches
    # the original hard-coded value.
    material_scale: float = 12.0

    # ── Optimizer ────────────────────────────────────────────────────────
    lr_initial: float = 1e-3       # Peak LR after warmup
    lr_backbone_mult: float = 1.0  # Backbone LR multiplier
    lr_head_mult: float = 0.5      # Policy / Value head LR multiplier
    lr_gru_mult: float = 0.3       # GRU encoder LR multiplier
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    warmup_steps: int = 2_000      # Cosine annealing warmup
    total_steps: int = 500_000     # Full cosine cycle length

    # ── Loss weights (phase-aware) ───────────────────────────────────────
    # Phase boundaries by total piece count on board
    opening_piece_threshold: int = 24   # > 24 pieces → opening
    endgame_piece_threshold: int = 12   # < 12 pieces → endgame

    alpha_p_opening: float = 1.5   # Policy loss weight in opening
    alpha_p_mid: float = 1.0       # Policy loss weight in middlegame
    alpha_p_endgame: float = 0.5   # Policy loss weight in endgame
    alpha_v_opening: float = 1.0   # Value loss weight in opening
    alpha_v_mid: float = 1.0       # Value loss weight in middlegame
    alpha_v_endgame: float = 1.8   # Value loss weight in endgame (accuracy critical)

    # Auxiliary value head supervision. The aux head is now a learnable
    # material-balance regressor (see model/heads.py ValueHead.piece_values)
    # whose target is the game outcome. Active for ALL training (not gated by
    # aux_steps any more — the field is kept for checkpoint compatibility).
    # L2-pull of `value_head.piece_values` back toward the classical
    # [P=1, N=3, B=3, R=5, Q=9, K=0] vector. Without it, the auxiliary head
    # is free to learn arbitrary piece valuations (the data only constrains
    # *combinations* well), and observed drift in early runs landed at
    # implausible values like Q≈6 or P≈2.5. A tiny anchor keeps the learned
    # values interpretable without locking them — set to 0.0 to disable.
    piece_value_anchor_weight: float = 1e-3

    aux_loss_weight: float = 2.0  # 0.2 → 0.6 → 0.9 → 2.0: at step ~30k the
                                  # value head fully regressed (v=0.0000 for
                                  # 11+ consecutive log lines). 0.9 wasn't
                                  # dominant enough — the WDL loss kept
                                  # pulling the tower back to the trivial
                                  # all-zero solution. 2.0 makes the
                                  # material-balance aux head the dominant
                                  # gradient source, FORCING the value
                                  # tower to learn something non-trivial.
                                  # Drop back to ~0.5 once the value-loss
                                  # has stayed above 0.1 for at least
                                  # 5000 consecutive steps.
    aux_steps: int = 5_000        # legacy: no longer consulted by the trainer

    # ── TD(λ) value blending ─────────────────────────────────────────────
    # Blend mc_return with the MCTS root Q on positions inside the window.
    # Window widened from 5..25 → 3..100: the original 5..25 left ~95 % of
    # half-moves in a 120-ply game on pure MC, which is the highest-variance
    # target. Endgame moves benefit most from variance reduction because
    # the search Q there is sharp and very close to the eventual outcome.
    # Move 0–2 stays pure-MC because the MCTS Q at game start is noisy
    # while the network is still cold.
    td_lambda: float = 0.8           # Fixed λ (adaptive deferred to future work)
    td_lambda_start_move: int = 3
    td_lambda_end_move: int = 100

    # ── MCTS ─────────────────────────────────────────────────────────────
    # Tuned for a single consumer GPU. 800-sim teacher rollouts are AlphaZero
    # farm-scale and pure waste on a sub-1600 net (it plays near-random); they
    # collapsed throughput ~7x (85→12 pos/s) after the bootstrap ramp. Lower
    # budgets give targets that are still far better than the 50-sim bootstrap
    # at a fraction of the cost. Raise these again later for end-game polish.
    mcts_sims: int = 128           # Standard simulation budget (was 200)
    mcts_sims_critical: int = 256  # Budget when |Q1-Q2| < dca_threshold
    mcts_sims_teacher: int = 256   # Budget for teacher rollout games (was 800)
    mcts_c_puct: float = 2.0       # UCB exploration constant
    # Self-play stays in the cheap 50-sim fast-bootstrap until this training
    # step (was hard-coded 500 — far too early; the net has barely learned
    # anything by then, so 200/800-sim rollouts just burn time). Keeping the
    # high-throughput bootstrap much longer fills the buffer ~7x faster.
    bootstrap_ramp_step: int = 8_000

    # Root Dirichlet noise
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25    # Fraction of prior replaced by noise

    # Dynamic Compute Allocation thresholds
    dca_q_threshold: float = 0.05  # |Q1-Q2| below this → use sims_critical
    dca_check_mult: float = 1.5    # Sim multiplier when in check

    # ── Self-play ────────────────────────────────────────────────────────
    parallel_games: int = 6  # hw-derived
    temperature_moves: int = 30    # τ=1.0 for first N moves, then τ→0
    temperature_final: float = 0.1 # Temperature after temperature_moves
    max_game_moves: int = 80       # Hard cap on game length (half-moves).
                                   # 150→120→80: at step ~30k the value head
                                   # had completely collapsed back to 0
                                   # because draws dominated. Cutting the
                                   # cap to 80 forces decisive results MUCH
                                   # earlier — any game where neither side
                                   # makes progress gets cut. Yes some
                                   # legitimate long games get truncated,
                                   # but at this stage we vastly prefer
                                   # 200 short decisive games over 50 long
                                   # drawn ones for value-head supervision.

    # ── Resign mechanism (caps wasted compute on already-lost games) ─────
    # When the side-to-move's MCTS root Q stays below -resign_q for
    # resign_streak consecutive moves, that side resigns. Shortens hopeless
    # tails and raises the share of self-play games with a decisive winner —
    # critical in early training when most games hit the move cap as draws,
    # collapsing the value-loss to 0. Set resign_streak=0 to disable.
    # 0.85/8 → 0.70/6 → 0.55/6 for bootstrap: with a weak value head the
    # absolute root-Q rarely reaches the higher thresholds, so a strict
    # gate never fires and every hopeless game runs to the move cap as a
    # draw. 0.55 means "side-to-move's Q says it's roughly 78 % losing for
    # six consecutive moves" — still conservative enough not to resign
    # equal positions, but it actually fires during bootstrap. The 20-move
    # gate (added in mcts/tree.py) keeps early-opening noise out.
    resign_q: float = 0.55
    resign_streak: int = 6

    # ── Opening diversity (anti-overfitting on a few openings) ───────────
    # Each self-play game starts from a position obtained by playing this many
    # uniform-random legal plies from the standard start (NOT recorded as
    # training targets). 0 = always start from the standard position
    # (unchanged behaviour). 8 → 14: bootstrap is draw-collapse-bound, and
    # the strongest lever against that is "ensure each game starts from a
    # MATERIALLY UNBALANCED position" so the value head sees decisive
    # outcomes. 14 random plies usually leaves one side a piece up or down,
    # which dramatically raises the share of decisive games.
    random_opening_plies: int = 20
    # 8 → 14 → 20: with 20 random plies most games START already with one
    # side a piece (or more) ahead, GUARANTEEING decisive games. The cost
    # is that the opening phase of the resulting games is unrealistic, but
    # at this training stage we need the value head to see win/loss
    # outcomes, not realistic openings. Drop back to ~8 once the model has
    # an actual opening preference (policy entropy < 3.2).
    # Optional path to a file with one FEN per line; if set, each game starts
    # from a random FEN drawn from it (applied before random_opening_plies).
    opening_book_path: str = ""

    # ── Teacher / Self-distillation ──────────────────────────────────────
    teacher_game_ratio: float = 0.10   # Fraction of games run at 800 sims
    teacher_batch_ratio: float = 0.30  # Fraction of training batch from teacher buf
    teacher_loss_weight: float = 1.0   # Scale of the teacher KL term in total loss

    # ── Replay buffer (Prioritized Experience Replay) ────────────────────
    replay_buffer_cap: int = 2000000  # hw-derived
    teacher_buffer_cap: int = 200_000
    replay_start_training: int = 20_000   # Wait until buffer has this many positions
    per_alpha: float = 0.6                # Priority exponent
    per_beta_start: float = 0.4           # IS correction start
    per_beta_end: float = 1.0             # IS correction end (annealed over training)
    # Down-weight positions from drawn games when sampling. Early in training
    # ~70 % of self-play games end as draws (200-ply cutoff + random play), so
    # the value-loss collapses to 0 (the trivial draw constant). Halving the
    # draw share keeps the data distribution but lets the value head see
    # win/loss positions relatively more often.  1.0 disables.
    #
    # Schedule: linearly ramp from `draw_priority_start_mult` (1.0, no
    # down-weighting) to `draw_priority_mult` over `draw_priority_ramp_steps`
    # training steps. Rationale: at step 0 the buffer is ~70 % draws and
    # there is no decisive supervision to learn from — down-weighting draws
    # there throws away the only data we have. Ramping in delays the
    # rebalancing until decisive games actually populate the buffer.
    draw_priority_mult: float = 0.1        # 0.5 → 0.3 → 0.1: with the value
                                            # head fully regressed at step
                                            # ~30k we need to almost completely
                                            # exclude draws from the training
                                            # distribution. 0.1 means a drawn
                                            # position has 10% the chance of
                                            # a decisive position to be sampled.
                                            # The buffer still contains them,
                                            # but the value head is now starved
                                            # of the "predict 0.5" shortcut.
    draw_priority_start_mult: float = 1.0
    draw_priority_ramp_steps: int = 25_000  # was 50_000 — at step 16k+ the
                                            # ramp was barely halfway in.
                                            # 25k pulls full effect by ~30k
                                            # steps so the bootstrap actually
                                            # benefits from the down-weight.

    # ── Opponent pool ────────────────────────────────────────────────────
    opponent_pool_size: int = 5
    arena_games: int = 200
    arena_every_n_steps: int = 10_000
    self_play_vs_pool_ratio: float = 0.30  # 30 % of games vs pool checkpoint

    # ── Entropy-Regulated Exploration Decay ──────────────────────────────
    entropy_target: float = 2.5    # Target policy entropy in nats
    entropy_check_every: int = 500   # was 5000 — at that cadence ERED fired
                                     # ~3× in 14K steps and the policy entropy
                                     # drifted uncorrected.
    entropy_window: int = 200        # was 1000 — responsive to current data
    dirichlet_eps_min: float = 0.10
    dirichlet_eps_max: float = 0.40
    alpha_v_global_min: float = 0.5
    alpha_v_global_max: float = 2.0

    # ── Training batch ───────────────────────────────────────────────────
    batch_size: int = 512  # hw-derived

    # ── DataLoader ───────────────────────────────────────────────────────
    dataloader_workers: int = 5  # hw-derived
    # prefetch_factor=4 instead of PyTorch's default 2: GPU is the bottleneck
    # here, so having a few extra batches ready to go avoids the data path
    # ever blocking the optimizer. 4 is a reasonable default; values >8 mostly
    # eat RAM without buying additional pipeline depth.
    prefetch_factor: int = 4
    # persistent_workers keeps the DataLoader workers alive between epochs
    # (~10-30 ms spawn cost saved per epoch boundary on Windows). The
    # trainer currently samples directly from the replay buffer, so this
    # field is consumed only when a real DataLoader path is wired up.
    persistent_workers: bool = True

    # ── Checkpointing / logging ──────────────────────────────────────────
    # 1000 → 200: with native C++ crashes (chess_ext extension has known
    # memory bugs that trigger on specific mate-in-2 positions), losing
    # even ~4h of compute per crash is painful. 200 steps ≈ 1h of wall
    # time at ~60 train_steps/h, bounding crash-loss to roughly the
    # batch-generation interval. Disk-cost is trivial (one .pt per file).
    checkpoint_every: int = 200
    # Buffer snapshots (replay + teacher → pickle.gz next to checkpoint) every
    # N checkpoints. With checkpoint_every=200 and buffer_save_every=1 a
    # buffer dump lands every 200 steps (about every 30 min). Native crashes
    # were eating fresh self-play data on every restart at the old cadence
    # of 5 because the buffer would rewind to its last-saved state on
    # resume from latest.pt; saving every checkpoint costs ~3 GB disk for
    # the rolling snapshot pair but means no replay-data loss on crash.
    buffer_save_every: int = 1
    # Lightweight emergency snapshot every N self-play batches (separate
    # from `checkpoint_every`). Overwrites `latest.pt` so disk stays bounded;
    # the supervisor wrapper reads this when auto-restarting after a crash
    # so the work-loss is bounded to N batches (≈ a few minutes).
    emergency_checkpoint_every_batches: int = 3
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    log_every: int = 100           # Training steps between console logs

    # ── Paths ─────────────────────────────────────────────────────────────
    # Set to directory containing .rtbw/.rtbz Syzygy tablebase files.
    # Leave empty to disable Syzygy probing.
    syzygy_path: str = ""

    # Path to a Stockfish UCI binary, used by tools/elo_vs_stockfish.py to
    # measure an objective ELO. Leave empty and pass --stockfish on the CLI.
    stockfish_path: str = "C:/Users/silas/OneDrive/Desktop/ChatKi/chess_ai/tools/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"

    # ── Endgame pretraining ──────────────────────────────────────────────
    pretrain_positions: int = 3_000_000
    pretrain_epochs: int = 5

    # ── ELO tracking ────────────────────────────────────────────────────
    initial_elo: float = 1000.0
    elo_k_factor: float = 32.0


# Singleton used throughout the codebase.
CONFIG = ChessAIConfig()


def get_config() -> ChessAIConfig:
    return CONFIG


def update_config_from_dict(d: dict) -> None:
    """Merge a dict of field overrides into the global CONFIG."""
    for k, v in d.items():
        if hasattr(CONFIG, k):
            setattr(CONFIG, k, v)
        else:
            raise ValueError(f"Unknown config key: {k!r}")
