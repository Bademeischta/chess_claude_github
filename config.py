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
    # "torch" (default, zero behaviour change) | "onnx".
    # ONNX needs onnxruntime-gpu + a GPU execution provider; it falls back to
    # torch automatically if anything is missing.
    inference_backend: str = "torch"
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

    aux_loss_weight: float = 0.6  # bumped from 0.2 to break the
                                  # "value head sits at 0 because everything
                                  # is a draw" deadlock during bootstrap.
                                  # The material-balance aux head gives the
                                  # value head a non-trivial gradient even
                                  # when WDL labels are all 0.5 (draws).
                                  # Once decisive games dominate the buffer
                                  # this can be lowered back to ~0.2.
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
    max_game_moves: int = 120      # Hard cap on game length (half-moves).
                                   # 150→120: at the bootstrap level the
                                   # tail beyond move 120 is almost always
                                   # shuffling that ends in a phantom-draw.
                                   # Cutting it saves ~20% self-play time
                                   # AND raises the share of decisive games
                                   # (resign more likely to fire before the
                                   # cap kicks in).

    # ── Resign mechanism (caps wasted compute on already-lost games) ─────
    # When the side-to-move's MCTS root Q stays below -resign_q for
    # resign_streak consecutive moves, that side resigns. Shortens hopeless
    # tails and raises the share of self-play games with a decisive winner —
    # critical in early training when most games hit the move cap as draws,
    # collapsing the value-loss to 0. Set resign_streak=0 to disable.
    # Lowered from 0.85/8 to 0.70/6 for bootstrap: with a weak value head
    # the absolute root-Q rarely reaches 0.85, so a strict threshold never
    # fires and every hopeless game runs to the move cap as a draw. The
    # 20-move gate (added in mcts/tree.py) keeps early-opening noise out.
    resign_q: float = 0.70
    resign_streak: int = 6

    # ── Opening diversity (anti-overfitting on a few openings) ───────────
    # Each self-play game starts from a position obtained by playing this many
    # uniform-random legal plies from the standard start (NOT recorded as
    # training targets). 0 = always start from the standard position
    # (unchanged behaviour). A value of 6–10 markedly diversifies replay data.
    random_opening_plies: int = 8
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
    draw_priority_mult: float = 0.5
    draw_priority_start_mult: float = 1.0
    draw_priority_ramp_steps: int = 50_000

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
    prefetch_factor: int = 2

    # ── Checkpointing / logging ──────────────────────────────────────────
    # 5000 steps was hours of unsaved compute on a single GPU (a crash =
    # total loss). 1000 keeps the worst case to well under an hour.
    checkpoint_every: int = 1_000
    # Buffer snapshots (replay + teacher → pickle.gz next to checkpoint) every
    # N checkpoints. Default 5 → every 5,000 steps. Bounded disk usage (a
    # single sliding snapshot pair, not a per-step copy). Set 0 to disable.
    buffer_save_every: int = 5
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
    pretrain_positions: int = 2_000_000
    pretrain_epochs: int = 3

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
