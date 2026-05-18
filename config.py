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

    # Auxiliary value head supervision (phase signal, first 5 K steps only)
    aux_loss_weight: float = 0.1
    aux_steps: int = 5_000

    # ── TD(λ) value blending ─────────────────────────────────────────────
    td_lambda: float = 0.8         # Fixed λ (adaptive deferred to future work)
    td_lambda_start_move: int = 5  # Apply TD(λ) from move N …
    td_lambda_end_move: int = 25   # … to move M; pure MC otherwise

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
    max_game_moves: int = 200      # Hard cap on game length (half-moves); random nets never terminate naturally

    # ── Opening diversity (anti-overfitting on a few openings) ───────────
    # Each self-play game starts from a position obtained by playing this many
    # uniform-random legal plies from the standard start (NOT recorded as
    # training targets). 0 = always start from the standard position
    # (unchanged behaviour). A value of 6–10 markedly diversifies replay data.
    random_opening_plies: int = 0
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

    # ── Opponent pool ────────────────────────────────────────────────────
    opponent_pool_size: int = 5
    arena_games: int = 200
    arena_every_n_steps: int = 10_000
    self_play_vs_pool_ratio: float = 0.30  # 30 % of games vs pool checkpoint

    # ── Entropy-Regulated Exploration Decay ──────────────────────────────
    entropy_target: float = 2.5    # Target policy entropy in nats
    entropy_check_every: int = 5_000
    entropy_window: int = 1_000    # Number of recent positions to measure
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
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    log_every: int = 100           # Training steps between console logs

    # ── Paths ─────────────────────────────────────────────────────────────
    # Set to directory containing .rtbw/.rtbz Syzygy tablebase files.
    # Leave empty to disable Syzygy probing.
    syzygy_path: str = ""

    # Path to a Stockfish UCI binary, used by tools/elo_vs_stockfish.py to
    # measure an objective ELO. Leave empty and pass --stockfish on the CLI.
    stockfish_path: str = ""

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
