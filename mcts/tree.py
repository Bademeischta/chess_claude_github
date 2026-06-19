"""
mcts/tree.py

MCTS search driver.  Orchestrates the Select → Expand → Evaluate → Backup loop
and interfaces between the C++ tree (chess_ext) and the Python neural network.

When chess_ext is available, MCTSTree delegates to C++ for:
  - Node management and UCB selection
  - Virtual loss
  - Fast backup

The neural network inference (policy + value) always runs in Python/PyTorch.

Key features:
  - Batch inference: collects leaf evaluations across N parallel games,
    sends them to the GPU as a single batch.
  - TD(λ) is handled at the training level (trainer.py), not here.  MCTS
    backup uses pure MC (alternating sign).
  - Dirichlet noise at root (α=0.3, ε=0.25).
  - Dynamic Compute Allocation via mcts/dca.py.
  - Temperature-sampled move selection.
"""

from __future__ import annotations

import math
import os
import random
import traceback
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

from engine.board import Board
from engine.movegen import policy_array_to_legal_priors, move_to_action_index
from engine.rules import get_game_result, GameResult, probe_tablebase
from mcts.dca import get_sim_budget, compute_extra_sims
from model.inference import TorchInferenceEngine, build_inference_engine


def fill_history(out: np.ndarray, board_history: list, T: int) -> np.ndarray:
    """
    Fill `out` (shape (T, planes, 8, 8)) in-place with the most recent board
    tensors from `board_history`, right-aligned; unused leading slots zeroed.
    Returns `out`. Single shared implementation for every history-build path.
    """
    n = min(len(board_history), T)
    if n < T:
        out[: T - n] = 0.0
    for i in range(n):
        out[T - n + i] = board_history[-(n - i)]
    return out


# ── Backend selection ─────────────────────────────────────────────────────

_DISABLE_CPP = os.environ.get("CHESS_AI_NO_CPP", "").lower() in ("1", "true", "yes")

try:
    if _DISABLE_CPP:
        raise ImportError("CHESS_AI_NO_CPP=1 — forcing python MCTS path")
    import chess_ext as _cx
    _USE_CPP = True
except ImportError:
    _cx = None
    _USE_CPP = False
    from mcts.node import MCTSNode as _PyNode


# ── Constants ─────────────────────────────────────────────────────────────

VIRTUAL_LOSS = 1  # Added to visit count during parallel selection


# ============================================================
#  Game record produced by a completed self-play game
# ============================================================

class GameRecord:
    """Stores the trajectory of a self-play game for the replay buffer."""

    __slots__ = (
        "board_tensors",     # list[(21,8,8) float32 ndarray] — one per move
        "history_tensors",   # list[(T,21,8,8) ndarray]       — GRU input per move
        "policy_targets",    # list[np.ndarray(4672,)]         — visit distribution
        "wdl_labels",        # list[float]                     — WDL from side's perspective
        "mcts_qs",           # list[float] — MCTS root Q in [-1,1] (TD bootstrap)
        "game_result",       # float: 1.0/0.5/0.0 from white's perspective
        "move_numbers",      # list[int]
        "sides",             # list[int] 0=white,1=black — side to move at each
                             # recorded position. Authoritative for WDL labels;
                             # mn%2 parity breaks when games start from
                             # Black-to-move FENs or odd random_opening_plies.
        "phases",            # list[int]  0=opening, 1=mid, 2=endgame
        "piece_counts",      # list[int]
        "is_teacher",        # bool: was this a teacher (high-sim) game?
    )

    def __init__(self) -> None:
        self.board_tensors:   list = []
        self.history_tensors: list = []
        self.policy_targets:  list = []
        self.wdl_labels:      list = []
        self.mcts_qs:         list = []
        self.game_result:     float = 0.5
        self.move_numbers:    list = []
        self.sides:           list = []
        self.phases:          list = []
        self.piece_counts:    list = []
        self.is_teacher:      bool = False


# ============================================================
#  MCTSTree
# ============================================================

class MCTSTree:
    """
    Manages one MCTS search tree for a single game.

    Usage pattern (per move):
        tree.begin_search(board)
        for _ in range(n_sims):
            tree.run_one_simulation()
        move, policy_target = tree.select_move(temperature)
        tree.advance(move, new_board)

    `ParallelMCTS` instead drives many trees via `select_leaf_only` /
    `apply_eval` so one batched GPU forward serves all of them.
    """

    def __init__(self, cfg, model: torch.nn.Module, device: torch.device,
                 engine=None) -> None:
        self.cfg    = cfg
        self.model  = model
        self.device = device
        # Inference goes through a swappable engine (Torch eager by default,
        # optionally ONNX). Callers still get torch tensors back.
        self.engine = engine if engine is not None \
            else TorchInferenceEngine(model)

        self.c_puct      = cfg.mcts_c_puct
        # Clamp dirichlet_alpha defensively: np.random.dirichlet crashes on
        # alpha <= 0, and the ERED controller in trainer.py can push the eps
        # towards zero — which would degenerate the distribution. Alpha is
        # a static cfg value here, but we guard it anyway.
        self.dirichlet_a = max(float(cfg.dirichlet_alpha), 0.01)
        self.dirichlet_e = max(float(cfg.dirichlet_eps), 0.0)

        # History buffer: keeps the last `history_len` board tensors
        self._board_history: list = []  # list of (21,8,8) np.ndarray

        # Pre-allocated, reused history scratch buffers (filled in-place every
        # leaf eval instead of allocating ~1000 short-lived arrays per search).
        _T = cfg.gru_history_len
        self._hist_np = np.zeros(
            (_T, cfg.input_planes, 8, 8), dtype=np.float32
        )
        self._hist_t = torch.zeros(
            (1, _T, cfg.input_planes, 8, 8),
            device=device, dtype=self._net_dtype(),
        )

        # Current search state
        self._root_board: Optional[Board] = None
        self._root_node = None  # C++ MCTSTree or Python MCTSNode root

        if _USE_CPP:
            self._cpp_tree = _cx.MCTSTree(c_puct=self.c_puct)
        else:
            self._cpp_tree = None

    # ── Setup ─────────────────────────────────────────────────────────────

    def reset(self, board: Board, keep_history: bool = False) -> None:
        """
        Start a fresh search tree from `board`.

        `keep_history=False` (default, new game): seed the GRU history with the
        current board. `keep_history=True` (subsequent moves of an ongoing
        game): preserve the running history maintained by `advance()` — it must
        NOT be wiped every move, otherwise the GRU encoder only ever sees a
        single board plus zero padding.
        """
        self._root_board = board
        if not keep_history:
            self._board_history = [board.to_tensor()]

        if _USE_CPP:
            self._root_node = self._cpp_tree.new_root(0, 1.0)
        else:
            self._root_node = _PyNode(move=0, parent=None, prior=1.0)

    def advance(self, played_move: int, new_board: Board) -> None:
        """
        Advance the tree root to the child corresponding to `played_move`.
        Preserves the subtree for future reuse.
        """
        self._root_board = new_board
        self._board_history.append(new_board.to_tensor())
        # Keep only the last `history_len` boards
        if len(self._board_history) > self.cfg.gru_history_len:
            self._board_history = self._board_history[-self.cfg.gru_history_len:]

        if _USE_CPP:
            # If select_move had to use the empty-children fallback, the C++
            # subtree is in an inconsistent state. Wipe it via new_root()
            # instead of advance_root() so the next selection cycle starts
            # from a fresh, well-formed node.
            if getattr(self, "_needs_cpp_reset", False):
                self._root_node = self._cpp_tree.new_root(0, 1.0)
                self._needs_cpp_reset = False
            else:
                self._root_node = self._cpp_tree.advance_root(played_move)
        else:
            child = self._root_node.children.get(played_move)
            if child is None:
                child = _PyNode(move=played_move, parent=None, prior=1.0)
            else:
                child.parent = None
            self._root_node = child

    # ── Single simulation ─────────────────────────────────────────────────

    def run_one_simulation(self) -> None:
        """
        Run one MCTS simulation (select → evaluate → expand → backup).
        Single-board path; `ParallelMCTS` interleaves `select_leaf_only` /
        `apply_eval` across trees to share one batched GPU forward instead.
        """
        pending = self.select_leaf_only()
        if pending is None:
            return  # terminal / tablebase leaf — already expanded + backed up
        value, policy_logits = self._nn_evaluate(pending["leaf_board"])
        self.apply_eval(pending, value, policy_logits)

    # ── Split simulation primitives (enable cross-tree batched inference) ──

    def select_leaf_only(self) -> Optional[dict]:
        """
        Selection phase only (pure CPU, applies virtual loss along the path).

        Returns a pending-eval dict for leaves that need the network, or
        ``None`` if the leaf was terminal/tablebase — those are expanded and
        backed up immediately here, so they never consume a GPU batch slot.
        """
        if _USE_CPP:
            result = self._cpp_tree.select_leaf(
                self._root_node, self._root_board._b
            )
            leaf_node  = result.leaf
            leaf_board = Board(result.board)
            path       = result.path()
        else:
            path, leaf_node, leaf_board = self._select_python(
                self._root_node, self._root_board
            )

        non_nn = self._resolve_non_nn(leaf_board)
        if non_nn is not None:
            value, _ = non_nn
            # Mirror the original expand logic exactly: zero policy, the
            # terminal/no-legal-move check inside _finish_leaf decides
            # mark_terminal vs. expand (preserves draw-with-legal-moves).
            zero = np.zeros(self.cfg.num_actions, dtype=np.float32)
            self._finish_leaf(path, leaf_node, leaf_board, value, zero)
            return None

        return {"path": path, "leaf_node": leaf_node, "leaf_board": leaf_board}

    def apply_eval(
        self, pending: dict, value: float, policy_logits: np.ndarray
    ) -> None:
        """Expand + backup a pending leaf using a network evaluation."""
        self._finish_leaf(
            pending["path"], pending["leaf_node"],
            pending["leaf_board"], value, policy_logits,
        )

    # ── Batched single-tree simulations ──────────────────────────────────

    @torch.inference_mode()
    def run_batched_sims(self, total_sims: int,
                         batch_size: int = 0) -> int:
        """
        Run up to ``total_sims`` MCTS simulations, sharing one NN forward
        across ``batch_size`` distinct leaves per round.

        How it can find DIFFERENT leaves inside the same tree without
        re-evaluating the same node: ``select_leaf_only`` applies a virtual
        loss along the selected path on each call, so the next selection
        cycle's UCB scores penalise that path and route to a different leaf.
        After the network evaluates the whole batch, ``apply_eval`` undoes
        the virtual loss and applies the real backup.

        This is the single-tree analogue of ``ParallelMCTS._batched_sims``.

        Why this matters for the web UI: a CUDA-Graph inference engine is
        captured at a fixed batch width (typically ``parallel_games``,
        e.g. 32). Calling ``run_one_simulation`` once at a time still pays
        the full graph cost — the engine pads the other 31 slots with
        zeros and computes them anyway. Batching the leaves brings the
        effective sims/s up by roughly that padding factor.

        Returns the number of sims actually completed (some leaves may be
        terminal / tablebase hits and are resolved inline without using a
        GPU slot, but still count as "one sim of progress").
        """
        if total_sims <= 0:
            return 0
        if batch_size <= 0:
            batch_size = int(getattr(self.engine, "batch_size", 0)) or 32
        batch_size = max(1, int(batch_size))

        dtype = self._net_dtype()
        completed = 0

        while completed < total_sims:
            remaining = total_sims - completed
            target = min(batch_size, remaining)

            # Collect up to `target` distinct leaves. select_leaf_only applies
            # virtual loss along each chosen path so subsequent calls in the
            # same round explore different subtrees.
            pendings: list = []
            attempts = 0
            # Allow a few extra attempts so terminal-leaf hits don't starve
            # the batch — but cap to avoid an infinite loop if every path
            # leads to a non-NN leaf (rare; means the search has converged
            # on terminal lines and there's no more work to do).
            attempts_budget = target * 2 + 4
            while len(pendings) < target and attempts < attempts_budget:
                attempts += 1
                try:
                    p = self.select_leaf_only()
                except Exception:
                    # Hard failure inside selection — count it and bail out
                    # of this round; outer loop may keep trying.
                    completed += 1
                    break
                if p is None:
                    # Terminal / tablebase leaf — resolved inline, no NN needed.
                    completed += 1
                else:
                    pendings.append(p)

            if not pendings:
                # Nothing left to evaluate — the search has saturated. Break
                # early instead of spinning. Caller can still inspect the
                # tree via root_analysis.
                if completed >= total_sims or attempts >= attempts_budget:
                    break
                continue

            # Build the batched NN input. All leaves in one tree share the
            # same root-relative history; broadcast one history copy across
            # the batch instead of stacking n identical arrays.
            try:
                boards_np = np.stack(
                    [p["leaf_board"].to_tensor() for p in pendings])
            except Exception:
                # If ANY leaf's to_tensor fails, fall back to single sims
                # to make progress on the others.
                for p in pendings:
                    try:
                        value, policy_logits = self._nn_evaluate(p["leaf_board"])
                        self.apply_eval(p, value, policy_logits)
                    except Exception:
                        pass
                    completed += 1
                continue

            hist_one = self.history_np()        # (T, planes, 8, 8) — reused
            hist_np  = np.broadcast_to(
                hist_one, (len(pendings),) + hist_one.shape
            ).copy()  # actual storage; CUDA-Graph engine .copy_()s into it

            board_t = torch.from_numpy(boards_np).to(self.device, dtype)
            hist_t  = torch.from_numpy(hist_np).to(self.device, dtype)

            try:
                policy_logits, values = self.engine.infer(board_t, hist_t)
            except Exception as e:  # noqa: BLE001  — fallback per-leaf eval
                # Engine failure (e.g. CUDA-Graph capture mismatch) — fall
                # back to single-leaf inference so the search still makes
                # progress, just slowly.
                for p in pendings:
                    try:
                        v, pl = self._nn_evaluate(p["leaf_board"])
                        self.apply_eval(p, v, pl)
                    except Exception:
                        pass
                    completed += 1
                continue

            # Fuse cast+copy.
            values_np  = values.to("cpu", torch.float32).numpy()
            logits_np  = policy_logits.to("cpu", torch.float32).numpy()

            for k, p in enumerate(pendings):
                self.apply_eval(p, float(values_np[k]), logits_np[k])
                completed += 1

        return completed

    def _finish_leaf(self, path, leaf_node, leaf_board,
                     value: float, policy_logits: np.ndarray) -> None:
        legal = leaf_board.legal_moves()
        is_terminal = getattr(leaf_node, "terminal", False) or not legal
        if _USE_CPP:
            if is_terminal:
                self._cpp_tree.mark_terminal(leaf_node, value)
            else:
                priors = self._compute_priors(
                    policy_logits, legal, leaf_board.side_to_move
                )
                self._cpp_tree.expand(leaf_node, legal, priors)
            self._cpp_tree.backup(
                path, value, self.cfg.td_lambda,
                self._root_board.fullmove_number,
                self.cfg.td_lambda_start_move,
                self.cfg.td_lambda_end_move,
            )
        else:
            if is_terminal:
                leaf_node.terminal = True
                leaf_node.terminal_value = value
            else:
                priors = self._compute_priors(
                    policy_logits, legal, leaf_board.side_to_move
                )
                for move, prior in zip(legal, priors):
                    leaf_node.children[move] = _PyNode(move, leaf_node, prior)
            self._backup_python(path, value)

    def history_np(self, exclude_last: bool = False) -> np.ndarray:
        """
        This tree's root GRU-history as a (T, planes, 8, 8) array, written
        into the reused scratch buffer. The caller must stack/copy it before
        the tree's next selection overwrites the buffer (ParallelMCTS stacks
        all trees' histories immediately after collection).

        exclude_last=True drops the running history's final entry (the
        current root) — used when the ROOT itself is being evaluated, so
        the history always holds positions strictly before the input board
        (audit P3-4).
        """
        T = self.cfg.gru_history_len
        hist = self._board_history[:-1] if exclude_last else self._board_history
        self._hist_np.fill(0.0)
        return fill_history(self._hist_np, hist, T)

    def _select_python(
        self, root: "_PyNode", board: Board
    ) -> Tuple[list, "_PyNode", Board]:
        path = [root]
        node = root
        b    = board
        while node.is_expanded() and not node.terminal:
            node = node.best_child(self.c_puct)
            node.add_virtual_loss()
            b = b.apply_move(node.move)
            path.append(node)
        return path, node, b

    def _backup_python(self, path: list, leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.undo_virtual_loss()
            node.W += value
            node.N += 1
            node.Q  = node.W / node.N
            value   = -value

    # ── Leaf evaluation ───────────────────────────────────────────────────

    def _resolve_non_nn(self, board: Board) -> Optional[Tuple[float, bool]]:
        """
        Resolve a leaf without the network. Returns (value, is_game_over) for
        terminal positions and tablebase hits, else ``None`` (network needed).
        is_game_over is informational; the terminal-vs-expand decision is made
        in _finish_leaf to preserve draw-with-legal-moves behaviour.
        """
        result = get_game_result(board)
        if result != GameResult.ONGOING:
            if result == GameResult.DRAW:
                # Contempt: when set > 0 (play mode), make terminal draws
                # look slightly worse than equal so MCTS prefers fighting
                # lines over shuffle-to-3fold lines. The engine still picks
                # a draw if every alternative is even worse, but it won't
                # repeat a queen-shuffle when there's a non-drawn line of
                # similar value. Keep this at 0.0 during training (configured
                # in config.py) to avoid biasing the value-head targets.
                value = -float(getattr(self.cfg, "mcts_contempt", 0.0))
            elif (result == GameResult.WHITE_WIN and board.side_to_move == 0) or \
                 (result == GameResult.BLACK_WIN and board.side_to_move == 1):
                value = 1.0
            else:
                value = -1.0
            return value, True

        tb_val = probe_tablebase(board)
        if tb_val is not None:
            # probe_tablebase already returns WDL in [0,1] from the side-to-move
            # perspective, so the [0,1] → [-1,1] map is the same for both
            # colours. (The old per-side branch double-flipped Black.)
            # NOTE: contempt is intentionally NOT applied to tablebase draws —
            # those are objectively drawn positions, no amount of contempt can
            # change their truth. Applying it here would just confuse MCTS in
            # endgames where there is no way to avoid the draw.
            value = 2.0 * tb_val - 1.0
            return value, False

        return None

    @torch.no_grad()
    def _nn_evaluate(self, board: Board,
                     exclude_last: bool = False) -> Tuple[float, np.ndarray]:
        """Single-board neural network evaluation (no batching).

        HISTORY CONVENTION (audit P3-4): the GRU history holds positions
        strictly BEFORE the evaluated board. For LEAF evaluations the
        running history ends at the root — the root *is* a position before
        the leaf, so the full history is passed (exclude_last=False). For
        ROOT evaluations the running history's last entry is the root
        itself and must be dropped (exclude_last=True), otherwise training
        data (which records pre-position history) and inference inputs
        diverge.
        """
        board_t = torch.from_numpy(board.to_tensor()).unsqueeze(0)  # (1,21,8,8)
        board_t = board_t.to(device=self.device, dtype=self._net_dtype())

        # GRU history
        history_t = self._make_history_tensor(exclude_last)  # (1, T, 21, 8, 8)

        policy_logits, value_scalar = self.engine.infer(board_t, history_t)

        policy_np = policy_logits[0].float().cpu().numpy()
        value     = float(value_scalar[0].cpu())
        return value, policy_np

    @torch.no_grad()
    def root_wdl(self) -> dict:
        """Win/Draw/Loss probabilities + material-aux scalar for the CURRENT
        root, from the side-to-move's POV. Uses the raw model (the CUDA-graph
        engine only returns the scalar value) — one small forward, called at
        most once per position by the web UI for the WDL breakdown panel.

        Returns {"win","draw","loss","aux"} or {} if the model can't provide
        WDL (e.g. a stub/older net without a 3-logit value head).
        """
        try:
            import torch.nn.functional as F
            board = self._root_board
            board_t = torch.from_numpy(board.to_tensor()).unsqueeze(0)
            board_t = board_t.to(device=self.device, dtype=self._net_dtype())
            # ROOT eval: history must EXCLUDE the current root (audit P3-4).
            hist_t = self._make_history_tensor(exclude_last=True)
            model = getattr(self.model, "_orig_mod", self.model)
            model.eval()
            out = model(board_t, hist_t)
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                return {}
            wdl_logits = out[1].float()
            if wdl_logits.ndim != 2 or wdl_logits.shape[-1] != 3:
                return {}  # net has no 3-class WDL head
            probs = F.softmax(wdl_logits, dim=-1)[0].cpu().tolist()
            aux = 0.0
            if len(out) >= 3 and out[2] is not None:
                try:
                    aux = float(out[2].float().reshape(-1)[0].cpu())
                except Exception:
                    aux = 0.0
            return {"win": float(probs[0]), "draw": float(probs[1]),
                    "loss": float(probs[2]), "aux": aux}
        except Exception:
            return {}

    def _net_dtype(self) -> torch.dtype:
        if self.cfg.precision == "bf16":   return torch.bfloat16
        elif self.cfg.precision == "fp16": return torch.float16
        return torch.float32

    def _make_history_tensor(self, exclude_last: bool = False
                             ) -> Optional[torch.Tensor]:
        T = self.cfg.gru_history_len
        hist = self._board_history[:-1] if exclude_last else self._board_history
        self._hist_np.fill(0.0)
        fill_history(self._hist_np, hist, T)
        # copy_ does the H2D transfer + dtype cast into the persistent buffer;
        # no per-call allocation. Safe because infer() consumes it synchronously
        # before the next leaf overwrites it.
        self._hist_t.copy_(torch.from_numpy(self._hist_np).unsqueeze(0))
        return self._hist_t

    # ── Priors ────────────────────────────────────────────────────────────

    def _compute_priors(
        self,
        policy_logits: np.ndarray,
        legal_moves: List[int],
        side_to_move: int,
    ) -> List[float]:
        probs = policy_array_to_legal_priors(policy_logits, legal_moves, side_to_move)
        return probs.tolist()

    # ── Dirichlet noise ───────────────────────────────────────────────────

    def add_dirichlet_noise(self) -> None:
        # Read eps LIVE from the config: the trainer's entropy controller
        # (ERED) adjusts CONFIG.dirichlet_eps during training. The previous
        # snapshot-at-__init__ meant ERED's noise regulation never reached
        # the search (audit P4-4).
        eps = max(float(getattr(self.cfg, "dirichlet_eps", self.dirichlet_e)), 0.0)
        if _USE_CPP:
            self._cpp_tree.add_dirichlet_noise(
                self._root_node, self.dirichlet_a, eps
            )
        else:
            if not self._root_node.children:
                return
            n = len(self._root_node.children)
            noise = np.random.dirichlet([self.dirichlet_a] * n).astype(np.float32)
            for child, eta in zip(self._root_node.children.values(), noise):
                child.P = (1.0 - eps) * child.P + eps * eta

    # ── Move selection ────────────────────────────────────────────────────

    def select_move(
        self, temperature: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        """
        Select a move and return (move_int, policy_target_array).
        policy_target_array has shape (num_actions,) with visit fractions.
        """
        # Defensive guard against a known native crash in chess_ext: the C++
        # MCTSNode::sample_move() starts with `assert(!children.empty())`
        # which is a no-op in release builds, then iterates over an empty
        # `children` map and hits undefined behaviour (Windows access
        # violation). The faulthandler catches it but the process still dies.
        # Prevent the call from ever happening on an unexpanded / leafless
        # root by checking the count first, and fall back to a uniform pick
        # from the board's legal moves when the tree is degenerate.
        if _USE_CPP:
            try:
                n_children = int(self._root_node.get_children_count())
            except Exception:
                n_children = 0
            if n_children == 0:
                # Root not expanded (this signals a search-path bug upstream).
                # Don't crash the whole self-play pool — pick any legal move
                # so the game can finish, and log loudly so the FEN is in the
                # transcript for diagnosis.
                try:
                    fen = self._root_board.to_fen()
                except Exception:
                    fen = "<no-fen>"
                print(f"[MCTSTree] WARN: root has 0 children, "
                      f"falling back to first legal move. FEN={fen}",
                      flush=True)
                from engine.movegen import generate_legal_moves
                legal = list(generate_legal_moves(self._root_board))
                if not legal:
                    # No legal moves — return a sentinel; caller will treat
                    # this as game-over on the next get_game_result() check.
                    return 0, np.zeros(self.cfg.num_actions, dtype=np.float32)
                move = legal[0]
                pt = np.zeros(self.cfg.num_actions, dtype=np.float32)
                stm = self._root_board.side_to_move
                idx = move_to_action_index(move, stm)
                if idx >= 0:
                    pt[idx] = 1.0
                # CRITICAL: mark the C++ tree as "needs a clean root next
                # advance" so we don't carry an inconsistent subtree forward.
                # advance() reads this flag and calls new_root() instead of
                # advance_root() when set, wiping any orphaned C++ state.
                self._needs_cpp_reset = True
                return move, pt

        # Same API on both backends (C++ chess_ext node and Python _PyNode).
        move  = self._root_node.sample_move(temperature)
        pairs = self._root_node.get_policy_target()

        policy_target = np.zeros(self.cfg.num_actions, dtype=np.float32)
        stm = self._root_board.side_to_move
        for m, frac in pairs:
            idx = move_to_action_index(m, stm)
            if idx >= 0:
                policy_target[idx] = frac

        return move, policy_target

    def root_value(self) -> float:
        """
        MCTS value estimate of the root position in [-1, 1], from the
        side-to-move's perspective (mean backed-up value = root Q). Used as the
        TD(λ) bootstrap target instead of the network's own prediction.
        """
        node = self._root_node
        if node is None:
            return 0.0
        try:
            return float(node.Q)
        except Exception:
            return 0.0

    @staticmethod
    def _children(node):
        """[(move, child_node), …] for either backend, or []."""
        if node is None:
            return []
        if hasattr(node, "children"):           # python fallback
            return list(node.children.items())
        try:                                    # C++ chess_ext node
            return list(node.get_children())
        except Exception:
            return []

    def root_analysis(self, top_k: int = 5, pv_len: int = 8) -> dict:
        """
        Inspect the searched tree (call right after a search, before advance):

          {"value": root Q in [-1,1] (side-to-move POV),
           "nodes": tree size, "depth": pv length,
           "top": [(move_int, visits, q), …]  most-visited first;
                  q is SIDE-TO-MOVE POV (higher = better for the player to
                  move), consistent with `value` and the eval bar,
           "pv":  [move_int, …]               principal variation,
           "ext": [{"move","n","q","p","u","line"}, …]  per-move detail,
                  aligned with `top`: p = policy prior, u = PUCT exploration
                  term, line = the move's own expected continuation (mini-PV,
                  move ints). For the web UI's decision-tracing panels.}

        Used for the UCI `info` line and the --play/--show readout.
        """
        root = self._root_node
        kids = sorted(self._children(root),
                      key=lambda mc: getattr(mc[1], "N", 0), reverse=True)
        # PERSPECTIVE: child.Q is stored from the CHILD's POV — i.e. the
        # opponent, who is to move AFTER this move. A move that is GOOD for
        # the player therefore has a NEGATIVE child.Q. Report -child.Q so the
        # per-move eval is in the player-to-move's frame (matching `value`).
        # Without this negation the best move (most visits, lowest child.Q)
        # displayed the LOWEST eval, so the engine looked like it played its
        # worst move while "preferring" a +Q blunder (user-reported via the
        # web UI). Move SELECTION was always correct (it is visit-based, and
        # PUCT already negates internally) — only this readout was inverted.
        top = [(m, int(getattr(c, "N", 0)), -float(getattr(c, "Q", 0.0)))
               for m, c in kids[:top_k]]

        # ── Per-move detail for decision tracing (web UI) ────────────────
        # p = policy prior P (the net's instinct BEFORE search);
        # u = PUCT exploration bonus c_puct·P·√N_parent/(1+N_child)
        #     (why a low-visit move is still being explored);
        # line = the move's own best continuation (most-visited descendants),
        #     i.e. what the engine expects AFTER playing it.
        parent_n = max(1, int(getattr(root, "N", 0)))
        sqrt_pn = math.sqrt(parent_n)
        ext = []
        for m, c in kids[:top_k]:
            n_c = int(getattr(c, "N", 0))
            p_c = float(getattr(c, "P", 0.0))
            u_c = self.c_puct * p_c * sqrt_pn / (1.0 + n_c)
            line = [m]
            node = c
            for _ in range(max(0, pv_len - 1)):
                ch = sorted(self._children(node),
                            key=lambda mc: getattr(mc[1], "N", 0),
                            reverse=True)
                if not ch or getattr(ch[0][1], "N", 0) <= 0:
                    break
                line.append(ch[0][0])
                node = ch[0][1]
            ext.append({
                "move": m,
                "n":    n_c,
                "q":    -float(getattr(c, "Q", 0.0)),
                "p":    p_c,
                "u":    u_c,
                "line": line,
            })

        pv, node = [], root
        for _ in range(pv_len):
            ch = sorted(self._children(node),
                        key=lambda mc: getattr(mc[1], "N", 0), reverse=True)
            if not ch or getattr(ch[0][1], "N", 0) <= 0:
                break
            pv.append(ch[0][0])
            node = ch[0][1]

        nodes = 0
        try:
            nodes = int(self._cpp_tree.tree_size()) if _USE_CPP else 0
        except Exception:
            nodes = 0

        return {"value": self.root_value(), "nodes": nodes,
                "depth": len(pv), "top": top, "pv": pv, "ext": ext}

    # ── Full search (run N simulations with DCA) ──────────────────────────

    def search(
        self,
        board: Board,
        n_sims: int,
        temperature: float = 1.0,
        use_dca: bool = True,
        keep_history: bool = False,
        add_noise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Run a full MCTS search from `board` and return the selected move.

        `keep_history` is forwarded to `begin_search`/`reset`: pass True for
        moves after the first in an ongoing game so the GRU history (maintained
        by `advance()`) is preserved instead of reset to a single board.

        `add_noise=False` skips root Dirichlet noise — use it for *real* play
        (vs humans / GUIs / arena): exploration noise only helps training-data
        diversity and measurably weakens actual play.

        Returns:
            (move, policy_target_array)
        """
        self.begin_search(board, keep_history=keep_history,
                          add_noise=add_noise)

        # Run base simulations
        for _ in range(n_sims):
            self.run_one_simulation()

        # DCA: run extra sims if position is ambiguous
        if use_dca:
            for _ in range(self.dca_extra(board, n_sims)):
                self.run_one_simulation()

        return self.select_move(temperature)

    def begin_search(self, board: Board, keep_history: bool = False,
                     add_noise: bool = True) -> None:
        """Reset, expand the root with network priors, (optionally) add noise."""
        self.reset(board, keep_history=keep_history)
        self._expand_root()
        if add_noise:
            self.add_dirichlet_noise()

    def dca_extra(self, board: Board, n_sims: int) -> int:
        """Dynamic-compute-allocation: how many extra sims this position needs."""
        return compute_extra_sims(
            self._root_node,
            already_run=n_sims,
            base_sims=self.cfg.mcts_sims,
            sims_critical=self.cfg.mcts_sims_critical,
            max_sims=self.cfg.mcts_sims_teacher,
            in_check=board.in_check(),
            q_threshold=self.cfg.dca_q_threshold,
            check_mult=self.cfg.dca_check_mult,
        )

    def _expand_root(self) -> None:
        """Evaluate root position and expand it (before Dirichlet noise)."""
        root = self._root_node
        if _USE_CPP:
            already = root.is_expanded()
        else:
            already = root.is_expanded()
        if already:
            return
        value, policy_logits = self._nn_evaluate(self._root_board,
                                                 exclude_last=True)
        legal = self._root_board.legal_moves()
        if not legal:
            return
        priors = self._compute_priors(policy_logits, legal, self._root_board.side_to_move)
        if _USE_CPP:
            self._cpp_tree.expand(root, legal, np.array(priors, dtype=np.float32))
        else:
            for move, prior in zip(legal, priors):
                root.children[move] = _PyNode(move, root, prior)


# ============================================================
#  Batch MCTS for 8 parallel games
# ============================================================

class ParallelMCTS:
    """
    Manages N parallel MCTS trees and batches their leaf evaluations
    for a single GPU forward pass.
    """

    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        device: torch.device,
        n_games: Optional[int] = None,
    ) -> None:
        self.cfg    = cfg
        self.model  = model
        self.device = device
        self.n      = n_games if n_games is not None else cfg.parallel_games
        self._opening_fens: Optional[list] = None  # lazy opening-book cache
        self._endgame_pool: Optional[list] = None  # lazy won-endgame FEN cache
        # One inference engine shared by every tree in the pool.
        self.engine = build_inference_engine(cfg, model, device)
        self.trees  = [MCTSTree(cfg, model, device, engine=self.engine)
                       for _ in range(self.n)]
        # Reusable fp32 scratch for fill_history; one allocation for the
        # lifetime of the pool instead of one per recorded position.
        # The fp16 copy at recording time is unavoidable (each record needs
        # its own storage in the replay buffer).
        self._hist_scratch = np.zeros(
            (cfg.gru_history_len, cfg.input_planes, 8, 8), dtype=np.float32
        )

        # ── C++ round driver ──────────────────────────────────────────
        # Runs the whole select→(NN)→expand/backup round in C++ with the
        # GIL released and a thread pool over trees. Measured motivation:
        # at pool width 96 the GPU sits at ~8.7k sims/s while its inference
        # ceiling is ~15k pos/s — the serial per-tree Python work (leaf
        # selection, prior gather via move_to_action_index, expand+backup
        # crossings) is the binding constraint. Tablebase probing stays
        # Python-only, so the driver is bypassed when Syzygy is active.
        self._round_driver = None
        self._driver_boards = None
        self._driver_pending = None
        if _USE_CPP and bool(getattr(cfg, "cpp_round_driver", True)):
            try:
                self._round_driver = _cx.RoundDriver(int(cfg.num_actions))
                self._driver_boards = np.zeros(
                    (self.n, cfg.input_planes, 8, 8), dtype=np.float32)
                self._driver_pending = np.zeros(self.n, dtype=np.uint8)
            except Exception as e:  # noqa: BLE001
                print(f"[ParallelMCTS] RoundDriver unavailable "
                      f"({type(e).__name__}: {e}) — using Python round loop.")
                self._round_driver = None

    @torch.inference_mode()
    def _batch_begin_search(
        self, active: List[int], boards: "List[Board]",
        fresh: Optional[set] = None,
    ) -> None:
        """
        Expand all active roots with ONE batched GPU call, then add Dirichlet
        noise.  Replaces N sequential `begin_search` calls (each firing its own
        single-board GPU forward) with a single batch forward over all N roots.

        `fresh` is the set of tree indices that just started a new game (move 0):
        those are reset and their GRU history is re-seeded. Every other active
        tree is at move > 0 — its previous `tree.advance()` already moved the
        root to the played child, preserving that subtree (visit counts / Q
        from earlier simulations). Rebuilding it would throw that work away
        every move. Unexpanded roots (a freshly reached leaf, or a move that
        wasn't searched) are still picked up by the needs_eval pass below;
        expanded roots are reused as-is.
        """
        fresh = fresh or set()
        for i in active:
            if i in fresh:
                self.trees[i].reset(boards[i], keep_history=False)

        # Find trees whose root still needs network evaluation. Compute
        # legal_moves() ONCE here and pass it on — the C++ Board.legal_moves
        # call allocates a fresh Python list every time, so calling it twice
        # per tree per round is pure overhead.
        needs_eval: list[int] = []
        legal_cache: list = []  # parallel to needs_eval
        for i in active:
            if self.trees[i]._root_node.is_expanded():
                continue
            lm = boards[i].legal_moves()
            if not lm:
                continue
            needs_eval.append(i)
            legal_cache.append(lm)

        if needs_eval:
            dtype = self.trees[0]._net_dtype()

            bt = np.stack([boards[i].to_tensor() for i in needs_eval])
            # Root evaluation: history must exclude the root itself (P3-4).
            ht = np.stack([self.trees[i].history_np(exclude_last=True)
                           for i in needs_eval])

            # Direct H2D transfer (no inline pin_memory — that would force a
            # synchronous pageable→pinned copy on this thread, paying more
            # than the resulting async transfer saves).
            board_t = torch.from_numpy(bt).to(self.device, dtype)
            hist_t  = torch.from_numpy(ht).to(self.device, dtype)

            policy_logits, values = self.engine.infer(board_t, hist_t)
            # Fuse the .float() cast with the D2H copy to skip one intermediate
            # GPU tensor allocation. `.to('cpu', dtype, non_blocking=False)`
            # synchronises just like .cpu() does because the consumer below
            # needs the data immediately.
            # Root values are not consumed (priors only) — skip their D2H copy.
            logits_np = policy_logits.to("cpu", torch.float32).numpy()

            for k, i in enumerate(needs_eval):
                tree  = self.trees[i]
                board = boards[i]
                legal = legal_cache[k]  # cached above; no second C++ call
                priors = tree._compute_priors(
                    logits_np[k], legal, board.side_to_move
                )
                if _USE_CPP:
                    tree._cpp_tree.expand(
                        tree._root_node, legal,
                        np.array(priors, dtype=np.float32),
                    )
                else:
                    for move, prior in zip(legal, priors):
                        tree._root_node.children[move] = \
                            _PyNode(move, tree._root_node, prior)

        # Dirichlet noise on every active root
        for i in active:
            self.trees[i].add_dirichlet_noise()

    @torch.inference_mode()
    def _evaluate_batch(
        self, pendings: List[Tuple[int, dict]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One GPU forward for every collected leaf across all trees.

        pendings: list of (tree_index, pending-dict). Returns
        (values[B], policy_logits[B, num_actions]) as numpy arrays.
        """
        dtype = self.trees[0]._net_dtype()

        # Stack once instead of B separate H2D copies. history_np() returns
        # each tree's reused scratch buffer; np.stack copies here, before any
        # tree's next selection overwrites it.
        bt = np.stack([p["leaf_board"].to_tensor() for _, p in pendings])
        ht = np.stack([self.trees[ti].history_np() for ti, _ in pendings])

        # Direct H2D — inline pin_memory() was a perf trap (synchronous
        # pageable→pinned copy on the main thread; the supposed async win
        # from non_blocking=True is gone if we paid for the pin first).
        board_t = torch.from_numpy(bt).to(self.device, dtype)
        hist_t  = torch.from_numpy(ht).to(self.device, dtype)

        policy_logits, value = self.engine.infer(board_t, hist_t)
        # Fuse the BF16→FP32 upcast with the D2H copy (one less GPU kernel
        # allocation than `.float().cpu()`). The caller consumes both
        # immediately, so the sync inside .to('cpu', ...) is unavoidable.
        return (value.to("cpu", torch.float32).numpy(),
                policy_logits.to("cpu", torch.float32).numpy())

    def _batched_sims(self, active: List[int], counts) -> None:
        """
        Run MCTS simulations for the given active trees, sharing one batched
        GPU forward per round. `counts` is an int (same for all) or a dict
        {tree_idx: n_sims} (DCA — per-tree budgets, shrinking active set).

        NOTE: a CPU/GPU double-buffering variant (splitting the trees into two
        disjoint groups and overlapping one group's GPU forward with the
        other's CPU select/backup) was implemented and benchmarked here. It was
        consistently *slower* (~0.6x): this workload is GPU-bound on the network
        forward, so halving the batch costs more GPU efficiency than the CPU
        overlap recovers. Kept serial deliberately.
        """
        if isinstance(counts, int):
            rounds = counts
            budget = None
        else:
            rounds = max(counts.values()) if counts else 0
            budget = counts

        # ── C++ fast path: whole round (select → NN → expand/backup) in
        # C++ with GIL released + thread pool. Syzygy endgame leaves are
        # probed in Python and resolved via driver.resolve_pending (the
        # probe only fires at ≤5 pieces, so it costs nothing in the
        # opening/midgame). Fails open to the legacy loop on any error.
        if self._round_driver is not None:
            try:
                self._batched_sims_cpp(active, rounds, budget)
                return
            except Exception as e:  # noqa: BLE001
                print(f"[ParallelMCTS] RoundDriver failed "
                      f"({type(e).__name__}: {e}) — falling back to "
                      f"Python round loop permanently.", flush=True)
                self._round_driver = None

        for s in range(rounds):
            pendings: List[Tuple[int, dict]] = []
            for i in active:
                if budget is not None and s >= budget[i]:
                    continue
                pending = self.trees[i].select_leaf_only()
                if pending is not None:
                    pendings.append((i, pending))

            if not pendings:
                continue

            values, logits = self._evaluate_batch(pendings)
            for k, (i, pending) in enumerate(pendings):
                self.trees[i].apply_eval(pending, float(values[k]), logits[k])

    def _batched_sims_cpp(self, active: List[int], rounds: int,
                          budget: Optional[Dict[int, int]]) -> None:
        """C++ round-driver variant of the loop above (same semantics)."""
        cfg      = self.cfg
        driver   = self._round_driver
        contempt = float(getattr(cfg, "mcts_contempt", 0.0))
        threads  = int(getattr(cfg, "cpp_round_threads", 0))
        dtype    = self.trees[0]._net_dtype()
        from engine.rules import is_tablebase_available
        tb_on    = is_tablebase_available()

        # Root node/board are CONSTANT across the rounds of one move —
        # build the per-tree argument lists once (per active-set shape).
        cur_idx: List[int] = []
        trees_cpp: list = []
        roots: list = []
        boards: list = []
        fullm: List[int] = []

        def _rebuild(idxs: List[int]) -> None:
            cur_idx[:] = idxs
            trees_cpp[:] = [self.trees[i]._cpp_tree for i in idxs]
            roots[:]     = [self.trees[i]._root_node for i in idxs]
            boards[:]    = [self.trees[i]._root_board._b for i in idxs]
            fullm[:]     = [int(self.trees[i]._root_board.fullmove_number)
                            for i in idxs]

        # ── History cache (the big win) ──────────────────────────────────
        # The GRU history is ROOT-relative, so it is identical for every leaf
        # of a tree across all `rounds` sims of this move. The old code
        # rebuilt it (np.stack of fill_history) AND re-transferred it H2D
        # every round — measured at ~64 % of self-play transfer volume and
        # the single largest cost (.to() = 53 % of wall time), because the
        # history tensor is 8× the board tensor (T=8 stacked planes).
        # Build it ONCE on the GPU here and gather the needed rows per round.
        device = self.device
        active_row = {ti: a for a, ti in enumerate(active)}
        hist_active = np.stack([self.trees[ti].history_np() for ti in active])
        hist_active_gpu = torch.from_numpy(hist_active).to(device, dtype)

        for s in range(rounds):
            idxs = [i for i in active
                    if budget is None or s < budget[i]]
            if not idxs:
                continue
            if idxs != cur_idx:
                _rebuild(idxs)
            m = len(idxs)

            n_pending = driver.select(
                trees_cpp, roots, boards, fullm,
                contempt, cfg.td_lambda,
                cfg.td_lambda_start_move, cfg.td_lambda_end_move,
                self._driver_boards[:m], self._driver_pending[:m],
                threads,
            )
            if n_pending == 0:
                continue

            pend = np.flatnonzero(self._driver_pending[:m])

            # ── Syzygy endgame leaves: probe in Python, resolve in C++ ──
            # piece_count = sum over the 12 piece planes (canonical encoding
            # planes 0-11). Probing only ≤5-piece leaves keeps this at zero
            # cost through the opening/midgame; resolved leaves bypass the
            # NN forward (mirrors _resolve_non_nn in the legacy path).
            if tb_on and pend.size:
                pc = self._driver_boards[pend, 0:12].sum(axis=(1, 2, 3))
                for j in pend[pc <= 5]:
                    tb_val = probe_tablebase(Board.from_fen(driver.slot_fen(int(j))))
                    if tb_val is not None:
                        driver.resolve_pending(int(j), 2.0 * tb_val - 1.0)
                        self._driver_pending[j] = 0
                pend = np.flatnonzero(self._driver_pending[:m])
                if pend.size == 0:
                    continue

            bt = self._driver_boards[pend]                       # (P,21,8,8)
            board_t = torch.from_numpy(bt).to(device, dtype)
            # History: gather the precomputed GPU rows for the pending trees
            # (no per-round rebuild, no per-round history H2D).
            rows = torch.as_tensor([active_row[idxs[j]] for j in pend],
                                   device=device, dtype=torch.long)
            hist_t = hist_active_gpu.index_select(0, rows)
            policy_logits, values = self.engine.infer(board_t, hist_t)

            logits_np = np.ascontiguousarray(
                policy_logits.to("cpu", torch.float32).numpy())
            values_np = np.ascontiguousarray(
                values.to("cpu", torch.float32).numpy())
            driver.finish(logits_np, values_np, threads)

    def _fresh_board(self) -> "Board":
        """
        Starting position for a (new or recycled) self-play game.

        Optional opening-book FEN (random line) then `random_opening_plies`
        uniform-random legal plies — these plies are NOT recorded as training
        targets (recording starts from the returned position), they only
        diversify the data so the net doesn't overfit a few openings.
        Defaults make this a no-op (standard start position).
        """
        from engine.board import Board, STARTPOS_FEN

        if getattr(self.cfg, "opening_book_path", "") and \
                self._opening_fens is None:
            try:
                with open(self.cfg.opening_book_path, encoding="utf-8") as fh:
                    self._opening_fens = [ln.strip() for ln in fh
                                          if ln.strip()]
            except OSError:
                self._opening_fens = []

        plies = getattr(self.cfg, "random_opening_plies", 0)
        # Re-deal if the random plies stumbled into a terminal position
        # (e.g. a fool's-mate pattern): a game seeded on a terminal board
        # produced a sentinel move 0 + a garbage training row (audit P4-2).
        for _attempt in range(10):
            if self._opening_fens:
                board = Board.from_fen(random.choice(self._opening_fens))
            else:
                board = Board.from_fen(STARTPOS_FEN)
            for _ in range(plies):
                if get_game_result(board) != GameResult.ONGOING:
                    break
                legal = board.legal_moves()
                if not legal:
                    break
                board = board.apply_move(random.choice(legal))
            if get_game_result(board) == GameResult.ONGOING:
                return board
        return Board.from_fen(STARTPOS_FEN)

    # ── Endgame curriculum ───────────────────────────────────────────────
    def _build_endgame_pool(self) -> list:
        """Pre-sample a pool of WINNABLE Syzygy endgame FENs (side to move is
        winning). Built lazily, once per pool instance. Random ≤max_pieces
        material is generated with python-chess (probe-only utility use — NOT
        move generation) and filtered by the tablebase so only positions the
        side to move actually wins are kept. The net therefore always practises
        CONVERTING a win into a real mate."""
        from engine.rules import is_tablebase_available, probe_tablebase
        from engine.board import Board

        if not is_tablebase_available():
            return []
        max_pieces = int(getattr(self.cfg, "endgame_curriculum_max_pieces", 5))
        target = int(getattr(self.cfg, "endgame_curriculum_pool_size", 1500))
        max_pieces = min(max_pieces, 5)  # Syzygy probe ceiling

        import chess
        # Instructive textbook wins to guarantee the pool is never empty and
        # always contains the canonical mating techniques.
        seed_fens = [
            "8/8/8/3k4/8/8/3R4/3K4 w - - 0 1",    # KR vs k
            "8/8/8/3k4/8/3Q4/8/3K4 w - - 0 1",    # KQ vs k
            "8/8/3k4/8/8/8/3KP3/8 w - - 0 1",     # KP vs k
            "8/3k4/8/8/8/3K4/3PP3/8 w - - 0 1",   # K2P vs k
            "8/8/8/2k5/8/8/2KBB3/8 w - - 0 1",    # K2B vs k
        ]
        pool: list = []
        for fen in seed_fens:
            try:
                if probe_tablebase(Board.from_fen(fen)) == 1.0:
                    pool.append(fen)
            except Exception:
                continue

        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN]
        attempts = 0
        max_attempts = target * 40
        while len(pool) < target and attempts < max_attempts:
            attempts += 1
            try:
                b = chess.Board.empty()
                k_sq = random.randint(0, 63)
                b.set_piece_at(k_sq, chess.Piece(chess.KING, chess.WHITE))
                ok_sq = random.randint(0, 63)
                while ok_sq == k_sq:
                    ok_sq = random.randint(0, 63)
                b.set_piece_at(ok_sq, chess.Piece(chess.KING, chess.BLACK))
                # max_pieces includes the two kings.
                for _ in range(random.randint(1, max_pieces - 2)):
                    sq = random.randint(0, 63)
                    while b.piece_at(sq):
                        sq = random.randint(0, 63)
                    pt = random.choice(piece_types)
                    color = random.choice([chess.WHITE, chess.BLACK])
                    # Pawns cannot live on the 1st/8th rank.
                    if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                        continue
                    b.set_piece_at(sq, chess.Piece(pt, color))
                b.turn = random.choice([chess.WHITE, chess.BLACK])
                b.clear_stack()
                if not b.is_valid():
                    continue
                board = Board.from_fen(b.fen())
                # Keep only positions the SIDE TO MOVE is winning → the net
                # learns to convert, and the game is guaranteed decisive.
                if probe_tablebase(board) == 1.0:
                    pool.append(b.fen())
            except Exception:
                continue
        return pool

    def _get_endgame_pool(self) -> list:
        pool = getattr(self, "_endgame_pool", None)
        if pool is None:
            pool = self._build_endgame_pool()
            self._endgame_pool = pool
        return pool

    def _endgame_seed_fen(self) -> Optional[str]:
        """A winnable endgame FEN with probability endgame_curriculum_fraction,
        else None (→ normal opening seed)."""
        frac = float(getattr(self.cfg, "endgame_curriculum_fraction", 0.0))
        if frac <= 0.0 or random.random() >= frac:
            return None
        pool = self._get_endgame_pool()
        return random.choice(pool) if pool else None

    def _finalize_record(self, rec: GameRecord, result: "GameResult") -> None:
        # Side to move at each recorded position. The recorded board side is
        # authoritative — mn%2 parity silently inverts EVERY label in a game
        # that starts from a Black-to-move opening FEN, an odd
        # random_opening_plies count, or after a stale-move side desync.
        # Parity remains only as a fallback for records without sides.
        sides = (rec.sides
                 if getattr(rec, "sides", None)
                 and len(rec.sides) == len(rec.move_numbers)
                 else [mn % 2 for mn in rec.move_numbers])
        if result == GameResult.WHITE_WIN:
            game_val = 1.0
        elif result == GameResult.BLACK_WIN:
            game_val = 0.0
        elif result == GameResult.ONGOING and rec.mcts_qs:
            # Move-cap hit: adjudicate from the final search Q instead of
            # blanket-labelling a draw. Labelling every >cap game as 0.5
            # corrupted value targets for all long games (audit P4-1).
            # rec.mcts_qs[-1] is from the perspective of the side to move
            # at the last recorded position.
            thr = float(getattr(self.cfg, "adjudicate_q_threshold", 0.0))
            q_last = rec.mcts_qs[-1]
            last_side = sides[-1] if sides else rec.move_numbers[-1] % 2
            q_white = q_last if last_side == 0 else -q_last
            if thr > 0.0 and q_white >= thr:
                game_val = 1.0
            elif thr > 0.0 and q_white <= -thr:
                game_val = 0.0
            else:
                game_val = 0.5
        else:
            game_val = 0.5  # genuine draw
        rec.game_result = game_val
        for side in sides:
            rec.wdl_labels.append(game_val if side == 0 else 1.0 - game_val)

    def game_stream(self, is_teacher: bool = False):
        """
        Infinite generator that yields one completed ``GameRecord`` at a time.

        ``self.n`` games are always kept in flight: the moment a game ends its
        slot is immediately re-seeded with a fresh game, so every batched GPU
        forward stays full width for the whole run. This removes the
        shrinking-cohort tail (where GPU utilisation collapsed once the first
        games of a fixed batch finished) — the pool never drains.

        Board / history tensors are stored as float16 to roughly halve replay
        RAM (training casts to bf16/fp16 anyway; the MCTS trees keep their own
        fp32 history internally for network accuracy).
        """
        n_sims = self.cfg.mcts_sims_teacher if is_teacher else self.cfg.mcts_sims
        T      = self.cfg.gru_history_len

        records   = [GameRecord() for _ in range(self.n)]
        move_nums = [0] * self.n
        # Per-game resign counter: streaks[i][0/1] = consecutive moves where
        # white/black saw root_value < -resign_q from their own perspective.
        resign_q     = getattr(self.cfg, "resign_q", 0.0)
        resign_limit = getattr(self.cfg, "resign_streak", 0)
        resign_off_p = float(getattr(self.cfg, "resign_disable_fraction", 0.0))
        streaks      = [[0, 0] for _ in range(self.n)]
        # Per-slot resign toggle. A `resign_off_p` fraction of games play out
        # to a real terminal (no resignation) so the net sees actual
        # conversions/checkmates in its training data (audit: 93%-draw arena).
        def _roll_resign() -> bool:
            return random.random() >= resign_off_p

        # Seed a fresh game slot. An `endgame_curriculum_fraction` slice starts
        # from a winnable Syzygy endgame and NEVER resigns — those games must
        # be played out to a real mate so the net learns conversion technique.
        # The rest get the normal opening seed with the usual resign roll.
        def _new_seed():
            from engine.board import Board
            fen = self._endgame_seed_fen()
            if fen is not None:
                return Board.from_fen(fen), False
            return self._fresh_board(), _roll_resign()

        boards    = [None] * self.n
        resign_on = [True] * self.n
        for _i in range(self.n):
            boards[_i], resign_on[_i] = _new_seed()
        for rec in records:
            rec.is_teacher = is_teacher

        active = list(range(self.n))
        fresh  = set(active)  # every slot starts a brand-new game

        while True:
          try:
            # 1) Expand roots with ONE batched GPU forward. `fresh` slots reset
            #    + re-seed history; the rest reuse their advanced subtree.
            self._batch_begin_search(active, boards, fresh=fresh)
            fresh = set()

            # 2) Base simulations — batched across all active trees.
            self._batched_sims(active, n_sims)

            # 3) DCA extra sims — per-tree budgets, still batched.
            extra = {i: self.trees[i].dca_extra(boards[i], n_sims)
                     for i in active}
            self._batched_sims(active, extra)

            # 4) Pick moves, record, advance, check termination per tree.
            for i in active:
                tree  = self.trees[i]
                board = boards[i]
                mn    = move_nums[i]
                temperature = (1.0 if mn < self.cfg.temperature_moves
                               else self.cfg.temperature_final)

                move, policy_target = tree.select_move(temperature)

                if move == 0:
                    # Sentinel: the root had no legal moves — the game is
                    # already over. Finalize WITHOUT recording a garbage row
                    # or applying the "pass move" that move-int 0 decodes to
                    # (audit P4-2 / P1-5).
                    finished = records[i]
                    self._finalize_record(finished, get_game_result(board))
                    boards[i], resign_on[i] = _new_seed()
                    records[i]   = GameRecord()
                    records[i].is_teacher = is_teacher
                    move_nums[i] = 0
                    streaks[i]   = [0, 0]
                    fresh.add(i)
                    if finished.move_numbers:  # only yield non-empty games
                        yield finished
                    continue

                rec = records[i]
                rec.board_tensors.append(board.to_tensor().astype(np.float16))
                rec.policy_targets.append(policy_target)
                q_root = tree.root_value()
                rec.mcts_qs.append(q_root)
                rec.move_numbers.append(mn)
                rec.sides.append(board.side_to_move)
                rec.phases.append(board.get_phase())
                rec.piece_counts.append(board.piece_count())

                # Reuse the pool-wide scratch; fill_history overwrites it
                # in place. Zero it first so stale frames from an older,
                # longer game don't leak through when board_history is short.
                # HISTORY CONVENTION (P3-4): record positions strictly BEFORE
                # the current board ([:-1] — the running history's last entry
                # is the position being recorded), matching what the network
                # sees at root/leaf evaluation time.
                self._hist_scratch.fill(0.0)
                fill_history(self._hist_scratch, tree._board_history[:-1], T)
                rec.history_tensors.append(
                    self._hist_scratch.astype(np.float16)  # owned copy
                )

                # Resign tracking — track consecutive low-Q moves PER SIDE.
                # The board's own side_to_move (NOT mn%2 parity — see
                # _finalize_record) is the side moving now; root_value is
                # from that side's perspective. Disable resign before move
                # 20 — early-opening Q is too noisy when the net is weak,
                # and false resigns bias the training distribution toward
                # pessimism in book positions.
                side = board.side_to_move
                if resign_limit > 0 and resign_on[i] and mn >= 20 and q_root < -resign_q:
                    streaks[i][side] += 1
                else:
                    streaks[i][side] = 0

                new_board = board.apply_move(move)
                tree.advance(move, new_board)
                boards[i] = new_board
                move_nums[i] += 1

                result = get_game_result(new_board)
                resign_loser = None
                if resign_limit > 0 and resign_on[i]:
                    if streaks[i][0] >= resign_limit:
                        resign_loser = 0  # white resigns
                    elif streaks[i][1] >= resign_limit:
                        resign_loser = 1  # black resigns
                    if resign_loser is not None and result == GameResult.ONGOING:
                        result = (GameResult.BLACK_WIN if resign_loser == 0
                                  else GameResult.WHITE_WIN)

                if (result != GameResult.ONGOING
                        or move_nums[i] >= self.cfg.max_game_moves):
                    self._finalize_record(rec, result)
                    finished = rec
                    # Re-seed this slot with a fresh game *before* yielding so
                    # the pool is already full again when the consumer resumes.
                    boards[i], resign_on[i] = _new_seed()
                    records[i]   = GameRecord()
                    records[i].is_teacher = is_teacher
                    move_nums[i] = 0
                    streaks[i]   = [0, 0]
                    fresh.add(i)
                    yield finished

            # `active` is constant (= the full pool); slots are recycled in
            # place rather than dropped, so the batch never shrinks.
          except Exception as e:
            # Surface ANY crash from inside the loop. Without this wrapper a
            # native-extension fault or unexpected exception kills the
            # generator silently, which from the consumer side looks like
            # the whole process just exited.
            print(f"[game_stream] FATAL: {type(e).__name__}: {e}", flush=True)
            try:
                cur_fens = [b.fen() if hasattr(b, "fen") else "<no-fen>"
                            for b in boards]
                print(f"[game_stream] boards at crash: {cur_fens}", flush=True)
                print(f"[game_stream] move_nums: {move_nums}", flush=True)
            except Exception:
                pass
            traceback.print_exc()
            raise
