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

    def history_np(self) -> np.ndarray:
        """
        This tree's root GRU-history as a (T, planes, 8, 8) array, written
        into the reused scratch buffer. The caller must stack/copy it before
        the tree's next selection overwrites the buffer (ParallelMCTS stacks
        all trees' histories immediately after collection).
        """
        T = self.cfg.gru_history_len
        return fill_history(self._hist_np, self._board_history, T)

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
                value = 0.0
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
            value = 2.0 * tb_val - 1.0
            return value, False

        return None

    @torch.no_grad()
    def _nn_evaluate(self, board: Board) -> Tuple[float, np.ndarray]:
        """Single-board neural network evaluation (no batching)."""
        board_t = torch.from_numpy(board.to_tensor()).unsqueeze(0)  # (1,21,8,8)
        board_t = board_t.to(device=self.device, dtype=self._net_dtype())

        # GRU history
        history_t = self._make_history_tensor()  # (1, T, 21, 8, 8)

        policy_logits, value_scalar = self.engine.infer(board_t, history_t)

        policy_np = policy_logits[0].float().cpu().numpy()
        value     = float(value_scalar[0].cpu())
        return value, policy_np

    def _net_dtype(self) -> torch.dtype:
        if self.cfg.precision == "bf16":   return torch.bfloat16
        elif self.cfg.precision == "fp16": return torch.float16
        return torch.float32

    def _make_history_tensor(self) -> Optional[torch.Tensor]:
        T = self.cfg.gru_history_len
        fill_history(self._hist_np, self._board_history, T)
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
        if _USE_CPP:
            self._cpp_tree.add_dirichlet_noise(
                self._root_node, self.dirichlet_a, self.dirichlet_e
            )
        else:
            if not self._root_node.children:
                return
            n = len(self._root_node.children)
            noise = np.random.dirichlet([self.dirichlet_a] * n).astype(np.float32)
            for child, eta in zip(self._root_node.children.values(), noise):
                child.P = (1.0 - self.dirichlet_e) * child.P + self.dirichlet_e * eta

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
           "top": [(move_int, visits, q), …]  most-visited first,
           "pv":  [move_int, …]               principal variation}

        Used for the UCI `info` line and the --play/--show readout.
        """
        root = self._root_node
        kids = sorted(self._children(root),
                      key=lambda mc: getattr(mc[1], "N", 0), reverse=True)
        top = [(m, int(getattr(c, "N", 0)), float(getattr(c, "Q", 0.0)))
               for m, c in kids[:top_k]]

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
                "depth": len(pv), "top": top, "pv": pv}

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
        value, policy_logits = self._nn_evaluate(self._root_board)
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

        # Find trees whose root still needs network evaluation
        needs_eval = [i for i in active
                      if not self.trees[i]._root_node.is_expanded()
                      and boards[i].legal_moves()]

        if needs_eval:
            dtype = self.trees[0]._net_dtype()
            non_blocking = (self.device.type == "cuda")

            bt = np.stack([boards[i].to_tensor() for i in needs_eval])
            ht = np.stack([self.trees[i].history_np() for i in needs_eval])

            board_t = (torch.from_numpy(bt).pin_memory()
                       .to(self.device, dtype, non_blocking=non_blocking)
                       if non_blocking
                       else torch.from_numpy(bt).to(self.device, dtype))
            hist_t = (torch.from_numpy(ht).pin_memory()
                      .to(self.device, dtype, non_blocking=non_blocking)
                      if non_blocking
                      else torch.from_numpy(ht).to(self.device, dtype))

            policy_logits, values = self.engine.infer(board_t, hist_t)
            values_np  = values.float().cpu().numpy()
            logits_np  = policy_logits.float().cpu().numpy()

            for k, i in enumerate(needs_eval):
                tree  = self.trees[i]
                board = boards[i]
                legal = board.legal_moves()
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
        non_blocking = (self.device.type == "cuda")

        # Stack once instead of B separate H2D copies. history_np() returns
        # each tree's reused scratch buffer; np.stack copies here, before any
        # tree's next selection overwrites it.
        bt = np.stack([p["leaf_board"].to_tensor() for _, p in pendings])
        ht = np.stack([self.trees[ti].history_np() for ti, _ in pendings])

        board_t = torch.from_numpy(bt).pin_memory().to(
            self.device, dtype, non_blocking=non_blocking
        ) if non_blocking else torch.from_numpy(bt).to(self.device, dtype)
        hist_t = torch.from_numpy(ht).pin_memory().to(
            self.device, dtype, non_blocking=non_blocking
        ) if non_blocking else torch.from_numpy(ht).to(self.device, dtype)

        policy_logits, value = self.engine.infer(board_t, hist_t)
        return (value.float().cpu().numpy(),
                policy_logits.float().cpu().numpy())

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

        if self._opening_fens:
            board = Board.from_fen(random.choice(self._opening_fens))
        else:
            board = Board.from_fen(STARTPOS_FEN)

        plies = getattr(self.cfg, "random_opening_plies", 0)
        for _ in range(plies):
            if get_game_result(board) != GameResult.ONGOING:
                break
            legal = board.legal_moves()
            if not legal:
                break
            board = board.apply_move(random.choice(legal))
        return board

    @staticmethod
    def _finalize_record(rec: GameRecord, result: "GameResult") -> None:
        if result == GameResult.WHITE_WIN:
            game_val = 1.0
        elif result == GameResult.BLACK_WIN:
            game_val = 0.0
        else:
            game_val = 0.5  # draw or forced draw by move limit
        rec.game_result = game_val
        for mn_j in rec.move_numbers:
            side = 0 if mn_j % 2 == 0 else 1  # 0=white, 1=black
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

        boards    = [self._fresh_board() for _ in range(self.n)]
        records   = [GameRecord() for _ in range(self.n)]
        move_nums = [0] * self.n
        # Per-game resign counter: streaks[i][0/1] = consecutive moves where
        # white/black saw root_value < -resign_q from their own perspective.
        resign_q     = getattr(self.cfg, "resign_q", 0.0)
        resign_limit = getattr(self.cfg, "resign_streak", 0)
        streaks      = [[0, 0] for _ in range(self.n)]
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

                rec = records[i]
                rec.board_tensors.append(board.to_tensor().astype(np.float16))
                rec.policy_targets.append(policy_target)
                q_root = tree.root_value()
                rec.mcts_qs.append(q_root)
                rec.move_numbers.append(mn)
                rec.phases.append(board.get_phase())
                rec.piece_counts.append(board.piece_count())

                # Reuse the pool-wide scratch; fill_history overwrites it
                # in place. Zero it first so stale frames from an older,
                # longer game don't leak through when board_history is short.
                self._hist_scratch.fill(0.0)
                fill_history(self._hist_scratch, tree._board_history, T)
                rec.history_tensors.append(
                    self._hist_scratch.astype(np.float16)  # owned copy
                )

                # Resign tracking — track consecutive low-Q moves PER SIDE.
                # `mn % 2` is the side to move BEFORE this move (0=white,1=black);
                # root_value is from that side's perspective. Disable resign
                # before move 20 — early-opening Q is too noisy when the net
                # is weak, and false resigns bias the training distribution
                # toward pessimism in book positions.
                side = mn % 2
                if resign_limit > 0 and mn >= 20 and q_root < -resign_q:
                    streaks[i][side] += 1
                else:
                    streaks[i][side] = 0

                new_board = board.apply_move(move)
                tree.advance(move, new_board)
                boards[i] = new_board
                move_nums[i] += 1

                result = get_game_result(new_board)
                resign_loser = None
                if resign_limit > 0:
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
                    boards[i]    = self._fresh_board()
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
