"""Regression tests for the 2026-06 audit findings (audit/AUDIT_REPORT.md).

Each test pins one fixed bug. They are deliberately cheap (CPU-only, no
checkpoints) so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from config import CONFIG
from engine.board import Board
from engine.movegen import _move_uci, move_to_action_index, action_index_to_move


# ── C1 / P3-5: PUCT must use the parent-perspective child Q ──────────────

class _StubEngine:
    """Uniform priors, value 0 — search outcomes are then purely structural."""

    def infer(self, board_t, hist_t):
        B = board_t.shape[0]
        return torch.zeros(B, CONFIG.num_actions), torch.zeros(B)


class _StubModel:
    pass


def test_puct_finds_mate_in_one_with_stub_net():
    """With value≡0 and uniform priors, the only signal is the terminal win
    behind the mating move. Correct PUCT concentrates visits there; the
    inverted form (audit P3-5) gave it 1 visit out of 300 and avoided it."""
    from mcts.tree import MCTSTree

    old_device, old_prec = CONFIG.device, CONFIG.precision
    CONFIG.device, CONFIG.precision = "cpu", "fp32"
    try:
        tree = MCTSTree(CONFIG, _StubModel(), torch.device("cpu"),
                        engine=_StubEngine())
        board = Board.from_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        tree.reset(board)
        tree._expand_root()
        for _ in range(200):
            tree.run_one_simulation()
        move, _ = tree.select_move(0.0)
        assert _move_uci(move) == "a1a8", (
            f"search chose {_move_uci(move)} over the mate — "
            f"PUCT child-Q perspective regressed (audit P3-5)")
    finally:
        CONFIG.device, CONFIG.precision = old_device, old_prec


def test_python_node_ucb_prefers_winning_child():
    from mcts.node import MCTSNode

    root = MCTSNode()
    root.N = 10
    win = MCTSNode(move=1, parent=root, prior=0.5)
    win.N, win.W = 3, -3.0       # child's own perspective: opponent lost
    win.Q = win.W / win.N        # -1.0 == win for the PARENT
    neutral = MCTSNode(move=2, parent=root, prior=0.5)
    neutral.N, neutral.W, neutral.Q = 3, 0.0, 0.0
    root.children = {1: win, 2: neutral}
    assert root.best_child(c_puct=2.0) is win


# ── C2 / P4-1: move-cap games are adjudicated, not blanket-drawn ──────────

def _finalize(rec, result):
    """Call ParallelMCTS._finalize_record without building a real pool."""
    import types
    from mcts.tree import ParallelMCTS
    dummy = types.SimpleNamespace(cfg=CONFIG)
    ParallelMCTS._finalize_record(dummy, rec, result)


def test_cap_hit_adjudicated_from_final_q():
    from mcts.tree import GameRecord
    from engine.rules import GameResult

    rec = GameRecord()
    rec.move_numbers = list(range(6))
    rec.mcts_qs = [0.0] * 5 + [0.8]   # last position: White to move (mn=5? no:
    # mn 5 is odd → Black to move; q=+0.8 for Black → White-persp -0.8
    _finalize(rec, GameResult.ONGOING)
    assert rec.game_result == 0.0, "Black-favoured cap game must adjudicate 0.0"

    rec2 = GameRecord()
    rec2.move_numbers = list(range(5))            # last mn=4 → White to move
    rec2.mcts_qs = [0.0] * 4 + [0.8]
    _finalize(rec2, GameResult.ONGOING)
    assert rec2.game_result == 1.0, "White-favoured cap game must adjudicate 1.0"

    rec3 = GameRecord()
    rec3.move_numbers = list(range(5))
    rec3.mcts_qs = [0.0] * 5                       # dead equal → draw
    _finalize(rec3, GameResult.ONGOING)
    assert rec3.game_result == 0.5


# ── D5 / P4-3: draw down-weighting actually affects priorities ────────────

def test_draw_positions_get_lower_priority():
    from training.replay_buffer import PrioritizedReplayBuffer, PositionRecord

    buf = PrioritizedReplayBuffer(
        capacity=8, draw_priority_mult=0.5,
        draw_priority_start_mult=0.5, draw_priority_ramp_steps=1,
    )

    def rec(label):
        pol = np.zeros(CONFIG.num_actions, dtype=np.float32)
        pol[0] = 1.0
        return PositionRecord(
            board_tensor=np.zeros((21, 8, 8), np.float16),
            history_tensor=np.zeros((1, 21, 8, 8), np.float16),
            policy_target=pol, wdl_label=label,
            phase=1, piece_count=20, move_number=5,
        )

    buf.add(rec(1.0))   # decisive
    buf.add(rec(0.5))   # draw
    leaf0 = buf.tree.tree[buf.capacity - 1 + 0]
    leaf1 = buf.tree.tree[buf.capacity - 1 + 1]
    assert leaf1 == pytest.approx(leaf0 * 0.5), (
        "draw multiplier not applied at add() — audit P4-3 regressed")
    # ...and survives a priority update
    buf.update_priorities(np.array([0, 1]), np.array([1.0, 1.0]))
    leaf0 = buf.tree.tree[buf.capacity - 1 + 0]
    leaf1 = buf.tree.tree[buf.capacity - 1 + 1]
    assert leaf1 == pytest.approx(leaf0 * 0.5)


# ── D4 / P5-1: OneCycleLR keeps per-group LR multipliers ─────────────────

def test_scheduler_preserves_group_lr_ratios():
    lin1 = torch.nn.Linear(4, 4)
    lin2 = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW([
        {"params": lin1.parameters(), "lr": 1e-3},
        {"params": lin2.parameters(), "lr": 3e-4},
    ])
    group_max = [g["lr"] for g in opt.param_groups]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=group_max, total_steps=1000, pct_start=0.1,
        anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4)
    sched.step(); sched.step()
    lrs = [g["lr"] for g in opt.param_groups]
    assert lrs[0] != lrs[1], "per-group LR multipliers were erased (P5-1)"
    assert lrs[1] / lrs[0] == pytest.approx(0.3, rel=1e-3)


# ── C3 / P2-1: aux loss must reach the value tower ────────────────────────

def test_aux_loss_gradient_reaches_value_tower():
    from model.network import ChessNet

    net = ChessNet(input_planes=21, gru_context=2, channels=32, num_blocks=1,
                   se_every=4, gru_hidden=16, gru_layers=1, history_len=2,
                   num_actions=CONFIG.num_actions, grad_ckpt_from=99,
                   policy_mid_channels=2).train()
    board = torch.zeros(2, 21, 8, 8)
    board[:, 12] = 1.0  # White to move
    hist = torch.zeros(2, 2, 21, 8, 8)
    _, _, aux = net(board, hist)
    loss = torch.nn.functional.mse_loss(aux.squeeze(1),
                                        torch.ones(2))
    loss.backward()
    tower_grads = [p.grad.abs().max().item()
                   for n, p in net.named_parameters()
                   if n.startswith(("value_head.fc1", "value_head.conv",
                                    "blocks.", "input_proj."))
                   and p.grad is not None]
    assert tower_grads and max(tower_grads) > 0, (
        "aux loss sends no gradient to the value tower (audit P2-1)")


# ── D1 / P2-2: value head emits logits; scalar stays in [-1, 1] ──────────

def test_wdl_head_emits_logits_not_probs():
    from model.heads import ValueHead

    vh = ValueHead(in_channels=32, hidden_size=16)
    x = torch.randn(4, 32, 8, 8)
    wdl_logits, aux = vh(x, material_diff=torch.zeros(4, 6))
    sums = wdl_logits.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3), (
        "WDL output looks softmaxed — forward must return raw logits (P2-2)")
    v = vh.scalar_value(wdl_logits)
    assert v.min() >= -1.0 and v.max() <= 1.0
    probs = vh.wdl_probs(wdl_logits)
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-5)


# ── P7-1d: a long winning series must move ELO substantially ─────────────

def test_elo_series_not_capped_at_single_update():
    from utils.elo import ELOSystem

    es = ELOSystem(initial_elo=1000.0, k=32.0)
    es.register("a"); es.register("b")
    new_a, _ = es.update_from_series("a", "b", wins=100, draws=0, losses=0)
    assert new_a > 1100, (
        f"100-0 sweep moved ELO only to {new_a:.0f} — series collapsed "
        f"into one K-update again (audit P7-1d)")


# ── P1-2: repetition marker (plane 20) fires on both history parities ─────

def test_plane20_both_parities():
    b = Board.startpos()
    seq = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3"]
    snap = {}
    for i, uci in enumerate(seq, start=1):
        mv = next(m for m in b.legal_moves() if _move_uci(m) == uci)
        b = b.apply_move(mv)
        snap[i] = float(b.to_tensor()[20].max())
    assert snap[4] == 1.0, "even-parity repetition not marked"
    assert snap[5] == 1.0, "odd-parity repetition not marked (audit P1-2)"


# ── P1-1: ep square only set when the capture is fully legal ─────────────

def test_pinned_ep_not_recorded():
    b = Board.from_fen("8/8/8/8/k2p3R/8/4P3/4K3 w - - 0 1")
    mv = next(m for m in b.legal_moves() if _move_uci(m) == "e2e4")
    after = b.apply_move(mv)
    assert after.ep_square == -1, (
        "ep square set although the only capturer is pinned (audit P1-1)")


# ── encoding bijectivity smoke test (audit T1.1) ──────────────────────────

def test_move_encoding_bijective_startpos_both_colors():
    for fen, stm in [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", 1),
    ]:
        b = Board.from_fen(fen)
        seen = {}
        for m in b.legal_moves():
            ai = move_to_action_index(m, stm)
            assert 0 <= ai < CONFIG.num_actions
            assert ai not in seen
            seen[ai] = m
            assert action_index_to_move(ai, b) == m
