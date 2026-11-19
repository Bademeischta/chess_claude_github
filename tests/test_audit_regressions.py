"""Regression tests for the 2026-06 audit findings (audit/AUDIT_REPORT.md).

Each test pins one fixed bug. They are deliberately cheap (CPU-only, no
checkpoints) so the suite stays fast.
"""

from __future__ import annotations

import json
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


# ── NEW-1 (2026-06-12): pool registration must persist immediately ───────

def test_pool_register_persists_state(tmp_path):
    """register_current() must write pool_state.json right away. Previously
    save_state() only ran after arenas (step ~10k) / graceful shutdown, so a
    restart found an empty pool and overwrote the net_step0 baseline with
    current weights (observed in the 2026-06-11 production run)."""
    import torch.nn as nn
    from training.pbt import OpponentPool

    pool = OpponentPool(pool_size=3, checkpoint_dir=str(tmp_path),
                        initial_elo=1000.0)
    pool.register_current(nn.Linear(4, 2), step=0, elo=1000.0)

    assert (tmp_path / "pool_state.json").exists(), \
        "register_current must persist pool state (audit NEW-1)"

    # A fresh pool (simulated restart) must see the baseline entry.
    pool2 = OpponentPool(pool_size=3, checkpoint_dir=str(tmp_path),
                         initial_elo=1000.0)
    pool2.load_state()
    assert pool2._pool, "restart must find the registered baseline"
    assert pool2._pool[0].player_id == "net_step0"


# ── NEW-2 (2026-06-12): WDL labels from recorded side, not mn%2 parity ───

def test_wdl_labels_use_recorded_sides():
    """A game seeded from a Black-to-move position has mn=0 ↔ Black. Parity-
    derived labels would invert every label in such games (odd
    random_opening_plies / Black-to-move opening-book FENs)."""
    import types
    from mcts.tree import GameRecord, ParallelMCTS
    from engine.rules import GameResult

    rec = GameRecord()
    rec.move_numbers = [0, 1, 2]
    rec.sides        = [1, 0, 1]     # Black to move first
    rec.mcts_qs      = [0.0, 0.0, 0.0]
    dummy = types.SimpleNamespace(cfg=CONFIG)
    ParallelMCTS._finalize_record(dummy, rec, GameResult.WHITE_WIN)
    # mn=0 was BLACK to move → label 1-1.0 = 0.0 (loss for the mover)
    assert rec.wdl_labels == [0.0, 1.0, 0.0], (
        f"labels must follow recorded sides, got {rec.wdl_labels}")

    # Cap adjudication must also flip via the recorded last side.
    rec2 = GameRecord()
    rec2.move_numbers = [0, 1]
    rec2.sides        = [0, 1]       # last recorded mover is Black
    rec2.mcts_qs      = [0.0, 0.8]   # +0.8 for Black → White-persp -0.8
    ParallelMCTS._finalize_record(dummy, rec2, GameResult.ONGOING)
    assert rec2.game_result == 0.0, "Black-favoured cap game must be 0.0"

    # Fallback: no sides recorded → legacy parity behaviour.
    rec3 = GameRecord()
    rec3.move_numbers = [0, 1]
    rec3.mcts_qs      = [0.0, 0.0]
    ParallelMCTS._finalize_record(dummy, rec3, GameResult.WHITE_WIN)
    assert rec3.wdl_labels == [1.0, 0.0]


# ── NEW-3 (2026-06-12): legacy buffer load survives capacity changes ─────

def test_legacy_buffer_load_capacity_change(tmp_path):
    """PrioritizedReplayBuffer.load() used to rebind tree.tree wholesale —
    a dump from a different capacity silently broke the SumTree leaf-offset
    invariant. It must rebuild from leaves instead."""
    from training.replay_buffer import PrioritizedReplayBuffer, PositionRecord

    def rec():
        pol = np.zeros(CONFIG.num_actions, dtype=np.float32)
        pol[0] = 1.0
        return PositionRecord(
            board_tensor=np.zeros((21, 8, 8), np.float16),
            history_tensor=np.zeros((1, 21, 8, 8), np.float16),
            policy_target=pol, wdl_label=1.0,
            phase=0, piece_count=32, move_number=0,
        )

    small = PrioritizedReplayBuffer(capacity=4)
    for _ in range(4):
        small.add(rec())
    p = tmp_path / "buf.pkl"
    small.save(str(p))

    big = PrioritizedReplayBuffer(capacity=16)
    big.load(str(p))
    assert len(big._data) == 16, "data list must match current capacity"
    assert big._size == 4
    # SumTree invariant: root total == sum of leaves, sampling works.
    leaves = big.tree.tree[big.tree.capacity - 1:]
    assert abs(big.tree.total_priority - leaves.sum()) < 1e-9
    records, idx, w = big.sample(8)
    assert all(r is not None for r in records), \
        "sampling after capacity-change load must never yield empty slots"


# ── 2026-08-04: CUDA-graph capture must not swallow the GRU weights ──────

def test_gru_flatten_is_guarded_on_graph_capture():
    """`GRUHistoryEncoder.forward` may only flatten OUTSIDE a graph capture.

    Inside a `torch.cuda.graph` capture, `flatten_parameters()` re-homes
    every gru.* Parameter into the graph's private memory pool: the weights
    then read as exact zeros outside a replay (dead encoder for every eager
    forward and for `state_dict()`), each replay reverts optimiser updates
    made after capture, and `load_state_dict` writes into graph-private
    memory (→ cudaErrorIllegalAddress).

    Dropping the call entirely is not the answer either — the 2026-08-11
    self-play run then warned "RNN module weights are not part of single
    contiguous chunk of memory" right after the first arena gate and
    recompacted on every forward from then on. So the call stays, guarded
    by `is_current_stream_capturing`. CPU-checkable: no GPU needed.
    """
    import inspect

    from model.attention import GRUHistoryEncoder

    src = inspect.getsource(GRUHistoryEncoder.forward)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "flatten_parameters" in code, \
        "flatten_parameters() must still run outside a capture"
    guard_above = code[:code.index("flatten_parameters")]
    assert "is_current_stream_capturing" in guard_above, (
        "flatten_parameters() in GRUHistoryEncoder.forward is no longer "
        "guarded by is_current_stream_capturing — a CUDA-graph capture "
        "will move the GRU weights into the graph pool")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gru_weights_survive_cuda_graph_capture():
    """End-to-end: after a capture the encoder must still hold its weights
    and the replayed graph must agree with an eager forward."""
    from model.network import build_model

    cfg = CONFIG
    if not getattr(cfg, "use_gru_history", True):
        pytest.skip("GRU-ablated build")
    dev = torch.device("cuda")
    net = build_model(cfg, compile_model=False).eval()
    enc = net.gru_encoder
    assert enc is not None

    before = max(float(p.detach().abs().max())
                 for n, p in enc.named_parameters() if n.startswith("gru."))
    assert before > 0.0, "freshly built GRU must have nonzero weights"

    B = 4
    board = torch.zeros(B, cfg.input_planes, 8, 8, device=dev)
    hist = torch.randn(B, cfg.gru_history_len, cfg.input_planes, 8, 8,
                       device=dev)
    with torch.inference_mode():
        pol_eager, val_eager = net.infer(board, hist)
        pol_eager, val_eager = pol_eager.clone(), val_eager.clone()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3):
            net.infer(board, hist)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        pol_g, val_g = net.infer(board, hist)

    after = max(float(p.detach().abs().max())
                for n, p in enc.named_parameters() if n.startswith("gru."))
    assert after == pytest.approx(before), (
        f"GRU weights lost to the graph pool: {before} -> {after}")

    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(pol_g, pol_eager, atol=1e-3), \
        "captured graph disagrees with the eager forward"
    assert torch.allclose(val_g, val_eager, atol=1e-3)


# ── 2026-08-04: teacher producers must emit a REAL GRU history ──────────

def test_sf_distill_records_carry_real_history():
    """`tools/stockfish_distill` must reconstruct the GRU history window.

    It used to write an all-zero (T, 21, 8, 8) tensor into every record.
    With `teacher_batch_ratio` at 0.30 that paired ~30% of each training
    batch with a context the net never sees while playing — and burned
    21.5 kB of zeros per position on disk.
    """
    chess = pytest.importorskip("chess")
    from tools.stockfish_distill import build_position_records

    board = chess.Board()
    ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5",
            "c2c3", "d7d6", "b1d2", "e8g8"]
    game: list = []
    for i, uci in enumerate(ucis):
        # First two plies stand in for the random-opening prefix: context
        # for what follows, but not teacher signal.
        game.append((board.fen(), None if i < 2 else uci,
                     0 if board.turn == chess.WHITE else 1))
        board.push(chess.Move.from_uci(uci))

    recs = build_position_records(game, 1.0, CONFIG)
    assert len(recs) == len(ucis) - 2, "opening plies must not become targets"

    T = CONFIG.gru_history_len
    for r in recs:
        assert r.history_tensor.shape == (T, CONFIG.input_planes, 8, 8)
        assert r.history_tensor.dtype == np.float16
        assert np.any(r.history_tensor), "history must not be all zeros"

    # Self-play convention (mcts.tree.fill_history): right-aligned, so the
    # last slot is the immediately preceding position, and the window grows
    # from the start of the game.
    assert np.array_equal(recs[-1].history_tensor[-1], recs[-2].board_tensor)
    filled = lambda r: sum(1 for t in range(T) if np.any(r.history_tensor[t]))
    assert filled(recs[0]) == 2, "first record sees only the two opening plies"
    assert filled(recs[-1]) == T, "window fills up and then stays saturated"


# ── 2026-08-06: harvest must stream to disk, not hoard in RAM ────────────

def _mk_record(seed: int):
    from training.replay_buffer import PositionRecord
    pol = np.zeros(CONFIG.num_actions, dtype=np.float32)
    pol[seed % CONFIG.num_actions] = 1.0
    return PositionRecord(
        board_tensor=np.full((21, 8, 8), seed % 7, dtype=np.float16),
        history_tensor=np.zeros((2, 21, 8, 8), np.float16),
        policy_target=pol, wdl_label=1.0,
        phase=0, piece_count=32, move_number=seed, is_teacher=True)


def test_record_stream_roundtrip_and_meta(tmp_path):
    """A streamed harvest file reads back complete, and its sidecar meta
    reports the count without touching the data."""
    from training.replay_buffer import (RecordStreamWriter, count_records,
                                        iter_record_chunks)

    path = tmp_path / "part.pkl"
    with RecordStreamWriter(str(path), chunk_positions=10) as w:
        for i in range(35):
            w.add(_mk_record(i))
    assert w.n_chunks == 4, "35 records at chunk=10 → 3 full + 1 tail"

    # Meta is the cheap path: present, complete, and correct.
    meta = json.loads((tmp_path / "part.pkl.meta.json").read_text())
    assert meta["positions"] == 35 and meta["complete"] is True
    assert count_records(str(path)) == 35

    chunks = list(iter_record_chunks(str(path)))
    assert [len(c) for c in chunks] == [10, 10, 10, 5]
    recs = [r for c in chunks for r in c]
    assert [r.move_number for r in recs] == list(range(35))


def test_record_stream_survives_a_killed_writer(tmp_path):
    """Ctrl+C mid-harvest: every flushed chunk must still be readable, and
    the meta must reflect what actually made it to disk."""
    from training.replay_buffer import (RecordStreamWriter, count_records,
                                        iter_record_chunks)

    path = tmp_path / "killed.pkl"
    w = RecordStreamWriter(str(path), chunk_positions=10)
    for i in range(25):
        w.add(_mk_record(i))
    # 2 chunks flushed, 5 records still buffered; simulate a hard kill by
    # dropping the writer without close().
    w._f.close()
    w._f = None

    assert count_records(str(path)) == 20, "meta counts only flushed chunks"
    recs = [r for c in iter_record_chunks(str(path)) for r in c]
    assert len(recs) == 20
    assert [r.move_number for r in recs] == list(range(20))


def test_legacy_buffer_file_still_readable(tmp_path):
    """The reader must keep understanding parts harvested before the switch —
    `distill_offensive --skip-harvest` runs on exactly those."""
    from training.replay_buffer import (PrioritizedReplayBuffer, count_records,
                                        iter_record_chunks)

    buf = PrioritizedReplayBuffer(capacity=64)
    for i in range(12):
        buf.add(_mk_record(i))
    path = tmp_path / "legacy.pkl"
    buf.save(str(path))

    assert count_records(str(path)) == 12
    recs = [r for c in iter_record_chunks(str(path)) for r in c]
    assert sorted(r.move_number for r in recs) == list(range(12))


def test_seed_teacher_reservoir_is_uniform_and_bounded(tmp_path):
    """seed_teacher must draw its sample without materialising the source."""
    import random as _random

    from tools.seed_teacher import _sample_records
    from training.replay_buffer import RecordStreamWriter

    path = tmp_path / "src.pkl"
    with RecordStreamWriter(str(path), chunk_positions=50) as w:
        for i in range(1000):
            w.add(_mk_record(i))

    _random.seed(7)
    recs, seen = _sample_records(path, 100)
    assert seen == 1000, "the full source must be counted"
    assert len(recs) == 100, "reservoir must fill to exactly the budget"
    assert len({r.move_number for r in recs}) == 100, "no duplicates"
    # Uniformity smoke test: a reservoir over 1000 items should not cluster
    # in the first chunk the way a naive `take the first N` would.
    assert max(r.move_number for r in recs) > 500

    # Budget larger than the source → everything comes back.
    allrecs, seen = _sample_records(path, 5000)
    assert seen == 1000 and len(allrecs) == 1000


# ── 2026-08-06: P4-1 in the distill harvester — cap hit ≠ draw ───────────

class _StubSF:
    """Minimal python-chess engine stand-in for adjudication tests."""

    def __init__(self, score):
        self._score = score
        self.skill_history: list = []
        self.analysed = 0

    def configure(self, opts):
        if "Skill Level" in opts:
            self.skill_history.append(opts["Skill Level"])

    def analyse(self, board, limit):
        self.analysed += 1
        return {"score": self._score}


def test_ply_cap_is_adjudicated_not_stamped_draw():
    """Games stopped by --max-plies must take their label from an evaluation.

    Measured on the 2026-08-06 harvest before the fix: 14.4% of games hit the
    cap, 94% of those were labelled 0.5, and — being the longest games — they
    supplied ~19% of every harvested position. That is audit finding P4-1,
    which was fixed for self-play but never reached this tool.
    """
    chess = pytest.importorskip("chess")
    from tools.stockfish_distill import adjudicate_final

    board = chess.Board()
    cases = [
        (chess.engine.PovScore(chess.engine.Cp(600), chess.WHITE), 1.0, "+6 → White"),
        (chess.engine.PovScore(chess.engine.Cp(-600), chess.WHITE), 0.0, "-6 → Black"),
        (chess.engine.PovScore(chess.engine.Cp(40), chess.WHITE), 0.5, "+0.4 → draw"),
        (chess.engine.PovScore(chess.engine.Cp(-40), chess.WHITE), 0.5, "-0.4 → draw"),
        (chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE), 1.0, "mate → White"),
        (chess.engine.PovScore(chess.engine.Mate(-3), chess.WHITE), 0.0, "mated → Black"),
    ]
    for score, expect, what in cases:
        sf = _StubSF(score)
        got = adjudicate_final(sf, board, skill=12, depth=8, win_cp=200)
        assert got == expect, f"{what}: expected {expect}, got {got}"
        # Full strength for the verdict, configured level restored after.
        assert sf.skill_history == [20, 12], \
            f"{what}: skill must go 20 → back to 12, got {sf.skill_history}"


def test_adjudication_failure_falls_back_to_draw():
    """An engine hiccup must not invent a decisive label."""
    pytest.importorskip("chess")
    import chess as _chess

    from tools.stockfish_distill import adjudicate_final

    class _Broken(_StubSF):
        def analyse(self, board, limit):
            raise RuntimeError("engine died")

    sf = _Broken(None)
    assert adjudicate_final(sf, _chess.Board(), skill=12, depth=8,
                            win_cp=200) == 0.5
    assert sf.skill_history[-1] == 12, "skill must be restored even on error"
