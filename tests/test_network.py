"""Network forward — shape contracts and the widened policy head."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.network import ChessNet


@pytest.fixture(scope="module")
def net():
    # Smallest sensible net — 2 blocks, fp32, CPU. Keeps the test seconds-fast.
    return ChessNet(
        input_planes=21, gru_context=4, channels=32, num_blocks=2,
        se_every=4, gru_hidden=32, gru_layers=1, history_len=4,
        num_actions=4672, grad_ckpt_from=99,  # disable checkpointing
        policy_mid_channels=8, material_scale=12.0,
    ).eval()


def test_forward_shapes(net):
    B = 2
    board = torch.zeros(B, 21, 8, 8)
    hist = torch.zeros(B, 4, 21, 8, 8)
    policy, wdl, aux = net(board, hist)
    assert policy.shape == (B, 4672)
    assert wdl.shape == (B, 3)
    assert aux.shape == (B, 1)
    # WDL row-sums to 1 (softmax).
    assert torch.allclose(wdl.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_forward_without_history(net):
    """The history-less code path must still produce well-shaped outputs."""
    B = 1
    board = torch.zeros(B, 21, 8, 8)
    policy, wdl, aux = net(board, None)
    assert policy.shape == (B, 4672)
    assert wdl.shape == (B, 3)


def test_policy_head_width_configurable():
    """policy_mid_channels actually changes the conv output width."""
    n = ChessNet(channels=16, num_blocks=1, gru_hidden=8, gru_layers=1,
                 history_len=2, policy_mid_channels=16)
    assert n.policy_head.conv.out_channels == 16
    assert n.policy_head.fc.in_features == 16 * 8 * 8


def test_masked_softmax_is_stable_for_huge_logits():
    """Large logits must not produce NaN/Inf and the mask must drop illegals."""
    from model.heads import PolicyHead
    head = PolicyHead(in_channels=16, num_actions=4672, mid_channels=8)
    logits = torch.full((1, 4672), 1e6)
    logits[0, 0] = 2e6  # one "winning" logit
    mask = torch.zeros(1, 4672, dtype=torch.bool)
    mask[0, :5] = True
    probs = head.masked_softmax(logits, mask)
    assert torch.isfinite(probs).all()
    # All illegal moves get probability 0 (within numerical tolerance).
    assert probs[0, 5:].sum().item() < 1e-6
    # Legal moves sum to 1.
    assert probs[0, :5].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_value_head_material_scale_in_state_dict():
    """Material scale travels with state_dict for reproducibility."""
    from model.heads import ValueHead
    h1 = ValueHead(material_scale=7.5)
    sd = h1.state_dict()
    assert "_material_scale" in sd
    assert float(sd["_material_scale"]) == pytest.approx(7.5)

    h2 = ValueHead(material_scale=99.0)
    h2.load_state_dict(sd)
    assert float(h2._material_scale) == pytest.approx(7.5)
