"""
model/onnx_export.py

Export the inference forward (policy logits + scalar value, no aux head) to a
single fp32 ONNX graph with a dynamic batch axis.

Why fp32: the live net is bf16 except the GRU encoder which is forced fp32
(cuDNN rejects bf16 RNN weights). Exporting one consistent fp32 graph sidesteps
that mixed-precision split entirely and avoids onnxruntime's weak bf16 CUDA
coverage. An fp16 variant (with the GRU kept fp32 via an op block list) is a
possible later optimisation, deliberately not done in this conservative cut.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class _InferWrapper(nn.Module):
    """Reproduces ChessNet.infer: (policy_logits, scalar value), no aux."""

    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def forward(self, board: torch.Tensor, history: torch.Tensor):
        policy_logits, wdl, _ = self.net.forward(board, history)
        value = self.net.value_head.scalar_value(wdl)
        return policy_logits, value


def export_onnx(model, cfg, path: str, opset: int = 17) -> str:
    """
    Write an fp32 ONNX graph of the inference forward to `path`.
    The live `model` is untouched (a CPU fp32 deep copy is traced).
    """
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    net = copy.deepcopy(base).to("cpu").float().eval()
    wrapper = _InferWrapper(net).eval()

    B = 4
    dummy_board = torch.zeros(B, cfg.input_planes, 8, 8, dtype=torch.float32)
    dummy_hist = torch.zeros(
        B, cfg.gru_history_len, cfg.input_planes, 8, 8, dtype=torch.float32
    )

    torch.onnx.export(
        wrapper,
        (dummy_board, dummy_hist),
        path,
        input_names=["board", "history"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board":         {0: "batch"},
            "history":       {0: "batch"},
            "policy_logits": {0: "batch"},
            "value":         {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        # Legacy TorchScript exporter: stable, predictable GRU/GroupNorm
        # handling, only needs `onnx` (not onnxscript). torch 2.11 defaults
        # to the dynamo exporter which pulls extra deps.
        dynamo=False,
    )
    return path
