"""
model/inference.py

Inference-engine abstraction for the MCTS / self-play / arena / --play path.

The MCTS code calls exactly one method — ``engine.infer(board_t, hist_t)`` —
and immediately does ``.float().cpu().numpy()`` on the results. So an engine
only has to return two torch tensors (any dtype/device); everything else is
unchanged.

Backends
--------
- ``TorchInferenceEngine``  : wraps the live PyTorch model. Bit-identical to
  the previous direct ``model.infer`` calls. Always available, the default.
- ``OnnxInferenceEngine``   : runs an exported ONNX graph via onnxruntime.
  Only worth it with a GPU execution provider (CUDA/TensorRT). Falls back
  transparently to Torch if onnxruntime / the GPU provider / export is
  unavailable, so self-play can never break because of it.

Training (``trainer.py``) keeps using the live PyTorch model directly and is
NOT affected by any of this.
"""

from __future__ import annotations

import os
import time
from typing import Tuple

import torch


class TorchInferenceEngine:
    """Eager PyTorch inference — identical to the original ``model.infer``."""

    def __init__(self, model) -> None:
        self.model = model

    def infer(self, board_t: torch.Tensor, hist_t: torch.Tensor):
        return self.model.infer(board_t, hist_t)

    def maybe_refresh(self, step: int) -> None:  # no-op for parity of API
        pass


class OnnxInferenceEngine:
    """
    onnxruntime-backed inference. Exports the model to fp32 ONNX once, then
    serves ``infer`` from an ORT session. Inputs/outputs go through numpy on
    CPU (the MCTS tensors are tiny and the callers copy to CPU anyway); the
    speedup comes from the optimised graph execution, not transfer tricks.
    """

    def __init__(self, model, cfg, device, onnx_path: str) -> None:
        import onnxruntime as ort  # raises if not installed → caught upstream

        self.cfg       = cfg
        self.device    = device
        self.onnx_path = onnx_path
        self._mtime    = 0.0

        from model.onnx_export import export_onnx
        if not os.path.exists(onnx_path):
            export_onnx(model, cfg, onnx_path)

        provider = {
            "cuda": "CUDAExecutionProvider",
            "trt":  "TensorrtExecutionProvider",
            "cpu":  "CPUExecutionProvider",
        }.get(getattr(cfg, "onnx_provider", "cuda"), "CUDAExecutionProvider")

        avail = ort.get_available_providers()
        if provider not in avail:
            raise RuntimeError(
                f"ONNX provider {provider!r} unavailable (have {avail}). "
                f"Install onnxruntime-gpu for the CUDA/TensorRT EP."
            )
        # Always keep CPU as a last-resort fallback provider.
        self._providers = [provider, "CPUExecutionProvider"]
        self._ort = ort
        # Fixed batch width: MCTS batch size varies every round (1..pool).
        # The ORT CUDA/TensorRT EP re-plans for every new shape → with varying
        # shapes it spends ~all its time re-optimising instead of computing.
        # Pad every call up to this single fixed width so ORT plans ONCE.
        self._fixed_b = max(1, int(getattr(cfg, "parallel_games", 122)))
        self._make_session()

    def _make_session(self) -> None:
        so = self._ort.SessionOptions()
        so.graph_optimization_level = \
            self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = self._ort.InferenceSession(
            self.onnx_path, sess_options=so, providers=self._providers
        )
        self._mtime = os.path.getmtime(self.onnx_path)
        self._in_board = self._sess.get_inputs()[0].name
        self._in_hist  = self._sess.get_inputs()[1].name

    def maybe_refresh(self, step: int) -> None:
        """Reload the session if the .onnx file was re-exported (between
        games only — never call mid-batch)."""
        try:
            if os.path.getmtime(self.onnx_path) > self._mtime:
                self._make_session()
        except OSError:
            pass

    @torch.no_grad()
    def infer(self, board_t: torch.Tensor, hist_t: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        b = board_t.detach().to("cpu", torch.float32).numpy()
        h = hist_t.detach().to("cpu", torch.float32).numpy()
        real = b.shape[0]

        # Pad up to the fixed batch width so ORT always sees ONE shape.
        F = self._fixed_b if real <= self._fixed_b else real
        if F != real:
            b = np.concatenate(
                [b, np.zeros((F - real, *b.shape[1:]), b.dtype)], axis=0)
            h = np.concatenate(
                [h, np.zeros((F - real, *h.shape[1:]), h.dtype)], axis=0)

        policy, value = self._sess.run(
            None, {self._in_board: b, self._in_hist: h}
        )
        # Drop the padding rows before handing results back.
        return (torch.from_numpy(policy[:real]),
                torch.from_numpy(value[:real]))


def build_inference_engine(cfg, model, device):
    """
    Pick the inference engine from ``cfg.inference_backend``.

    Default ``"torch"`` → zero behaviour change. ``"onnx"`` tries the ONNX
    engine and, on ANY failure (missing onnxruntime, no GPU provider, export
    error), logs the reason and silently falls back to Torch — self-play must
    never die because of an optional accelerator.
    """
    backend = getattr(cfg, "inference_backend", "torch")
    if backend != "onnx":
        return TorchInferenceEngine(model)

    onnx_path = os.path.join(cfg.checkpoint_dir, "model.onnx")
    try:
        eng = OnnxInferenceEngine(model, cfg, device, onnx_path)
        print(f"[inference] ONNX engine active "
              f"({cfg.onnx_provider} provider, {onnx_path}).")
        return eng
    except Exception as e:  # noqa: BLE001 — fallback must catch everything
        print(f"[inference] ONNX unavailable ({type(e).__name__}: {e}) — "
              f"falling back to eager PyTorch.")
        return TorchInferenceEngine(model)
