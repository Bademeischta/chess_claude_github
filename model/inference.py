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


class CudaGraphInferenceEngine:
    """
    PyTorch inference accelerated via captured CUDA Graphs.

    Eliminates kernel-launch latency (typically 5-20 μs per kernel × ~60
    kernels per forward = 300-1200 μs / forward) by replaying a pre-captured
    sequence of GPU kernels. For our 8×8-board model the launch overhead is
    a substantial fraction of total forward time, so the captured-replay
    path is consistently 1.3-2× faster than eager. Static-shape capture
    requirements:

      - Fixed batch size (we pad smaller batches up to the captured size
        with zeros, oversized batches fall back to the direct eager path)
      - Static input/output memory addresses (we preallocate and `copy_` into
        the static buffers every call)
      - Model in eval() with no autograd (handled by inference_mode wrapping
        capture + every replay)

    Weight updates via in-place writes (`.copy_(...)`) preserve graph
    validity because CUDA Graphs capture data-pointers, not the data itself.
    Replacing tensors entirely (`load_state_dict` on a brand-new module) DOES
    invalidate a graph — call `recapture()` afterwards if that happens.
    """

    def __init__(self, model, cfg, device,
                 batch_size: int, dtype: torch.dtype) -> None:
        if device.type != "cuda":
            raise RuntimeError("CudaGraphInferenceEngine needs a CUDA device.")
        self.model      = model
        self.cfg        = cfg
        self.device     = device
        self.batch_size = int(batch_size)
        self.dtype      = dtype

        input_planes = int(cfg.input_planes)
        history_len  = int(cfg.gru_history_len)

        # Static buffers — addresses captured into the graph; mutated in-place
        # before every replay.
        self._board = torch.zeros(
            (self.batch_size, input_planes, 8, 8),
            device=device, dtype=dtype,
        )
        self._hist = torch.zeros(
            (self.batch_size, history_len, input_planes, 8, 8),
            device=device, dtype=dtype,
        )

        self._policy: torch.Tensor | None = None
        self._value:  torch.Tensor | None = None
        self._graph:  torch.cuda.CUDAGraph | None = None
        self._capture()

    def _capture(self) -> None:
        """(Re-)capture the inference graph. Safe to call after any in-place
        weight update; required after wholesale tensor replacement."""
        torch.cuda.synchronize()

        # Warmup pass on a side stream — required for CUDA Graph capture so
        # cuBLAS / cuDNN have a chance to JIT-pick kernels for this shape.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                _ = self.model.infer(self._board, self._hist)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        # `torch.cuda.graph` is the documented capture context. inference_mode
        # is required so autograd does NOT record the captured ops.
        with torch.cuda.graph(self._graph), torch.inference_mode():
            self._policy, self._value = self.model.infer(self._board, self._hist)

    def infer(self, board_t: torch.Tensor, hist_t: torch.Tensor):
        B = int(board_t.shape[0])
        if B > self.batch_size:
            # Oversize: skip the graph, do a direct eager forward. Callers
            # almost never hit this in steady state (parallel pool is fixed),
            # so the fallback cost is irrelevant.
            return self.model.infer(board_t, hist_t)

        # Match captured shape via in-place copy + zero-pad the tail.
        if B == self.batch_size:
            self._board.copy_(board_t, non_blocking=True)
            self._hist.copy_(hist_t,  non_blocking=True)
        else:
            self._board[:B].copy_(board_t, non_blocking=True)
            self._board[B:].zero_()
            self._hist[:B].copy_(hist_t,  non_blocking=True)
            self._hist[B:].zero_()

        self._graph.replay()
        # The static output tensors get overwritten on the next replay, but
        # the MCTS caller does `.float().cpu().numpy()` immediately, which is
        # a synchronising D2H — by the next infer() the host has the data.
        # `.narrow` is a zero-copy view, cheaper than `.clone()`.
        return self._policy.narrow(0, 0, B), self._value.narrow(0, 0, B)

    def maybe_refresh(self, step: int) -> None:  # API parity with siblings
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

    Default ``"torch"`` → zero behaviour change. ``"cudagraph"`` captures the
    forward as a CUDA Graph for the configured parallel-games batch size.
    ``"onnx"`` tries the ONNX engine. Each accelerator silently falls back to
    Torch on ANY error (missing dep, capture failure, no GPU provider) — the
    self-play loop must never die because of an optional accelerator.
    """
    backend = getattr(cfg, "inference_backend", "torch")

    if backend == "cudagraph":
        # Pool capture: every MCTS round forwards exactly `parallel_games`
        # boards (with a zero-padded tail when fewer trees need eval), so the
        # graph nails the hottest steady-state shape.
        if device.type != "cuda":
            print("[inference] cudagraph backend requested but device is "
                  f"{device.type} — falling back to eager PyTorch.")
            return TorchInferenceEngine(model)
        try:
            # Match the dtype the model was cast to (bf16/fp16/fp32). Use the
            # input_proj weight as a stable proxy — the model wrapper may be a
            # torch.compile module, so reach for `_orig_mod` first.
            base = getattr(model, "_orig_mod", model)
            dtype = next(base.input_proj.parameters()).dtype
            bs = int(getattr(cfg, "parallel_games", 16))
            eng = CudaGraphInferenceEngine(model, cfg, device, bs, dtype)
            print(f"[inference] CUDA-Graph engine active "
                  f"(batch={bs}, dtype={dtype}).")
            return eng
        except Exception as e:  # noqa: BLE001 — fallback must catch everything
            print(f"[inference] CUDA-Graph capture failed "
                  f"({type(e).__name__}: {e}) — falling back to eager PyTorch.")
            return TorchInferenceEngine(model)

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
