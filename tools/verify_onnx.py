#!/usr/bin/env python3
"""
tools/verify_onnx.py  —  numerical parity check: eager PyTorch vs ONNX.

Read-only. Exports the current architecture to a temporary fp32 ONNX file,
runs both backends on board/history batches of sizes {1, 17, 122} and asserts:
  - policy logits      allclose(atol=2e-3, rtol=1e-3)
  - value              allclose(atol=2e-3)
  - argmax(policy)      100 % identical   (the load-bearing MCTS property)

Exit code 0 = parity OK (safe to set config.inference_backend = "onnx").

    python tools/verify_onnx.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch

from config import CONFIG
from model.network import build_model
from model.onnx_export import export_onnx


def main() -> int:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — cannot verify. "
              "(pip install onnxruntime  or  onnxruntime-gpu)")
        return 1

    model = build_model(CONFIG, compile_model=False).eval()

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    export_onnx(model, CONFIG, onnx_path)

    sess = ort.InferenceSession(onnx_path,
                                providers=["CPUExecutionProvider"])
    in_b = sess.get_inputs()[0].name
    in_h = sess.get_inputs()[1].name

    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    ref = base.float().eval()  # fp32 reference (matches the exported graph)
    dev = next(ref.parameters()).device

    ok = True
    rng = np.random.default_rng(0)
    for B in (1, 17, 122):
        b = rng.standard_normal(
            (B, CONFIG.input_planes, 8, 8)).astype(np.float32)
        h = rng.standard_normal(
            (B, CONFIG.gru_history_len, CONFIG.input_planes, 8, 8)
        ).astype(np.float32)

        with torch.no_grad():
            p_t, v_t = ref.infer(
                torch.from_numpy(b).to(dev), torch.from_numpy(h).to(dev)
            )
        p_t = p_t.float().cpu().numpy()
        v_t = v_t.float().cpu().numpy().reshape(-1)

        p_o, v_o = sess.run(None, {in_b: b, in_h: h})
        v_o = np.asarray(v_o).reshape(-1)

        p_d = float(np.abs(p_t - p_o).max())
        v_d = float(np.abs(v_t - v_o).max())
        argmax_same = bool((p_t.argmax(-1) == p_o.argmax(-1)).all())
        passed = (p_d <= 2e-3 + 1e-3 * np.abs(p_t).max()
                  and v_d <= 2e-3 and argmax_same)
        ok &= passed
        print(f"  B={B:4d}  policy|d|={p_d:.2e}  value|d|={v_d:.2e}  "
              f"argmax_same={argmax_same}  {'PASS' if passed else 'FAIL'}")

    Path(onnx_path).unlink(missing_ok=True)
    print("\nONNX parity:", "OK ✓" if ok else "FAILED ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
