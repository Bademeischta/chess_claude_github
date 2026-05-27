#!/usr/bin/env python3
"""
Hardware probing utility. Probes GPU / CPU / RAM and writes optimally
derived hyperparameters back to config.py.  Run standalone before training.
"""

from __future__ import annotations

import os
import sys
import math
import subprocess
import textwrap
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def _probe_gpu() -> dict:
    info = {
        "name": "unknown",
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "compute_cap": "0.0",
        "driver_version": "unknown",
        "cuda_version": "unknown",
        "bf16_supported": False,
        "flash_attn_supported": False,
        "torch_compile_ok": False,
        "available": False,
    }
    try:
        import torch

        info["cuda_version"] = torch.version.cuda or "n/a"
        if not torch.cuda.is_available():
            return info

        info["available"] = True
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["vram_total_mb"] = props.total_memory // (1024 ** 2)
        info["compute_cap"] = f"{props.major}.{props.minor}"
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["flash_attn_supported"] = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

        # Free VRAM
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(0)
        info["vram_free_mb"] = free // (1024 ** 2)

        # torch.compile on GPU requires Triton for kernel codegen.
        # torch._dynamo is present on 2.x, but Triton has no Windows build,
        # so compile silently falls back to eager on this platform.
        import torch._dynamo  # noqa: F401  – present in 2.x
        try:
            import triton  # noqa: F401
            info["torch_compile_ok"] = True
        except ImportError:
            info["torch_compile_ok"] = False  # no Triton → eager only
    except Exception as e:
        # GPU probe failed unexpectedly — log so the user knows they're
        # silently falling back to CPU defaults (previously this was just
        # `except: pass` and the failure was invisible).
        warnings.warn(
            f"[system_probe] GPU probe failed ({type(e).__name__}: {e}); "
            f"using CPU defaults.",
            RuntimeWarning,
            stacklevel=2,
        )

    # nvidia-smi for driver version
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        info["driver_version"] = out.split("\n")[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # nvidia-smi missing or non-NVIDIA GPU — expected on Intel/AMD/Apple,
        # not worth warning about (info["driver_version"] stays "unknown").
        pass

    return info


def _probe_cpu() -> dict:
    info = {"physical_cores": 4, "logical_cores": 8, "model": "unknown"}
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False) or 4
        info["logical_cores"] = psutil.cpu_count(logical=True) or 8
    except ImportError:
        info["logical_cores"] = os.cpu_count() or 8
        info["physical_cores"] = max(1, info["logical_cores"] // 2)
    try:
        import platform
        info["model"] = platform.processor() or "unknown"
    except Exception as e:  # noqa: BLE001  — non-fatal best-effort lookup
        warnings.warn(
            f"[system_probe] CPU model lookup failed ({type(e).__name__}: {e})",
            RuntimeWarning,
            stacklevel=2,
        )
    return info


def _probe_ram() -> dict:
    info = {"total_gb": 8.0, "available_gb": 4.0}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["total_gb"] = vm.total / 1024 ** 3
        info["available_gb"] = vm.available / 1024 ** 3
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------------------
# Config derivation
# ---------------------------------------------------------------------------

def _derive_config(gpu: dict, cpu: dict, ram: dict) -> dict:
    """Map hardware → optimal hyperparameter values."""

    vram_free_mb = gpu.get("vram_free_mb", 4000)
    vram_total_mb = gpu.get("vram_total_mb", 4000)
    phys_cores = cpu["physical_cores"]
    log_cores = cpu["logical_cores"]

    # ── Precision ──────────────────────────────────────────────────────────
    if gpu["bf16_supported"]:
        precision = "bf16"
    else:
        precision = "fp16"

    # ── Batch size ─────────────────────────────────────────────────────────
    # Activation memory per sample at BF16 with 20 ResBlocks 256ch ≈ 2.4 MB
    # Target: use ≤ 50 % of free VRAM for activations; the rest is model /
    # optimizer / MCTS / replay buffer.
    activation_per_sample_mb = 2.4  # empirical estimate BF16 20-block 256ch
    vram_for_activations_mb = vram_free_mb * 0.50
    batch_size = int(vram_for_activations_mb / activation_per_sample_mb)
    # Clamp BEFORE log2 — a 0 estimate (e.g. 0 MB free VRAM reported) would
    # otherwise crash math.log2. Round down to nearest power of 2 in [64, 512].
    batch_size = max(64, batch_size)
    batch_size = min(512, 2 ** int(math.log2(batch_size)))

    # ── Parallel games ─────────────────────────────────────────────────────
    # Self-play MCTS is GPU-latency-bound: every simulation round fires one
    # batched forward whose batch size == number of concurrent games. A handful
    # of games (the old CPU-core heuristic) leaves the GPU at ~50 % and starves
    # as games finish. Size the concurrent pool to fill the GPU instead — the
    # net is tiny (8x8 boards), so a wide pool is cheap and keeps utilisation
    # high. Bounded so CPU-side leaf selection stays the non-bottleneck.
    parallel_games = max(16, min(128, vram_free_mb // 90))

    # ── DataLoader workers ─────────────────────────────────────────────────
    dataloader_workers = max(2, min(8, log_cores // 3))

    # ── Replay buffer ──────────────────────────────────────────────────────
    # Real per-position cost AFTER the sparse-policy compression in
    # replay_buffer.PositionRecord:
    #   * history (8 × 21 × 64 × 2 B fp16) ≈ 21.5 KB  (still dense)
    #   * board   (21 × 64 × 2 B fp16)     ≈  2.6 KB
    #   * policy target (sparse, ~50 nz):
    #       int16 idx (50 × 2 B) + float32 val (50 × 4 B) ≈ 0.3 KB
    #     (was 18.7 KB dense — ~50× smaller; the trainer expands on read)
    #   * Python object overhead + scalars                ≈ 0.6 KB
    # Total ≈ 25 KB / position. The old 46 KB estimate was correct BEFORE
    # the sparse policy compression; after it the buffer can grow ~1.8× at
    # the same RAM budget. Cap at 60 % of available RAM (was 40 %) — the
    # DataLoader prefetch is small enough that this leaves plenty of
    # headroom for activations / GRU state / OS.
    # NOTE: 25 KB is the theoretical packed size; in practice Python object
    # overhead, numpy header padding, and the per-record metadata push the
    # realised footprint closer to 40 KB. The RAM share is also dropped from
    # 0.60 → 0.40 because the ParallelMCTS pool itself can hold several GB of
    # board-history state in the trees (n_games × T × board objects) on top
    # of the replay buffer — silent process exits during self-play have
    # historically been Windows OOM-kills triggered by exactly this overlap.
    bytes_per_pos = 40_000
    max_buf = int(ram["available_gb"] * 0.40 * 1024 ** 3 / bytes_per_pos)
    replay_buffer_cap = max(50_000, min(2_000_000, max_buf))

    # ── Compile ────────────────────────────────────────────────────────────
    torch_compile = gpu["torch_compile_ok"] and gpu["available"]

    return {
        "device": "cuda" if gpu["available"] else "cpu",
        "precision": precision,
        "torch_compile": torch_compile,
        "batch_size": batch_size,
        "parallel_games": parallel_games,
        "dataloader_workers": dataloader_workers,
        "replay_buffer_cap": replay_buffer_cap,
        "pin_memory": gpu["available"],
    }


# ---------------------------------------------------------------------------
# config.py writer
# ---------------------------------------------------------------------------

def write_config(derived: dict, config_path: Path | None = None) -> None:
    """Merge derived hardware values into config.py without touching manual overrides."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.py"

    if not config_path.exists():
        print(f"[system_probe] config.py not found at {config_path}, skipping write.")
        return

    src = config_path.read_text(encoding="utf-8")
    lines = src.splitlines()
    patched = list(lines)

    # For each derived key, look for the corresponding line and patch it.
    mapping = {
        "device": "device",
        "precision": "precision",
        "torch_compile": "torch_compile",
        "batch_size": "batch_size",
        "parallel_games": "parallel_games",
        "dataloader_workers": "dataloader_workers",
        "replay_buffer_cap": "replay_buffer_cap",
        "pin_memory": "pin_memory",
    }

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        for cfg_key, hw_key in mapping.items():
            if stripped.startswith(cfg_key + ":") or stripped.startswith(cfg_key + " ="):
                val = derived[hw_key]
                # Preserve indentation
                indent = len(line) - len(stripped)
                spaces = " " * indent
                if isinstance(val, str):
                    patched[idx] = f'{spaces}{cfg_key}: str = "{val}"  # hw-derived'
                elif isinstance(val, bool):
                    patched[idx] = f"{spaces}{cfg_key}: bool = {val}  # hw-derived"
                else:
                    patched[idx] = f"{spaces}{cfg_key}: int = {val}  # hw-derived"
                break

    config_path.write_text("\n".join(patched) + "\n", encoding="utf-8")
    print(f"[system_probe] config.py updated at {config_path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _bar(value: float, max_val: float, width: int = 20) -> str:
    filled = int(round(value / max_val * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_summary(gpu: dict, cpu: dict, ram: dict, derived: dict) -> None:
    print()
    print("=" * 60)
    print("  Chess AI — System Probe")
    print("=" * 60)

    # GPU
    print(f"\n  GPU  : {gpu['name']}")
    vt = gpu["vram_total_mb"]
    vf = gpu["vram_free_mb"]
    print(f"  VRAM : {vt/1024:.1f} GB total  |  {vf/1024:.1f} GB free")
    print(f"         {_bar(vt - vf, vt)} used")
    print(f"  CUDA : {gpu['cuda_version']}  |  Compute {gpu['compute_cap']}")
    compile_str = "yes" if gpu["torch_compile_ok"] else "no (Triton unavailable)"
    print(f"  BF16 : {'yes' if gpu['bf16_supported'] else 'no'}"
          f"  |  FlashAttn: {'yes' if gpu['flash_attn_supported'] else 'no'}"
          f"  |  compile: {compile_str}")

    # CPU
    print(f"\n  CPU  : {cpu['model']}")
    print(f"  Cores: {cpu['physical_cores']} physical  /  {cpu['logical_cores']} logical")

    # RAM
    ra = ram["available_gb"]
    rt = ram["total_gb"]
    print(f"\n  RAM  : {rt:.1f} GB total  |  {ra:.1f} GB available")
    print(f"         {_bar(rt - ra, rt)} used")

    # Derived config
    print(f"\n  --- Derived Config -------------------------------------------")
    print(f"  Precision        : {derived['precision']}")
    print(f"  Batch size       : {derived['batch_size']}")
    print(f"  Parallel games   : {derived['parallel_games']}")
    print(f"  DataLoader wkrs  : {derived['dataloader_workers']}")
    print(f"  Replay buf cap   : {derived['replay_buffer_cap']:,}")
    print(f"  torch.compile    : {'on' if derived['torch_compile'] else 'off'}")

    # Speed estimate — NN-bound. Every MCTS simulation needs one neural-net
    # forward pass; that GPU forward is the floor. The C++ extension only
    # accelerates move-gen / tree ops (negligible vs. the net), so there is NO
    # large "C++ multiplier" — the old ~15x estimate was fiction.
    #
    # nn_evals_per_s: a 20-block / 256-ch net at a wide batch runs at roughly
    #   ~9-13 K evals/s on a modern (Ampere/Ada/Blackwell) GPU in eager bf16.
    # efficiency: Python drives select/expand serially between forwards, so the
    #   GPU is idle ~50-60 % of wall time → measured ~0.4 of the NN ceiling.
    nn_evals_per_s = 10_000
    efficiency     = 0.40
    sims_per_move  = 50   # Phase-1 fast-bootstrap budget (rises to 200 later)
    pos_per_s  = nn_evals_per_s * efficiency / sims_per_move
    pos_per_hr = int(pos_per_s * 3600)
    print(f"\n  --- Performance Estimate (rough, NN-bound) ---------------")
    print(f"  NN forward ceiling : ~{nn_evals_per_s:,} evals/s (batch {derived['parallel_games']})")
    print(f"  Positions / hour   : ~{pos_per_hr:,}  @ {sims_per_move} sims/move (bootstrap)")
    print(f"                       ~{pos_per_hr // 4:,}  @ 200 sims/move (standard)")
    print(f"  Note: C++ ext accelerates move-gen only, NOT the GPU forward.")
    print()
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_probe(write: bool = True) -> dict:
    gpu = _probe_gpu()
    cpu = _probe_cpu()
    ram = _probe_ram()
    derived = _derive_config(gpu, cpu, ram)
    print_summary(gpu, cpu, ram, derived)
    if write:
        write_config(derived)
    return derived


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    run_probe(write=True)
