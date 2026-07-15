"""Audit P2/P5 runtime: parameters, dtypes, checkpoint, init, grads, scheduler.

Read-only w.r.t. training state (latest.pt loaded onto CPU copy / fresh models).
Run: python audit/scripts/audit_network.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config import CONFIG, update_config_from_dict
from utils.system_probe import run_probe
from utils.ckpt_arch import match_arch, latest_checkpoint

OUT: dict = {}

# ── replicate main.py init exactly ──────────────────────────────────────
derived = run_probe(write=False)
update_config_from_dict(derived)
OUT["hw_derived_config"] = {k: derived[k] for k in
                            ("device", "precision", "torch_compile", "batch_size",
                             "parallel_games", "replay_buffer_cap")}

ckpt_path = latest_checkpoint(str(ROOT / "checkpoints"))
OUT["latest_checkpoint"] = ckpt_path
match_arch(CONFIG, ckpt_path)
OUT["effective_arch"] = {
    "num_res_blocks": CONFIG.num_res_blocks,
    "num_channels": CONFIG.num_channels,
    "policy_mid_channels": CONFIG.policy_mid_channels,
    "material_scale": CONFIG.material_scale,
    "gru": [CONFIG.gru_hidden, CONFIG.gru_layers, CONFIG.gru_history_len],
}

# ── checkpoint inspection (CPU, read-only) ──────────────────────────────
state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
sd = state["model"]
opt = state.get("optimizer", {})
n_model_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
model_dtypes = {}
for k, v in sd.items():
    if hasattr(v, "dtype"):
        model_dtypes[str(v.dtype)] = model_dtypes.get(str(v.dtype), 0) + v.numel()
opt_dtypes: dict = {}
opt_state = opt.get("state", {})
for pid, s in opt_state.items():
    for name, v in s.items():
        if torch.is_tensor(v):
            key = f"{name}:{v.dtype}"
            opt_dtypes[key] = opt_dtypes.get(key, 0) + v.numel()
OUT["checkpoint"] = {
    "step": state.get("step"),
    "keys": sorted(state.keys()),
    "model_param_count": int(n_model_params),
    "model_dtype_histogram": model_dtypes,
    "optimizer_state_dtype_histogram": opt_dtypes,
    "dirichlet_eps": state.get("dirichlet_eps"),
    "alpha_v_global": state.get("alpha_v_global"),
    "replay_step": state.get("replay_step"),
    "has_rng_states": any("rng" in k.lower() for k in state.keys()),
    "scheduler_last_epoch": state.get("scheduler", {}).get("last_epoch"),
    "file_mb": round(Path(ckpt_path).stat().st_size / 1e6, 1),
}

# ── live model exactly as main.py builds it ─────────────────────────────
from model.network import build_model  # noqa: E402

device = torch.device(CONFIG.device)
model = build_model(CONFIG, compile_model=CONFIG.torch_compile)
base = model._orig_mod if hasattr(model, "_orig_mod") else model

comp = {}
for name, mod in [("gru_encoder", base.gru_encoder),
                  ("input_proj", base.input_proj),
                  ("blocks", base.blocks),
                  ("policy_head", base.policy_head),
                  ("value_head", base.value_head)]:
    comp[name] = sum(p.numel() for p in mod.parameters())
total = base.param_count()
param_dtypes: dict = {}
for p in base.parameters():
    param_dtypes[str(p.dtype)] = param_dtypes.get(str(p.dtype), 0) + p.numel()
OUT["live_model"] = {
    "total_params": total, "by_component": comp,
    "param_dtype_histogram": param_dtypes,
    "fp32_size_mb": round(total * 4 / 1e6, 1),
}

# ── init behavior (fresh model, eval) ───────────────────────────────────
model.eval()
dt = torch.bfloat16 if CONFIG.precision == "bf16" else torch.float32
from engine.board import Board  # noqa: E402
b0 = torch.from_numpy(Board.startpos().to_tensor()).unsqueeze(0).to(device, dt)
h0 = torch.zeros(1, CONFIG.gru_history_len, 21, 8, 8, device=device, dtype=dt)
with torch.no_grad():
    pol, wdl, aux = model(b0, h0)
    q = (wdl[:, 0] - wdl[:, 2]).float()
OUT["init_behavior"] = {
    "wdl_at_init": [round(float(x), 4) for x in wdl[0].float().cpu()],
    "q_at_init": round(float(q[0]), 4),
    "policy_logit_std": round(float(pol.float().std()), 4),
    "policy_logit_absmax": round(float(pol.float().abs().max()), 4),
}

# ── batch-1 vs batch-N and train vs eval consistency ────────────────────
bN = b0.repeat(8, 1, 1, 1)
hN = h0.repeat(8, 1, 1, 1, 1)
with torch.no_grad():
    polN, wdlN, _ = model(bN, hN)
    model.train()
    polT, wdlT, _ = model(bN, hN)
    model.eval()
OUT["consistency"] = {
    "batch1_vs_batchN_policy_maxdiff": float((polN[0] - pol[0]).float().abs().max()),
    "train_vs_eval_policy_maxdiff": float((polT - polN).float().abs().max()),
}

# ── gradient flow with the trainer's actual loss composition ────────────
model.train()
bB = b0.repeat(4, 1, 1, 1).requires_grad_(False)
hB = h0.repeat(4, 1, 1, 1, 1)
polB, wdlB, auxB = model(bB, hB)
target_pol = torch.full((4, CONFIG.num_actions), 1.0 / CONFIG.num_actions,
                        device=device)
logp = torch.nn.functional.log_softmax(polB.float(), dim=-1)
policy_loss = -(target_pol * logp).sum(-1).mean()
qB = (wdlB[:, 0] - wdlB[:, 2]).float()
value_loss = torch.nn.functional.mse_loss(qB, torch.zeros_like(qB))
aux_loss = torch.nn.functional.mse_loss(auxB.squeeze(1).float(),
                                        torch.zeros(4, device=device))
loss = policy_loss + value_loss + 2.0 * aux_loss
loss.backward()
no_grad, zero_grad = [], []
for n, p in base.named_parameters():
    if p.grad is None:
        no_grad.append(n)
    elif float(p.grad.float().abs().max()) == 0.0:
        zero_grad.append(n)
OUT["grad_flow"] = {"params_without_grad": no_grad,
                    "params_with_zero_grad": zero_grad[:20],
                    "n_zero_grad": len(zero_grad)}

# aux-isolation proof: gradient of ONLY the aux loss
model.zero_grad(set_to_none=True)
polB, wdlB, auxB = model(bB, hB)
aux_only = torch.nn.functional.mse_loss(auxB.squeeze(1).float(),
                                        torch.ones(4, device=device))
aux_only.backward()
touched = [n for n, p in base.named_parameters()
           if p.grad is not None and float(p.grad.float().abs().max()) > 0]
OUT["aux_loss_gradient_reaches"] = touched  # expect ONLY value_head.piece_values

# ── OneCycleLR group-lr clobbering proof (P5-1) ─────────────────────────
pg = base.parameter_groups(lr=CONFIG.lr_initial,
                           backbone_mult=1.0, head_mult=0.5, gru_mult=0.3)
opt2 = torch.optim.AdamW(pg, weight_decay=1e-4)
before = [g["lr"] for g in opt2.param_groups]
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt2, max_lr=CONFIG.lr_initial, total_steps=CONFIG.total_steps,
    pct_start=CONFIG.warmup_steps / CONFIG.total_steps,
    anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4)
after_init = [g["lr"] for g in opt2.param_groups]
sched.step(); sched.step()
after_2steps = [g["lr"] for g in opt2.param_groups]
OUT["onecycle_clobbering_P5_1"] = {
    "group_lrs_as_constructed": before,
    "after_scheduler_init": after_init,
    "after_2_steps": after_2steps,
    "multipliers_survive": len(set(round(x, 12) for x in after_2steps)) > 1,
}

# ── VRAM for a training-shaped fwd+bwd at hw batch size ─────────────────
if device.type == "cuda":
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    B = CONFIG.batch_size
    bb = torch.randn(B, 21, 8, 8, device=device, dtype=dt)
    hh = torch.randn(B, CONFIG.gru_history_len, 21, 8, 8, device=device, dtype=dt)
    model.zero_grad(set_to_none=True)
    p3, w3, a3 = model(bb, hh)
    l3 = torch.nn.functional.log_softmax(p3.float(), -1).mean() + w3.float().mean()
    l3.backward()
    torch.cuda.synchronize()
    OUT["vram_train_shape"] = {
        "batch": B,
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
    }

out = Path(__file__).with_name("audit_network.json")
out.write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print(json.dumps(OUT, indent=2, default=str))
print(f"\nResults -> {out}")
