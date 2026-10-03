#!/usr/bin/env python3
"""
Measure how far apex-flash-1's experts moved from GLM-5.3-Flash, per expert.

apex-flash-1 was trained with "rank-256 LoRA across all experts and routers, combined
with full-parameter updates to 16 experts selected based on activation" (cantina.security).
The LoRA is merged into the released weights, so every expert differs from the base, but
not by the same amount: the 16 fully-updated ones should stand well clear of the rest.

That matters for quantization. The upstream GLM-5.3-Flash recipe spends 4 bits on every
routed expert. If 4-bit reconstruction error is the same size as the delta the RL
produced, quantizing at 4 bits puts the fine-tune back in the noise and we are serving
something closer to the base model. This script measures both quantities on the same
tensors so the bitrate can be chosen on evidence.

Both are reported as relative Frobenius norms:

    rl_delta   = ||W_apex - W_base||  / ||W_base||
    quant_err  = ||W_apex - W_dequant|| / ||W_apex||     (with --quant-dir)

Read the two side by side. quant_err comfortably below rl_delta means 4 bits preserves
the fine-tune. Anywhere it is close or above, that expert needs more bits.

usage:
  weight_delta.py --base DIR --apex DIR --layer 3
  weight_delta.py --base DIR --apex DIR --layer 3 --quant-dir OUT --json out.json
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

PROJS = ("gate_proj", "up_proj", "down_proj")


def shard_map(model_dir):
    with open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf8") as f:
        return json.load(f)["weight_map"]


class Reader:
    """Lazy per-shard safetensors reader; keeps one file open at a time."""

    def __init__(self, model_dir):
        from safetensors import safe_open

        self._open = safe_open
        self.dir = model_dir
        self.map = shard_map(model_dir)
        self._handle = None
        self._shard = None

    def available(self, key):
        shard = self.map.get(key)
        return bool(shard) and os.path.exists(os.path.join(self.dir, shard))

    def get(self, key):
        shard = self.map[key]
        if shard != self._shard:
            self._handle = self._open(os.path.join(self.dir, shard), framework="pt")
            self._shard = shard
        return self._handle.get_tensor(key)


def expert_keys(layer, n_experts):
    for e in range(n_experts):
        for proj in PROJS:
            yield e, proj, f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}.weight"


def n_experts_of(model_dir):
    with open(os.path.join(model_dir, "config.json"), encoding="utf8") as f:
        cfg = json.load(f)
    return cfg["text_config"]["n_routed_experts"]


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="GLM-5.3-Flash-BF16 directory")
    p.add_argument("--apex", required=True, help="apex-flash-1 directory")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--quant-dir", default=None,
                   help="converted EXL3 output, to also report 4-bit reconstruction error")
    p.add_argument("--json", default=None)
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    base, apex = Reader(args.base), Reader(args.apex)
    n_exp = n_experts_of(args.apex)

    # Control. Cantina trained "rank-256 LoRA across all experts and routers" plus full
    # updates to a few experts; attention was not touched. Those tensors should therefore
    # be bit-identical between the two checkpoints. If they are not, the comparison is
    # measuring something other than the fine-tune and every number below is suspect.
    control = [k for k in base.map
               if f".layers.{args.layer}." in k
               and ".mlp.experts." not in k
               and k.endswith(".weight")
               and base.available(k) and apex.available(k)]
    if control:
        worst = 0.0
        worst_key = None
        for k in control:
            wb = base.get(k).to(torch.float32)
            wa = apex.get(k).to(torch.float32)
            d = torch.linalg.matrix_norm(wa - wb).item() if wb.ndim == 2 \
                else (wa - wb).norm().item()
            n = torch.linalg.matrix_norm(wb).item() if wb.ndim == 2 else wb.norm().item()
            rel = d / max(n, 1e-12)
            if rel > worst:
                worst, worst_key = rel, k
        print(f"control: {len(control)} frozen tensors in layer {args.layer}, "
              f"largest relative difference {worst:.2e}")
        if worst == 0.0:
            print("         bit-identical, as expected -- the comparison is sound\n")
        else:
            print(f"         NOT identical ({worst_key}); treat the deltas below "
                  f"with caution\n")
    else:
        print("control: no frozen tensors available locally to check against\n")

    rows = []
    skipped = 0
    for e, proj, key in expert_keys(args.layer, n_exp):
        if not (base.available(key) and apex.available(key)):
            skipped += 1
            continue
        wb = base.get(key).to(torch.float32)
        wa = apex.get(key).to(torch.float32)
        nb = torch.linalg.matrix_norm(wb).item()
        delta = torch.linalg.matrix_norm(wa - wb).item() / max(nb, 1e-12)
        rows.append({"expert": e, "proj": proj, "rl_delta": delta})

    if not rows:
        sys.exit(f"No expert tensors for layer {args.layer} found in both directories "
                 f"({skipped} keys missing -- are the right shards downloaded?)")
    if skipped:
        print(f"note: {skipped} expert tensors skipped, shards not present locally\n")

    # Per expert, averaged over the three projections.
    per_expert = {}
    for r in rows:
        per_expert.setdefault(r["expert"], []).append(r["rl_delta"])
    avg = {e: statistics.mean(v) for e, v in per_expert.items()}
    ordered = sorted(avg.items(), key=lambda kv: -kv[1])

    vals = list(avg.values())
    med = statistics.median(vals)
    print(f"layer {args.layer}: {len(avg)} experts compared "
          f"({len(rows)} tensors)\n")
    print(f"  relative RL delta   median {med:.5f}   "
          f"min {min(vals):.5f}   max {max(vals):.5f}")

    # The fully-updated experts should separate from the LoRA-only population. Flag
    # anything well above the median rather than assuming exactly 16.
    outliers = [(e, d) for e, d in ordered if d > med * 3]
    print(f"  experts above 3x the median: {len(outliers)}")
    if outliers:
        print(f"  -> consistent with full-parameter updates to a small subset")

    print(f"\n  top {args.top} by delta:")
    for e, d in ordered[: args.top]:
        mark = "  <- outlier" if d > med * 3 else ""
        print(f"    expert {e:3d}  {d:.5f}{mark}")

    out = {"layer": args.layer, "median_rl_delta": med,
           "experts": {str(e): d for e, d in ordered},
           "outliers": [e for e, _ in outliers]}

    if args.quant_dir:
        print()
        qerr = quant_error(args, apex, n_exp, torch)
        if qerr:
            out["median_quant_err"] = statistics.median(qerr.values())
            verdict(med, out["median_quant_err"])

    if args.json:
        with open(args.json, "w", encoding="utf8") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.json}")


def quant_error(args, apex, n_exp, torch):
    """Relative error between apex's BF16 experts and the converted checkpoint."""
    try:
        from exllamav3 import Config, Model
    except ImportError:
        print("exllamav3 not importable; skipping quantization error", file=sys.stderr)
        return {}

    cfg = Config.from_directory(args.quant_dir)
    model = Model.from_config(cfg)
    errs = {}
    for e, proj, key in expert_keys(args.layer, n_exp):
        if not apex.available(key):
            continue
        mod = model.find_module(key.removesuffix(".weight")) if hasattr(model, "find_module") else None
        if mod is None:
            continue
        try:
            mod.load(cfg.stc)
            deq = mod.get_weight_tensor().to(torch.float32)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  could not dequantize {key}: {exc}", file=sys.stderr)
            continue
        wa = apex.get(key).to(torch.float32)
        errs[(e, proj)] = (torch.linalg.matrix_norm(wa - deq).item()
                           / max(torch.linalg.matrix_norm(wa).item(), 1e-12))
        mod.unload()
    if errs:
        print(f"  relative 4-bit error   median {statistics.median(errs.values()):.5f}")
    return errs


def verdict(rl, qe):
    ratio = qe / max(rl, 1e-12)
    print(f"\n  quant error / RL delta = {ratio:.2f}")
    if ratio < 0.5:
        print("  -> 4 bits preserves the fine-tune comfortably")
    elif ratio < 1.0:
        print("  -> 4 bits is marginal; consider more bits on the high-delta experts")
    else:
        print("  -> 4-bit error exceeds the RL delta: raise the bitrate, "
              "or the quantized model drifts back toward the base")


if __name__ == "__main__":
    main()
