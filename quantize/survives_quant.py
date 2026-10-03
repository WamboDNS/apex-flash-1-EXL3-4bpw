#!/usr/bin/env python3
"""
Does 4-bit quantization preserve the difference between apex-flash-1 and its base?

The fine-tune is small in norm: the routed experts sit about 8e-5 away from
GLM-5.3-Flash in relative Frobenius terms. EXL3 at 4 bits introduces its own error,
typically a couple of orders of magnitude larger. That does not by itself mean the
fine-tune is lost -- the RL delta is coherent and low-rank while quantization error is
incoherent -- but it does mean the question has to be answered by measurement.

So: quantize both checkpoints with the identical recipe and compare what comes out.

  r1 = ||W_apex - W_base||     / ||W_base||     the fine-tune, before quantization
  r2 = ||Q(apex) - Q(base)||   / ||W_base||     what is left of it after

Read r2 carefully, because the naive reading is wrong. For a perturbation d much smaller
than the quantizer's step s, a weight's code changes only with probability ~d/s, and when
it does it moves by a whole s. So the expected difference between the two quantized
models is around sqrt(d*s), not d -- larger than r1, not smaller.

  r2 << r1   both models landed on the same codes: the fine-tune was erased outright
  r2 ~= r1   the delta came through roughly intact
  r2 >> r1   the delta survives in magnitude but has been scattered across code
             boundaries -- the models still differ, but much of the difference is now
             quantization noise rather than the direction the training chose

Only the first case is unambiguous. The third is the one to expect here, and it is why
this test informs the bitrate decision rather than settling it: a weight-space norm
cannot tell you whether the *coherent*, low-rank part of the delta still does its job.
That takes a behavioural comparison of the two quantized models on real tasks.

The trellis codes are also compared, but read that number in one direction only. EXL3
encodes each tile as a sequential trellis, so a change early in a tile makes the path
diverge and most downstream code words flip. Measured here, an 8e-5 weight delta flips
86% of the code words -- which is what an arbitrarily small perturbation would do, and so
says nothing about whether the delta carried information. Code divergence is therefore
evidence of erasure when it is *absent* (identical codes mean the fine-tune was rounded
away exactly) and evidence of nothing when it is present.

usage:
  survives_quant.py --base-out DIR --apex-out DIR --layer 3
  survives_quant.py --base-out DIR --apex-out DIR --layer 3 \\
                    --base-src DIR --apex-src DIR     # adds r1/r2
"""

import argparse
import collections
import json
import os
import statistics
import sys

PROJS = ("gate_proj", "up_proj", "down_proj")


def index_of(d):
    with open(os.path.join(d, "model.safetensors.index.json"), encoding="utf8") as f:
        return json.load(f)["weight_map"]


class Store:
    def __init__(self, d):
        from safetensors import safe_open

        self._open = safe_open
        self.dir, self.map = d, index_of(d)
        self._h = self._s = None

    def has(self, k):
        s = self.map.get(k)
        return bool(s) and os.path.exists(os.path.join(self.dir, s))

    def get(self, k):
        s = self.map[k]
        if s != self._s:
            self._h = self._open(os.path.join(self.dir, s), framework="pt")
            self._s = s
        return self._h.get_tensor(k)


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--base-out", required=True, help="converted base checkpoint")
    p.add_argument("--apex-out", required=True, help="converted apex checkpoint")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--base-src", help="unquantized base, enables r1/r2")
    p.add_argument("--apex-src", help="unquantized apex, enables r1/r2")
    p.add_argument("--n-experts", type=int, default=288)
    p.add_argument("--json")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    qb, qa = Store(args.base_out), Store(args.apex_out)

    identical = 0
    compared = 0
    code_diff = []

    for e in range(args.n_experts):
        for proj in PROJS:
            key = (f"model.language_model.layers.{args.layer}"
                   f".mlp.experts.{e}.{proj}.trellis")
            if not (qb.has(key) and qa.has(key)):
                continue
            tb, ta = qb.get(key), qa.get(key)
            compared += 1
            same = torch.equal(tb, ta)
            identical += same
            if not same:
                code_diff.append((tb != ta).float().mean().item())

    if not compared:
        sys.exit(f"No trellis tensors for layer {args.layer} in both conversions.")

    print(f"layer {args.layer}: {compared} expert tensors in both conversions\n")
    pct = 100.0 * identical / compared
    print(f"  bit-identical trellis codes : {identical}/{compared}  ({pct:.1f}%)")
    if code_diff:
        print(f"  where they differ, fraction of code words changed: "
              f"median {statistics.median(code_diff) * 100:.2f}%")

    print()
    if pct > 90:
        print("  -> the codes are mostly identical: 4 bits rounds the fine-tune away")
        print("     outright, and the quantized model is essentially the quantized base.")
    else:
        print("  -> the codes differ, which only means the trellis path diverged. EXL3")
        print("     encodes sequentially, so any perturbation -- informative or not --")
        print("     flips most downstream code words. This does NOT show the fine-tune")
        print("     survived; compare r1 and r2 below, and then test behaviour.")

    out = {"layer": args.layer, "compared": compared,
           "identical_trellis": identical, "identical_pct": pct}

    if args.base_src and args.apex_src:
        out.update(norm_test(args, torch))

    if args.json:
        with open(args.json, "w", encoding="utf8") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.json}")


def norm_test(args, torch):
    """r1 and r2, via exllamav3's dequantization."""
    try:
        from exllamav3 import Config, Model
    except ImportError:
        print("\nexllamav3 not importable; skipping the r1/r2 norms", file=sys.stderr)
        return {}

    sb, sa = Store(args.base_src), Store(args.apex_src)
    models = {}
    for label, d in (("base", args.base_out), ("apex", args.apex_out)):
        cfg = Config.from_directory(d)
        models[label] = (cfg, Model.from_config(cfg))

    r1s, r2s = [], []
    for e in range(args.n_experts):
        for proj in PROJS:
            stem = f"model.language_model.layers.{args.layer}.mlp.experts.{e}.{proj}"
            wkey = stem + ".weight"
            if not (sb.has(wkey) and sa.has(wkey)):
                continue
            wb = sb.get(wkey).to(torch.float32)
            wa = sa.get(wkey).to(torch.float32)
            nb = torch.linalg.matrix_norm(wb).item()

            deq = {}
            ok = True
            for label in ("base", "apex"):
                cfg, model = models[label]
                mod = model.find_module(stem)
                if mod is None:
                    ok = False
                    break
                try:
                    # load() takes the device positionally; the safetensors collection
                    # goes through as a keyword.
                    mod.load(torch.device(args.device), stc=cfg.stc)
                    # get_weight_tensor lives on the inner LinearEXL3, not the Linear,
                    # and returns (in_features, out_features) -- transposed from the
                    # layout the source checkpoint stores.
                    w = mod.inner.get_weight_tensor().to(torch.float32)
                    if w.shape != wb.shape and w.T.shape == wb.shape:
                        w = w.T
                    deq[label] = w
                    mod.unload()
                except Exception as exc:                           # noqa: BLE001
                    if not getattr(norm_test, "_warned", False):
                        print(f"  dequantize failed on {stem} ({label}): "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)
                        norm_test._warned = True
                    ok = False
                    break
            if not ok:
                continue
            r1s.append(torch.linalg.matrix_norm(wa - wb).item() / max(nb, 1e-12))
            r2s.append(torch.linalg.matrix_norm(deq["apex"] - deq["base"]).item()
                       / max(nb, 1e-12))

    if not r1s:
        print("\ncould not dequantize; trellis identity above is the result", file=sys.stderr)
        return {}

    r1, r2 = statistics.median(r1s), statistics.median(r2s)
    print(f"\n  r1  fine-tune before quantization : {r1:.3e}")
    print(f"  r2  fine-tune after quantization  : {r2:.3e}")
    print(f"  r2/r1                             : {r2 / max(r1, 1e-12):.3f}")
    return {"r1": r1, "r2": r2, "ratio": r2 / max(r1, 1e-12)}


if __name__ == "__main__":
    main()
