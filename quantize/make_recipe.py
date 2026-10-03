#!/usr/bin/env python3
"""
Write the per-tensor bitrate recipe for apex-flash-1.

The scope matches Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw exactly: the routed experts
(mlp.experts.N.gate_proj / up_proj / down_proj) go to 4 bits, and every other weight
stays unquantized in BF16 -- attention, the shared expert, the router gate, the dense
MLPs of the first three layers, the KDA projections, the DSA indexer, lm_head, the
embedding, and the vision tower.

That split is not a guess. Reading the 150,226 tensor names in the upstream checkpoint's
model.safetensors.index.json, exactly 37,152 modules carry the EXL3 suffixes
(trellis/suh/svh/mcg) and all 37,152 are routed-expert projections; the remaining 1,618
tensors are stored as plain weights. apex-flash-1 has the same architecture and an
isomorphic tensor set (38,770 = 37,152 + 1,618), so the same recipe reproduces the layout.

The keys are read off the model that exllamav3 builds from config.json rather than
pattern-matched against the index, because create_q_strategy_from_recipe() requires the
recipe to cover the budgeted tensors exactly, and only the model knows which Linears
are budgeted.

usage: make_recipe.py <in_dir> <out.yaml> [--expert-bits 4]
"""

import argparse
import re
import sys

EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$")

# Stored unquantized. exllamav3 accepts 16 as "keep the original weight" in a recipe.
BF16 = 16


def fmt_bits(v):
    """Whole bitrates must be written as ints.

    exllamav3 passes a recipe's bitrate straight through to its C++ binding as K, which
    is typed int and also used as `65536 >> K`. A YAML `4.0` reaches it as a Python float
    and the call fails with "incompatible function arguments" partway into the first MoE
    layer. Half rates (1.5 / 2.5 / 3.5, mul1 codebook only) stay floats, as intended.
    """
    f = float(v)
    return int(f) if f.is_integer() else f


def budgeted_linears(model):
    """Every Linear the allocator expects a 'bits' entry for, in layout order."""
    from exllamav3.modules.linear import Linear

    found = []

    def walk(module):
        if isinstance(module, Linear) and module.qmap is not None:
            if getattr(module, "qbits_key", None) == "bits":
                found.append(module)
        for sm in module.modules:
            walk(sm)

    walk(model)
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("in_dir")
    p.add_argument("out_yaml")
    p.add_argument("--expert-bits", type=float, default=4)
    args = p.parse_args()

    from exllamav3 import Config, Model

    config = Config.from_directory(args.in_dir)
    model = Model.from_config(config)

    # Only the main model is budgeted through the recipe. The MTP head's Linears carry
    # qbits_key="mtp_bits" and the vision tower's are given a fixed rate, so neither can
    # be addressed here -- they are set with convert.py's --mtp_bits / --vision_bits.
    # That is why this covers 36,288 expert projections and not the checkpoint's 37,152:
    # the remaining 864 belong to the MTP head and follow --mtp_bits.
    models = [("main", model)]

    tensors = {}
    n_expert = 0
    numel_q = 0
    numel_bf16 = 0

    for name, m in models:
        for lin in budgeted_linears(m):
            if EXPERT_RE.search(lin.key):
                tensors[lin.key] = args.expert_bits
                n_expert += 1
                numel_q += lin.weights_numel()
            else:
                tensors[lin.key] = BF16
                numel_bf16 += lin.weights_numel()

    if not n_expert:
        sys.exit("No routed-expert projections found -- wrong model or changed naming.")

    # Reported bpw covers the quantized tensors only, which is what --bits means to the
    # converter when a recipe drives the allocation.
    target_bpw = args.expert_bits

    with open(args.out_yaml, "w", encoding="utf8") as f:
        f.write("# apex-flash-1 EXL3: routed experts quantized, everything else BF16.\n")
        f.write(f"# {n_expert} expert projections @ {args.expert_bits} bpw, "
                f"{len(tensors) - n_expert} other Linears unquantized.\n")
        f.write(f"target_bpw: {fmt_bits(target_bpw)}\n")
        f.write("head_bits: 16\n")
        f.write("tensors:\n")
        for k in sorted(tensors):
            f.write(f"  {k}: {fmt_bits(tensors[k])}\n")

    print(f"wrote {args.out_yaml}")
    print(f"  expert projections : {n_expert} @ {args.expert_bits} bpw "
          f"({numel_q / 1e9:.1f}G params -> {numel_q * args.expert_bits / 8 / 1e9:.1f} GB)")
    print(f"  kept BF16          : {len(tensors) - n_expert} Linears "
          f"({numel_bf16 / 1e9:.1f}G params -> {numel_bf16 * 2 / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
