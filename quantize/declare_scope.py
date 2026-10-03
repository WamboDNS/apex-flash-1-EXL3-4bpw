#!/usr/bin/env python3
"""
Declare in config.json what the conversion actually did, in the terms TensorFold checks.

TensorFold will not read an EXL3 checkpoint that does not say which weights were
quantized. exllamav3 writes `bits`, `codebook`, `head_bits` and so on, but not the scope,
so a checkpoint straight out of the converter is refused:

    tensorfold: GLM-5.3-Flash's CUDA engine reads EXL3 checkpoints with 4-bit
    mcg-codebook routed experts and BF16 elsewhere; this one has bits 4.0,
    codebook mcg, scope None.

That is a reasonable guard rather than a formality -- an EXL3 checkpoint with a different
scope has a tensor layout the GLM kernels cannot read, and it is better to say so at load
time than to half-load it. The published GLM-5.3-Flash EXL3 checkpoint carries these
fields; this adds them.

Everything written here is a statement of fact about the conversion, so only run it on a
checkpoint the rest of the pipeline produced:

  scope                     glm53_routed_experts_only  -- make_recipe.py quantizes only
                            mlp.experts.*, and restore_mtp_bf16.py puts the MTP head's
                            non-expert weights back, so this is true of the output.
  non_routed_dtype_policy   official_source_native     -- unfuse.py restores each plain
                            weight to the dtype the source holds it in.
  bits                      written as an int, matching the reference.

usage: declare_scope.py --out-dir DIR [--scope ...] [--check]
"""

import argparse
import json
import os
import sys

SUFFIXES = {"trellis", "suh", "svh", "mcg"}


def verify_scope(out_dir):
    """Confirm only routed experts are quantized, before claiming it."""
    idx = os.path.join(out_dir, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return None
    with open(idx, encoding="utf8") as f:
        keys = json.load(f)["weight_map"]
    offenders = sorted({
        k.rsplit(".", 1)[0] for k in keys
        if k.rsplit(".", 1)[-1] in SUFFIXES and ".mlp.experts." not in k
    })
    return offenders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--scope", default="glm53_routed_experts_only")
    p.add_argument("--dtype-policy", default="official_source_native")
    p.add_argument("--check", action="store_true",
                   help="only report; do not write")
    args = p.parse_args()

    cfg_path = os.path.join(args.out_dir, "config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"no config.json in {args.out_dir}")

    offenders = verify_scope(args.out_dir)
    if offenders:
        print(f"!! {len(offenders)} quantized module(s) are not routed experts, so "
              f"'{args.scope}' would be a false claim:", file=sys.stderr)
        for o in offenders[:5]:
            print(f"     {o}", file=sys.stderr)
        print("   Run restore_mtp_bf16.py first.", file=sys.stderr)
        return 1
    if offenders is None:
        print("no index to check against; declaring scope unverified", file=sys.stderr)

    with open(cfg_path, encoding="utf8") as f:
        cfg = json.load(f)
    q = cfg.get("quantization_config")
    if not q:
        sys.exit("config.json has no quantization_config -- is this a converted model?")

    before = dict(q)
    if isinstance(q.get("bits"), float) and float(q["bits"]).is_integer():
        q["bits"] = int(q["bits"])
    q["scope"] = args.scope
    q["non_routed_dtype_policy"] = args.dtype_policy
    q.setdefault("serving_reader_qualified", False)

    changed = {k: v for k, v in q.items() if before.get(k) != v}
    if not changed:
        print("already declared, nothing to do")
        return 0

    if args.check:
        print("would set:", json.dumps(changed, sort_keys=True))
        return 0

    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=4)
    print(f"declared in {cfg_path}: {json.dumps(changed, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
