#!/usr/bin/env python3
"""
Put the MTP head's non-expert weights back in BF16.

convert.py allocates every Linear in the MTP head through `--mtp_bits`, so there is no
setting that quantizes its 864 routed experts at 4 bits while leaving the handful of other
projections alone. `--mtp_bits 4` gets the experts right and takes nine extra modules with
it; `--mtp_bits 16` leaves all of them unquantized and adds about 10.9 GB for a head only
`DRAFTER=mtp` ever loads. The published GLM-5.3-Flash checkpoint has it both ways -- 4-bit
experts, BF16 everywhere else in that layer -- so this does the same afterwards:

    model.language_model.layers.45.eh_proj                     .trellis/.suh/.svh/.mcg
    model.language_model.layers.45.mlp.shared_experts.*_proj     -> .weight (BF16)
    model.language_model.layers.45.self_attn.{o,q_a,q_b,kv_a_proj_with_mqa,indexer.wq_b}

The replacement weights are copied from the source model, not dequantized from what the
converter wrote -- a dequantized tensor carries 4-bit error in a 16-bit container, which
would be the worst of both. That means the source shards holding layer 45 must still be
on disk; `stream_shards.py` pins them because they also hold tensors belonging to no
layer, but check before relying on it.

Without this the checkpoint has 36 tensors the serving side does not expect and is missing
the 9 it does, so `DRAFTER=mtp` fails to load. The default drafter never touches the MTP
head and would not notice.

usage: restore_mtp_bf16.py --out-dir DIR --src-dir DIR [--layer 45] [--dry-run]
"""

import argparse
import json
import os
import sys

SUFFIXES = ("trellis", "suh", "svh", "mcg")


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--src-dir", required=True)
    p.add_argument("--layer", type=int, default=45, help="the MTP layer")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_idx_path = os.path.join(args.out_dir, "model.safetensors.index.json")
    with open(out_idx_path, encoding="utf8") as f:
        out_map = json.load(f)["weight_map"]
    with open(os.path.join(args.src_dir, "model.safetensors.index.json"),
              encoding="utf8") as f:
        src_map = json.load(f)["weight_map"]

    prefix = f"model.language_model.layers.{args.layer}."
    # Quantized modules in the MTP layer that are not routed experts.
    stems = sorted({
        k.rsplit(".", 1)[0] for k in out_map
        if k.startswith(prefix) and k.rsplit(".", 1)[-1] in SUFFIXES
        and ".mlp.experts." not in k
    })
    if not stems:
        print(f"nothing to restore in layer {args.layer}")
        return 0

    print(f"{len(stems)} module(s) to put back in BF16:")
    for s in stems:
        print(f"  {s.replace(prefix, '')}")

    missing = [s for s in stems if s + ".weight" not in src_map]
    if missing:
        sys.exit(f"source has no .weight for: {missing[:3]}")
    unavailable = [s for s in stems
                   if not os.path.exists(os.path.join(args.src_dir,
                                                      src_map[s + ".weight"]))]
    if unavailable:
        sys.exit("source shards for layer "
                 f"{args.layer} are not on disk ({len(unavailable)} module(s) affected); "
                 "re-fetch them before running this.")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    # Group the work by the output shard each quantized tensor lives in.
    by_shard = {}
    for stem in stems:
        for suf in SUFFIXES:
            key = f"{stem}.{suf}"
            if key in out_map:
                by_shard.setdefault(out_map[key], []).append(key)

    src_cache = {}

    def source_weight(stem):
        path = os.path.join(args.src_dir, src_map[stem + ".weight"])
        if path not in src_cache:
            src_cache.clear()
            src_cache[path] = safe_open(path, framework="pt")
        return src_cache[path].get_tensor(stem + ".weight")

    new_map = dict(out_map)
    for shard, keys in sorted(by_shard.items()):
        path = os.path.join(args.out_dir, shard)
        with safe_open(path, framework="pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        dropped = 0
        for key in keys:
            tensors.pop(key, None)
            new_map.pop(key, None)
            dropped += 1
        for stem in stems:
            if any(out_map.get(f"{stem}.{s}") == shard for s in SUFFIXES):
                w = source_weight(stem)
                tensors[stem + ".weight"] = w
                new_map[stem + ".weight"] = shard
        save_file(tensors, path, metadata={"format": "pt"})
        print(f"{shard}: dropped {dropped} quantized tensor(s), wrote BF16 weights")

    total = sum(os.path.getsize(os.path.join(args.out_dir, s))
                for s in os.listdir(args.out_dir) if s.endswith(".safetensors"))
    with open(out_idx_path, "w", encoding="utf8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": new_map}, f)
    print(f"\nindex rewritten: {len(new_map):,} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
