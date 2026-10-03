#!/usr/bin/env python3
"""
Make every tensor's dtype match the published checkpoint's.

exllamav3 stores some weights in a wider dtype than the source holds them in -- the
hyper-connection `hc_*_fn` tensors come out float32 where both the source and the
published GLM-5.3-Flash EXL3 checkpoint keep bfloat16. The serving kernels are written
against the published layout and check:

    TORCH_CHECK(fn.scalar_type() == at::kBFloat16 && fn.size(0) == 24 && ...,
                "hc_partial: fn [24, 16384] bf16");

so a float32 tensor of the right shape fails in the forward pass, long after loading.

unfuse.py restores dtypes from the source model, but only for tensors whose source shard
is still on disk -- and with streaming, most have been evicted by the time it runs. The
result is a checkpoint where a handful of layers are right and the rest are not, which is
worse than uniformly wrong because it loads and then dies partway through a layer.

This reconciles against a reference checkpoint's index instead of the source, so it does
not care what streaming left behind. Casts are checked for exactness first: a value that
does not survive the round trip is a real difference, not a storage choice, and this stops
rather than quietly losing it.

usage:
  fix_dtypes.py --out-dir DIR --reference-dtypes FILE [--dry-run]
  fix_dtypes.py --dump-reference DIR --out FILE      # build the map from a checkpoint
"""

import argparse
import collections
import glob
import json
import os
import struct
import sys

NAMED = {"BF16": "bfloat16", "F16": "float16", "F32": "float32",
         "I16": "int16", "I32": "int32", "U8": "uint8", "I8": "int8"}


def load_map(path):
    """Read a dtype map, gzipped or not.

    The uncompressed map is ~10 MB of very repetitive JSON for 150,226 tensors; gzipped it
    is a few hundred KB, which is a better thing to keep in a repository.
    """
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8") as f:
        return json.load(f)


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def dump_reference(model_dir, out_path):
    """Record name -> dtype for every tensor, without loading any of them."""
    dtypes = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        for key, meta in read_header(path).items():
            if key == "__metadata__":
                continue
            dtypes[key] = meta["dtype"]
    import gzip
    opener = gzip.open if out_path.endswith(".gz") else open
    with opener(out_path, "wt", encoding="utf8") as f:
        json.dump(dtypes, f)
    counts = collections.Counter(dtypes.values())
    print(f"wrote {out_path}: {len(dtypes):,} tensors  {dict(counts)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir")
    p.add_argument("--reference-dtypes")
    p.add_argument("--dump-reference")
    p.add_argument("--out")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Dumping a reference map only reads safetensors headers, so it runs anywhere --
    # including on a box with no torch installed.
    if args.dump_reference:
        if not args.out:
            sys.exit("--dump-reference needs --out")
        dump_reference(args.dump_reference, args.out)
        return 0

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not (args.out_dir and args.reference_dtypes):
        sys.exit("need --out-dir and --reference-dtypes")

    ref = load_map(args.reference_dtypes)

    # Find mismatches without loading tensor data.
    wrong = collections.defaultdict(list)
    summary = collections.Counter()
    for path in sorted(glob.glob(os.path.join(args.out_dir, "*.safetensors"))):
        for key, meta in read_header(path).items():
            if key == "__metadata__":
                continue
            want = ref.get(key)
            if want and want != meta["dtype"]:
                wrong[os.path.basename(path)].append((key, meta["dtype"], want))
                summary[(key.rsplit(".", 1)[-1], meta["dtype"], want)] += 1

    if not wrong:
        print("every dtype already matches the reference")
        return 0

    total = sum(len(v) for v in wrong.values())
    print(f"{total:,} tensor(s) in {len(wrong)} shard(s) differ from the reference:")
    for (suffix, got, want), n in summary.most_common(10):
        print(f"   {n:5d}x  *.{suffix:16s} {got} -> {want}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    fixed = 0
    for shard, items in sorted(wrong.items()):
        path = os.path.join(args.out_dir, shard)
        with safe_open(path, framework="pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for key, got, want in items:
            target = getattr(torch, NAMED.get(want, want.lower()), None)
            if target is None:
                sys.exit(f"unknown reference dtype {want} for {key}")
            src = tensors[key]
            cast = src.to(target)
            # Only a storage change is acceptable here. If the value moves, the wider
            # dtype was carrying something real and narrowing it would lose it.
            if not torch.equal(cast.to(src.dtype), src):
                d = (cast.to(src.dtype).float() - src.float()).abs().max().item()
                sys.exit(f"{key}: {got} -> {want} is lossy (max {d:.3e}); refusing")
            tensors[key] = cast
            fixed += 1
        save_file(tensors, path, metadata={"format": "pt"})
        print(f"{shard}: {len(items)} tensor(s) recast")

    total_size = sum(os.path.getsize(os.path.join(args.out_dir, s))
                     for s in os.listdir(args.out_dir) if s.endswith(".safetensors"))
    idx_path = os.path.join(args.out_dir, "model.safetensors.index.json")
    with open(idx_path, encoding="utf8") as f:
        idx = json.load(f)
    idx["metadata"] = {"total_size": total_size}
    with open(idx_path, "w", encoding="utf8") as f:
        json.dump(idx, f)
    print(f"\n{fixed:,} tensor(s) recast, all losslessly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
