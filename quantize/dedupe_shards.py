#!/usr/bin/env python3
"""
Make the shards contain exactly what the index says -- no duplicates, no orphans.

exllamav3 writes some of the MTP layer's tensors into two shards. `enorm.weight` and
`hnorm.weight` come out byte-identical in both, and TensorFold refuses to load the result:

    tensorfold: CUDA startup memory geometry could not be established on every rank:
    ValueError: duplicate checkpoint tensor: model.language_model.layers.45.enorm.weight

An index maps each name to one shard, so a duplicate is invisible there -- it only shows up
in a reader that walks the files. Worse, rewriting a shard afterwards (unfuse.py,
restore_mtp_bf16.py) drops one copy and strands the other: present on disk, absent from the
index, served to nobody but still breaking the duplicate check.

So this takes the index as the authority and makes the files match it: a tensor stays only
in the shard the index names, and anything else is dropped. Duplicate copies are compared
before one is discarded -- if they ever disagree, that is a real problem and this stops
rather than silently picking one.

usage:
  dedupe_shards.py --out-dir DIR [--dry-run]
"""

import argparse
import collections
import glob
import json
import os
import sys


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    idx_path = os.path.join(args.out_dir, "model.safetensors.index.json")
    with open(idx_path, encoding="utf8") as f:
        index = json.load(f)["weight_map"]

    where = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.out_dir, "*.safetensors"))):
        shard = os.path.basename(path)
        with safe_open(path, framework="pt") as fh:
            for key in fh.keys():
                where[key].append(shard)

    dups = {k: v for k, v in where.items() if len(v) > 1}
    orphans = sorted(set(where) - set(index))
    missing = sorted(set(index) - set(where))

    print(f"index: {len(index):,} tensors   on disk: {len(where):,} distinct")
    print(f"  duplicated across shards : {len(dups)}")
    print(f"  on disk but not indexed  : {len(orphans)}")
    print(f"  indexed but not on disk  : {len(missing)}")
    for k, v in list(dups.items())[:6]:
        print(f"     dup  {k}  {v}")
    for k in orphans[:6]:
        print(f"     orphan  {k}  {where[k]}")

    if missing:
        sys.exit(f"\n{len(missing)} indexed tensor(s) are not on disk, e.g. {missing[:3]}. "
                 "The checkpoint is incomplete; do not try to repair it here.")
    if not dups and not orphans:
        print("\nalready consistent, nothing to do")
        return 0

    # Confirm duplicate copies agree before discarding any of them.
    for key, shards in dups.items():
        tensors = []
        for shard in shards:
            with safe_open(os.path.join(args.out_dir, shard), framework="pt") as fh:
                tensors.append(fh.get_tensor(key))
        for other in tensors[1:]:
            if other.shape != tensors[0].shape or not torch.equal(other, tensors[0]):
                sys.exit(f"copies of {key} differ between {shards}; refusing to guess "
                         "which one is right")
    if dups:
        print(f"\nall {len(dups)} duplicated tensor(s) are byte-identical")

    # Which shards need rewriting: those holding a copy the index does not point at.
    to_fix = sorted({
        shard for key, shards in where.items() for shard in shards
        if index.get(key) != shard
    })
    if args.dry_run:
        print(f"\nwould rewrite {len(to_fix)} shard(s): {to_fix}")
        return 0

    for shard in to_fix:
        path = os.path.join(args.out_dir, shard)
        with safe_open(path, framework="pt") as fh:
            keep = {k: fh.get_tensor(k) for k in fh.keys() if index.get(k) == shard}
            dropped = len(fh.keys()) - len(keep)
        if not keep:
            sys.exit(f"{shard} would be left empty; that is not expected, stopping")
        save_file(keep, path, metadata={"format": "pt"})
        print(f"{shard}: dropped {dropped}, kept {len(keep)}")

    total = sum(os.path.getsize(os.path.join(args.out_dir, s))
                for s in os.listdir(args.out_dir) if s.endswith(".safetensors"))
    with open(idx_path, "w", encoding="utf8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": index}, f)
    print(f"\nindex unchanged at {len(index):,} tensors; total_size refreshed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
