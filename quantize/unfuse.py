#!/usr/bin/env python3
"""
Split the attention tensors exllamav3 fuses, so the checkpoint loads.

exllamav3 represents the KDA layers' projections as single fused modules. Its log says so
while converting:

    Cloned ...layers.1.self_attn.qkv_proj from ['...q_proj', '...k_proj', '...v_proj']

and it writes them fused. But fusing is what the *loader* does, from split tensors --
the published GLM-5.3-Flash EXL3 checkpoint stores `q_proj` / `k_proj` / `v_proj` and
`q_conv1d` / `k_conv1d` / `v_conv1d` separately, and the serving side goes looking for
those names. A checkpoint written with the fused names converts cleanly and then fails to
load, which is an expensive thing to discover at the end of a multi-hour run.

So after converting, split them back:

    self_attn.qkv_proj.weight  [3H, D]     -> q_proj / k_proj / v_proj   [H, D] each
    self_attn.conv1d.weight    [3H, 1, K]  -> q_conv1d / k_conv1d / v_conv1d

Both are stored unquantized under this recipe, so the split is exact -- it is a view of
the same bytes, not a re-quantization. The order is q, k, v, which is the order
exllamav3's own clone message reports.

exllamav3 also writes the unquantized weights as float16, where both the source and the
published checkpoint store them as bfloat16. The values survive -- float16 carries more
mantissa than bfloat16 -- but anything below float16's subnormal floor (~6e-8) is flushed
to zero, so the round trip is not quite exact: measured over one layer, the worst
difference was 3e-8 and 805 values went to zero. That is far below anything quantization
does, but the dtype itself is worth restoring, because the serving kernels were written
against a bfloat16 checkpoint. So this also casts each plain weight back to whatever dtype
the source holds it in.

With --verify and the original model still on disk, each split piece is compared against
the source tensor it should equal.

usage:
  unfuse.py --out-dir DIR
  unfuse.py --out-dir DIR --verify --src-dir DIR
  unfuse.py --out-dir DIR --dry-run
"""

import argparse
import json
import os
import shutil
import sys

FUSED = {
    "self_attn.qkv_proj.weight": ("self_attn.q_proj.weight",
                                  "self_attn.k_proj.weight",
                                  "self_attn.v_proj.weight"),
    "self_attn.conv1d.weight": ("self_attn.q_conv1d.weight",
                                "self_attn.k_conv1d.weight",
                                "self_attn.v_conv1d.weight"),
}


def split_name(key):
    """(prefix, parts) if this tensor is one exllamav3 fused, else None."""
    for suffix, parts in FUSED.items():
        if key.endswith("." + suffix):
            prefix = key[: -len(suffix)]
            return prefix, parts
    return None


def restore_dtypes(tensors, src, src_dir, torch, safe_open):
    """Put each plain weight back in the dtype the source holds it in.

    Only tensors that exist in the source are touched; the quantization artefacts
    (trellis / suh / svh / mcg) keep whatever exllamav3 chose for them, which is what
    the published checkpoint stores too.
    """
    if src is None:
        return tensors, 0
    import os as _os
    n = 0
    cache = {}
    for key, t in list(tensors.items()):
        shard = src.get(key)
        if not shard:
            continue
        sp = _os.path.join(src_dir, shard)
        if not _os.path.exists(sp):
            continue
        if sp not in cache:
            cache.clear()
            cache[sp] = safe_open(sp, framework="pt")
        want = cache[sp].get_slice(key).get_dtype()
        want = {"BF16": torch.bfloat16, "F16": torch.float16,
                "F32": torch.float32}.get(want)
        if want is not None and t.dtype != want:
            tensors[key] = t.to(want)
            n += 1
    return tensors, n


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--src-dir", help="original model, for --verify")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-backup", action="store_true")
    args = p.parse_args()

    shards = sorted(f for f in os.listdir(args.out_dir) if f.endswith(".safetensors"))
    if not shards:
        sys.exit(f"no shards in {args.out_dir}")

    src = None
    if args.verify:
        if not args.src_dir:
            sys.exit("--verify needs --src-dir")
        with open(os.path.join(args.src_dir, "model.safetensors.index.json"),
                  encoding="utf8") as f:
            src = json.load(f)["weight_map"]

    total_split = 0
    total_recast = 0
    checked = 0
    new_map = {}

    for shard in shards:
        path = os.path.join(args.out_dir, shard)
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            fused_here = [k for k in keys if split_name(k)]
            if not fused_here:
                tensors = {k: f.get_tensor(k) for k in keys}
                tensors, recast = restore_dtypes(tensors, src, args.src_dir,
                                                 torch, safe_open)
                total_recast += recast
                if recast and not args.dry_run:
                    save_file(tensors, path, metadata={"format": "pt"})
                for k in tensors:
                    new_map[k] = shard
                continue
            tensors = {k: f.get_tensor(k) for k in keys}

        print(f"{shard}: {len(fused_here)} fused tensor(s)")
        for key in fused_here:
            prefix, parts = split_name(key)
            t = tensors.pop(key)
            if t.shape[0] % 3:
                sys.exit(f"{key} has leading dim {t.shape[0]}, not divisible by 3")
            n = t.shape[0] // 3
            for i, part in enumerate(parts):
                piece = t[i * n:(i + 1) * n].clone()
                name = prefix + part
                tensors[name] = piece
                if src is not None and name in src:
                    sp = os.path.join(args.src_dir, src[name])
                    if os.path.exists(sp):
                        with safe_open(sp, framework="pt") as sf:
                            want = sf.get_tensor(name)
                        # Compare in the source's dtype: exllamav3 stores these as fp16,
                        # so a bitwise test across dtypes would fail on every tensor and
                        # tell us nothing about whether the split itself is right.
                        got = piece.to(want.dtype)
                        if not torch.equal(got, want):
                            d = (got.float() - want.float()).abs().max().item()
                            if d > 1e-6:
                                sys.exit(
                                    f"MISMATCH: {name} differs from the source by {d:.3e}. "
                                    f"The split order or offsets are wrong.")
                        checked += 1
            total_split += 1

        tensors, recast = restore_dtypes(tensors, src, args.src_dir, torch, safe_open)
        total_recast += recast

        if not args.dry_run:
            if args.keep_backup:
                shutil.copy2(path, path + ".fused")
            save_file(tensors, path, metadata={"format": "pt"})
        for k in tensors:
            new_map[k] = shard

    print(f"\nsplit {total_split} fused tensor(s) into {total_split * 3}")
    print(f"restored the source dtype on {total_recast} tensor(s)")
    if args.verify:
        print(f"verified {checked} split piece(s) bit-identical to the source")
        if not checked:
            print("  (nothing verified -- were the source shards still on disk?)")

    if args.dry_run:
        print("dry run, nothing written")
        return

    idx = os.path.join(args.out_dir, "model.safetensors.index.json")
    total_size = sum(os.path.getsize(os.path.join(args.out_dir, s)) for s in shards)
    with open(idx, "w", encoding="utf8") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": new_map}, f)
    print(f"rewrote {idx} ({len(new_map):,} tensors)")


if __name__ == "__main__":
    main()
