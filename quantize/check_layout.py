#!/usr/bin/env python3
"""
Check that a converted checkpoint has the layout the serving patches expect.

TensorFold reads this checkpoint through the EXL3 storage ABI that the GLM-5.3-Flash
EXL3 checkpoint was published with: every quantized Linear stored as four tensors
(trellis, suh, svh, mcg) under its weight name minus the ".weight", and everything else
stored as a plain tensor under its original name. If our converter names things even
slightly differently, the server fails to load weights it cannot find.

Rather than trust that, this compares our output's tensor names against the published
GLM-5.3-Flash checkpoint's index. The two models have isomorphic tensor sets -- same
38,770 names -- so for every layer we converted, our names should match theirs exactly.

A partial conversion (--max_module) only produces some layers, so by default only the
layers present in our output are compared, and the rest are reported as not yet built.

usage:
  check_layout.py --out-dir DIR
  check_layout.py --out-dir DIR --reference-index mia-index.json
"""

import argparse
import collections
import json
import os
import re
import sys
import urllib.request

REFERENCE = ("https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw"
             "/resolve/main/model.safetensors.index.json")
EXL3_SUFFIXES = {"trellis", "suh", "svh", "mcg"}
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def load_index(path_or_url, cache=None):
    if os.path.exists(path_or_url):
        with open(path_or_url, encoding="utf8") as f:
            return json.load(f)["weight_map"]
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf8") as f:
            return json.load(f)["weight_map"]
    print(f"fetching reference index from {path_or_url}")
    with urllib.request.urlopen(path_or_url, timeout=300) as r:
        data = r.read()
    if cache:
        with open(cache, "wb") as f:
            f.write(data)
    return json.loads(data)["weight_map"]


def check_dtypes(args, ours):
    """Compare stored dtypes against the reference.

    Names matching is not enough. exllamav3 stores some weights wider than the source
    does -- float32 where the published checkpoint has bfloat16 -- and the serving kernels
    check dtype explicitly, so a checkpoint can pass a name-only comparison and then die
    inside a layer with something like "hc_partial: fn [24, 16384] bf16". This caught 615
    such tensors once.
    """
    import glob as _glob
    import struct as _struct

    import gzip
    opener = gzip.open if args.reference_dtypes.endswith(".gz") else open
    with opener(args.reference_dtypes, "rt", encoding="utf8") as f:
        ref = json.load(f)

    mismatched = collections.Counter()
    total = 0
    for path in sorted(_glob.glob(os.path.join(args.out_dir, "*.safetensors"))):
        with open(path, "rb") as fh:
            n = _struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            want = ref.get(key)
            if want and want != meta["dtype"]:
                mismatched[(key.rsplit(".", 1)[-1], meta["dtype"], want)] += 1
                total += 1

    if not total:
        print("\n   dtypes match the reference")
        return 0
    print(f"\n!! {total:,} tensor(s) have a different dtype than the reference:")
    for (suffix, got, want), n in mismatched.most_common(8):
        print(f"     {n:5d}x  *.{suffix:24s} {got} -> expected {want}")
    print("   Run fix_dtypes.py. The serving kernels check dtype, so this loads and then")
    print("   fails inside a layer.")
    return total


def scan_shards(d):
    """Tensor names straight out of the .safetensors files, for a partial conversion."""
    try:
        from safetensors import safe_open
    except ImportError:
        return set()
    names = set()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(d, fn), framework="pt") as f:
            names.update(f.keys())
    return names


def layers_of(names):
    out = set()
    for n in names:
        m = LAYER_RE.search(n)
        if m:
            out.add(int(m.group(1)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--reference-index", default=REFERENCE)
    p.add_argument("--cache", default="/tmp/reference-index.json")
    p.add_argument("--reference-dtypes",
                   help="dtype map from fix_dtypes.py --dump-reference; enables dtype checking")
    p.add_argument("--full", action="store_true",
                   help="require every layer, not just the ones converted so far")
    args = p.parse_args()

    idx = os.path.join(args.out_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf8") as f:
            ours = set(json.load(f)["weight_map"])
    else:
        # A partial run (--max_module) writes shards but no index until it finishes, and
        # the fused-module question is worth answering before committing to a full run.
        ours = scan_shards(args.out_dir)
        if not ours:
            sys.exit(f"no index and no shards in {args.out_dir} -- nothing written yet")
        print(f"(no index yet; read {len(ours):,} names straight from the shards)\n")
    ref = set(load_index(args.reference_index, args.cache))

    print(f"ours      : {len(ours):,} tensors")
    print(f"reference : {len(ref):,} tensors")

    suf = collections.Counter(n.rsplit(".", 1)[-1] for n in ours)
    print(f"\nsuffixes in ours: {dict(suf.most_common(8))}")

    done = layers_of(ours)
    # Only demand the EXL3 suffixes where the reference actually has them. The first
    # three layers are dense and carry no quantized tensors at all, so a partial run that
    # has not reached a sparse layer yet legitimately has none.
    ref_in_scope = {n for n in ref
                    if not LAYER_RE.search(n)
                    or int(LAYER_RE.search(n).group(1)) in done}
    ref_suf = set(n.rsplit(".", 1)[-1] for n in ref_in_scope) & EXL3_SUFFIXES
    got = set(suf) & EXL3_SUFFIXES
    if ref_suf and got != ref_suf:
        print(f"\n!! FAIL: the reference stores {sorted(ref_suf)} in these layers, "
              f"we wrote {sorted(got) or 'none'}")
        print("   The codebook or storage format is wrong. Was -cb mcg passed?")
        return 1
    if not ref_suf:
        print("   (no quantized tensors expected in the layers converted so far)")
    scope = ref if args.full else ref_in_scope
    print(f"\nlayers converted: {len(done)} ({min(done)}-{max(done)})" if done else "")
    if not args.full:
        print(f"comparing against the {len(scope):,} reference tensors in those layers")

    extra = ours - ref
    missing = scope - ours

    dtype_bad = check_dtypes(args, ours) if args.reference_dtypes else 0
    nonlayer_pre = {n for n in missing if not LAYER_RE.search(n)}
    if not extra and not dtype_bad and not (missing - (set() if args.full else nonlayer_pre)):
        print("\n   PASS: every tensor name matches the published checkpoint.")
        print("   The serving patches will find what they expect.")
        return 0

    # On a partial conversion the output holds only per-layer shards, so the embedding,
    # lm_head, the final norm and the vision tower are legitimately absent. Separate those
    # from tensors that are genuinely missing from a layer we did convert.
    nonlayer = {n for n in missing if not LAYER_RE.search(n)}
    real_missing = missing - nonlayer
    if nonlayer and not args.full:
        print(f"\n   {len(nonlayer):,} non-layer tensor(s) absent (embedding, lm_head, "
              f"norm, vision tower)")
        print("   -- expected for a partial run; re-check with --full once it finishes")
    if real_missing:
        print(f"\n!! {len(real_missing):,} tensor(s) the reference has and we do not:")
        for n in sorted(real_missing)[:15]:
            print(f"     {n}")
    if extra:
        print(f"\n!! {len(extra):,} tensor(s) we have and the reference does not:")
        for n in sorted(extra)[:15]:
            print(f"     {n}")
        fused = [n for n in extra if "qkv" in n or "gate_up" in n]
        if fused:
            print(f"\n   {len(fused)} look like fused modules (e.g. {fused[0]}).")
            print("   exllamav3 fuses q/k/v internally; if it also writes them fused,")
            print("   the serving side needs them split back out.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
