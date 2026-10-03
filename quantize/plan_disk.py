#!/usr/bin/env python3
"""
Work out how much disk the streaming conversion actually needs.

stream_shards.py holds a shard until every layer inside it is behind the converter.
That is not the same as holding it for one layer: the shards are numbered by
*lexicographic* layer name, so one shard can hold layers 4 and 40, and it has to stay
on disk from layer 4 until layer 40 is done. Guessing at the peak gets it wrong by a
wide margin, so this replays the eviction schedule against the real index and reports
what the run will actually need.

Pass --sizes to use exact shard sizes from the Hub; otherwise the index's total is
divided evenly, which is close enough for planning.

usage:
  plan_disk.py --index model.safetensors.index.json [--sizes] [--lookahead 2]
"""

import argparse
import collections
import json
import re
import urllib.request

LAYER_RE = re.compile(r"layers\.(\d+)\.")
OUT_GB = 176.0          # the converted checkpoint
WORK_GB = 176.0         # exllamav3 keeps a copy of the output in the working directory


def hub_sizes(repo):
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    with urllib.request.urlopen(url, timeout=120) as r:
        meta = json.loads(r.read())
    return {s["rfilename"]: (s.get("size") or 0) for s in meta.get("siblings", [])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--repo", default="cantina-security/apex-flash-1")
    p.add_argument("--sizes", action="store_true", help="fetch exact sizes from the Hub")
    p.add_argument("--lookahead", type=int, default=2)
    args = p.parse_args()

    with open(args.index, encoding="utf8") as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    shard_layers = collections.defaultdict(set)
    layer_shards = collections.defaultdict(set)
    for key, shard in wm.items():
        m = LAYER_RE.search(key)
        layer = int(m.group(1)) if m else -1
        shard_layers[shard].add(layer)
        layer_shards[layer].add(shard)

    if args.sizes:
        size = hub_sizes(args.repo)
    else:
        total = idx.get("metadata", {}).get("total_size", 0)
        per = total / max(len(shard_layers), 1)
        size = {s: per for s in shard_layers}

    pinned = set(layer_shards[-1])
    max_layer = max(l for l in layer_shards if l >= 0)
    naive = sum(size.get(s, 0) for s in shard_layers)

    print(f"{len(shard_layers)} shards, layers 0-{max_layer}, "
          f"{len(pinned)} pinned (no layer of their own)")
    print(f"source in full: {naive / 1e9:.1f} GB\n")
    print(f"{'lookahead':>10} {'peak source':>13} {'at layer':>9} {'total peak':>12}")

    for look in range(1, args.lookahead + 2):
        resident = set(pinned)
        peak, peak_at = 0.0, 0
        for cur in range(max_layer + 1):
            window = set(pinned)
            for L in range(cur, min(cur + look + 1, max_layer + 1)):
                window |= layer_shards.get(L, set())
            resident |= window
            for s in list(resident):
                if s in pinned or s in window:
                    continue
                if max(shard_layers[s]) < cur:
                    resident.discard(s)
            tot = sum(size.get(s, 0) for s in resident)
            if tot > peak:
                peak, peak_at = tot, cur
        print(f"{look:>10} {peak / 1e9:>12.1f}G {peak_at:>9} "
              f"{(peak / 1e9 + OUT_GB + WORK_GB):>11.0f}G")

    print(f"\nwithout streaming: {naive / 1e9:.0f}G source + {OUT_GB + WORK_GB:.0f}G "
          f"= {naive / 1e9 + OUT_GB + WORK_GB:.0f}G")
    print("A Spark's NVMe is 916 GB, so the naive layout does not fit.")


if __name__ == "__main__":
    main()
