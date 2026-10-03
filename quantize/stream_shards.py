#!/usr/bin/env python3
"""
Fetch and drop source shards around a chunked conversion.

apex-flash-1 is 642.7 GB of BF16 across 62 shards. Keeping the whole source on disk
alongside the working directory and the output needs roughly 1 TB, which a Spark's 916 GB
NVMe does not have. But exllamav3 walks the model in layout order and never looks back,
so only the shards for the layers it is working on have to be present.

The obvious implementation -- download shards in the background while one long conversion
runs -- does not work, and it is worth saying why, because it looks like it should.
exllamav3 globs the source directory once, when it builds its SafetensorsCollection
(loader/safetensors.py, `add_tensor_files`). Shards that appear afterwards are never
indexed, so the run dies with

    ValueError: Required tensor model.language_model.layers.5.hc_attn_fn
                not found in any *.safetensors file in .../src

even though the file is sitting right there, downloaded twenty minutes earlier.

What does work: convert in chunks. `--max_module` is read fresh from the command line on
every invocation, including resumes (convert_model.py assigns it after the override
table), and every invocation re-globs the source directory. So the driver alternates
between fetching the next chunk's shards and running the converter over them, and each
run sees the files that arrived since the last one.

This script is the fetch-and-drop half of that, driven by run.sh:

    stream_shards.py ensure --in-dir DIR --layers 6-11      # before a chunk
    stream_shards.py evict  --in-dir DIR --before 6         # after one

Shards holding tensors that belong to no layer -- the embedding, the final norm,
lm_head, the vision tower -- are read at both ends of the run and never evicted.

One wrinkle if you adapt this: the shards are numbered by *lexicographic* layer name, so
`layers.10` sorts before `layers.2` and a single shard can hold layers 4 and 40. Nothing
here keys off shard numbers; everything is driven off the index.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import threading
import time

LAYER_RE = re.compile(r"layers\.(\d+)\.")
HF = "https://huggingface.co/{repo}/resolve/{rev}/{f}"


def log(msg):
    print(f"[stream] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_maps(in_dir):
    path = os.path.join(in_dir, "model.safetensors.index.json")
    with open(path, encoding="utf8") as f:
        wm = json.load(f)["weight_map"]
    shard_layers = collections.defaultdict(set)
    layer_shards = collections.defaultdict(set)
    for key, shard in wm.items():
        m = LAYER_RE.search(key)
        layer = int(m.group(1)) if m else -1
        shard_layers[shard].add(layer)
        layer_shards[layer].add(shard)
    return shard_layers, layer_shards


class Fetcher:
    def __init__(self, in_dir, repo, rev, parallel, attempts=12):
        self.in_dir, self.repo, self.rev, self.parallel = in_dir, repo, rev, parallel
        self.attempts = attempts
        self.lock = threading.Lock()
        self.failed = []

    def have(self, shard):
        p = os.path.join(self.in_dir, shard)
        return os.path.exists(p) and os.path.getsize(p) > 0

    def fetch(self, shard):
        tmp = os.path.join(self.in_dir, f".{shard}.part")
        url = HF.format(repo=self.repo, rev=self.rev, f=shard)
        for attempt in range(1, self.attempts + 1):
            r = subprocess.run(
                # --http1.1 on purpose. Over HTTP/2 these 10 GB downloads fail with
                # "curl: (92) stream was not closed cleanly: CANCEL (err 8)" often enough
                # that a shard can lose several attempts in a row; HTTP/1.1 does not
                # multiplex and does not hit it. -C - resumes the partial file, so a
                # retry continues rather than starting over.
                ["curl", "-fsSL", "--http1.1", "--retry", "5", "--retry-delay", "5",
                 "--connect-timeout", "30", "-C", "-", "-o", tmp, url],
                capture_output=True,
            )
            if r.returncode == 0:
                os.replace(tmp, os.path.join(self.in_dir, shard))
                log(f"fetched {shard}")
                return
            got = os.path.getsize(tmp) / 1e9 if os.path.exists(tmp) else 0
            log(f"fetch failed {shard} (attempt {attempt}/{self.attempts}, "
                f"{got:.1f} GB so far): {r.stderr.decode().strip()[:140]}")
            time.sleep(min(15 * attempt, 120))
        with self.lock:
            self.failed.append(shard)

    def ensure(self, shards):
        todo = [s for s in sorted(shards) if not self.have(s)]
        if not todo:
            log(f"all {len(shards)} shard(s) already present")
            return
        log(f"fetching {len(todo)} shard(s)")
        while todo:
            batch, todo = todo[: self.parallel], todo[self.parallel:]
            threads = [threading.Thread(target=self.fetch, args=(s,)) for s in batch]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        if self.failed:
            sys.exit(f"could not fetch: {self.failed}")


def parse_range(text):
    if "-" in text:
        a, b = text.split("-", 1)
        return int(a), int(b)
    return int(text), int(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["ensure", "evict", "plan"])
    p.add_argument("--in-dir", required=True)
    p.add_argument("--layers", help="ensure: inclusive layer range, e.g. 6-11")
    p.add_argument("--before", type=int, help="evict: drop shards entirely below this layer")
    p.add_argument("--repo", default="cantina-security/apex-flash-1")
    p.add_argument("--revision", default="main")
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--attempts", type=int, default=12,
                   help="per-shard download attempts before giving up")
    args = p.parse_args()

    shard_layers, layer_shards = load_maps(args.in_dir)
    pinned = set(layer_shards[-1])
    max_layer = max(k for k in layer_shards if k >= 0)

    if args.mode == "plan":
        print(f"{len(shard_layers)} shards, layers 0-{max_layer}, "
              f"{len(pinned)} pinned: {sorted(pinned)}")
        return

    if args.mode == "ensure":
        if not args.layers:
            sys.exit("ensure needs --layers")
        lo, hi = parse_range(args.layers)
        want = set(pinned)
        for L in range(lo, min(hi, max_layer) + 1):
            want |= layer_shards.get(L, set())
        log(f"layers {lo}-{hi}: {len(want)} shard(s) needed")
        Fetcher(args.in_dir, args.repo, args.revision, args.parallel,
                args.attempts).ensure(want)
        log("ready")
        return

    # evict
    if args.before is None:
        sys.exit("evict needs --before")
    freed = 0
    for shard in sorted(shard_layers):
        if shard in pinned:
            continue
        path = os.path.join(args.in_dir, shard)
        if not os.path.exists(path):
            continue
        # Only drop a shard once every layer inside it is behind the converter. A shard
        # can hold two layers that are far apart, so this is not the same as "one layer
        # back" -- some shards stay resident much longer than others.
        if max(shard_layers[shard]) < args.before:
            size = os.path.getsize(path)
            os.remove(path)
            freed += size
            log(f"evicted {shard} ({size / 1e9:.1f} GB, layers {sorted(shard_layers[shard])})")
    log(f"freed {freed / 1e9:.1f} GB")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
