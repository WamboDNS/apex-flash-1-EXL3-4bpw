#!/usr/bin/env python3
"""
Retype whole bitrates in a resumed job's args.json.

convert.py stores the recipe strategy in work/args.json when the job starts, and a
--resume replays those stored values rather than re-reading the recipe. So a job that
began with float bitrates keeps using them even after the recipe is corrected, and keeps
failing in the same place. This rewrites them as ints.

Note the trap this walks around: in Python `int(4.0) == 4.0` is True, so a "did the value
change" check never fires. Compare types, not values.

usage: fix_job_args.py <work_dir>/args.json
"""

import collections
import json
import sys


def norm(v):
    return int(v) if isinstance(v, float) and v.is_integer() else v


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip().splitlines()[-1])
    path = sys.argv[1]
    with open(path, encoding="utf8") as f:
        d = json.load(f)

    changed = 0
    for key in ("bits", "head_bits", "mtp_bits", "vision_bits", "final_bits"):
        if key in d:
            nv = norm(d[key])
            changed += type(nv) is not type(d[key])
            d[key] = nv

    strategy = d.get("recipe_strategy") or {}
    for key, v in list(strategy.items()):
        nv = norm(v)
        changed += type(nv) is not type(v)
        strategy[key] = nv

    if not changed:
        print("already integral, nothing to do")
        return

    with open(path, "w", encoding="utf8") as f:
        json.dump(d, f)
    print(f"retyped {changed} bitrate(s) in {path}")
    if strategy:
        print("  strategy values:",
              dict(collections.Counter(repr(v) for v in strategy.values())))


if __name__ == "__main__":
    main()
