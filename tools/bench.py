#!/usr/bin/env python3
"""
Measure decode, prefill and prompt reuse through the OpenAI API.

Reports the same shapes as the upstream GLM-5.3-Flash recipe so the two tables can be
read side by side: aggregate and per-request decode at 1-4 concurrent requests with time
to first token, prefill at a range of prompt sizes, and the cold/warm gap on a repeated
prompt.

Decode rate is measured from the first token onward, so time to first token is not folded
into it. Streaming is required; the server must be reachable from wherever this runs.

  tools/bench.py --base-url http://<head>:8888/v1 --model apex-flash-1
  tools/bench.py --base-url ... --only decode --concurrency 1,4
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

PROSE = ("Write a detailed essay about the history of cryptography, from classical "
         "ciphers through to post-quantum key exchange. Use several paragraphs.")
STRUCTURED = ("List 25 well-known CVEs as a JSON array. Each element must be an object "
              'with keys "id", "year", "component" and "impact". Reply with JSON only.')

# The upstream recipe's "structured" column is constrained decoding, not a prompt asking
# nicely for JSON. Without response_format the server free-runs and the number measures
# something else entirely, so send the schema and make xgrammar do the work.
STRUCTURED_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "cves",
        "schema": {
            "type": "object",
            "properties": {
                "cves": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "year": {"type": "integer"},
                            "component": {"type": "string"},
                            "impact": {"type": "string"},
                        },
                        "required": ["id", "year", "component", "impact"],
                    },
                }
            },
            "required": ["cves"],
        },
    },
}


def health(base_url):
    """Server-side counters, or None."""
    url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())
    except Exception:                                              # noqa: BLE001
        return None


def post_stream(base_url, model, messages, max_tokens, extra=None, timeout=1800):
    """Returns (ttft_seconds, total_seconds, n_chunks).

    The third value counts SSE chunks, NOT tokens. With speculative decoding the server
    emits a whole accepted draft block in one chunk, so chunks undercount tokens by
    exactly the drafting speedup -- measured here as 21.9 "tok/s" against the server's own
    46.8, a factor of 2.12 that happened to equal tokens-per-round. Token counts come from
    /health instead; this is kept only for time-to-first-token.
    """
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True}
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - start
                n += 1
    total = time.perf_counter() - start
    if ttft is None:
        raise RuntimeError("no tokens returned")
    return ttft, total, n


def decode(args):
    print("\n## Decode\n")
    print("| Concurrent | Prose | Prose, per request | TTFT | Structured | "
          "Structured, per request | TTFT |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in args.concurrency:
        row = [str(c)]
        for prompt, fmt in ((PROSE, None), (STRUCTURED, STRUCTURED_FORMAT)):
            results = []
            errors = []

            def one(prompt=prompt, fmt=fmt):
                try:
                    results.append(post_stream(
                        args.base_url, args.model,
                        [{"role": "user", "content": prompt}], args.max_tokens,
                        extra={"response_format": fmt} if fmt else None))
                except Exception as e:                      # noqa: BLE001
                    errors.append(e)

            threads = [threading.Thread(target=one) for _ in range(c)]
            before = health(args.base_url)
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0
            after = health(args.base_url)
            if errors or not results:
                print(f"  !! {len(errors)} request(s) failed: {errors[:1]}", file=sys.stderr)
                row += ["err", "err", "err"]
                continue

            ttfts = [t for t, _, _ in results]
            decode_window = wall - min(ttfts)
            if before and after:
                toks = after["completion_tokens_total"] - before["completion_tokens_total"]
            else:
                # No counters: fall back to chunks and say so, rather than quietly
                # reporting a number that is too low by the drafting speedup.
                toks = sum(n for _, _, n in results)
                print("  !! /health unavailable; counting SSE chunks, which undercounts "
                      "tokens when drafting is on", file=sys.stderr)
            row += [f"{toks / decode_window:.1f} tok/s",
                    f"{toks / c / decode_window:.1f} tok/s",
                    f"{statistics.mean(ttfts) * 1000:.0f} ms"]
        print("| " + " | ".join(row) + " |")


def filler(tokens, nonce=None):
    """Rough sizing only -- token_count() reports what the server actually saw.

    The nonce matters. The server keeps prompt states and reuses them, so repeating the
    same filler across runs measures the cache, not prefill: a 6,154-token prompt came
    back in 0.14 s at a nonsensical 42,538 tok/s on the second run. A unique prefix per
    run forces real work.
    """
    head = f"session {nonce} " if nonce else ""
    return head + " ".join(["security"] * int(tokens * 0.75))


def token_count(base_url, text):
    """Ask the server how many tokens a prompt is, via /tokenize.

    Reporting a requested size rather than the real one would put made-up token counts
    in the results table, and prefill rate is tokens divided by time -- so a wrong count
    is a wrong throughput, not just a wrong label.
    """
    req = urllib.request.Request(
        base_url.rstrip("/").removesuffix("/v1") + "/tokenize",
        # The endpoint takes "prompt"; "text" is rejected with a 400 and the fallback
        # below then reports the requested size instead of the real one -- which silently
        # turns every prefill rate into a wrong number rather than a missing one.
        data=json.dumps({"prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read())
    except Exception:                                              # noqa: BLE001
        return None
    for key in ("count", "n_tokens", "num_tokens"):
        if isinstance(body.get(key), int):
            return body[key]
    toks = body.get("tokens")
    return len(toks) if isinstance(toks, list) else None


def prefill(args):
    print("\n## Prefill\n")
    print("| Prompt | Prefill | Time to first token |")
    print("| ---: | ---: | ---: |")
    nonce = args.nonce
    for size in args.prefill_sizes:
        text = filler(size, nonce) + "\n\nSummarise the above in one line."
        msg = [{"role": "user", "content": text}]
        actual = token_count(args.base_url, text)
        ttft, _, _ = post_stream(args.base_url, args.model, msg, 16)
        n = actual if actual else size
        note = "" if actual else " (requested size; /tokenize unavailable)"
        print(f"| {n:,} tokens{note} | {n / ttft:,.1f} tok/s | {ttft:.2f} s |")


def reuse(args):
    print("\n## Prompt reuse\n")
    print("| Prompt | First time | Next time |")
    print("| --- | ---: | ---: |")
    size = args.reuse_size
    msg = [{"role": "user", "content": filler(size, args.nonce)
            + "\n\nReply with the single word OK."}]
    first, _, _ = post_stream(args.base_url, args.model, msg, 8)
    second, _, _ = post_stream(args.base_url, args.model, msg, 8)
    print(f"| An identical {size // 1000}k-token prompt, sent again "
          f"| {first:.2f} s | {second:.2f} s |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="apex-flash-1")
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--concurrency", default="1,2,3,4")
    p.add_argument("--prefill-sizes", default="8192,32768,131072,262144")
    p.add_argument("--reuse-size", type=int, default=64000)
    p.add_argument("--only", default="decode,prefill,reuse")
    p.add_argument("--nonce", default=None,
                   help="unique prefix for prefill prompts; defaults to the clock, so "
                        "repeat runs do not measure the server's prompt cache")
    p.add_argument("--warmup", type=int, default=1,
                   help="throwaway requests before measuring; the first request after a "
                        "start pays for CUDA graph capture and would skew the first row")
    args = p.parse_args()
    args.concurrency = [int(x) for x in args.concurrency.split(",")]
    args.prefill_sizes = [int(x) for x in args.prefill_sizes.split(",")]

    if args.nonce is None:
        args.nonce = f"{time.time():.0f}"
    which = set(args.only.split(","))

    for i in range(args.warmup):
        try:
            post_stream(args.base_url, args.model,
                        [{"role": "user", "content": "Say OK."}], 8)
        except Exception as e:                                     # noqa: BLE001
            sys.exit(f"warmup request failed: {e}")

    print(f"# {args.model}")
    print(f"\nmeasured {time.strftime('%Y-%m-%d %H:%M')} against {args.base_url}")
    if "decode" in which:
        decode(args)
    if "prefill" in which:
        prefill(args)
    if "reuse" in which:
        reuse(args)


if __name__ == "__main__":
    main()
