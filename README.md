<h1 align="center">apex-flash-1 · EXL3 4-bit</h1>

<p align="center">
  <sub>built on <a href="https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold">Mia's AI Lab's GLM-5.3-Flash recipe</a></sub>
</p>

Serve [**apex-flash-1**](https://huggingface.co/cantina-security/apex-flash-1) — Cantina
Security's open-weights model for security research — quantized to EXL3 4-bit, through an
OpenAI-compatible API, with **4 concurrent requests** and a **262,144-token context**, on
two NVIDIA DGX Sparks linked by their ConnectX-7 ports. It runs
[TensorFold](https://github.com/ashhart/TensorFold) v0.6.0 on both Sparks, one rank each,
in NVIDIA's PyTorch container, with the 53 patches from the upstream recipe: DFlash2 and
copy drafts, 4-bit dense weights, an FP8 KV cache, faster prompt kernels, a one-shot RoCE
all-gather between the Sparks, several requests over one shared cache pool, tool calling,
`/tokenize` and `/metrics`.

apex-flash-1 ships as 642.7 GB of BF16 and nothing else, so this repository also contains
the conversion that produces the checkpoint it serves.

- Checkpoint: [`wambosec/apex-flash-1-EXL3-4bpw`](https://huggingface.co/wambosec/apex-flash-1-EXL3-4bpw)
  — EXL3 routed experts at 4 bits a weight, BF16 everywhere else, ~167 GB, 150,226 tensors
- Drafter: [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2),
  or the checkpoint's own MTP head (`DRAFTER`, see [Configuration](#configuration))
- API model id: `apex-flash-1`
- Context: **262,144 tokens** a request; the 4 requests share an FP8 KV pool of
  **1,288,192 tokens**
- Tool calling, structured outputs (xgrammar), `/tokenize`, and `reasoning_effort`
  `low` / `high` / `max`
- One command on the head Spark: `./start.sh` sets up both and starts both ranks;
  `./stop.sh` stops them

**This is a derivative of [Mia's AI Lab's recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).**
The TensorFold container, the two-rank launch over the CX7 link and all 53 files in
`patches/` are their work, carried over unmodified. What is new here is the EXL3
conversion in `quantize/`, the checkpoint it produces, and the defaults. See
[NOTICE](NOTICE).

## Performance

Two DGX Sparks at this recipe's defaults: a 262,144-token window, 4 streams, FP8 KV cache,
4-bit dense weights, DFlash2 plus copy drafts, vision off, GPU clocks left alone
(~2.4 GHz). Measured with `tools/bench.py` through the OpenAI API. Token counts come from
the server's counters rather than from counting stream chunks, since with drafting on one
chunk carries a whole accepted block.

**Decode** (aggregate across the concurrent requests, per request, and time to first token)

| Concurrent requests | Prose | Prose, per request | TTFT | Structured | Structured, per request | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.6 tok/s | 54.6 tok/s | 180 ms | 58.7 tok/s | 58.7 tok/s | 210 ms |
| 2 | 79.1 tok/s | 39.5 tok/s | 307 ms | 84.1 tok/s | 42.0 tok/s | 341 ms |
| 3 | 112.7 tok/s | 37.6 tok/s | 386 ms | 157.1 tok/s | 52.4 tok/s | 290 ms |
| 4 | 177.9 tok/s | 44.5 tok/s | 275 ms | 198.0 tok/s | 49.5 tok/s | 309 ms |

Structured decoding overtakes prose from three streams up: a JSON schema makes the next
token easier to draft, so more of each block is accepted. Drafting carries 2.1 to 2.5
tokens a round.

**Prefill** (prompt sizes as the server's `/tokenize` reports them; each run uses a fresh
prefix, since repeating a prompt measures the prompt cache instead)

| Prompt | Prefill | Time to first token |
| ---: | ---: | ---: |
| 6,161 tokens | 1,889.3 tok/s | 3.26 s |
| 24,593 tokens | 2,495.9 tok/s | 9.85 s |
| 98,321 tokens | 2,324.9 tok/s | 42.29 s |
| 196,625 tokens | 3,027.0 tok/s | 64.96 s |

**Prompt reuse** (the server resumes from a kept prompt state instead of prefilling)

| Prompt | First time | Next time |
| --- | ---: | ---: |
| An identical 64k-token prompt, sent again | 12.38 s | 0.19 s |

**Memory**: ~87.6 GiB on each Spark at startup, leaving 18 to 19 GiB free per Spark.

**Quality**: not benchmarked. See
[Choosing the bitrate](quantize/README.md#choosing-the-bitrate) if you are weighing 4 bits
against more.

For reference, the upstream recipe reports 60.4 tok/s prose at one stream and 108.8 at
four for 4 bpw base GLM-5.3-Flash on the same engine and hardware. That is a different
model at a 1,048,576-token window with vision on, so the two tables are not a controlled
comparison in either direction.

## Requirements

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory), with nothing else
  large on their GPUs: each needs ~95 GiB free when the server starts.
- **A direct ConnectX-7 link:** a QSFP cable between the CX7 ports and an IPv4 address on
  each end in one private subnet, with a RoCE v2 GID (`start.sh` checks). One QSFP port
  reaches the GB10 over two PCIe Gen5 x4 links, so it appears as two netdevs and two RoCE
  devices, and the two twins need **different** subnets.
- **Key-based ssh** from the head Spark to the worker:
  `ssh -o BatchMode=yes user@<worker> true` must succeed.
- Docker with the NVIDIA container runtime, your user in the `docker` group, and `rsync`,
  on both Sparks.
- **Disk, on each Spark:** ~195 GB — ~167 GB for the checkpoint and ~2.3 GB for DFlash2
  under `~/.cache/huggingface`, ~25 GB for the image under Docker's root.

`prepare.sh` downloads the checkpoint on the first run. Building it yourself instead needs
~500 GB free on one Spark and about 17 hours; see [quantize/](quantize/).

## Quick start

On the head Spark:

```bash
git clone https://github.com/WamboDNS/apex-flash-1-EXL3-4bpw.git
cd apex-flash-1-EXL3-4bpw
cp scripts/local.sh.example scripts/local.sh     # set WORKER=user@<worker address>
./start.sh
```

`WORKER` is the worker's ssh target. The ranks talk over the route to it; if that is not
the CX7 link, set `FABRIC_PEER` to the worker's CX7 address. The worker needs no copy of
this repository.

The first run sets up both Sparks: the image (~25 GB) on each, the checkpoint copied to
the worker over the link, then the CUDA kernels compile once per image. Later starts take
3 to 5 minutes to load ~88 GiB of weights on each Spark. `start.sh` shows each step, runs
a smoke test through both ranks, and prints the endpoint.

Any OpenAI client works with `base_url = "http://<head-address>:8888/v1"` and the model
`apex-flash-1`. The model thinks before it answers (`reasoning_content`), so give replies
enough `max_tokens`.

```bash
curl -s http://<head-address>:8888/v1/models
curl -s http://<head-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "apex-flash-1",
  "messages": [{"role": "user", "content": "Explain this stack trace."}],
  "max_tokens": 2000
}'

./start.sh restart                              # restart both ranks after a setting change
./stop.sh                                       # stop both ranks and free their memory
docker logs -f apex-flash-tf                    # rank 0's log
ssh <worker> docker logs -f apex-flash-tf       # rank 1's log
curl -s http://<head-address>:8888/health       # streams, pool tokens, draft acceptance
```

## Configuration

Settings are read from the environment, then `scripts/local.sh`, then `.env`, then the
defaults in `scripts/config.sh`. The ones that differ from the upstream recipe:

| | upstream | here | why |
| --- | ---: | ---: | --- |
| `CONTEXT` | 1048576 | 262144 | 256k covers agent and code work; the rest is memory back |
| `MEMORY_RESERVE_GIB` | 14.5 | 24 | leaves room on each Spark for other GPU work |
| `VISION` | 1 | 0 | the model card reports no image or video evaluation |

Everything else follows upstream: `PARALLEL` 4, `KV` fp8, `DENSE` q4, `DRAFTER` dflash2,
`THINKING` 1, `MAX_TOKENS` 32768, `PORT` 8888.

Three profiles ship in `profiles/`:

```bash
. profiles/headroom.sh    && ./start.sh restart   # the default
. profiles/max-context.sh && ./start.sh restart   # full 1M window, vision on, no headroom
. profiles/commercial.sh  && ./start.sh restart   # no DFlash2, one request at a time
```

### Drafter and licensing

`DRAFTER=dflash2` is the default and is what the concurrency numbers need: the batched
verify window handles several streams only when DFlash2 drafts them, so `DRAFTER=mtp`
drops `PARALLEL` to 1.

**DFlash2 is CC BY-NC-ND 4.0 — non-commercial use only.** apex-flash-1 is MIT and
TensorFold is Apache-2.0, so DFlash2 is the only part of the stack that restricts
commercial use. `profiles/commercial.sh` drops it and drafts with the checkpoint's own MTP
head instead.

### Memory and the KV pool

TensorFold sizes the shared KV pool from the memory free at start, minus
`MEMORY_RESERVE_GIB`, capped at `KV_POOL_GIB`. At the defaults that comes out at
1,288,192 pooled tokens across 4 streams, with the server itself using ~87.6 GiB a Spark.
Raise `MEMORY_RESERVE_GIB` to leave more for other work, lower it for a larger pool.

## The checkpoint

apex-flash-1 is a GRPO post-train of GLM-5.3-Flash. The two configs differ only in the
base's FP8 `quantization_config` block, and the tensor sets are isomorphic — both have
38,770 tensors under the same names — so the serving stack needs no changes and the
patches carry over unmodified.

The conversion quantizes what the published GLM-5.3-Flash EXL3 checkpoint quantizes and
nothing else: the **routed experts** (`mlp.experts.N.gate_proj` / `up_proj` / `down_proj`)
at 4 bits with the `mcg` codebook, 37,152 modules in all. Attention, the shared expert,
the router gate, the dense MLPs of the first three layers, the KDA projections, the DSA
indexer, `lm_head`, the embedding and the vision tower stay BF16.

`quantize/` builds it in about 17 hours on one Spark, streaming the 642.7 GB source so the
job fits on a 916 GB disk, and verifies the result against the published checkpoint's
layout — tensor names and dtypes — before calling it done. See [quantize/](quantize/).

## Checks

```bash
tools/bench.py --base-url http://<head>:8888/v1 --model apex-flash-1
tools/toolcheck.py          # tool calling and structured output
tools/needle.py             # long-context retrieval
tools/end_of_turn.py        # end-of-turn detection
```

TensorFold describes its EXL3 support as experimental; replies are exact, but the MLX path
it recommends instead is not available on NVIDIA hardware.

## Repository layout

```
quantize/        EXL3 conversion: recipe, streaming fetch, verification, analysis
patches/         53 TensorFold patches, unmodified from the upstream recipe
scripts/         config, prepare, node and link helpers
profiles/        headroom (default), max-context, commercial
tools/           benchmark and validation clients
start.sh stop.sh two-rank launch and shutdown
```

## License

Apache-2.0, inherited from the upstream recipe. See [LICENSE](LICENSE) and
[NOTICE](NOTICE), which records what this project changed and the third-party terms that
still apply — in particular DFlash2's non-commercial restriction.

apex-flash-1 is MIT, and the EXL3 checkpoint carries the same.

## Credits

[Mia's AI Lab](https://github.com/MiaAI-Lab) wrote the recipe this is built on, including
every one of the 53 patches that make a GLM-5.3-Flash-family model run across two Sparks.
[Cantina Security](https://www.cantina.security/apex-flash) and Yeta Labs trained and
released apex-flash-1. [turboderp](https://github.com/turboderp-org/exllamav3) wrote
ExLlamaV3. [TensorFold](https://github.com/ashhart/TensorFold) is the serving engine.
