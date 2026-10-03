# Building the EXL3 checkpoint

`wambosec/apex-flash-1-EXL3-4bpw` is published, so you only need this if you want to
reproduce it, change the bitrate, or convert a different post-train of GLM-5.3-Flash.

```bash
tmux new -s quant
ROOT=$HOME/apex-quant ./run.sh
```

Expect a long GPU run. It resumes: re-run with the same `ROOT` and it continues from the
last checkpoint in `work/`.

## What you need

- One GB10 / Blackwell box with enough memory to hold a single layer (a DGX Spark).
- **About 500 GB free.** Run `plan_disk.py --index <index> --sizes` for the exact schedule
  on your model; for apex-flash-1 it peaks near 146 GB of streamed source plus ~176 GB of
  working directory plus ~176 GB of output.
- Docker with GPU access and the NVIDIA PyTorch container.

## The pieces

| | |
| --- | --- |
| `run.sh` | drives everything: image, recipe, streaming, conversion |
| `make_recipe.py` | decides which tensors get quantized and at what bitrate |
| `stream_shards.py` | fetches source shards ahead of the converter, deletes them behind it |
| `plan_disk.py` | replays the eviction schedule so you know the real peak before starting |
| `check_layout.py` | verifies the output's tensor names against the published GLM checkpoint |
| `weight_delta.py` | measures how far the post-train moved each expert |
| `survives_quant.py` | checks whether quantization preserves that movement |
| `restore_mtp_bf16.py` | puts the MTP head's non-expert weights back in BF16 |
| `declare_scope.py` | records the quantization scope the server checks for |
| `dedupe_shards.py` | makes the shards contain exactly what the index says |
| `install_local.sh` | makes a locally built checkpoint servable without uploading it |

## The conversion

`run.sh` converts in chunks rather than one long run, and that is not an optimisation --
it is the only thing that works. exllamav3 globs the source directory once, when it
builds its `SafetensorsCollection`, so shards downloaded after the converter starts are
never indexed:

```
ValueError: Required tensor model.language_model.layers.5.hc_attn_fn
            not found in any *.safetensors file in .../src
```

with the file sitting in that directory, downloaded twenty minutes earlier. A background
downloader feeding one long conversion cannot work.

What does: `--max_module` is read straight from the command line on every invocation,
including resumes, and every invocation re-globs the directory. So the driver fetches a
chunk's shards, converts that far, drops what is behind, and resumes:

```
stream_shards.py ensure --layers 6-11
convert.py --in_dir src --work_dir work --out_dir out \
  --recipe recipe.yaml --bits 4 --head_bits 16 --mtp_bits 4 --vision_bits 16 \
  --codebook mcg --shard_size 8192 --checkpoint_interval 120 \
  --max_module 13 --resume
stream_shards.py evict --before 12
```

Module index is layer + 2, since the embedding and `hc_expand` come first. `CHUNK`
controls how many layers go in each pass (default 6); each pass costs one model reload,
so smaller chunks mean less source on disk and slightly more overhead.

Three of those flags are not defaults and matter:

- **`-cb mcg`.** Current exllamav3 defaults to the `mul1` codebook. The published
  GLM-5.3-Flash EXL3 checkpoint — the one the serving patches were written against — uses
  `mcg`. Getting this wrong produces a checkpoint that converts cleanly and then fails to
  serve.
- **`-mb 4`.** The MTP head's 873 Linears are allocated through `mtp_bits`, not the
  recipe, so `make_recipe.py` cannot reach its 864 routed experts. At 4 they match the
  rest; at 16 they add ~10.9 GB for a head that only `DRAFTER=mtp` loads. The cost is the
  head's handful of non-expert Linears, which the published checkpoint keeps in BF16.
- **`-hb 16` / `-vb 16`.** `lm_head` and the vision tower stay BF16, as upstream.

## Serving a checkpoint you built yourself

`start.sh` addresses the checkpoint by Hugging Face repo id and looks for it in the local
Hub cache, so a freshly converted directory has nowhere to go until it has been pushed --
an awkward requirement when you only want to benchmark it, and a 176 GB upload besides.
`install_local.sh` lays the output into the cache in the shape the serving scripts expect,
hard-linking rather than copying so it costs no extra disk:

```bash
quantize/install_local.sh ~/apex-quant/out
MODEL_ID=wambosec/apex-flash-1-EXL3-4bpw MODEL_REVISION=local ./start.sh
```

## Verifying the result

```bash
check_layout.py --out-dir out --full
```

This is the check that matters. It diffs every tensor name in the output against the
published GLM-5.3-Flash EXL3 checkpoint's index. The two models have isomorphic tensor
sets, so the names should match exactly; if they do, the serving patches will find what
they expect. A mismatch here is the difference between a checkpoint that loads and one
that does not, and it costs seconds to check against hours to discover at serving time.

Watch in particular for fused modules. exllamav3 represents q/k/v as a single internal
`qkv_proj`; the serving side needs them under their original separate names.

## Choosing the bitrate

apex-flash-1 was trained with rank-256 LoRA over all experts and routers plus
full-parameter updates to 16 experts. That makes the usual uniform 4 bpw worth
questioning: the routed experts sit only ~8e-5 away from the base model in relative
Frobenius terms, which is well inside the error 4-bit quantization introduces.

```bash
weight_delta.py   --base <GLM-5.3-Flash-BF16> --apex <apex-flash-1> --layer 3
survives_quant.py --base-out <converted base> --apex-out <converted apex> --layer 3
```

`weight_delta.py` includes a control: attention was frozen during the post-train, so
those tensors must be bit-identical between the two checkpoints. If they are not, the
comparison is measuring something other than the fine-tune.

Be careful reading these. A weight-space norm cannot tell you whether the coherent,
low-rank part of the delta still does its job — the post-train's direction can survive
noise much larger than itself. The numbers narrow the choice; only running both quantized
models on real tasks settles it.
