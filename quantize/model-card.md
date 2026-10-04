---
base_model: cantina-security/apex-flash-1
base_model_relation: quantized
license: mit
language:
  - en
  - zh
pipeline_tag: image-text-to-text
library_name: exllamav3
tags:
  - exl3
  - quantized
  - 4-bit
  - glm5_next
  - security
---

# apex-flash-1 · EXL3 4-bit

[`cantina-security/apex-flash-1`](https://huggingface.co/cantina-security/apex-flash-1)
quantized to EXL3, so it fits on two NVIDIA DGX Sparks (or any pair of 128 GB GB10
systems). The original release is BF16 only — 642.7 GB — which does not fit.

**~167 GB · 150,226 tensors · 23 shards**

Recipe, serving stack and benchmarks:
**https://github.com/WamboDNS/apex-flash-1-EXL3-4bpw**

## What is quantized

Routed experts only, matching the layout of
[`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
so the same serving stack reads it:

| | |
| --- | --- |
| `mlp.experts.N.{gate,up,down}_proj` | **4 bits**, `mcg` codebook — 37,152 modules |
| attention, shared expert, router gate | BF16 |
| dense MLPs (first three layers), KDA projections, DSA indexer | BF16 |
| `lm_head`, embedding, vision tower | BF16 |
| MTP head | experts 4-bit, the rest BF16 |

Produced with [ExLlamaV3](https://github.com/turboderp-org/exllamav3) v1.5.3. Tensor names
and dtypes were verified against the published GLM-5.3-Flash EXL3 checkpoint before
release.

## Running it

It needs an engine that reads EXL3 checkpoints with 4-bit `mcg` routed experts — the
recipe above uses [TensorFold](https://github.com/ashhart/TensorFold) v0.6.0 across two
Sparks over a ConnectX-7 link, which is what these numbers come from:

| Concurrent requests | Prose | Structured |
| ---: | ---: | ---: |
| 1 | 54.6 tok/s | 58.7 tok/s |
| 4 | 177.9 tok/s | 198.0 tok/s |

262,144-token window, FP8 KV cache, ~87.6 GiB per Spark. Prefill runs 1,900–3,000 tok/s.

## Quality

Not benchmarked — the BF16 original does not fit on the hardware this was built for, so
there was nothing to compare against locally. Whether it holds the security-task advantage
Cantina report for the original (66.7% vs 60% for base GLM-5.3-Flash) is an open question;
if you have a suitable benchmark, the result is worth publishing either way.

Worth a glance if you are choosing a bitrate: the post-train is small in weight-space next
to what 4-bit quantization does, though EXL3 places its error where the activations are
weakest, which is not where a trained delta lives. The
[recipe](https://github.com/WamboDNS/apex-flash-1-EXL3-4bpw/blob/main/quantize/README.md#choosing-the-bitrate)
has the numbers.

## License

MIT, inherited from apex-flash-1.

## Credits

[Cantina Security](https://www.cantina.security/apex-flash) and Yeta Labs trained and
released apex-flash-1. [Mia's AI Lab](https://github.com/MiaAI-Lab) wrote the
GLM-5.3-Flash serving recipe this is built to work with.
[turboderp](https://github.com/turboderp-org/exllamav3) wrote ExLlamaV3.
