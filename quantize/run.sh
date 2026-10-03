#!/usr/bin/env bash
# Build the EXL3 checkpoint for apex-flash-1 on a single DGX Spark.
#
# Runs exllamav3's converter inside the same NVIDIA PyTorch container the serving image
# is built from, with stream_shards.py feeding it source shards and deleting them again
# as it advances. Resumable: re-run with the same ROOT and it picks up from the last
# checkpoint in work/.
#
# Expect this to take many hours. Run it under tmux.
#
#   tmux new -s quant
#   quantize/run.sh
#
set -euo pipefail

ROOT="${ROOT:-$HOME/apex-quant}"
REPO="${REPO:-cantina-security/apex-flash-1}"
REVISION="${REVISION:-main}"
EXPERT_BITS="${EXPERT_BITS:-4}"
MTP_BITS="${MTP_BITS:-4}"          # see the note on -mb below
CODEBOOK="${CODEBOOK:-mcg}"          # matches the GLM-5.3-Flash EXL3 checkpoint's storage ABI
SHARD_SIZE="${SHARD_SIZE:-8192}"     # MB
LOOKAHEAD="${LOOKAHEAD:-2}"          # layers of source to stay ahead by
PARALLEL="${PARALLEL:-6}"            # concurrent shard downloads
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
EXL3_REF="${EXL3_REF:-v1.5.3}"
MAX_MODULE="${MAX_MODULE:-}"         # set for a short trial run

SRC="$ROOT/src"
WORK="$ROOT/work"
OUT="$ROOT/out"
LOG="$ROOT/convert.log"
RECIPE="$ROOT/recipe.yaml"

mkdir -p "$SRC" "$WORK" "$OUT"

# The budget is free space plus whatever this job already holds -- on a resume the source
# shards and the working directory are part of the peak, not on top of it.
have_budget() {
  local need_gb="$1" free_gb used_gb total
  free_gb=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc '0-9')
  used_gb=$(du -sB1G "$SRC" "$WORK" "$OUT" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  total=$(( free_gb + used_gb ))
  (( total >= need_gb )) || {
    echo "Only ${total} GB available under $ROOT (${free_gb} free + ${used_gb} already" >&2
    echo "in src/work/out); want at least ${need_gb} GB." >&2
    echo "Peak is the chunk's source shards plus ~176 GB work and ~176 GB output." >&2
    echo "Run quantize/plan_disk.py for the schedule." >&2
    return 1
  }
  echo "disk: ${total} GB available (${free_gb} free + ${used_gb} in use)"
}
have_budget "${NEED_GB:-500}"

# --- model metadata -----------------------------------------------------------------
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         chat_template.jinja processor_config.json model.safetensors.index.json LICENSE; do
  [[ -s "$SRC/$f" ]] && continue
  echo "fetching $f"
  curl -fsSL -o "$SRC/$f" "https://huggingface.co/$REPO/resolve/$REVISION/$f"
done

# --- container with exllamav3 -------------------------------------------------------
# The tag carries a patch level as well as the exllamav3 ref. Without it, an image built
# before these patches existed still satisfies `docker image inspect` and gets used
# silently -- the build is skipped and the conversion fails exactly as it did before the
# fix. Bump this whenever the Dockerfile below changes.
PATCH_LEVEL="${PATCH_LEVEL:-p1}"
TAG="exl3-convert:${EXL3_REF}-${PATCH_LEVEL}"
if ! docker image inspect "$TAG" >/dev/null 2>&1; then
  echo "building $TAG"
  docker build -t "$TAG" -f - . <<DOCKERFILE
FROM $IMAGE
# GB10 is Blackwell, compute capability 12.1.
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCH:-12.1}"
ENV MAX_JOBS=16
RUN pip install --no-cache-dir safetensors pyyaml huggingface_hub
RUN git clone --depth 1 --branch $EXL3_REF https://github.com/turboderp-org/exllamav3 /opt/exllamav3 \
 && cd /opt/exllamav3 && pip install --no-cache-dir -e .
# convert.py takes --bits / --mtp_bits / --head_bits / --vision_bits as argparse
# type=float, so a whole bitrate like 4 arrives as 4.0, and a resumed job replays whatever
# types it stored in work/args.json. K is handed straight to the C++ bindings, which are
# typed int and shift by it, so the first MoE layer dies with "incompatible function
# arguments" -- about ten minutes in, with a message that points nowhere near the cause.
# make_quant_args is the single place K enters, so normalise it there; it covers the
# recipe, the MTP head and the ordinary allocator alike. Fractional rates stay floats.
RUN sed -i 's/^def make_quant_args(args, idx, K, devices, device_ratios = None):/&\n    K = int(K) if isinstance(K, float) and K.is_integer() else K/' \
      /opt/exllamav3/exllamav3/conversion/convert_model.py \
 && sed -n '/^def make_quant_args/,+2p' /opt/exllamav3/exllamav3/conversion/convert_model.py
WORKDIR /opt/exllamav3
DOCKERFILE
fi

CONTAINER="${CONTAINER:-apex-exl3-convert}"

# Everything the container writes under $ROOT -- the working directory, the checkpoints,
# the converted shards -- lands as root otherwise, and you need sudo to so much as delete
# your own output. Map the invoking user in.
DOCKER_USER=(--user "$(id -u):$(id -g)")

run_in() {
  docker run --rm --gpus all --ipc=host "${DOCKER_USER[@]}" \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$ROOT:$ROOT" -w /opt/exllamav3 "$TAG" "$@"
}

# The conversion itself runs under a fixed name. Without one, killing the shell (or the
# tmux session) leaves the container running: it keeps its share of the GPU, and the next
# attempt dies with "CUDA error: out of memory" while the real culprit sits in docker ps.
run_convert() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run --rm --name "$CONTAINER" --gpus all --ipc=host "${DOCKER_USER[@]}" \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$ROOT:$ROOT" -w /opt/exllamav3 "$TAG" "$@"
}

# A container can exit before the driver releases its GPU memory, and the next chunk then
# fails to allocate even a few kilobytes. Wait for the device to actually come free.
wait_for_gpu() {
  local i busy out
  for i in $(seq 1 60); do
    out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    # grep -c prints 0 AND exits non-zero when nothing matches, so "|| echo 0" would
    # append a second line and the arithmetic test below would choke on it.
    busy=$(printf '%s' "$out" | grep -c . || true)
    busy=${busy:-0}
    if (( busy == 0 )); then
      (( i > 1 )) && echo "gpu free after $(( i * 5 ))s"
      return 0
    fi
    sleep 5
  done
  echo "!! gpu still busy after 5 minutes; continuing anyway" >&2
}

# Stop the converter with the script, however the script ends.
cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  return 0
}
trap cleanup EXIT INT TERM

# --- recipe -------------------------------------------------------------------------
# Needs only config.json, so it runs before any weights are on disk.
if [[ ! -s "$RECIPE" ]]; then
  cp "$(dirname "$0")/make_recipe.py" "$ROOT/make_recipe.py"
  run_in python3 "$ROOT/make_recipe.py" "$SRC" "$RECIPE" --expert-bits "$EXPERT_BITS"
fi

# --- convert, in chunks -------------------------------------------------------------
# One long run with a background downloader does not work: exllamav3 globs the source
# directory once when it builds its SafetensorsCollection, so shards that arrive later are
# invisible and the job dies on a tensor whose file is already on disk. --max_module is
# read fresh from the command line every invocation, though, and every invocation
# re-globs. So: fetch a chunk's shards, convert that far, drop what is behind, repeat.
cp "$(dirname "$0")/stream_shards.py" "$ROOT/stream_shards.py"

RESUMING=0
[[ -s "$WORK/args.json" ]] && RESUMING=1
if (( RESUMING )); then
  cp "$(dirname "$0")/fix_job_args.py" "$ROOT/fix_job_args.py"
  run_in python3 "$ROOT/fix_job_args.py" "$WORK/args.json"
fi

stream() { python3 "$ROOT/stream_shards.py" "$@" --in-dir "$SRC" --repo "$REPO" \
             --revision "$REVISION" --parallel "$PARALLEL"; }

# Module index = layer + 2 (the embedding and hc_expand come first), so converting
# through layer L means --max_module (L+2).
LAST_LAYER="${LAST_LAYER:-45}"       # 44 transformer layers plus the MTP layer
CHUNK="${CHUNK:-6}"

# Resume where the converter actually got to, not from zero. Restarting the loop at layer
# 0 would re-fetch the shards for chunks that are already converted and evicted -- tens of
# gigabytes of download to produce nothing. The working directory names one file per
# finished module, so the highest layer there is the last one done.
start_layer() {
  local last
  last=$(ls "$WORK/qtensors" 2>/dev/null \
         | sed -n 's/.*layers\.\([0-9]*\)\.safetensors/\1/p' \
         | sort -n | tail -1)
  # 10# so a padded number is never read as octal.
  [[ -n "$last" ]] && echo $(( 10#$last + 1 )) || echo 0
}

layer="${START_LAYER:-$(start_layer)}"
(( layer > 0 )) && echo "resuming at layer ${layer} (layers 0-$(( layer - 1 )) already converted)"
while (( layer <= LAST_LAYER )); do
  end=$(( layer + CHUNK - 1 ))
  (( end > LAST_LAYER )) && end=$LAST_LAYER

  # A shard download that exhausts its retries should not end a run that is hours in.
  # Give the fetch step several goes of its own, with a pause between them -- the partial
  # file resumes, so a later attempt picks up where the last one stopped.
  fetched=0
  for fetch_try in 1 2 3 4; do
    if stream ensure --layers "${layer}-${end}"; then fetched=1; break; fi
    echo "!! shard fetch failed for layers ${layer}-${end} (try ${fetch_try}); waiting" >&2
    sleep $(( fetch_try * 120 ))
  done
  if (( ! fetched )); then
    echo "!! could not fetch the shards for layers ${layer}-${end} after 4 tries." >&2
    echo "   The partials are kept; re-run this script to continue." >&2
    exit 1
  fi

  ARGS=(--in_dir "$SRC" --work_dir "$WORK" --out_dir "$OUT"
        --recipe "$RECIPE" --bits "$EXPERT_BITS"
        --head_bits 16 --mtp_bits "$MTP_BITS" --vision_bits 16
        --codebook "$CODEBOOK" --shard_size "$SHARD_SIZE" --checkpoint_interval 120)
  # The final chunk runs uncapped so the tail modules and the output get written.
  if (( end < LAST_LAYER )); then
    ARGS+=(--max_module $(( end + 2 )))
  fi
  (( RESUMING )) && ARGS+=(--resume)

  echo "=== converting layers ${layer}-${end} ==="
  rc=1
  for attempt in 1 2 3; do
    wait_for_gpu
    # A retry always resumes: by now the job has written a checkpoint, even if the first
    # attempt died early, and starting over would discard it.
    if (( attempt > 1 )) && [[ ! " ${ARGS[*]} " == *" --resume "* ]]; then
      ARGS+=(--resume)
    fi
    run_convert python3 convert.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    (( rc == 0 )) && break
    # A chunk boundary is the one place a CUDA OOM is expected and harmless: the previous
    # container's memory may not be released yet, and even a tiny allocation then fails.
    # The checkpoint is intact, so retrying costs a model reload, not a chunk.
    if grep -q "CUDA error: out of memory" <(tail -40 "$LOG"); then
      echo "!! CUDA OOM on attempt ${attempt} for layers ${layer}-${end}; waiting and retrying" >&2
      sleep 60
      continue
    fi
    break
  done
  if (( rc != 0 )); then
    echo "!! conversion exited $rc while on layers ${layer}-${end}" >&2
    exit "$rc"
  fi

  RESUMING=1
  layer=$(( end + 1 ))
  if (( layer <= LAST_LAYER )); then
    stream evict --before "$layer"
  fi
done

# --- split the fused attention tensors -----------------------------------------------
# exllamav3 writes the KDA layers' q/k/v projections and conv1d weights fused into one
# tensor each. The serving side looks for them split, the way the published GLM-5.3-Flash
# checkpoint stores them, so a fused checkpoint loads nothing. Both are unquantized here,
# so the split is exact.
cp "$(dirname "$0")/unfuse.py" "$ROOT/unfuse.py"
VERIFY=()
# Source shards survive only if streaming was off or the run was short; verify when we can.
[[ -s "$SRC/model.safetensors.index.json" ]] && VERIFY=(--verify --src-dir "$SRC")
run_in python3 "$ROOT/unfuse.py" --out-dir "$OUT" "${VERIFY[@]}"

# --- put the MTP head's non-expert weights back in BF16 ------------------------------
# --mtp_bits covers every Linear in the MTP head, so 4 takes nine extra modules along with
# the 864 experts. The published checkpoint keeps those nine unquantized, and DRAFTER=mtp
# will not load a checkpoint that has them the other way. Needs the source shards for the
# MTP layer, which the streamer pins.
cp "$(dirname "$0")/restore_mtp_bf16.py" "$ROOT/restore_mtp_bf16.py"
run_in python3 "$ROOT/restore_mtp_bf16.py" --out-dir "$OUT" --src-dir "$SRC" \
  --layer "$LAST_LAYER" || echo "!! could not restore the MTP head; DRAFTER=mtp will not load" >&2

# --- make the shards match the index -------------------------------------------------
# exllamav3 writes some MTP-layer tensors into two shards. An index cannot express that,
# so it is invisible until a reader walks the files and refuses: "duplicate checkpoint
# tensor". Rewriting shards above also strands copies the index no longer points at.
cp "$(dirname "$0")/dedupe_shards.py" "$ROOT/dedupe_shards.py"
run_in python3 "$ROOT/dedupe_shards.py" --out-dir "$OUT"

# --- say what the conversion did, in the terms the server checks ---------------------
# TensorFold refuses an EXL3 checkpoint that does not declare which weights were
# quantized, and exllamav3 does not write that field. This adds it -- after verifying it
# is true, so the claim is never ahead of the checkpoint.
cp "$(dirname "$0")/declare_scope.py" "$ROOT/declare_scope.py"
run_in python3 "$ROOT/declare_scope.py" --out-dir "$OUT"

# --- reconcile dtypes against the reference ------------------------------------------
# unfuse.py restores dtypes from the source model, but streaming evicts most source shards
# before it runs, so most tensors keep exllamav3's wider storage. The serving kernels check
# dtype, so this loads and then dies inside a layer. REFERENCE_DTYPES is a name -> dtype map
# built with fix_dtypes.py --dump-reference from a known-good checkpoint.
cp "$(dirname "$0")/fix_dtypes.py" "$ROOT/fix_dtypes.py"
if [[ -s "${REFERENCE_DTYPES:-$(dirname "$0")/ref-dtypes.json.gz}" ]]; then
  run_in python3 "$ROOT/fix_dtypes.py" --out-dir "$OUT" \
    --reference-dtypes "${REFERENCE_DTYPES:-$(dirname "$0")/ref-dtypes.json.gz}"
else
  echo "!! no reference dtype map; skipping the dtype reconciliation." >&2
  echo "   Build one with: fix_dtypes.py --dump-reference <good checkpoint> --out ref-dtypes.json.gz" >&2
fi

# --- check the layout before anyone tries to serve it --------------------------------
cp "$(dirname "$0")/check_layout.py" "$ROOT/check_layout.py"
CHECK_ARGS=(--out-dir "$OUT" --full)
[[ -s "${REFERENCE_DTYPES:-$(dirname "$0")/ref-dtypes.json.gz}" ]] &&
  CHECK_ARGS+=(--reference-dtypes "${REFERENCE_DTYPES:-$(dirname "$0")/ref-dtypes.json.gz}")
if run_in python3 "$ROOT/check_layout.py" "${CHECK_ARGS[@]}"; then
  echo
  echo "done. output in $OUT"
else
  echo
  echo "!! The layout does not match the published checkpoint. Do not serve this yet." >&2
fi
du -sh "$OUT"
