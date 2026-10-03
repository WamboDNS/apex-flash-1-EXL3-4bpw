#!/usr/bin/env bash
# Make a locally converted checkpoint servable without uploading it first.
#
# start.sh and scripts/prepare.sh address the checkpoint by Hugging Face repo id and look
# for it under $HF_CACHE/hub/models--<owner>--<name>/snapshots/<revision>/. There is no
# local-path option, so a checkpoint you built yourself has nowhere to go until it has
# been pushed to the Hub -- which is a poor trade when you only want to benchmark it, and
# a 176 GB upload besides.
#
# This lays the converted output into the cache in the shape those scripts expect. Files
# are hard-linked, not copied, so it costs no extra disk as long as the output and the
# cache share a filesystem.
#
#   quantize/install_local.sh ~/apex-quant/out
#   quantize/install_local.sh ~/apex-quant/out wambosec/apex-flash-1-EXL3-4bpw local
#
# Then serve it with the id you installed under:
#
#   MODEL_ID=wambosec/apex-flash-1-EXL3-4bpw MODEL_REVISION=local ./start.sh
#
set -euo pipefail

OUT_DIR="${1:?usage: install_local.sh <converted out dir> [model id] [revision]}"
MODEL_ID="${2:-wambosec/apex-flash-1-EXL3-4bpw}"
REVISION="${3:-local}"
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"

die() { echo "$*" >&2; exit 1; }

[[ -f "$OUT_DIR/config.json" ]] ||
  die "No config.json in $OUT_DIR -- did the conversion finish?"
[[ -f "$OUT_DIR/model.safetensors.index.json" ]] ||
  die "No index in $OUT_DIR -- run quantize/unfuse.py first."

CACHE_DIR="$HF_CACHE/hub/models--${MODEL_ID//\//--}"
SNAP="$CACHE_DIR/snapshots/$REVISION"
mkdir -p "$SNAP" "$CACHE_DIR/refs"

linked=0
copied=0
for f in "$OUT_DIR"/*; do
  [[ -f "$f" ]] || continue
  name=$(basename "$f")
  [[ -e "$SNAP/$name" ]] && continue
  if ln "$f" "$SNAP/$name" 2>/dev/null; then
    linked=$(( linked + 1 ))
  else
    # Different filesystem: fall back to copying, which does cost the full size.
    cp "$f" "$SNAP/$name"
    copied=$(( copied + 1 ))
  fi
done

echo "$REVISION" > "$CACHE_DIR/refs/main"

echo "installed $MODEL_ID@$REVISION"
echo "  $SNAP"
echo "  $linked file(s) hard-linked${copied:+, $copied copied}"
du -shL "$SNAP" 2>/dev/null | awk '{print "  " $1 " (shared with the conversion output)"}'
echo
echo "serve it with:"
echo "  MODEL_ID=$MODEL_ID MODEL_REVISION=$REVISION ./start.sh"
