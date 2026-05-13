#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$REPO_DIR"
PRETRAINED_DIR="${PRETRAINED_DIR:-$ROOT_DIR/pretrained}"
PYTHON_BIN="${PYTHON_BIN:-python}"

download() {
  local output="$1"
  shift
  local urls=("$@")
  local part="${output}.part"
  local url

  mkdir -p "$(dirname "$output")"

  if [ -s "$output" ]; then
    echo "[skip] $output"
    return 0
  fi

  if [ -s "$part" ]; then
    echo "[resume] $output from $(stat -c%s "$part" 2>/dev/null || echo unknown) bytes"
  else
    echo "[download] $output"
  fi

  for url in "${urls[@]}"; do
    echo "[source] $url"
    if command -v curl >/dev/null 2>&1; then
      if curl -L --fail --retry 5 --retry-all-errors --retry-delay 2 --connect-timeout 20 --continue-at - -o "$part" "$url"; then
        mv "$part" "$output"
        return 0
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget -c -O "$part" "$url"; then
        mv "$part" "$output"
        return 0
      fi
    else
      echo "Error: need curl or wget" >&2
      exit 1
    fi
    echo "[warn] failed from $url" >&2
  done

  echo "Error: failed to download $output from all sources" >&2
  return 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: missing command '$1'" >&2
    exit 1
  }
}

need_cmd "$PYTHON_BIN"

mkdir -p \
  "$PRETRAINED_DIR/amt_model" \
  "$PRETRAINED_DIR/raft_model/models" \
  "$PRETRAINED_DIR/vtss" \
  "$PRETRAINED_DIR/LanguageBind_Video_FT"

# CLIP
download \
  "$PRETRAINED_DIR/ViT-B-32.pt" \
  "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

# DINO
download \
  "$PRETRAINED_DIR/dino_vitbase16_pretrain.pth" \
  "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"

# RAFT
# Prefer direct checkpoint download to avoid Dropbox ZIP instability.
RAFT_PTH="$PRETRAINED_DIR/raft_model/models/raft-things.pth"
RAFT_ZIP="$PRETRAINED_DIR/raft_model/models.zip"

if [ -s "$RAFT_PTH" ]; then
  echo "[skip] $RAFT_PTH"
else
  if ! download \
    "$RAFT_PTH" \
    "https://huggingface.co/RaphaelLiu/EvalCrafter-Models/resolve/main/RAFT/models/raft-things.pth"
  then
    download \
      "$RAFT_ZIP" \
      "https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip?dl=1" \
      "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
    "$PYTHON_BIN" - <<PY
from pathlib import Path
import zipfile

zip_path = Path(r"$RAFT_ZIP")
out_dir = Path(r"$PRETRAINED_DIR/raft_model")
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(out_dir)
zip_path.unlink(missing_ok=True)
PY
  fi
fi

# AMT
cp "$REPO_DIR/narrastream_bench/third_party/amt/cfgs/AMT-S.yaml" "$PRETRAINED_DIR/amt_model/AMT-S.yaml"
download \
  "$PRETRAINED_DIR/amt_model/amt-s.pth" \
  "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth"

# VTSS
download \
  "$PRETRAINED_DIR/vtss/infer.pth" \
  "https://huggingface.co/Koala-36M/Training_Suitability_Assessment/resolve/main/infer.pth"

# LanguageBind video model
LANGUAGEBIND_BASE="https://huggingface.co/LanguageBind/LanguageBind_Video_FT/resolve/main"
for name in \
  config.json \
  merges.txt \
  pytorch_model.bin \
  special_tokens_map.json \
  tokenizer.json \
  tokenizer_config.json \
  vocab.json
do
  download "$PRETRAINED_DIR/LanguageBind_Video_FT/$name" "$LANGUAGEBIND_BASE/$name"
done

echo
echo "Weights downloaded into: $PRETRAINED_DIR"
echo "You can verify expected files with: ls -R $PRETRAINED_DIR"
