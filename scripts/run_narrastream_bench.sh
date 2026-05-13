#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$REPO_DIR"
CALLER_PWD="$PWD"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_PRETRAINED_DIR="${PRETRAINED_DIR:-$ROOT_DIR/pretrained}"

RUN_NAME=""
GPU_ID=""
VIDEO_DIR=""
PROMPTS=""
EVAL_DATA=""
SEGMENT_DURATION="10"
OUTPUT_ROOT="$ROOT_DIR/runs"
CONFIG_PATH="$REPO_DIR/configs/default.yaml"
PATH_CONFIG="$REPO_DIR/configs/paths.yaml"
VLM_MODEL=""
API_WORKERS="4"
RESUME=0
METRICS=()
METRICS_SET=0

usage() {
  cat <<'USAGE_EOF'
Usage:
  bash scripts/run_narrastream_bench.sh --run-name RUN --video-dir DIR --prompts FILE [options]
  bash scripts/run_narrastream_bench.sh --run-name RUN --eval-data FILE [options]

Required:
  --run-name RUN         Name of this run
  One of:
    --video-dir DIR      Directory containing sample_0.mp4, sample_1.mp4, ...
    --prompts FILE       Prompt json/jsonl aligned with sample index
  Or:
    --eval-data FILE     Existing processed eval_data.json

Options:
  --segment-duration N   Segment duration in seconds. Default: 10
  --output-root DIR      Output root. Default: ./runs
  --config FILE          Override configs/default.yaml
  --path-config FILE     Override configs/paths.yaml
                       Default weights path points to ./pretrained
  --gpu-id ID           Bind this run to physical GPU ordinal (0-based, e.g. 4)
                       Auto-resolves against amd-smi/nvidia-smi when available
  --api-workers N       API metric workers. Default: 4
  --resume              Resume from output/results_latest.json
  --vlm-model MODEL      Override VLM model name
  --metrics M1 [M2 ...]  Metrics to run. Default: all
  -h, --help             Show this help
USAGE_EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

resolve_path() {
  local raw="$1"
  if [ -z "$raw" ]; then
    return 0
  fi
  if [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s\n' "$CALLER_PWD/$raw"
  fi
}

print_cmd() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

log_stage() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

resolve_gpu_exports() {
  local requested="$1"
  "$PYTHON_BIN" - "$requested" <<'PY'
import csv
import shlex
import shutil
import subprocess
import sys

requested = [int(x) for x in sys.argv[1].split(',') if x]


def emit(key: str, value: str) -> None:
    print(f"export {key}={shlex.quote(value)}")


def parse_bdf(value: str):
    domain, bus, rest = value.split(':')
    device, function = rest.split('.')
    return (int(domain, 16), int(bus, 16), int(device, 16), int(function, 16))


def resolve_amd() -> None:
    out = subprocess.check_output(["amd-smi", "list", "--csv"], text=True)
    rows = list(csv.DictReader(out.splitlines()))
    rows.sort(key=lambda row: parse_bdf(row["gpu_bdf"]))
    max_idx = len(rows) - 1
    resolved = []
    mapping = []
    for physical in requested:
        if physical < 0 or physical > max_idx:
            raise SystemExit(f"Requested physical GPU {physical} out of range 0..{max_idx}")
        row = rows[physical]
        runtime_idx = row["gpu"]
        resolved.append(str(runtime_idx))
        mapping.append(f"{physical}->{runtime_idx}({row['gpu_bdf']})")
    emit("GPU_BINDING_KIND", "amd")
    emit("GPU_BINDING_MAPPING", "; ".join(mapping))
    emit("ROCR_VISIBLE_DEVICES", ",".join(resolved))
    emit("HIP_VISIBLE_DEVICES", ",".join(str(i) for i in range(len(resolved))))


def resolve_nvidia() -> None:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(',', 3)]
        if len(parts) < 4:
            continue
        index, uuid, bus_id, name = parts
        rows.append({"index": index, "uuid": uuid, "bus_id": bus_id, "name": name})
    rows.sort(key=lambda row: parse_bdf(row["bus_id"]))
    max_idx = len(rows) - 1
    uuids = []
    mapping = []
    for physical in requested:
        if physical < 0 or physical > max_idx:
            raise SystemExit(f"Requested physical GPU {physical} out of range 0..{max_idx}")
        row = rows[physical]
        uuids.append(row["uuid"])
        mapping.append(f"{physical}->{row['index']}({row['bus_id']})")
    emit("GPU_BINDING_KIND", "nvidia")
    emit("GPU_BINDING_MAPPING", "; ".join(mapping))
    emit("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    emit("CUDA_VISIBLE_DEVICES", ",".join(uuids))


if shutil.which("amd-smi"):
    resolve_amd()
elif shutil.which("nvidia-smi"):
    resolve_nvidia()
else:
    emit("GPU_BINDING_KIND", "raw")
    emit("GPU_BINDING_MAPPING", sys.argv[1])
    emit("CUDA_VISIBLE_DEVICES", sys.argv[1])
PY
}

while (($#)); do
  case "$1" in
    --run-name)
      [ $# -ge 2 ] || die "--run-name requires a value"
      RUN_NAME="$2"
      shift 2
      ;;
    --video-dir)
      [ $# -ge 2 ] || die "--video-dir requires a value"
      VIDEO_DIR="$2"
      shift 2
      ;;
    --prompts)
      [ $# -ge 2 ] || die "--prompts requires a value"
      PROMPTS="$2"
      shift 2
      ;;
    --eval-data)
      [ $# -ge 2 ] || die "--eval-data requires a value"
      EVAL_DATA="$2"
      shift 2
      ;;
    --segment-duration)
      [ $# -ge 2 ] || die "--segment-duration requires a value"
      SEGMENT_DURATION="$2"
      shift 2
      ;;
    --output-root)
      [ $# -ge 2 ] || die "--output-root requires a value"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --config)
      [ $# -ge 2 ] || die "--config requires a value"
      CONFIG_PATH="$2"
      shift 2
      ;;
    --path-config)
      [ $# -ge 2 ] || die "--path-config requires a value"
      PATH_CONFIG="$2"
      shift 2
      ;;
    --gpu-id)
      [ $# -ge 2 ] || die "--gpu-id requires a value"
      GPU_ID="$2"
      shift 2
      ;;
    --api-workers)
      [ $# -ge 2 ] || die "--api-workers requires a value"
      API_WORKERS="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --vlm-model)
      [ $# -ge 2 ] || die "--vlm-model requires a value"
      VLM_MODEL="$2"
      shift 2
      ;;
    --metrics)
      shift
      METRICS=()
      METRICS_SET=1
      while (($#)) && [[ "$1" != --* ]]; do
        METRICS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[ -n "$RUN_NAME" ] || die "--run-name is required"

if [ -n "$EVAL_DATA" ] && { [ -n "$VIDEO_DIR" ] || [ -n "$PROMPTS" ]; }; then
  die "use either --eval-data or (--video-dir and --prompts)"
fi

if [ -z "$EVAL_DATA" ]; then
  [ -n "$VIDEO_DIR" ] || die "--video-dir is required when --eval-data is absent"
  [ -n "$PROMPTS" ] || die "--prompts is required when --eval-data is absent"
fi

if [ "$METRICS_SET" -eq 1 ] && [ "${#METRICS[@]}" -eq 0 ]; then
  die "--metrics requires at least one metric"
fi

if [ -n "$GPU_ID" ] && [[ ! "$GPU_ID" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  die "--gpu-id must be a GPU index or comma-separated indices, e.g. 4 or 1,4"
fi

if [[ ! "$API_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  die "--api-workers must be a positive integer"
fi

VIDEO_DIR="$(resolve_path "$VIDEO_DIR")"
PROMPTS="$(resolve_path "$PROMPTS")"
EVAL_DATA="$(resolve_path "$EVAL_DATA")"
OUTPUT_ROOT="$(resolve_path "$OUTPUT_ROOT")"
CONFIG_PATH="$(resolve_path "$CONFIG_PATH")"
PATH_CONFIG="$(resolve_path "$PATH_CONFIG")"

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
PROCESSED_DIR="$RUN_DIR/processed"
RESULTS_DIR="$RUN_DIR/results"
PREPROCESS_SIGNATURE_FILE="$PROCESSED_DIR/.preprocess_signature"
PREPROCESS_EVAL_DATA="$PROCESSED_DIR/eval_data.json"
CURRENT_PREPROCESS_SIGNATURE="$(printf 'video_dir=%s\nprompts=%s\nsegment_duration=%s\n' "$VIDEO_DIR" "$PROMPTS" "$SEGMENT_DURATION")"

[ -f "$CONFIG_PATH" ] || die "config file not found: $CONFIG_PATH"
[ -f "$PATH_CONFIG" ] || die "path config file not found: $PATH_CONFIG"
if [ -n "$GPU_ID" ]; then
  eval "$(resolve_gpu_exports "$GPU_ID")"
fi
if [ ! -d "$DEFAULT_PRETRAINED_DIR" ]; then
  echo "Warning: default weights directory not found: $DEFAULT_PRETRAINED_DIR" >&2
  echo "         Run scripts/download_weights.sh first, or pass --path-config with custom paths." >&2
fi

mkdir -p "$RUN_DIR"
cd "$REPO_DIR"

log_stage "Run Context"
echo "Run name:      $RUN_NAME"
echo "Repo dir:      $REPO_DIR"
echo "Python bin:    $PYTHON_BIN"
echo "Python exec:   $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Python ver:    $("$PYTHON_BIN" -V 2>&1)"
echo "Output root:   $OUTPUT_ROOT"
echo "Results dir:   $RESULTS_DIR"
[ -n "$VIDEO_DIR" ] && echo "Video dir:     $VIDEO_DIR"
[ -n "$PROMPTS" ] && echo "Prompts:       $PROMPTS"
[ -n "$EVAL_DATA" ] && echo "Eval data:     $EVAL_DATA"
echo "Segment sec:   $SEGMENT_DURATION"
echo "API workers:   $API_WORKERS"

if [ -n "$GPU_ID" ]; then
  echo "GPU binding:   physical GPU(s) $GPU_ID via ${GPU_BINDING_KIND:-unknown}"
  echo "GPU mapping:   ${GPU_BINDING_MAPPING:-unknown}"
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "CUDA visible:  $CUDA_VISIBLE_DEVICES"
  [ -n "${HIP_VISIBLE_DEVICES:-}" ] && echo "HIP visible:   $HIP_VISIBLE_DEVICES"
  [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && echo "ROCR visible:  $ROCR_VISIBLE_DEVICES"
fi

if [ -z "$EVAL_DATA" ]; then
  log_stage "Preprocess"
  mkdir -p "$PROCESSED_DIR"
  if [ -f "$PREPROCESS_EVAL_DATA" ]; then
    if [ -f "$PREPROCESS_SIGNATURE_FILE" ]; then
      STORED_PREPROCESS_SIGNATURE="$(cat "$PREPROCESS_SIGNATURE_FILE")"
      if [ "$STORED_PREPROCESS_SIGNATURE" != "$CURRENT_PREPROCESS_SIGNATURE" ]; then
        die "existing processed data in $PROCESSED_DIR was created from different inputs; remove it or use a new --run-name"
      fi
    else
      echo "Warning: found existing $PREPROCESS_EVAL_DATA without signature; reusing it for resume." >&2
    fi
    echo "Preprocess mode: resume"
    echo "Preprocess data: $PREPROCESS_EVAL_DATA"
    EVAL_DATA="$PREPROCESS_EVAL_DATA"
  else
    echo "Preprocess mode: fresh"
    PREPROCESS_CMD=(
      "$PYTHON_BIN" -m narrastream_bench.core.preprocess
      --video_dir "$VIDEO_DIR"
      --prompts "$PROMPTS"
      --output "$PROCESSED_DIR"
      --segment_duration "$SEGMENT_DURATION"
    )
    print_cmd "${PREPROCESS_CMD[@]}"
    "${PREPROCESS_CMD[@]}"
    printf '%s' "$CURRENT_PREPROCESS_SIGNATURE" > "$PREPROCESS_SIGNATURE_FILE"
    EVAL_DATA="$PREPROCESS_EVAL_DATA"
  fi
fi

mkdir -p "$RESULTS_DIR"
log_stage "Evaluate"
if [ "$METRICS_SET" -eq 1 ]; then
  echo "Metrics:       ${METRICS[*]}"
else
  echo "Metrics:       all"
fi
EVAL_CMD=(
  "$PYTHON_BIN" -m narrastream_bench.core.evaluate
  --eval_data "$EVAL_DATA"
  --output "$RESULTS_DIR"
  --config "$CONFIG_PATH"
  --path_config "$PATH_CONFIG"
  --api-workers "$API_WORKERS"
)

if [ "$METRICS_SET" -eq 1 ]; then
  EVAL_CMD+=(--metrics "${METRICS[@]}")
fi

if [ "$RESUME" -eq 1 ]; then
  EVAL_CMD+=(--resume)
fi

if [ -n "$VLM_MODEL" ]; then
  EVAL_CMD+=(--vlm_model "$VLM_MODEL")
fi

print_cmd "${EVAL_CMD[@]}"
"${EVAL_CMD[@]}"

log_stage "Run Complete"
echo
echo "Run complete."
echo "Processed data: $PROCESSED_DIR"
echo "Results:        $RESULTS_DIR"
echo "Weights root:   $DEFAULT_PRETRAINED_DIR"
if [ -n "$GPU_ID" ]; then
  echo "GPU mapping:    ${GPU_BINDING_MAPPING:-unknown}"
fi
