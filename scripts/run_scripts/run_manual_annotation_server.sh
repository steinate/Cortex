#!/bin/bash
set -euo pipefail

# Start the browser-based correction stage after VLM initial annotation.
# Example:
#   bash scripts/run_scripts/run_manual_annotation_server.sh
#   PORT=9000 bash scripts/run_scripts/run_manual_annotation_server.sh

dataset_root="${DATASET_ROOT:-assets/dump_bin_bigbin}"
camera="${CAMERA:-observation.images.cam_high}"
dataset_name="$(basename "${dataset_root}")"
video_dir="${VIDEO_DIR:-${dataset_root}/videos/chunk-000/${camera}}"
output_dir="${OUTPUT_DIR:-annotations/${dataset_name}/manual}"
host="${ANNOTATOR_HOST:-127.0.0.1}"
port="${ANNOTATOR_PORT:-8765}"

if [ ! -d "${video_dir}" ]; then
  echo "[ERROR] video directory does not exist: ${video_dir}"
  exit 1
fi

mkdir -p "${output_dir}"

echo "[INFO] video_dir: ${video_dir}"
echo "[INFO] output_dir: ${output_dir}"
echo "[INFO] annotator_url: http://${host}:${port}"

exec python -m cortex.annotation.annotator_server \
  --video-dir "${video_dir}" \
  --output-dir "${output_dir}" \
  --host "${host}" \
  --port "${port}" \
  "$@"
