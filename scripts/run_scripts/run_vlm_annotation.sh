#!/bin/bash
set -euo pipefail

##################################################
# usage
##################################################
# Run VLM initial subtask annotation on the example dataset:
#   OPENAI_API_KEY=... OPENAI_MODEL=<vision-model> \
#   bash scripts/run_scripts/run_vlm_annotation.sh 0
#
# Run multiple episodes:
#   OPENAI_API_KEY=... OPENAI_MODEL=<vision-model> \
#   bash scripts/run_scripts/run_vlm_annotation.sh 0 1 2
#
# Or pass episodes through an environment variable:
#   EPISODES="0 1 2" OPENAI_API_KEY=... OPENAI_MODEL=<vision-model> \
#   bash scripts/run_scripts/run_vlm_annotation.sh
#
# Useful overrides:
#   DATASET_ROOT=assets/dump_bin_bigbin
#   CAMERA=observation.images.cam_high
#   OUTPUT_DIR=annotations/dump_bin_bigbin/manual
#   OPENAI_BASE_URL=https://api.openai.com/v1
#   VIDEO_VIEW_MODE=head_wrists
#   MAX_SAMPLE_FRAMES=32
#   OPENAI_TIMEOUT_SECONDS=300

dataset_root="${DATASET_ROOT:-assets/dump_bin_bigbin}"
camera="${CAMERA:-observation.images.cam_high}"
dataset_name="$(basename "${dataset_root}")"

output_dir="${OUTPUT_DIR:-annotations/${dataset_name}/manual}"
check_frames_root="${CHECK_FRAMES_ROOT:-exp/cortex/annotation/${dataset_name}/check_frames}"
subtask_clips_root="${SUBTASK_CLIPS_ROOT:-exp/cortex/annotation/${dataset_name}/subtask_clips}"

openai_base_url="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
openai_api_key="${OPENAI_API_KEY:-${API_KEY:-}}"
openai_model="${OPENAI_MODEL:-${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}}"
max_sample_frames="${MAX_SAMPLE_FRAMES:-32}"
openai_timeout_seconds="${OPENAI_TIMEOUT_SECONDS:-300}"
video_view_mode="${VIDEO_VIEW_MODE:-head_wrists}"

export OPENAI_TIMEOUT_SECONDS="${openai_timeout_seconds}"

# OpenAI-compatible APIs conventionally expose chat completions below /v1.
# A bare host often serves an HTML dashboard instead of the API.
if [[ "${openai_base_url}" =~ ^https?://[^/]+/?$ ]]; then
  openai_base_url="${openai_base_url%/}/v1"
fi

if [ -z "${openai_api_key}" ] || [ "${openai_api_key}" = "sk-xxx" ] || [ "${openai_api_key}" = "EMPTY" ]; then
  echo "[ERROR] Set OPENAI_API_KEY to a valid token for OPENAI_BASE_URL."
  echo "[HINT] OPENAI_API_KEY=... OPENAI_BASE_URL=... bash $0 ${*:-0}"
  exit 1
fi

if [ -z "${openai_model}" ]; then
  echo "[ERROR] OPENAI_MODEL must be set to an OpenAI-compatible vision model."
  exit 1
fi

if [ ! -d "${dataset_root}" ]; then
  echo "[ERROR] DATASET_ROOT does not exist: ${dataset_root}"
  exit 1
fi

video_dir="${dataset_root}/videos/chunk-000/${camera}"
if [ ! -d "${video_dir}" ]; then
  echo "[ERROR] camera video directory does not exist: ${video_dir}"
  exit 1
fi

meta_path="${dataset_root}/meta/episodes.jsonl"
if [ ! -f "${meta_path}" ]; then
  echo "[WARN] meta file not found; task instruction will be empty: ${meta_path}"
fi

python - <<'PY'
import sys

try:
    import openai  # noqa: F401
except Exception as exc:
    print("[ERROR] Failed to import the OpenAI Python SDK.", file=sys.stderr)
    print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
    print(
        "[HINT] Reinstall compatible OpenAI/Pydantic packages:\n"
        "       python -m pip install -U --force-reinstall 'openai' 'pydantic>=2.7,<3'",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

if [ "$#" -gt 0 ]; then
  episodes=("$@")
elif [ -n "${EPISODES:-}" ]; then
  read -r -a episodes <<< "${EPISODES}"
else
  episodes=(0)
fi

mkdir -p "${output_dir}" "${check_frames_root}" "${subtask_clips_root}"

format_episode_name() {
  local raw="$1"
  if [[ "${raw}" =~ ^episode_([0-9]+)$ ]]; then
    printf "episode_%06d" "$((10#${BASH_REMATCH[1]}))"
  else
    printf "episode_%06d" "$((10#${raw}))"
  fi
}

read_task_instruction() {
  local episode_index="$1"
  local fallback="${TASK_INSTRUCTION:-}"
  if [ -n "${fallback}" ] || [ ! -f "${meta_path}" ]; then
    printf "%s" "${fallback}"
    return
  fi

  python - "${meta_path}" "${episode_index}" <<'PY'
import json
import sys

meta_path, episode_index = sys.argv[1], int(sys.argv[2])
with open(meta_path, "r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("episode_index", -1)) == episode_index:
            tasks = row.get("tasks") or []
            print(str(tasks[0]) if tasks else "")
            break
PY
}

echo "[INFO] dataset_root: ${dataset_root}"
echo "[INFO] camera: ${camera}"
echo "[INFO] output_dir: ${output_dir}"
echo "[INFO] openai_base_url: ${openai_base_url}"
echo "[INFO] openai_model: ${openai_model}"
echo "[INFO] max_sample_frames: ${max_sample_frames}"
echo "[INFO] openai_timeout_seconds: ${openai_timeout_seconds}"
echo "[INFO] video_view_mode: ${video_view_mode}"
echo "[INFO] episodes: ${episodes[*]}"

for raw_episode in "${episodes[@]}"; do
  episode_name="$(format_episode_name "${raw_episode}")"
  episode_digits="${episode_name#episode_}"
  episode_index="$((10#${episode_digits}))"

  video_path="${video_dir}/${episode_name}.mp4"
  output_path="${output_dir}/${episode_name}.json"
  check_frames_dir="${check_frames_root}/${episode_name}"
  subtask_clips_dir="${subtask_clips_root}/${episode_name}"
  task_instruction="$(read_task_instruction "${episode_index}")"

  if [ ! -f "${video_path}" ]; then
    echo "[ERROR] video not found: ${video_path}"
    exit 1
  fi

  echo "[RUN] ${episode_name}"
  echo "[INFO] task_instruction: ${task_instruction}"

  cmd=(
    python -m cortex.annotation.vlm_annotation
    --video_path "${video_path}"
    --output_path "${output_path}"
    --sample_id "${episode_name}"
    --max_sample_frames "${max_sample_frames}"
    --check_frames_dir "${check_frames_dir}"
    --subtask_clips_dir "${subtask_clips_dir}"
    --ceph_video_view_mode "${video_view_mode}"
    --base_url "${openai_base_url}"
    --api_key "${openai_api_key}"
    --model "${openai_model}"
  )

  if [ -n "${task_instruction}" ]; then
    cmd+=(--task_instruction "${task_instruction}")
  fi

  "${cmd[@]}"
  echo "[DONE] annotation saved to ${output_path}"
done
