#!/bin/bash
#SBATCH --job-name=episode_level_eval
#SBATCH -p xxx
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

##################################################
# usage
##################################################
# Single-task episode-level eval. DATASET_TAG can be galaxea, agibot, or behavior.
# The script reuses cortex/inference/config/sys2_subtask_val.json for dataset root,
# video keys, and timing knobs. TASK_NAME is the first positional argument.
#
# Behavior:
#   MODEL_NAME_OR_PATH=/path/to/checkpoint DATASET_TAG=behavior \
#   sbatch scripts/run_scripts/episode_level.sh task-0000
#
# Galaxea:
#   MODEL_NAME_OR_PATH=/path/to/checkpoint DATASET_TAG=galaxea \
#   sbatch scripts/run_scripts/episode_level.sh Arrange_The_Sofa_Cushions_20250722_008
#
# Override dataset config or dataset fields explicitly:
#   EVAL_DATASET_CONFIG=/path/to/sys2_subtask_val.json DATASET_TAG=agibot \
#   MODEL_NAME_OR_PATH=/path/to/checkpoint \
#   sbatch scripts/run_scripts/episode_level.sh task_422
#
# Useful quick-run knobs:
#   MAX_EPISODES=1 SAVE_VISUAL_VIDEO=False

##################################################
# task / suite setting
##################################################
TASK_NAME="${1:-${TASK_NAME:-}}"
EVAL_SUITE="${EVAL_SUITE:-${DATASET_TAG:-galaxea}}"
DATASET_TAG="${DATASET_TAG:-${EVAL_SUITE}}"
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG:-cortex/inference/config/sys2_subtask_val.json}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-${EVAL_DATASET:-}}"

CONFIG_DATASET_NAME=""
CONFIG_DATASET_ROOT=""
CONFIG_VIDEO_KEYS=""
CONFIG_TRANSITION_TAIL_SEC=""
CONFIG_TRANSITION_HEAD_SEC=""
CONFIG_LAST_TAIL_SEC=""
CONFIG_IGNORE_BOUNDARY_SEC=""

if [ -n "${EVAL_DATASET_CONFIG}" ] && [ -f "${EVAL_DATASET_CONFIG}" ]; then
  CONFIG_EXPORTS="$(python - "${EVAL_DATASET_CONFIG}" "${DATASET_TAG}" "${EVAL_DATASET_NAME}" <<'PYCFG'
import json
import shlex
import sys
from pathlib import Path

config_path, dataset_tag, dataset_name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(config_path, "r", encoding="utf-8") as f:
    configs = json.load(f)
if isinstance(configs, dict):
    configs = [configs]

def normalize_name(value):
    return str(value or "").replace("-", "_").lower()

TASK_TYPE_SUFFIXES = ("_spatial", "_counting", "_long")

def base_dataset_name(value):
    name = normalize_name(value)
    for suffix in TASK_TYPE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name

tag = normalize_name(dataset_tag)
wanted = normalize_name(dataset_name)
selected = None
if wanted:
    for cfg in configs:
        if normalize_name(cfg.get("dataset_name")) == wanted:
            selected = cfg
            break
else:
    exact_base = []
    prefixed = []
    for cfg in configs:
        name = normalize_name(cfg.get("dataset_name"))
        base_name = base_dataset_name(name)
        if base_name == tag or base_name == f"{tag}_subtask_val":
            exact_base.append(cfg)
        elif name.startswith(f"{tag}_"):
            prefixed.append(cfg)
    selected = (exact_base or prefixed or [None])[0]
if selected is None:
    sys.exit(0)

mapping = {
    "CONFIG_DATASET_NAME": selected.get("dataset_name", ""),
    "CONFIG_DATASET_ROOT": selected.get("data_path", ""),
    "CONFIG_VIDEO_KEYS": selected.get("video_keys", ""),
    "CONFIG_TRANSITION_TAIL_SEC": selected.get("transition_tail_sec", ""),
    "CONFIG_TRANSITION_HEAD_SEC": selected.get("transition_head_sec", ""),
    "CONFIG_LAST_TAIL_SEC": selected.get("last_tail_sec", ""),
    "CONFIG_IGNORE_BOUNDARY_SEC": selected.get("ignore_boundary_sec", ""),
}
for key, value in mapping.items():
    print(f"{key}={shlex.quote(str(value))}")
PYCFG
)"
  eval "${CONFIG_EXPORTS}"
fi

case "${DATASET_TAG}" in
  agibot)
    DATASET_ROOT="${DATASET_ROOT:-${CONFIG_DATASET_ROOT:-/path/to/agibotworld}}"
    VIDEO_KEYS="${VIDEO_KEYS:-${CONFIG_VIDEO_KEYS:-observation.images.head,observation.images.hand_left,observation.images.hand_right}}"
    TRANSITION_TAIL_SEC="${TRANSITION_TAIL_SEC:-${CONFIG_TRANSITION_TAIL_SEC:-0.7}}"
    TRANSITION_HEAD_SEC="${TRANSITION_HEAD_SEC:-${CONFIG_TRANSITION_HEAD_SEC:-0.5}}"
    LAST_TAIL_SEC="${LAST_TAIL_SEC:-${CONFIG_LAST_TAIL_SEC:-1.0}}"
    ;;
  galaxea)
    DATASET_ROOT="${DATASET_ROOT:-${CONFIG_DATASET_ROOT:-/path/to/lerobot_opensource/}}"
    VIDEO_KEYS="${VIDEO_KEYS:-${CONFIG_VIDEO_KEYS:-observation.images.head_rgb,observation.images.left_wrist_rgb,observation.images.right_wrist_rgb}}"
    TRANSITION_TAIL_SEC="${TRANSITION_TAIL_SEC:-${CONFIG_TRANSITION_TAIL_SEC:-1.7}}"
    TRANSITION_HEAD_SEC="${TRANSITION_HEAD_SEC:-${CONFIG_TRANSITION_HEAD_SEC:-0.4}}"
    LAST_TAIL_SEC="${LAST_TAIL_SEC:-${CONFIG_LAST_TAIL_SEC:-1.5}}"
    ;;
  behavior)
    DATASET_ROOT="${DATASET_ROOT:-${CONFIG_DATASET_ROOT:-${BEHAVIOR_DATASET_ROOT:-}}}"
    VIDEO_KEYS="${VIDEO_KEYS:-${CONFIG_VIDEO_KEYS:-observation.images.rgb.head,observation.images.rgb.left_wrist,observation.images.rgb.right_wrist}}"
    TRANSITION_TAIL_SEC="${TRANSITION_TAIL_SEC:-${CONFIG_TRANSITION_TAIL_SEC:-0.7}}"
    TRANSITION_HEAD_SEC="${TRANSITION_HEAD_SEC:-${CONFIG_TRANSITION_HEAD_SEC:-1.0}}"
    LAST_TAIL_SEC="${LAST_TAIL_SEC:-${CONFIG_LAST_TAIL_SEC:-1.5}}"
    ;;
  *)
    DATASET_ROOT="${DATASET_ROOT:-${CONFIG_DATASET_ROOT:-}}"
    VIDEO_KEYS="${VIDEO_KEYS:-${CONFIG_VIDEO_KEYS:-}}"
    TRANSITION_TAIL_SEC="${TRANSITION_TAIL_SEC:-${CONFIG_TRANSITION_TAIL_SEC:-0.1}}"
    TRANSITION_HEAD_SEC="${TRANSITION_HEAD_SEC:-${CONFIG_TRANSITION_HEAD_SEC:-0.2}}"
    LAST_TAIL_SEC="${LAST_TAIL_SEC:-${CONFIG_LAST_TAIL_SEC:-1.0}}"
    ;;
esac
IGNORE_BOUNDARY_SEC="${IGNORE_BOUNDARY_SEC:-${CONFIG_IGNORE_BOUNDARY_SEC:-0.0}}"

if [ -z "${TASK_NAME}" ]; then
  echo "[ERROR] TASK_NAME must be set, or pass it as the first positional argument."
  exit 1
fi
if [ -z "${DATASET_ROOT}" ]; then
  echo "[ERROR] DATASET_ROOT must be set for DATASET_TAG=${DATASET_TAG}."
  exit 1
fi
if [ -z "${VIDEO_KEYS}" ]; then
  echo "[ERROR] VIDEO_KEYS must be set."
  exit 1
fi

DATASET_PATH="${DATASET_PATH:-${DATASET_ROOT%/}/${TASK_NAME}}"

##################################################
# dist setting (single node)
##################################################
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
export NNODES="${NNODES:-${SLURM_NNODES:-1}}"
export NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
export MASTER_PORT="${MASTER_PORT:-$((RANDOM % 101 + 20000))}"

##################################################
# model / eval setting
##################################################
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-${BASE_MODEL:-/path/to/Qwen3-VL-8B-Instruct}}"
PROCESSOR_NAME_OR_PATH="${PROCESSOR_NAME_OR_PATH:-${BASE_MODEL_NAME_OR_PATH}}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${JUDGE_MODEL:-/path/to/Qwen3.5-9B/}}"

SEED="${SEED:-42}"
NUM_EVAL_EPISODES_PER_TASK="${NUM_EVAL_EPISODES_PER_TASK:-10}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-15}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-0.0}"
MAX_SAMPLES_PER_EPISODE="${MAX_SAMPLES_PER_EPISODE:-0}"
MAX_EPISODES="${MAX_EPISODES:-0}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
ENABLE_CLOSED_LOOP_EVAL="${ENABLE_CLOSED_LOOP_EVAL:-True}"
SAVE_TASK_SUMMARY="${SAVE_TASK_SUMMARY:-True}"
SAVE_EPISODE_SUMMARY="${SAVE_EPISODE_SUMMARY:-True}"
SAVE_VISUAL_VIDEO="${SAVE_VISUAL_VIDEO:-True}"
JUDGE_DEVICE="${JUDGE_DEVICE:-auto}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-256}"
IGNORE_BOUNDARY_SEC="${IGNORE_BOUNDARY_SEC:-0.0}"
VISUAL_VIDEO_FPS="${VISUAL_VIDEO_FPS:-2}"
POLICY_BACKEND="${POLICY_BACKEND:-local}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-10097}"
POLICY_API_KEY="${POLICY_API_KEY:-}"
POLICY_PING_INTERVAL="${POLICY_PING_INTERVAL:-}"
POLICY_PING_TIMEOUT="${POLICY_PING_TIMEOUT:-}"
POLICY_MAX_NEW_TOKENS="${POLICY_MAX_NEW_TOKENS:-256}"

##Inference Harness Engineering.
#1.coarse abstract goals
#USE_DETAILED_INSTRUCTION=False,USE_SUBTASK_LIST=False
#2.detailed procedural descriptions
#USE_DETAILED_INSTRUCTION=True,USE_SUBTASK_LIST=False
#3.explicit subtask lists
#USE_DETAILED_INSTRUCTION=False,USE_SUBTASK_LIST=True
USE_DETAILED_INSTRUCTION="${USE_DETAILED_INSTRUCTION:-False}" 
USE_SUBTASK_LIST="${USE_SUBTASK_LIST:-True}"


if [ -z "${MODEL_NAME_OR_PATH}" ]; then
  echo "[ERROR] MODEL_NAME_OR_PATH must be set to a local checkpoint/model path or remote model label."
  exit 1
fi
if [ "${POLICY_BACKEND}" = "local" ] && [ -z "${BASE_MODEL_NAME_OR_PATH}" ]; then
  echo "[ERROR] BASE_MODEL_NAME_OR_PATH or BASE_MODEL must be set when POLICY_BACKEND=local."
  exit 1
fi
if [ "${ENABLE_CLOSED_LOOP_EVAL}" = "True" ] && [ -z "${JUDGE_MODEL_PATH}" ]; then
  echo "[ERROR] JUDGE_MODEL_PATH or JUDGE_MODEL must be set when ENABLE_CLOSED_LOOP_EVAL=True."
  exit 1
fi

##################################################
# output setting
##################################################
MODEL_NAME="$(basename "${MODEL_NAME_OR_PATH%/}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp/cortex/eval_results/sys2_episode_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${DATASET_TAG}/${TASK_NAME}/${MODEL_NAME}}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/predictions.jsonl}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

exec > >(tee -a "${OUTPUT_DIR}/job.out") 2> >(tee -a "${OUTPUT_DIR}/job.err" >&2)

echo "[INFO] model_name_or_path: ${MODEL_NAME_OR_PATH}"
echo "[INFO] base_model_name_or_path: ${BASE_MODEL_NAME_OR_PATH}"
echo "[INFO] dataset_tag: ${DATASET_TAG}"
echo "[INFO] eval_dataset_config: ${EVAL_DATASET_CONFIG}"
echo "[INFO] config_dataset_name: ${CONFIG_DATASET_NAME}"
echo "[INFO] task_name: ${TASK_NAME}"
echo "[INFO] dataset_path: ${DATASET_PATH}"
echo "[INFO] video_keys: ${VIDEO_KEYS}"
echo "[INFO] output_path: ${OUTPUT_PATH}"
echo "[INFO] nproc_per_node: ${NPROC_PER_NODE}"
echo "[INFO] policy_backend: ${POLICY_BACKEND}"
echo "[INFO] policy_host: ${POLICY_HOST}"
echo "[INFO] policy_port: ${POLICY_PORT}"

##################################################
# run
##################################################
INFER_ARGS=(
  --dataset_path "${DATASET_PATH}"
  --video_keys "${VIDEO_KEYS}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --base_model_name_or_path "${BASE_MODEL_NAME_OR_PATH}"
  --processor_name_or_path "${PROCESSOR_NAME_OR_PATH}"
  --output_path "${OUTPUT_PATH}"
  --sample_interval "${SAMPLE_INTERVAL}"
  --sample_interval_sec "${SAMPLE_INTERVAL_SEC}"
  --max_samples_per_episode "${MAX_SAMPLES_PER_EPISODE}"
  --max_episodes "${MAX_EPISODES}"
  --episode_indices "${EPISODE_INDICES}"
  --seed "${SEED}"
  --num_eval_episodes_per_task "${NUM_EVAL_EPISODES_PER_TASK}"
  --enable_closed_loop_eval "${ENABLE_CLOSED_LOOP_EVAL}"
  --save_task_summary "${SAVE_TASK_SUMMARY}"
  --save_episode_summary "${SAVE_EPISODE_SUMMARY}"
  --save_visual_video "${SAVE_VISUAL_VIDEO}"
  --judge_model_path "${JUDGE_MODEL_PATH}"
  --judge_device "${JUDGE_DEVICE}"
  --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}"
  --max_new_tokens "${POLICY_MAX_NEW_TOKENS}"
  --use_detailed_instruction "${USE_DETAILED_INSTRUCTION}"
  --use_subtask_list_instruction "${USE_SUBTASK_LIST}"
  --ignore_boundary_sec "${IGNORE_BOUNDARY_SEC}"
  --transition_tail_sec "${TRANSITION_TAIL_SEC}"
  --transition_head_sec "${TRANSITION_HEAD_SEC}"
  --last_tail_sec "${LAST_TAIL_SEC}"
  --visual_video_fps "${VISUAL_VIDEO_FPS}"
  --policy_backend "${POLICY_BACKEND}"
  --policy_host "${POLICY_HOST}"
  --policy_port "${POLICY_PORT}"
)

if [ -n "${POLICY_API_KEY}" ]; then
  INFER_ARGS+=(--policy_api_key "${POLICY_API_KEY}")
fi
if [ -n "${POLICY_PING_INTERVAL}" ]; then
  INFER_ARGS+=(--policy_ping_interval "${POLICY_PING_INTERVAL}")
fi
if [ -n "${POLICY_PING_TIMEOUT}" ]; then
  INFER_ARGS+=(--policy_ping_timeout "${POLICY_PING_TIMEOUT}")
fi
if [ "${POLICY_BACKEND}" = "local" ]; then
  INFER_ARGS+=(--bf16)
fi

echo "[INFO] launch: srun --nodes=1 --ntasks=1 torchrun --nproc_per_node ${NPROC_PER_NODE}"
srun --nodes=1 --ntasks=1 torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  cortex/inference/episode_level_eval.py \
  "${INFER_ARGS[@]}"
