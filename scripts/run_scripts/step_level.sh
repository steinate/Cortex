#!/bin/bash
#SBATCH --job-name=step_level_eval
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
# Basic local-checkpoint eval. By default this runs all datasets and all three
# step-level task types: spatial, counting, and long.
#   BASE_MODEL=/path/to/base-vlm JUDGE_MODEL=/path/to/judge-vlm \
#   sbatch scripts/run_scripts/step_level.sh /path/to/checkpoint-xxxx
#
# Run one task type across all datasets:
#   EVAL_TASK_TYPE=spatial BASE_MODEL=/path/to/base-vlm JUDGE_MODEL=/path/to/judge-vlm \
#   sbatch scripts/run_scripts/step_level.sh /path/to/checkpoint-xxxx
#
# Run one task type on one dataset:
#   EVAL_TASK_TYPE=counting EVAL_DATASET_TAG=behavior \
#   BASE_MODEL=/path/to/base-vlm JUDGE_MODEL=/path/to/judge-vlm \
#   sbatch scripts/run_scripts/step_level.sh /path/to/checkpoint-xxxx
#
# Supported filters:
#   EVAL_TASK_TYPE=all|spatial|counting|long
#   EVAL_DATASET_TAG=all|galaxea|agibot|behavior
#
# Use a custom eval config. The config may contain all slices; the script filters
# it into a temporary JSON according to EVAL_TASK_TYPE/EVAL_DATASET_TAG.
#   EVAL_DATASET_CONFIG=/path/to/sys2_subtask_val.json EVAL_TASK_TYPE=long \
#   BASE_MODEL=/path/to/base-vlm JUDGE_MODEL=/path/to/judge-vlm \
#   sbatch scripts/run_scripts/step_level.sh /path/to/checkpoint-xxxx
#
# Evaluate with a pre-started remote policy server:
#   POLICY_BACKEND=websocket POLICY_HOST=127.0.0.1 POLICY_PORT=10094 \
#   BASE_MODEL=/path/to/base-vlm JUDGE_MODEL=/path/to/judge-vlm \
#   sbatch scripts/run_scripts/step_level.sh gemini

CHECKPOINT_DIR="${1:-${CHECKPOINT_DIR:-}}"
POLICY_BACKEND_EARLY="${POLICY_BACKEND:-local}"
if [ -z "${CHECKPOINT_DIR}" ]; then
  echo "Usage: sbatch $0 <checkpoint_dir_or_label>"
  exit 1
fi

##################################################
# dist setting (single node)
##################################################
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-${NPROC_PER_NODE}}"
export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
export MASTER_PORT="${MASTER_PORT:-$((RANDOM % 101 + 20000))}"

##################################################
# dataset / model setting
##################################################
base_model="${BASE_MODEL:-/path/to/Qwen3-VL-8B-Instruct}"
judge_model="${JUDGE_MODEL:-/path/to/Qwen3.5-9B/}"
eval_dataset_name="${EVAL_DATASET:-}"
eval_dataset_config="${EVAL_DATASET_CONFIG:-cortex/inference/config/sys2_subtask_val.json}"
eval_task_type="${EVAL_TASK_TYPE:-all}"
eval_dataset_tag="${EVAL_DATASET_TAG:-${DATASET_TAG:-all}}"

if [ -z "${base_model}" ]; then
  echo "[ERROR] BASE_MODEL must be set to the base VLM path/name."
  exit 1
fi
if [ -z "${eval_dataset_name}" ] && [ -z "${eval_dataset_config}" ]; then
  echo "[ERROR] EVAL_DATASET_CONFIG must be set, or EVAL_DATASET must be an inline/path dataset config."
  exit 1
fi
if [ -z "${judge_model}" ]; then
  echo "[ERROR] JUDGE_MODEL or auto_eval_judge_model must be set for judge scoring."
  exit 1
fi

if [ -n "${eval_dataset_config}" ] && [ -f "${eval_dataset_config}" ]; then
  filtered_eval_dataset_config="${eval_dataset_config}"
  if [ "${eval_task_type}" != "all" ] || [ "${eval_dataset_tag}" != "all" ]; then
    filtered_eval_dataset_config="$(mktemp /tmp/sys2_subtask_eval.XXXXXX.json)"
    python - "${eval_dataset_config}" "${filtered_eval_dataset_config}" "${eval_task_type}" "${eval_dataset_tag}" <<'PYCFG'
import json
import sys

src, dst, task_type, dataset_tag = sys.argv[1:5]
task_type = task_type.strip().lower()
dataset_tag = dataset_tag.strip().lower()
with open(src, "r", encoding="utf-8") as f:
    configs = json.load(f)
if isinstance(configs, dict):
    configs = [configs]

selected = []
for cfg in configs:
    name = str(cfg.get("dataset_name", "")).lower()
    cfg_task_type = str(cfg.get("eval_task_type", "")).lower()
    keep = True
    if task_type != "all":
        keep = keep and (cfg_task_type == task_type or name.endswith(f"_{task_type}"))
    if dataset_tag != "all":
        keep = keep and (name == dataset_tag or name.startswith(f"{dataset_tag}_"))
    if keep:
        selected.append(cfg)
if not selected:
    raise SystemExit(
        f"No eval dataset configs matched EVAL_TASK_TYPE={task_type!r} "
        f"and EVAL_DATASET_TAG={dataset_tag!r} in {src}"
    )
with open(dst, "w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)
    f.write("\n")
PYCFG
    eval_dataset_config="${filtered_eval_dataset_config}"
  fi
fi

per_device_eval_batch_size="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
max_samples="${MAX_SAMPLES:-1000}"
max_videos="${MAX_VIDEOS:-0}"
seed="${SEED:-42}"

##Inference Harness Engineering.
#1.coarse abstract goals
#USE_DETAILED_INSTRUCTION=False,USE_SUBTASK_LIST=False
#2.detailed procedural descriptions
#USE_DETAILED_INSTRUCTION=True,USE_SUBTASK_LIST=False
#3.explicit subtask lists
#USE_DETAILED_INSTRUCTION=False,USE_SUBTASK_LIST=True
use_detailed_instruction="${USE_DETAILED_INSTRUCTION:-True}"
use_subtask_list="${USE_SUBTASK_LIST:-False}"
max_samples_mode="per_dataset_equal"

# mostly fixed knobs
policy_device="${POLICY_DEVICE:-auto}"
policy_backend="${POLICY_BACKEND:-local}"
policy_host="${POLICY_HOST:-127.0.0.1}"
policy_port="${POLICY_PORT:-10097}"
policy_api_key="${POLICY_API_KEY:-}"
policy_ping_interval="${POLICY_PING_INTERVAL:-}"
policy_ping_timeout="${POLICY_PING_TIMEOUT:-}"
judge_device="${JUDGE_DEVICE:-auto}"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-8}"
policy_max_new_tokens="${POLICY_MAX_NEW_TOKENS:-256}"
judge_max_new_tokens="${JUDGE_MAX_NEW_TOKENS:-256}"

##################################################
# output setting
##################################################
ckpt_name="$(basename "${CHECKPOINT_DIR}")"
out_root="${OUTPUT_ROOT:-exp/cortex/eval_results/sys2_step_eval}"
eval_output_group="${eval_dataset_tag}_${eval_task_type}"
output_dir="${OUTPUT_DIR:-${out_root}/${ckpt_name}/${eval_output_group}}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# log
exec > >(tee -a "${output_dir}/job.out") 2> >(tee -a "${output_dir}/job.err" >&2)

echo "[INFO] checkpoint_dir: ${CHECKPOINT_DIR}"
echo "[INFO] output_dir: ${output_dir}"
echo "[INFO] nproc_per_node: ${NPROC_PER_NODE}"
echo "[INFO] eval_dataset_config: ${eval_dataset_config}"
echo "[INFO] eval_task_type: ${eval_task_type}"
echo "[INFO] eval_dataset_tag: ${eval_dataset_tag}"
echo "[INFO] use_detailed_instruction: ${use_detailed_instruction}"
echo "[INFO] policy_backend: ${policy_backend}"
echo "[INFO] policy_host: ${policy_host}"
echo "[INFO] policy_port: ${policy_port}"

##################################################
# run
##################################################
EVAL_ARGS=(
  --model_name_or_path "${base_model}"
  --checkpoint_path "${CHECKPOINT_DIR}"
  --eval_dataset "${eval_dataset_name}"
  --output_dir "${output_dir}"
  --per_device_eval_batch_size "${per_device_eval_batch_size}"
  --dataloader_num_workers "${dataloader_num_workers}"
  --enable_checkpoint_auto_subtask_eval True
  --enable_neighbor_subtask_compare True
  --judge_model_path "${judge_model}"
  --policy_device "${policy_device}"
  --policy_backend "${policy_backend}"
  --policy_host "${policy_host}"
  --policy_port "${policy_port}"
  --judge_device "${judge_device}"
  --max_new_tokens "${policy_max_new_tokens}"
  --judge_max_new_tokens "${judge_max_new_tokens}"
  --max_samples_mode "${max_samples_mode}"
  --max_samples "${max_samples}"
  --max_videos "${max_videos}"
  --seed "${seed}"
  --use_detailed_instruction "${use_detailed_instruction}"
  --use_subtask_list "${use_subtask_list}"
)

if [ -n "${eval_dataset_config}" ]; then
  EVAL_ARGS+=(--eval_dataset_config "${eval_dataset_config}")
fi

if [ "${policy_backend}" = "local" ]; then
  EVAL_ARGS+=(--bf16)
fi
if [ -n "${policy_api_key}" ]; then
  EVAL_ARGS+=(--policy_api_key "${policy_api_key}")
fi
if [ -n "${policy_ping_interval}" ]; then
  EVAL_ARGS+=(--policy_ping_interval "${policy_ping_interval}")
fi
if [ -n "${policy_ping_timeout}" ]; then
  EVAL_ARGS+=(--policy_ping_timeout "${policy_ping_timeout}")
fi

echo "[INFO] launch: srun --nodes=1 --ntasks=1 torchrun --nproc_per_node ${NPROC_PER_NODE}"
srun --nodes=1 --ntasks=1 torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes 1 \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  cortex/inference/step_level_eval.py \
  "${EVAL_ARGS[@]}"
