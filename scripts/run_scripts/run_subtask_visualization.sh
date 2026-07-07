#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="${REPO_ROOT}/visualize_subtask_sampling.py"

# Behavior
python "${PY_SCRIPT}" \
    --dataset-name behavior_subtask_train \
    --task-id "${BEHAVIOR_TASK_ID:-task-0002}" \
    --episode-index "${BEHAVIOR_EPISODE_INDEX:-20010}" \
    --font-size 0 \
    --text-position top-left \
    --resize-max-width 960 \
    --crf 24 \
    --preset veryfast \
    --output vis_behavior.mp4
# Agibot
python "${PY_SCRIPT}" \
    --dataset-name agibot_subtask_train \
    --task-id "${AGIBOT_TASK_ID:-task_327}" \
    --episode-index "${AGIBOT_EPISODE_INDEX:-1}" \
    --font-size 0 \
    --text-position top-left \
    --resize-max-width 960 \
    --crf 24 \
    --preset veryfast \
    --output vis_agibot.mp4

# Galaxea
python "${PY_SCRIPT}" \
    --dataset-name galaxea_subtask_train \
    --task-id "${GALAXEA_TASK_ID:-Adjust_The_Air_Conditioner_Temperature_20250711_006}" \
    --episode-index "${GALAXEA_EPISODE_INDEX:-1}" \
    --font-size 0 \
    --text-position top-left \
    --resize-max-width 960 \
    --crf 24 \
    --preset veryfast \
    --output vis_galaxea.mp4
