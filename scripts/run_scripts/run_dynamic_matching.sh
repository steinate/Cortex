#!/bin/bash
set -euo pipefail

dataset_root="${DATASET_ROOT:-assets/dump_bin_bigbin}"
annotation_dir="${ANNOTATION_DIR:-annotations/dump_bin_bigbin/manual}"
summary_dir="${SUMMARY_DIR:-exp/cortex/annotation/dump_bin_bigbin/summaries}"
visualization_dir="${VISUALIZATION_DIR:-exp/cortex/annotation/dump_bin_bigbin/visualizations}"

read -r -a manual_episodes <<< "${MANUAL_EPISODES:-0 1 2 3 4}"
read -r -a protect_episodes <<< "${PROTECT_EPISODES:-${MANUAL_EPISODES:-0 1 2 3 4}}"
read -r -a cameras <<< "${CAMERAS:-observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist}"
read -r -a state_columns <<< "${STATE_COLUMNS:-observation.state action}"

cmd=(
  python -m cortex.annotation.dynamic_matching
  --dataset-root "${dataset_root}"
  --annotation-dir "${annotation_dir}"
  --manual-episodes "${manual_episodes[@]}"
  --protect-episodes "${protect_episodes[@]}"
  --max-new 0
  --state-columns "${state_columns[@]}"
  --cameras "${cameras[@]}"
  --preview-camera "${cameras[0]}"
  --summary-dir "${summary_dir}"
  --visualization-dir "${visualization_dir}"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  cmd+=(--dry-run --no-write-visualizations)
fi
if [ "${OVERWRITE:-0}" = "1" ]; then
  cmd+=(--overwrite --no-skip-existing)
fi

echo "[INFO] dataset_root: ${dataset_root}"
echo "[INFO] annotation_dir: ${annotation_dir}"
echo "[INFO] manual_episodes: ${manual_episodes[*]}"
echo "[INFO] protected_episodes: ${protect_episodes[*]}"
echo "[INFO] cameras: ${cameras[*]}"
echo "[INFO] state_columns: ${state_columns[*]}"
echo "[INFO] dry_run: ${DRY_RUN:-0}"
echo "[INFO] overwrite: ${OVERWRITE:-0}"

"${cmd[@]}"
