# Subtask Annotation

This guide describes the Cortex subtask annotation pipeline. The pipeline has
three stages:

1. Use a VLM to generate initial subtask annotations.
2. Correct the first few episodes with the browser annotator.
3. Use the corrected episodes as few-shot examples to refine boundaries for
   the remaining episodes with visual, action, and state features.

Follow [installation.md](installation.md) before running the commands below.

## Expected Data Layout

The few-shot refinement stage expects a LeRobot-style dataset:

```text
/path/to/dataset/
  meta/
    episodes.jsonl
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
  videos/
    chunk-000/
      observation.images.cam_high/
        episode_000000.mp4
        episode_000001.mp4
      observation.images.cam_left_wrist/
        episode_000000.mp4
        episode_000001.mp4
      observation.images.cam_right_wrist/
        episode_000000.mp4
        episode_000001.mp4
```

The manual web annotator only needs one video folder containing
`episode_*.mp4`. The few-shot boundary refinement stage reads both the parquet
state/action features and the video folders.

## Output Format

All stages write one JSON file per episode:

```json
{
  "episode_index": 0,
  "tasks": ["wash the beaker"],
  "length": 1234,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 120,
      "action_text": "Pick up the beaker.",
      "skill": "Pick"
    }
  ]
}
```

`start_frame` is inclusive and `end_frame` is exclusive. Segments must be
contiguous, and the last segment must end at `length`.

## Stage 1: VLM Initial Annotation

`vlm_annotation.py` calls an OpenAI-compatible VLM endpoint and converts the
model output into the JSON format above.

```bash
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL=your_vision_model
```

Annotate one episode:

```bash
python -m cortex.annotation.vlm_annotation \
  --video_path /path/to/dataset/videos/chunk-000/observation.images.cam_high/episode_000000.mp4 \
  --output_path annotations/manual/episode_000000.json \
  --sample_id episode_000000 \
  --task_instruction "wash the beaker" \
  --max_sample_frames 200 \
  --ceph_video_view_mode head_wrists \
  --check_frames_dir exp/cortex/annotation/check_frames/episode_000000 \
  --subtask_clips_dir exp/cortex/annotation/subtask_clips/episode_000000
```

Annotate the first few episodes before manual correction:

```bash
for idx in 0 1 2; do
  episode=$(printf "episode_%06d" "${idx}")
  python -m cortex.annotation.vlm_annotation \
    --video_path "/path/to/dataset/videos/chunk-000/observation.images.cam_high/${episode}.mp4" \
    --output_path "annotations/manual/${episode}.json" \
    --sample_id "${episode}" \
    --task_instruction "wash the beaker" \
    --max_sample_frames 200 \
    --ceph_video_view_mode head_wrists \
    --check_frames_dir "exp/cortex/annotation/check_frames/${episode}" \
    --subtask_clips_dir "exp/cortex/annotation/subtask_clips/${episode}"
done
```

You can also pass `--base_url`, `--model`, and `--api_key` directly instead of
using environment variables.

For the bundled 10-episode `dump_bin_bigbin` demo, annotate the first few seed
episodes with:

```bash
OPENAI_API_KEY=your_api_key \
OPENAI_MODEL=your_vision_model \
bash scripts/run_scripts/run_vlm_annotation.sh 0 1 2
```

The script defaults to `assets/dump_bin_bigbin`, uses
`observation.images.cam_high` as the head view, stitches the matching left and
right wrist videos when available, and writes outputs to
`annotations/dump_bin_bigbin/manual/`. Override `DATASET_ROOT`,
`CAMERA`, `VIDEO_VIEW_MODE`, `OUTPUT_DIR`, `OPENAI_BASE_URL`, or
`MAX_SAMPLE_FRAMES` as needed. The example script samples at most 32 stitched
multiview frames by default and allows up to 300 seconds for a model response.
Use `OPENAI_TIMEOUT_SECONDS` to adjust the request timeout for your endpoint.

## Stage 2: Manual Correction

Start the annotation server on the same video folder used for VLM initial
annotation:

```bash
bash scripts/run_scripts/run_manual_annotation_server.sh
```

Open `http://127.0.0.1:8765` and correct the first few VLM-generated episodes.
The server reads and writes `annotations/manual/episode_XXXXXX.json`.

Correct boundaries and subtask text for a representative set of episodes, then
click `Save JSON`. The server maintains contiguous segments and preserves the
per-episode JSON format. For remote use, keep the default loopback host and
forward the port through SSH instead of exposing the unauthenticated server.

## Stage 3: Few-Shot Boundary Refinement

Run boundary refinement after manually correcting several seed episodes:

```bash
python -m cortex.annotation.dynamic_matching \
  --dataset-root /path/to/dataset \
  --annotation-dir annotations/manual \
  --manual-episodes 0 1 2 \
  --protect-episodes 0 1 2 \
  --episodes 3 4 5 6 7 \
  --summary-dir exp/cortex/annotation/summaries \
  --visualization-dir exp/cortex/annotation/visualizations \
  --cameras observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist \
  --preview-camera observation.images.cam_high
```

For the bundled `dump_bin_bigbin` demo, episodes `0-4` can be used as
protected manual seeds and all remaining episodes can be annotated in one run:

```bash
conda activate InternVLA
bash scripts/run_scripts/run_dynamic_matching.sh
```

The runner uses `observation.state` and `action` from the example parquet
files, all three camera views, and `--max-new 0`. Set `DRY_RUN=1` to validate
template selection without writing annotations. Manual seeds may contain
different numbers of subtasks; the matcher fits separate templates and selects
the closest template for each target episode.

Important arguments:

- `--manual-episodes`: corrected examples used to fit the boundary model.
- `--protect-episodes`: episodes that must not be overwritten.
- `--episodes`: explicit episode indices to refine. Omit this to select
  unannotated episodes automatically.
- `--max-new`: when `--episodes` is omitted, refine at most this many new
  episodes. Use `0` for all remaining episodes.
- `--cameras`: video folders under `videos/chunk-000/` used for visual
  features. Pass a single camera if your dataset has only one view.
- `--state-columns`: parquet columns used for state/action features. Override
  this if your LeRobot schema uses different column names.
- `--no-write-visualizations`: skip JPEG visualization output.
- `--dry-run`: run the selection and inference path without writing annotation
  files.

The refinement script groups manual episodes by their number of subtasks and
fits one ordered boundary model per template. For every target episode, it
matches state, visual, and trajectory-length descriptors against the manual
examples, selects the corresponding variable-length template, and then uses:

- visual features from sampled camera frames,
- state/action features from parquet columns,
- duration priors from manual segment lengths,
- boundary priors from manual segment start ratios.

It solves the selected template's boundaries with dynamic programming and
snaps each boundary toward nearby low-motion frames. Manual episodes with two
and four subtasks can therefore be passed together. Outputs are written back to
`--annotation-dir`, and summaries are written to:

```text
exp/cortex/annotation/summaries/
  boundary_refinement_summary.json
  boundary_refinement_summary.csv
  _cache/
```

Visualization images are written to `--visualization-dir` and can be inspected
after each run.

## Using the Annotations

For System-2 evaluation packaging, keep annotation paths relative inside
`cortex/inference/config/sys2_subtask_val.json` so the repository remains
portable. The released validation annotations in this repository follow the
same convention under `annotations/`.

For a new dataset, keep raw per-episode annotations in a separate directory
such as `annotations/manual/`, then convert or package them into the JSONL
format expected by your evaluation dataset config.

## Troubleshooting

- If the manual server starts but the episode list is empty, check that
  `--video-dir` points directly to the camera folder containing `episode_*.mp4`.
- If refinement reports missing state/action columns, inspect one parquet file
  and pass the matching column names with `--state-columns`.
- If refinement reports missing videos, make sure each camera passed to
  `--cameras` exists under `videos/chunk-000/`.
- Manual examples with different numbers of subtasks are supported. Correct
  the boundaries and semantic fields for each template before refinement.
- Object-storage batch annotation is optional. For local annotation, do not set
  `--ceph_path`. If your environment uses object storage, set `PETREL_CONF` and
  pass the storage root explicitly.
