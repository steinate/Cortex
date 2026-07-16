# Datasets

This document describes the raw datasets and Cortex annotations used by the
data loader. Download the raw dataset first, then download the corresponding
Cortex JSONL annotation files. The annotation paths in
`cortex/dataloader/qwenvl_llavajson/qwen_data_config.py` are relative to the
repository root and expect the JSONL files under `annotations/`.

## Prerequisites

Install the Hugging Face CLI and log in before downloading gated datasets:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

For gated datasets, accept the provider's terms on the dataset page before
running the download command.

## Raw Datasets

| Dataset key | Raw dataset on Hugging Face | Suggested local directory | Notes |
| --- | --- | --- | --- |
| `agiworld` | [agibot-world/AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | `data/agiworld` | Gated. |
| `galaxea` | [OpenGalaxea/Galaxea-Open-World-Dataset](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) | `data/galaxea` | Gated. |
| `behavior` | [behavior-1k/2025-challenge-demos](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos) | `data/behavior` | Public. |
| `robocerebra` | [qiukingballball/RoboCerebra](https://huggingface.co/datasets/qiukingballball/RoboCerebra) | `data/robocerebra` | Public. Convert the released layout to the LeRobot layout used by the loader before training. |
| `agiworld26` | [agibot-world/AgiBotWorld2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) | `data/agiworld26` | Public. |

Download a raw dataset with the Hugging Face CLI. Replace the repository and
local directory as needed:

```bash
huggingface-cli download agibot-world/AgiBotWorld-Beta \
  --repo-type dataset \
  --local-dir data/agiworld
```

The loader expects a LeRobot-style layout with `data/`, `meta/`, and `videos/`
under the configured `data_path`. Set each `data_path` in
`qwen_data_config.py` to the location where the corresponding raw dataset was
downloaded or converted.

## Cortex Annotations

Download all released JSONL files from
[Steinate/Cortex](https://huggingface.co/datasets/Steinate/Cortex) before
training or evaluating with the data configuration:

```bash
huggingface-cli download Steinate/Cortex \
  --repo-type dataset \
  --include "*.jsonl" \
  --local-dir annotations
```

The download creates the following files directly under `annotations/`:

| Raw dataset key | Cortex annotation files |
| --- | --- |
| `agiworld` | `agibot_norm_mem_train.jsonl`, `agibot_norm_mem_val.jsonl` |
| `agiworld26` | `agibot26_norm_mem_train.jsonl`, `agibot26_norm_mem_val.jsonl` |
| `galaxea` | `galaxea_norm_mem_train.jsonl`, `galaxea_norm_mem_val.jsonl` |
| `behavior` | `behavior_norm_mem_train.jsonl`, `behavior_norm_mem_val.jsonl` |
| `robocerebra` | `robocerebra_norm_mem.jsonl` |
| `robotwin` | `robotwin_norm_mem_train.jsonl`, `robotwin_norm_mem_val.jsonl` |

`robotwin` annotations are released together with the Cortex annotations. Set
its `data_path` to a locally prepared RoboTwin LeRobot dataset before using the
`robotwin_subtask_*` entries.

## Data Configuration

The following keys are available in
`cortex/dataloader/qwenvl_llavajson/qwen_data_config.py`:

```text
agibot_subtask_train       agibot_subtask_val
agibot26_subtask_train     agibot26_subtask_val
galaxea_subtask_train      galaxea_subtask_val
behavior_subtask_train     behavior_subtask_val
robocerebra_subtask_train
robotwin_subtask_train     robotwin_subtask_val
```

The config ships with portable placeholders such as
`/path/to/datasets/agiworld`. Replace only `data_path` with your local raw-data
directory. Do not change the relative `annotation_path` values after placing
the released JSONL files in `annotations/`.
