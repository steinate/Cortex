<div align="center">

# Cortex

### A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation

[![Project Page](https://img.shields.io/badge/Project-Page-2f80ed)](https://steinate.github.io/cortex.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-2607.05377-b31b1b)](https://arxiv.org/abs/2607.05377)
[![Code](https://img.shields.io/badge/GitHub-Cortex-181717?logo=github)](https://github.com/steinate/Cortex)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Steinate%2FCortex-ffcc4d?logo=huggingface)](https://huggingface.co/Steinate/Cortex)
[![Video](https://img.shields.io/badge/Demo-Video-8e44ad)](#cortex-in-action)

<p align="center">
  <video src="https://github.com/user-attachments/assets/8126cadc-b9ca-45f5-bdb8-e80527853faf" controls muted></video>
</p>

</div>

**Cortex (aka InternVLA-M1.5)** is a bidirectionally aligned embodied agent framework for long-horizon manipulation. The core idea is to let a high-level VLM act as a cognitive orchestrator that tracks progress, updates semantic memory, and emits executable subtasks, while a low-level VLA focuses on reactive physical execution.


## Highlights

<div align="center">
<img src="assets/architecture.png" width="95%" alt="Cortex architecture">
<br>
<sub><b>Architecture.</b> System-2 observes the scene and language memory, then streams executable subtasks to System-1.</sub>
</div>

**A bidirectionally aligned agent framework.** Cortex aligns high-level cognitive planning with low-level manipulation execution through a shared subtask interface.

**A standardized long-horizon subtask space.** The paper standardizes manipulation behavior into 32 canonical skill primitives and augments subtasks with spatial, numerical, and object-attribute grounding.

**Event-balanced System-2 training.** Instead of only sampling uniformly from trajectories, Cortex emphasizes transition regions where the planner must verify completion and update memory.

**Closed-loop long-horizon deployment.** In real-world chemistry-style tasks, Cortex uses memory and visual verification to preserve task order, avoid premature switching, and recover from local execution ambiguity.

## Result

### Performance comparison with VLM.

Step-level:

| Model | Avg. Total | Spatial Total | Long-horizon Total | Counting Total |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B-Instruct | 6.739 | 6.424 | 6.775 | 7.018 |
| GPT-5 | 6.268 | 6.422 | 6.163 | 6.220 |
| Gemini-3.1-Pro | 6.925 | 6.697 | 6.920 | 7.159 |
| Cortex | 8.318 | 8.053 | 8.160 | 8.741 |

Episode-level:

| Model | Avg. Total | Spatial Total | Long-horizon Total | Counting Total |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B-Instruct | 6.292 | 6.468 | 6.021 | 6.388 |
| GPT-5 | 7.231 | 7.321 | 6.996 | 7.376 |
| Gemini-3.1-Pro | 6.860 | 6.929 | 6.644 | 7.006 |
| Cortex | 7.810 | 7.587 | 7.380 | 8.464 |

### Performance comparison with VLA.

| Method | LIBERO-long | RoboTwin 2.0 | Real-world |
| --- | ---: | ---: | ---: |
| pi05 | 92.4 | 82.7 | 22.1 |
| Cortex | 95.5 | 86.8 | 76.8 |

<div align="center">
<img src="assets/robotwin.png" width="70%" alt="RoboTwin performance">
<br>
<sub><b>Simulation.</b> Cortex improves long-horizon task execution by providing explicit subtask routing and progress verification.</sub>
</div>

## Evaluation
Follow the environment setup in [docs/installation.md](docs/installation.md) before running evaluation. The examples below use the released System-2 checkpoint hosted on [Hugging Face](https://huggingface.co/Steinate/Cortex) as `CHECKPOINT_DIR`:

```bash
CHECKPOINT_DIR=Steinate/Cortex
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct
JUDGE_MODEL=/path/to/Qwen3.5-9B
EVAL_DATASET_CONFIG=cortex/inference/config/sys2_subtask_val.json
```
The System-2 evaluation is organized around spatial grounding, long-horizon logical consistency, and object counting accuracy. The full system is evaluated on long-horizon simulation suites and zero-shot real-world manipulation tasks.

### Step-Level Evaluation

Frame-level evaluation measures subtask and memory prediction accuracy with ground-truth memory. The default command evaluates all supported datasets and all three task types: `spatial`, `counting`, and `long`.

```bash
BASE_MODEL="${BASE_MODEL}" \
JUDGE_MODEL="${JUDGE_MODEL}" \
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG}" \
OUTPUT_ROOT=exp/cortex/eval/step \
MAX_SAMPLES=3000 \
USE_DETAILED_INSTRUCTION=False \
USE_SUBTASK_LIST=True \
sbatch scripts/run_scripts/step_level.sh "${CHECKPOINT_DIR}"
```

Outputs are saved to `exp/cortex/eval/step/<checkpoint_name>/<dataset_tag>_<task_type>/`. To reproduce one slice, set `EVAL_TASK_TYPE` and `EVAL_DATASET_TAG`:

```bash
EVAL_TASK_TYPE=spatial \
EVAL_DATASET_TAG=galaxea \
BASE_MODEL="${BASE_MODEL}" \
JUDGE_MODEL="${JUDGE_MODEL}" \
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG}" \
OUTPUT_ROOT=exp/cortex/eval/step \
MAX_SAMPLES=3000 \
USE_DETAILED_INSTRUCTION=False \
USE_SUBTASK_LIST=True \
sbatch scripts/run_scripts/step_level.sh "${CHECKPOINT_DIR}"
```

### Episode-Level Evaluation

Closed-loop episode evaluation measures semantic drift when the planner reads its own previous memory. Choose a `DATASET_TAG` and pass one task name from that dataset as the final script argument.

```bash
MODEL_NAME_OR_PATH="${CHECKPOINT_DIR}" \
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL}" \
PROCESSOR_NAME_OR_PATH="${BASE_MODEL}" \
JUDGE_MODEL_PATH="${JUDGE_MODEL}" \
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG}" \
OUTPUT_ROOT=exp/cortex/eval/episode \
DATASET_TAG=galaxea \
NUM_EVAL_EPISODES_PER_TASK=10 \
USE_DETAILED_INSTRUCTION=False \
USE_SUBTASK_LIST=True \
sbatch scripts/run_scripts/episode_level.sh Adjust_The_Air_Conditioner_Temperature_20250711_006
```

Outputs are saved to `exp/cortex/eval/episode/<dataset_tag>/<task_name>/<model_name>/`. Set `DATASET_TAG` to `galaxea`, `agibot`, or `behavior`; set `MAX_EPISODES=1` for a quick smoke test.

```bash
# Optional: evaluate a pre-started WebSocket policy instead of a local checkpoint.
POLICY_BACKEND=websocket \
POLICY_HOST=127.0.0.1 \
POLICY_PORT=10094 \
sbatch scripts/run_scripts/step_level.sh gpt-5
```

### Visualization

```bash
bash scripts/run_scripts/run_subtask_visualization.sh
```

<p align="center">
  <video src="https://github.com/user-attachments/assets/4bf6487a-44ec-4b30-98fa-26355ad55642" controls muted></video>
</p>

## Real-world Long-horizon Deployment

The real-world experiments emphasize capabilities that are difficult to obtain from monolithic end-to-end policies: preserving procedural order, verifying completion before switching, using memory to disambiguate similar visual stages, and adapting to local execution uncertainty.

<div align="center">
<img src="assets/real_world.png" width="92%" alt="Real-world Cortex deployment">
<br>
<sub><b>Real-world deployment.</b> Zero-shot multi-stage chemistry task with fine-grained subtask prediction and memory tracking.</sub>
</div>

## TODO

- [x] Release System-2 evaluation code.
- [ ] Release System-1/2 evaluation code for LIBERO and RoboTwin.
- [ ] Release the subtask dataset.
- [ ] Release System-2 training code.

## Citation

If you find this project useful, please cite:

```bibtex
@misc{peng2026cortex,
  title={Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation},
  author={Jiaqi Peng and Xiqian Yu and Delin Feng and Yuqiang Yang and Wenzhe Cai and Jing Xiong and Ganlin Yang and Jinliang Zheng and Jiafei Cao and Xueyuan Wei and Jiangmiao Pang and Yuan Shen and Tai Wang},
  year={2026},
  eprint={2607.05377},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2607.05377}
}
```
