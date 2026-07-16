# Installation

This repository keeps only the Cortex System-2 evaluation, optional WebSocket serving, and subtask visualization paths.

## Conda Environment

```bash
conda create -n cortex python=3.10 gcc_linux-64 gxx_linux-64 -c conda-forge -y
conda activate cortex
```

## Dependencies

```bash
conda install -c conda-forge av==15.0.0 sentencepiece==0.2.1 tiktoken
pip install -r requirements.txt
pip install -e .
```

If your datasets are stored in an object store such as S3, Ceph, or Petrel,
install the storage backend required by your environment and make sure
`mmengine.fileio` can read the configured `data_path` and `annotation_path`
URIs. For local files, no extra storage backend is required.

Install FlashAttention 2 for local Qwen-VL inference:

```bash
pip install flash-attn --no-build-isolation
```

If building from source fails, install a wheel matching the local CUDA/PyTorch/Python versions. Example for CUDA 12, PyTorch 2.6, Python 3.10:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install ./flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -c "import flash_attn; print(f'version: {flash_attn.__version__}')"
```

## Quick Checks

```bash
python -m py_compile cortex/inference/step_level_eval.py
python -m py_compile cortex/inference/episode_level_eval.py
python -m py_compile cortex/annotation/vlm_annotation.py
python -m py_compile cortex/annotation/annotator_server.py
python -m py_compile cortex/annotation/dynamic_matching.py
python -m py_compile visualize_subtask_sampling.py
bash -n scripts/run_scripts/run_vlm_annotation.sh
bash -n scripts/run_scripts/run_manual_annotation_server.sh
bash -n scripts/run_scripts/run_dynamic_matching.sh
bash -n scripts/run_scripts/step_level.sh
bash -n scripts/run_scripts/episode_level.sh
bash -n scripts/run_scripts/run_subtask_visualization.sh
```
