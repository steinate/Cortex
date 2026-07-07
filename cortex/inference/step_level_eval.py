# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import bisect
import copy
import itertools
import json
import logging
import os
import random
import re
import sys
import time
from datetime import timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from cortex.inference.eval_subtask_dataset import EvalSubtaskDataset
from cortex.inference.eval_utils import IGNORE_INDEX, rank0_print  # noqa: E402
from cortex.inference.subtask_eval import (  # noqa: E402
    _extract_current_subtask_text as _shared_extract_current_subtask_text,
    _get_subtask_action_text as _shared_get_subtask_action_text,
    _load_judge_model as _shared_load_judge_model,
    _normalize_text as _shared_normalize_text,
    _score_subtask_with_existing_judge as _shared_score_subtask_with_existing_judge,
    run_checkpoint_auto_eval,
    run_neighbor_compare_from_detail_file,
)

local_rank = 0

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint path for inference model loading."},
    )


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    eval_dataset: str = field(default="")
    eval_dataset_config: str = field(
        default="",
        metadata={
            "help": (
                "Evaluation dataset config JSON. Use this for open-source eval to avoid "
                "depending on legacy qwen_data_config dataset names."
            )
        },
    )
    sample_interleave: int = field(default=8)
    transition_tail_sec: float = field(default=0.5)
    transition_head_sec: float = field(default=0.5)
    last_tail_sec: float = field(default=1.0)
    video_key: str = field(default="observation.images.head")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = field(default=2)


@dataclass
class InferenceArguments:
    output_dir: str = field(default="./exp/cortex/inference")
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=8192)
    per_device_eval_batch_size: int = field(default=1)
    dataloader_num_workers: int = field(default=8)
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)

    max_new_tokens: int = field(default=256)
    do_sample: bool = field(default=False)
    temperature: float = field(default=0.0)
    top_p: float = field(default=1.0)
    num_beams: int = field(default=1)
    repetition_penalty: float = field(default=1.0)

    attn_implementation: str = field(default="flash_attention_2")
    max_samples: int = field(default=0)
    max_samples_mode: str = field(default="global")
    log_every_steps: int = field(default=10)
    seed: int = field(default=42)

    enable_checkpoint_auto_subtask_eval: bool = field(default=True)
    checkpoint_root_dir: Optional[str] = field(default=None)
    monitor_new_checkpoints: bool = field(default=False)
    monitor_interval_sec: int = field(default=60)
    monitor_timeout_sec: int = field(default=0)
    monitor_max_checkpoints: int = field(default=0)
    judge_model_path: str = field(default="")
    policy_device: str = field(default="auto")
    policy_backend: str = field(default="local")
    policy_host: str = field(default="127.0.0.1")
    policy_port: int = field(default=10094)
    policy_api_key: str = field(default="")
    policy_ping_interval: Optional[float] = field(default=None)
    policy_ping_timeout: Optional[float] = field(default=None)
    distributed_timeout_sec: float = field(default=7200.0)
    judge_device: str = field(default="cpu")
    judge_max_new_tokens: int = field(default=256)
    enable_neighbor_subtask_compare: bool = field(default=False)
    max_videos: int = field(default=0)
    monitor_state_file: str = field(default="checkpoint_auto_eval_state.json")
    use_detailed_instruction: bool = field(
        default=False,
        metadata={"help": "Whether checkpoint auto-eval should switch to the detailed system message when detailed task text is available."},
    )
    use_subtask_list: bool =field(default=False)
    resample_seed_with_checkpoint_step: bool = field(default=False)

def _init_distributed(timeout_sec: float = 7200.0) -> tuple[int, int, int]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1 and torch.distributed.is_available() and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=max(1.0, float(timeout_sec))),
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _infer_model_type(model_name_or_path: str) -> str:
    lowered = model_name_or_path.lower()
    if "qwen3" in lowered:
        return "qwen3vl"
    if "qwen2.5" in lowered:
        return "qwen2.5vl"
    return "qwen2vl"


def _load_model_tokenizer_processor(model_args: ModelArguments, infer_args: InferenceArguments):
    model_load_path = model_args.checkpoint_path or model_args.model_name_or_path
    model_key = model_args.model_name_or_path.lower()
    model_name = Path(model_args.model_name_or_path.rstrip("/")).name.lower()

    dtype = None
    if infer_args.bf16:
        dtype = torch.bfloat16
    elif infer_args.fp16:
        dtype = torch.float16

    model_kwargs = {
        "cache_dir": infer_args.cache_dir,
        "attn_implementation": infer_args.attn_implementation,
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    if "qwen3" in model_key and "a" in model_name:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_load_path, **model_kwargs)
    elif "qwen3" in model_key:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_load_path, **model_kwargs)
    elif "qwen2.5" in model_key:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_load_path, **model_kwargs)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_load_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=infer_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=infer_args.cache_dir,
        model_max_length=infer_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.config.use_cache = True

    rank0_print(f"[inference] model_load_path = {model_load_path}")
    rank0_print(f"[inference] model class = {model.__class__.__name__}")

    return model, tokenizer, processor




def _load_tokenizer_processor(model_args: ModelArguments, infer_args: InferenceArguments):
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=infer_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=infer_args.cache_dir,
        model_max_length=infer_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, processor





class PromptOnlyDataset(Dataset):
    def __init__(self, base_dataset: Dataset, max_samples: int = 0):
        self.base_dataset = base_dataset
        total = len(base_dataset)
        if max_samples is not None and max_samples > 0:
            total = min(total, max_samples)
        self.indices = list(range(total))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        real_idx = self.indices[idx]
        sample = self.base_dataset[real_idx]

        input_ids: torch.Tensor = sample["input_ids"]
        labels: Optional[torch.Tensor] = sample.get("labels", None)

        if labels is not None:
            non_ignore = torch.nonzero(labels.ne(IGNORE_INDEX), as_tuple=False)
            first_target = int(non_ignore[0].item()) if non_ignore.numel() > 0 else int(input_ids.shape[0])
        else:
            first_target = int(input_ids.shape[0])
        first_target = max(first_target, 1)

        reference_ids = torch.empty(0, dtype=torch.long)
        if labels is not None:
            ref_mask = labels.ne(IGNORE_INDEX)
            if torch.any(ref_mask):
                reference_ids = labels[ref_mask]

        return {
            "sample_idx": real_idx,
            "input_ids": input_ids[:first_target],
            "pixel_values": sample.get("pixel_values", None),
            "image_grid_thw": sample.get("image_grid_thw", None),
            "pixel_values_videos": sample.get("pixel_values_videos", None),
            "video_grid_thw": sample.get("video_grid_thw", None),
            "reference_ids": reference_ids,
        }


@dataclass
class InferenceDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="left",
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "sample_indices": torch.tensor([instance["sample_idx"] for instance in instances], dtype=torch.long),
            "reference_ids": [instance["reference_ids"] for instance in instances],
        }

        images = list(
            itertools.chain(*(instance["pixel_values"] for instance in instances if instance.get("pixel_values") is not None))
        )
        if len(images) > 0:
            batch["pixel_values"] = torch.cat([img for img in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(instance["image_grid_thw"] for instance in instances if instance.get("image_grid_thw") is not None)
                )
            )
            batch["image_grid_thw"] = torch.stack(grid_thw, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        videos = list(
            itertools.chain(
                *(instance["pixel_values_videos"] for instance in instances if instance.get("pixel_values_videos") is not None)
            )
        )
        if len(videos) > 0:
            batch["pixel_values_videos"] = torch.cat([vid for vid in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(instance["video_grid_thw"] for instance in instances if instance.get("video_grid_thw") is not None)
                )
            )
            batch["video_grid_thw"] = torch.stack(video_grid_thw, dim=0)
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None

        return batch



def _prepare_raw_eval_dataset(tokenizer, processor, data_args: DataArguments) -> Dataset:
    eval_data_args = copy.deepcopy(data_args)
    eval_data_args.model_type = _infer_model_type(data_args.model_name_or_path) if hasattr(data_args, "model_name_or_path") else None

    return EvalSubtaskDataset(tokenizer=tokenizer, processor=processor, data_args=eval_data_args)


def _build_dataloader(dataset: Dataset, tokenizer, infer_args: InferenceArguments, rank: int, world_size: int) -> DataLoader:
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    return DataLoader(
        dataset,
        batch_size=infer_args.per_device_eval_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=infer_args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=InferenceDataCollator(tokenizer=tokenizer),
    )


def _decode_text(tokenizer, token_ids: torch.Tensor) -> str:
    if token_ids is None or token_ids.numel() == 0:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _normalize_text(text: Any) -> str:
    return _shared_normalize_text(text)


def _extract_current_subtask_text(text: str) -> str:
    return _shared_extract_current_subtask_text(text)


def _resolve_judge_device(device: str, local_rank: int) -> str:
    if device == "auto":
        return f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        logging.warning("judge_device=%s but CUDA is unavailable, fallback to cpu.", device)
        return "cpu"

    return str(device)


def _load_neighbor_compare_judge(infer_args: InferenceArguments, local_rank: int):
    judge_device = _resolve_judge_device(infer_args.judge_device, local_rank)
    judge_model, judge_tokenizer = _shared_load_judge_model(
        judge_model_path=infer_args.judge_model_path,
        device=judge_device,
    )
    return judge_model, judge_tokenizer, judge_device


def _score_subtask_with_judge(
    judge_model,
    judge_tokenizer,
    judge_device: str,
    max_new_tokens: int,
    reference_subtask: str,
    candidate_subtask: str,
) -> float:
    return _shared_score_subtask_with_existing_judge(
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        judge_device=judge_device,
        judge_max_new_tokens=max_new_tokens,
        reference_subtask=reference_subtask,
        candidate_subtask=candidate_subtask,
    )


def _get_subtask_action_text(action_cfg: Sequence[Dict[str, Any]], subtask_id: int, allow_completion: bool = True) -> str:
    if subtask_id < 0:
        return ""
    if subtask_id >= len(action_cfg):
        return "task_completed" if allow_completion else ""
    return _shared_get_subtask_action_text(action_cfg, subtask_id)


def _block_kind_to_sample_kind(block_kind: str) -> str:
    kind = str(block_kind or "").strip().lower()
    if kind == "uniform":
        return "subtask"
    if kind == "transition_dense":
        return "transition"
    if kind == "final_tail_dense":
        return "final_tail"
    return "unknown"


def _resolve_neighbor_subtasks(subtask_dataset: EvalSubtaskDataset, sample_idx: int) -> Optional[Dict[str, str]]:
    if sample_idx < 0 or sample_idx >= len(subtask_dataset):
        return None

    try:
        dataset_id = bisect.bisect_right(subtask_dataset.dataset_cum_sizes, sample_idx)
        dataset_prev = 0 if dataset_id == 0 else subtask_dataset.dataset_cum_sizes[dataset_id - 1]
        local_idx = sample_idx - dataset_prev

        block = subtask_dataset.dataset_blocks[dataset_id]
        sampled_indices = block.get("sampled_indices")
        sample_id = sampled_indices[local_idx] if sampled_indices is not None else local_idx

        seg_id = bisect.bisect_right(block["cum_counts"], sample_id)
        sampling_block = block["sampling_blocks"][seg_id]
        episode_pos = int(sampling_block[0])
        target_subtask_id = int(sampling_block[1])
        block_kind = str(sampling_block[6]) if len(sampling_block) > 6 else ""

        episode = block["episodes"][episode_pos]
        action_config = episode.get("action_config", [])
        if not action_config:
            return None

        return {
            "sample_kind": _block_kind_to_sample_kind(block_kind),
            "reference_subtask": _get_subtask_action_text(action_config, target_subtask_id, allow_completion=True),
            "previous_subtask": _get_subtask_action_text(action_config, target_subtask_id - 1, allow_completion=False),
            "next_subtask": _get_subtask_action_text(action_config, target_subtask_id + 1, allow_completion=False),
        }
    except Exception:
        return None


def _analyze_prediction_against_neighbors(
    sample_idx: int,
    pred_text: str,
    ref_text: str,
    subtask_dataset: EvalSubtaskDataset,
    judge_model,
    judge_tokenizer,
    judge_device: str,
    judge_max_new_tokens: int,
    low_score_threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    pred_subtask = _extract_current_subtask_text(pred_text)
    ref_subtask = _extract_current_subtask_text(ref_text)

    ctx = _resolve_neighbor_subtasks(subtask_dataset, sample_idx)
    sample_kind = ctx.get("sample_kind", "unknown") if ctx else "unknown"
    if not ref_subtask and ctx:
        ref_subtask = ctx.get("reference_subtask", "")

    if not pred_subtask and not ref_subtask:
        return None

    subtask_score = _score_subtask_with_judge(
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        judge_device=judge_device,
        max_new_tokens=judge_max_new_tokens,
        reference_subtask=ref_subtask,
        candidate_subtask=pred_subtask,
    )

    result: Dict[str, Any] = {
        "sample_kind": sample_kind,
        "pred_current_subtask": pred_subtask,
        "ref_current_subtask": ref_subtask,
        "subtask_judge_score": subtask_score,
    }

    if subtask_score >= low_score_threshold:
        result["low_score_neighbor_relation"] = "not_needed"
        return result

    if not ctx:
        result["low_score_neighbor_relation"] = "unknown"
        return result

    prev_subtask = ctx.get("previous_subtask", "")
    next_subtask = ctx.get("next_subtask", "")
    prev_score = (
        _score_subtask_with_judge(
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            judge_device=judge_device,
            max_new_tokens=judge_max_new_tokens,
            reference_subtask=prev_subtask,
            candidate_subtask=pred_subtask,
        )
        if _normalize_text(prev_subtask)
        else None
    )
    next_score = (
        _score_subtask_with_judge(
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            judge_device=judge_device,
            max_new_tokens=judge_max_new_tokens,
            reference_subtask=next_subtask,
            candidate_subtask=pred_subtask,
        )
        if _normalize_text(next_subtask)
        else None
    )

    relation = "neither"
    best_score = max(prev_score if prev_score is not None else -1.0, next_score if next_score is not None else -1.0)
    if best_score < low_score_threshold:
        relation = "neither"
    elif prev_score is not None and next_score is not None:
        if abs(prev_score - next_score) < 1e-6:
            relation = "neither"
        else:
            relation = "previous_subtask" if prev_score > next_score else "next_subtask"
    elif prev_score is not None:
        relation = "previous_subtask"
    elif next_score is not None:
        relation = "next_subtask"
    else:
        relation = "unknown"

    result["neighbor_previous_subtask"] = prev_subtask
    result["neighbor_next_subtask"] = next_subtask
    result["neighbor_prev_judge_score"] = prev_score
    result["neighbor_next_judge_score"] = next_score
    result["low_score_neighbor_relation"] = relation
    return result


def _run_neighbor_subtask_compare(
    merged_file: str,
    subtask_dataset: EvalSubtaskDataset,
    infer_args: InferenceArguments,
    local_rank: int,
) -> str:
    judge_model, judge_tokenizer, judge_device = _load_neighbor_compare_judge(infer_args, local_rank)

    input_path = Path(merged_file)
    output_path = input_path.with_name(input_path.stem + ".neighbor_compare.jsonl")
    summary_path = output_path.with_name(output_path.stem + ".summary.json")
    low_score_threshold = 0.5

    total = 0
    analyzed = 0
    failed = 0
    relation_keys = ["previous_subtask", "next_subtask", "neither", "unknown"]
    failed_relation_counts: Dict[str, int] = {k: 0 for k in relation_keys}
    by_sample_kind: Dict[str, Dict[str, Any]] = {}

    def _ensure_kind_stat(kind: str) -> Dict[str, Any]:
        key = str(kind or "unknown")
        if key not in by_sample_kind:
            by_sample_kind[key] = {
                "analyzed": 0,
                "failed": 0,
                "failed_relation_counts": {k: 0 for k in relation_keys},
            }
        return by_sample_kind[key]

    def _ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    compare_pbar = None
    try:
        try:
            from tqdm.auto import tqdm

            compare_pbar = tqdm(
                total=None,
                desc="[neighbor-compare]",
                dynamic_ncols=True,
                leave=True,
            )
        except Exception:
            compare_pbar = None

        with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                if compare_pbar is not None:
                    compare_pbar.update(1)

                record = json.loads(line)
                pred_text = str(record.get("prediction", ""))
                ref_text = str(record.get("reference", ""))
                sample_idx = int(record.get("sample_idx", -1))

                if ref_text and sample_idx >= 0:
                    analysis = _analyze_prediction_against_neighbors(
                        sample_idx=sample_idx,
                        pred_text=pred_text,
                        ref_text=ref_text,
                        subtask_dataset=subtask_dataset,
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        judge_device=judge_device,
                        judge_max_new_tokens=infer_args.judge_max_new_tokens,
                        low_score_threshold=low_score_threshold,
                    )
                    if analysis:
                        record.update(analysis)
                        analyzed += 1

                        sample_kind = str(analysis.get("sample_kind", "unknown") or "unknown")
                        kind_stat = _ensure_kind_stat(sample_kind)
                        kind_stat["analyzed"] += 1

                        try:
                            subtask_score = float(analysis.get("subtask_judge_score", 0.0))
                        except Exception:
                            subtask_score = 0.0

                        if subtask_score < low_score_threshold:
                            failed += 1
                            kind_stat["failed"] += 1
                            relation = str(analysis.get("low_score_neighbor_relation", "unknown") or "unknown")
                            if relation not in failed_relation_counts:
                                relation = "unknown"
                            failed_relation_counts[relation] += 1
                            kind_stat["failed_relation_counts"][relation] += 1

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if compare_pbar is not None:
            compare_pbar.close()

    for kind, stat in by_sample_kind.items():
        analyzed_count = int(stat.get("analyzed", 0))
        failed_count = int(stat.get("failed", 0))
        stat["failed_ratio"] = _ratio(failed_count, analyzed_count)
        relation_counts = dict(stat.get("failed_relation_counts", {}))
        stat["failed_relation_ratio"] = {
            key: _ratio(int(relation_counts.get(key, 0)), failed_count)
            for key in relation_keys
        }

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "low_score_threshold": low_score_threshold,
        "total_records": total,
        "analyzed_records": analyzed,
        "failed_records": failed,
        "failed_ratio": _ratio(failed, analyzed),
        "failed_relation_counts": failed_relation_counts,
        "failed_relation_ratio": {
            key: _ratio(int(failed_relation_counts.get(key, 0)), failed)
            for key in relation_keys
        },
        "by_sample_kind": {
            kind: by_sample_kind[kind]
            for kind in sorted(by_sample_kind.keys())
        },
    }

    with summary_path.open("w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, ensure_ascii=False, indent=2)

    logging.info(
        "neighbor subtask compare finished: analyzed=%d/%d, failed=%d (%.4f), saved=%s, summary=%s",
        analyzed,
        total,
        failed,
        _ratio(failed, analyzed),
        output_path,
        summary_path,
    )
    return str(output_path)


def _merge_rank_outputs(output_dir: str, world_size: int) -> str:
    all_records: List[Dict] = []
    for rank in range(world_size):
        rank_file = Path(output_dir) / f"predictions.rank{rank}.jsonl"
        if not rank_file.exists():
            continue
        with rank_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    all_records.sort(key=lambda x: x.get("sample_idx", 0))

    merged_file = Path(output_dir) / "predictions.jsonl"
    with merged_file.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return str(merged_file)


def _parse_checkpoint_step(checkpoint_dir: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", checkpoint_dir.name)
    if match:
        return int(match.group(1))
    return -1


def _resolve_single_checkpoint_dir(model_args: ModelArguments, infer_args: InferenceArguments) -> Path:
    ckpt_path = infer_args.checkpoint_root_dir or model_args.checkpoint_path
    if not ckpt_path:
        raise ValueError("Please pass checkpoint path via --checkpoint_path or --checkpoint_root_dir.")

    checkpoint_dir = Path(str(ckpt_path)).expanduser()
    policy_backend = str(getattr(infer_args, "policy_backend", "local") or "local").strip().lower()
    if policy_backend == "local":
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"checkpoint path is invalid: {checkpoint_dir}")
        return checkpoint_dir

    # Remote policy backends do not load local checkpoint weights; keep the user-supplied
    # label for output naming/state tracking even if it is not an on-disk directory.
    return checkpoint_dir


def _load_processed_checkpoints(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    try:
        with state_file.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return set()

    paths = obj.get("processed_checkpoints", []) if isinstance(obj, dict) else []
    return {str(Path(p)) for p in paths}


def _save_processed_checkpoints(state_file: Path, processed: set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_checkpoints": sorted(processed),
        "updated_at": int(time.time()),
    }
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _checkpoint_already_evaluated(checkpoint_dir: Path, infer_args: InferenceArguments) -> bool:
    output_dir = Path(infer_args.output_dir)
    state_file = output_dir / infer_args.monitor_state_file
    ckpt_key = str(checkpoint_dir)

    processed = _load_processed_checkpoints(state_file)
    if ckpt_key in processed:
        logging.warning("[checkpoint-auto-eval] already evaluated (state): %s", checkpoint_dir)
        return True

    summary_path = output_dir / f"{checkpoint_dir.name}_summary.json"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = None

        if isinstance(summary, dict):
            saved_dir = str(summary.get("checkpoint_dir", "")).strip()
            if not saved_dir or os.path.abspath(saved_dir) == os.path.abspath(ckpt_key):
                processed.add(ckpt_key)
                _save_processed_checkpoints(state_file, processed)
                logging.warning("[checkpoint-auto-eval] already evaluated (summary): %s", checkpoint_dir)
                return True

    return False


def _run_single_checkpoint_auto_subtask_eval(
    model_args: ModelArguments,
    data_args: DataArguments,
    infer_args: InferenceArguments,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    checkpoint_dir = _resolve_single_checkpoint_dir(model_args, infer_args)
    os.makedirs(infer_args.output_dir, exist_ok=True)

    skip_eval = False
    if rank == 0:
        skip_eval = _checkpoint_already_evaluated(checkpoint_dir, infer_args)

    if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
        skip_holder = [skip_eval]
        torch.distributed.broadcast_object_list(skip_holder, src=0)
        skip_eval = bool(skip_holder[0])

    if skip_eval:
        if rank == 0:
            logging.warning("[checkpoint-auto-eval] skip checkpoint: %s", checkpoint_dir)
        return

    tokenizer, processor = _load_tokenizer_processor(model_args, infer_args)
    data_args.model_name_or_path = model_args.model_name_or_path
    data_args.model_type = _infer_model_type(model_args.model_name_or_path)
    eval_dataset = _prepare_raw_eval_dataset(tokenizer, processor, data_args)

    if not isinstance(eval_dataset, EvalSubtaskDataset):
        raise ValueError(
            "Checkpoint auto subtask eval requires agibot_subtask eval_dataset "
            f"(got {type(eval_dataset).__name__})."
        )

    step = _parse_checkpoint_step(checkpoint_dir)
    sampling_seed = infer_args.seed
    if infer_args.resample_seed_with_checkpoint_step and step >= 0:
        sampling_seed = infer_args.seed + step

    run_checkpoint_auto_eval(
        checkpoint_dir=str(checkpoint_dir),
        base_model_path=model_args.model_name_or_path,
        eval_dataset=eval_dataset,
        judge_model_path=infer_args.judge_model_path,
        output_dir=infer_args.output_dir,
        sampling_seed=sampling_seed,
        max_videos=infer_args.max_videos,
        max_samples=infer_args.max_samples,
        max_samples_mode=infer_args.max_samples_mode,
        policy_device=infer_args.policy_device,
        policy_backend=infer_args.policy_backend,
        policy_host=infer_args.policy_host,
        policy_port=infer_args.policy_port,
        policy_api_key=infer_args.policy_api_key,
        policy_ping_interval=infer_args.policy_ping_interval,
        policy_ping_timeout=infer_args.policy_ping_timeout,
        judge_device=infer_args.judge_device,
        policy_max_new_tokens=infer_args.max_new_tokens,
        judge_max_new_tokens=infer_args.judge_max_new_tokens,
        attn_implementation=infer_args.attn_implementation,
        policy_batch_size=max(1, int(infer_args.per_device_eval_batch_size)),
        distributed_rank=rank,
        distributed_world_size=world_size,
        enable_neighbor_subtask_compare=infer_args.enable_neighbor_subtask_compare,
        neighbor_low_score_threshold=0.5,
        use_detailed_instruction=bool(getattr(infer_args, "use_detailed_instruction", False)),
        use_subtask_list=bool(getattr(infer_args, "use_subtask_list", False)),
    )

    if rank == 0:
        state_file = Path(infer_args.output_dir) / infer_args.monitor_state_file
        processed = _load_processed_checkpoints(state_file)
        processed.add(str(checkpoint_dir))
        _save_processed_checkpoints(state_file, processed)
        logging.info("[checkpoint-auto-eval] finished checkpoint: %s", checkpoint_dir)


@torch.no_grad()
def run_batch_inference():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InferenceArguments))
    model_args, data_args, infer_args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    rank, world_size, local_rank = _init_distributed(timeout_sec=infer_args.distributed_timeout_sec)

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if remaining and rank == 0:
        logging.warning(f"Ignored extra args (train-related): {remaining}")

    random.seed(infer_args.seed)
    torch.manual_seed(infer_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(infer_args.seed)

    if infer_args.enable_checkpoint_auto_subtask_eval:
        _run_single_checkpoint_auto_subtask_eval(
            model_args=model_args,
            data_args=data_args,
            infer_args=infer_args,
            rank=rank,
            world_size=world_size,
        )
        _barrier()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    # explicit non-auto mode: skip policy inference; optionally run compare from existing detail file.
    if rank == 0:
        logging.info("enable_checkpoint_auto_subtask_eval=False, skip policy inference.")

    if infer_args.enable_neighbor_subtask_compare:
        detail_file = ""
        try:
            checkpoint_dir = _resolve_single_checkpoint_dir(model_args, infer_args)
            detail_file = str(Path(infer_args.output_dir) / f"{checkpoint_dir.name}_details.jsonl")
        except Exception:
            if rank == 0:
                logging.exception("failed to resolve checkpoint dir for deriving detail file")

        detail_exists = bool(detail_file) and Path(detail_file).exists()
        if detail_exists:
            try:
                tokenizer, processor = _load_tokenizer_processor(model_args, infer_args)
                data_args.model_name_or_path = model_args.model_name_or_path
                data_args.model_type = _infer_model_type(model_args.model_name_or_path)
                eval_dataset = _prepare_raw_eval_dataset(tokenizer, processor, data_args)

                if not isinstance(eval_dataset, EvalSubtaskDataset):
                    raise ValueError(
                        "neighbor compare from detail requires agibot_subtask eval_dataset "
                        f"(got {type(eval_dataset).__name__})."
                    )

                summary = run_neighbor_compare_from_detail_file(
                    detail_file=detail_file,
                    eval_dataset=eval_dataset,
                    judge_model_path=infer_args.judge_model_path,
                    judge_device=infer_args.judge_device,
                    judge_max_new_tokens=infer_args.judge_max_new_tokens,
                    low_score_threshold=0.5,
                    distributed_rank=rank,
                    distributed_world_size=world_size,
                )
                if rank == 0:
                    logging.info(
                        "detail compare finished: output=%s, summary=%s",
                        str(summary.get("output_file", "")),
                        str(summary.get("summary_file", "")),
                    )
            except Exception:
                if rank == 0:
                    logging.exception("detail compare failed")
        else:
            if rank == 0:
                logging.warning(
                    "enable_neighbor_subtask_compare=True but detail file not found: %s",
                    detail_file,
                )

    _barrier()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return


def start_debugpy_once() -> None:
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    if os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        start_debugpy_once()
    run_batch_inference()
