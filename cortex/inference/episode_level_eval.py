import io
import json
import logging
import os
import re
import sys
import hashlib
import random
import math
import time
import textwrap
from fractions import Fraction
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from mmengine import fileio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

try:
    import av
except ImportError:
    av = None

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from cortex.inference.eval_utils import (
    ROBOT_ATOMIC_SKILLS,
    SYSTEM_MESSAGE,
    is_ceph_path as _is_ceph_path,
    join_path as _join_path,
    rank0_print,
    read_json as _read_json,
    read_jsonl as _read_jsonl,
)

from cortex.inference.subtask_eval import (
    SubtaskEvalSample as _JudgeSubtaskEvalSample,
    _build_judge_prompt as _build_shared_judge_prompt,
    _clean_active_language_memory_text as _clean_eval_memory_text,
    _extract_current_subtask_text as _extract_eval_current_subtask_text,
    _get_subtask_action_text as _get_eval_subtask_action_text,
    _load_judge_model as _load_shared_judge_model,
    _normalize_text as _normalize_eval_text,
    _run_judge as _run_shared_judge,
)

DEFAULT_INPUT_EPISODES_FILENAME = "meta/episodes_norm_lang_mem.jsonl"

@dataclass
class InferenceArguments:
    video_keys: str = field(metadata={"help": "Comma-separated multi-view video keys."})
    model_name_or_path: str = field(metadata={"help": "Path to the trained model or checkpoint."})

    dataset_path: str = field(default="", metadata={"help": "LeRobot dataset root, used in lerobot mode."})
    mode: str = field(default="lerobot")
    output_path: str = field(default="./exp/cortex/inference_sys2/predictions.jsonl")
    base_model_name_or_path: Optional[str] = field(default=None)
    processor_name_or_path: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    task_text: Optional[str] = field(default=None)
    ordered_subtask_plan: Optional[str] = field(default=None, metadata={"help": "Ordered subtask plan separated by | or newline."})
    detailed_task: Optional[str] = field(default=None, metadata={"help": "Detailed global task instruction."})
    simple_video_paths: Optional[str] = field(default=None)
    task_key: Optional[str] = field(default=None)
    initial_memory: str = field(default="This is the first subtask, and no subtasks have been completed yet.")

    sample_interval: int = field(default=12, metadata={"help": "Sampling interval in steps."})
    sample_interval_sec: float = field(default=0.0, metadata={"help": "Sampling interval in seconds. Takes priority when > 0."})
    start_index: int = field(default=0)
    include_last: bool = field(default=True)
    episode_indices: str = field(default="")
    max_episodes: int = field(default=0)
    max_samples_per_episode: int = field(default=0)
    num_eval_episodes_per_task: int = field(default=10)
    seed: int = field(default=42)
    ignore_boundary_sec: float = field(default=0.0)
    transition_tail_sec: float = field(default=0.1)
    transition_head_sec: float = field(default=0.2)
    last_tail_sec: float = field(default=1.0)

    model_max_length: int = field(default=8192)
    max_new_tokens: int = field(default=256)
    do_sample: bool = field(default=False)
    temperature: float = field(default=0.0)
    top_p: float = field(default=1.0)
    num_beams: int = field(default=1)
    repetition_penalty: float = field(default=1.0)
    attn_implementation: str = field(default="flash_attention_2")
    device: str = field(default="auto")
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    policy_backend: str = field(
        default="local",
        metadata={"help": "Inference backend: local or websocket."},
    )
    policy_host: str = field(
        default="127.0.0.1",
        metadata={"help": "WebSocket policy server host when policy_backend=websocket."},
    )
    policy_port: int = field(
        default=10094,
        metadata={"help": "WebSocket policy server port when policy_backend=websocket."},
    )
    policy_api_key: str = field(
        default="",
        metadata={"help": "Optional API key for the WebSocket policy server."},
    )
    policy_ping_interval: Optional[float] = field(
        default=None,
        metadata={"help": "WebSocket keepalive ping interval in seconds. Set to None to disable."},
    )
    policy_ping_timeout: Optional[float] = field(
        default=None,
        metadata={"help": "WebSocket keepalive ping timeout in seconds. Set to None to disable."},
    )

    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = field(default=30.0)
    visual_video_fps: Optional[float] = field(default=None)
    use_detailed_instruction:bool = field(default=False)
    use_subtask_list_instruction: bool = field(default=False)
    save_visual_video: bool = field(default=True)
    visual_video_dir: Optional[str] = field(default=None)
    save_episode_summary: bool = field(default=True)
    enable_closed_loop_eval: bool = field(default=True)
    judge_model_path: str = field(default="")
    judge_device: str = field(default="auto")
    judge_max_new_tokens: int = field(default=256)
    save_task_summary: bool = field(default=True)
    shard_sync_timeout_sec: float = field(
        default=7200.0,
        metadata={"help": "Timeout in seconds for rank 0 waiting on output shard files from other ranks."},
    )


def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(fileio.get(path)))


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA is unavailable, falling back to cpu from device=%s", device)
        return "cpu"
    return str(device)


def _resolve_device_for_local_rank(device: str, local_rank: int) -> str:
    resolved = _resolve_device(device)
    if resolved == "cuda":
        return f"cuda:{max(0, int(local_rank))}"
    return resolved


def _init_distributed() -> Tuple[int, int, int]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1 and torch.distributed.is_available() and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _wait_for_files(paths: Sequence[Path], timeout_sec: float, poll_interval_sec: float = 5.0) -> None:
    deadline = time.time() + max(0.0, float(timeout_sec))
    pending = [Path(path) for path in paths]
    while pending:
        missing = [path for path in pending if not path.exists()]
        if not missing:
            return
        if time.time() >= deadline:
            missing_str = ", ".join(str(path) for path in missing)
            raise TimeoutError(
                f"Timed out after {timeout_sec:.1f}s while waiting for shard files: {missing_str}"
            )
        logging.info(
            "Waiting for %d shard files before merge: %s",
            len(missing),
            ", ".join(path.name for path in missing),
        )
        time.sleep(max(0.1, float(poll_interval_sec)))


def _infer_model_cls_from_config(config) -> type:
    architectures = [str(item).lower() for item in getattr(config, "architectures", [])]
    model_type = str(getattr(config, "model_type", "")).lower()

    if any("qwen3vlmoe" in item for item in architectures) or "qwen3_vl_moe" in model_type:
        return Qwen3VLMoeForConditionalGeneration
    if any("qwen3vl" in item for item in architectures) or "qwen3_vl" in model_type:
        return Qwen3VLForConditionalGeneration
    if any("qwen2_5_vl" in item or "qwen2.5" in item for item in architectures) or "qwen2_5_vl" in model_type:
        return Qwen2_5_VLForConditionalGeneration
    return Qwen2VLForConditionalGeneration


def _load_config_with_fallback(args: InferenceArguments):
    candidates = [
        args.model_name_or_path,
        args.base_model_name_or_path,
        args.processor_name_or_path,
    ]
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            config = AutoConfig.from_pretrained(candidate, cache_dir=args.cache_dir)
            return config, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise ValueError("Failed to load config from any candidate path. Errors: " + " | ".join(errors))


def _load_processor_tokenizer_with_fallback(args: InferenceArguments, config_source: str, config):
    config_name_or_path = str(getattr(config, "_name_or_path", "") or "").strip()
    candidates = []
    for candidate in (
        args.processor_name_or_path,
        args.base_model_name_or_path,
        config_name_or_path,
        config_source,
        args.model_name_or_path,
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    errors = []
    for candidate in candidates:
        try:
            logging.info("Trying processor/tokenizer path: %s", candidate)
            processor = AutoProcessor.from_pretrained(candidate, cache_dir=args.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                candidate,
                cache_dir=args.cache_dir,
                model_max_length=args.model_max_length,
                padding_side="left",
                use_fast=False,
            )
            return processor, tokenizer, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise OSError(
        "Failed to load processor/tokenizer from any candidate path. "
        "Please pass --processor_name_or_path or --base_model_name_or_path to the exact base model used for this checkpoint. "
        + " | ".join(errors)
    )


def _load_model_processor_tokenizer(args: InferenceArguments):
    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    config, config_source = _load_config_with_fallback(args)
    model_cls = _infer_model_cls_from_config(config)
    model_kwargs = {
        "cache_dir": args.cache_dir,
        "attn_implementation": args.attn_implementation,
        "config": config,
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    logging.info("Loading model weights from %s", args.model_name_or_path)
    logging.info("Resolved config from %s", config_source)
    logging.info("Resolved model class: %s", model_cls.__name__)
    model = model_cls.from_pretrained(args.model_name_or_path, **model_kwargs)

    processor, tokenizer, resolved_processor_path = _load_processor_tokenizer_with_fallback(args, config_source, config)
    logging.info("Resolved processor/tokenizer path: %s", resolved_processor_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    processor.tokenizer = tokenizer
    processor.tokenizer.model_max_length = args.model_max_length
    processor = update_processor_pixels(processor, args)

    device = _resolve_device(args.device)
    model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, processor, tokenizer, device

def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor

class LeRobotEpisodeReader:
    def __init__(self, dataset_path: str, task_key: Optional[str] = None):
        self.dataset_path = dataset_path.rstrip("/")
        if '2025-challenge-demos' in self.dataset_path:
            task_name = os.path.basename(self.dataset_path)
            self.dataset_path = os.path.dirname(self.dataset_path)
            behavior_episode_filename = f'annotations/{task_name}/episodes_lang_mem.jsonl'
            self.episodes = _read_json(_join_path(self.dataset_path, behavior_episode_filename))
        else:
            self.episodes = _read_jsonl(_join_path(self.dataset_path, DEFAULT_INPUT_EPISODES_FILENAME))
        self.info = _read_json(_join_path(self.dataset_path, "meta/info.json"))
        self.tasks = _read_jsonl(_join_path(self.dataset_path, "meta/tasks.jsonl"))
        self.task_map = {
            int(item["task_index"]): str(item.get("task", "")).strip()
            for item in self.tasks
            if "task_index" in item
        }
        self.task_column = str(task_key).strip() if task_key else None
        self.dataset_fps = self._resolve_dataset_fps()

    @property
    def chunk_size(self) -> int:
        return int(self.info["chunks_size"])

    def _resolve_dataset_fps(self) -> float:
        info = self.info or {}
        try:
            if "fps" in info:
                return float(info["fps"])
        except Exception:
            pass

        features = info.get("features", {}) or {}
        for meta in features.values():
            try:
                if "video_info" in meta and "video.fps" in meta["video_info"]:
                    return float(meta["video_info"]["video.fps"])
                if "info" in meta and "video.fps" in meta["info"]:
                    return float(meta["info"]["video.fps"])
            except Exception:
                continue

        return 30.0

    def _resolve_task_index_from_dataframe(self, df: pd.DataFrame, row_index: int) -> str:
        candidate_columns: List[str] = []
        if self.task_column:
            candidate_columns.append(self.task_column)
        candidate_columns.extend([
            "task_index",
            "annotation.human.action.task_description",
            "annotation.task_index",
        ])

        seen = set()
        for column in candidate_columns:
            if not column or column in seen or column not in df.columns:
                continue
            seen.add(column)
            try:
                task_idx = int(df[column].iloc[row_index])
            except Exception:
                continue
            task_text = self.task_map.get(task_idx, "")
            if task_text:
                return task_text
        return ""

    def get_parquet_path(self, episode_index: Any) -> str:
        episode_index = _normalize_episode_index(episode_index)
        rel_path = self.info["data_path"].format(
            episode_chunk=episode_index // self.chunk_size,
            episode_index=episode_index,
        )
        return _join_path(self.dataset_path, rel_path)

    def get_video_path(self, episode_index: Any, video_key: str) -> str:
        episode_index = _normalize_episode_index(episode_index)
        rel_path = self.info["video_path"].format(
            episode_chunk=episode_index // self.chunk_size,
            episode_index=episode_index,
            video_key=video_key,
        )
        return _join_path(self.dataset_path, rel_path)

    def load_episode_dataframe(self, episode_index: Any) -> pd.DataFrame:
        return _read_parquet(self.get_parquet_path(episode_index))

    def resolve_task_text_from_episode_metadata(self, episode: Optional[Dict[str, Any]]) -> str:
        if not isinstance(episode, dict):
            return ""

        tasks = episode.get("tasks", "")
        if isinstance(tasks, (list, tuple)):
            tasks = next((str(item).strip() for item in tasks if str(item).strip()), "")
        else:
            tasks = str(tasks).strip()
        if tasks:
            return tasks.split("|")[0].strip()

        for candidate in ("task", "task_description", "language_instruction"):
            value = episode.get(candidate, "")
            if isinstance(value, (list, tuple)):
                value = value[0] if value else ""
            value = str(value).strip()
            if value:
                return value.split("|")[0].strip()

        task_index_candidates = [self.task_column, "task_index", "annotation.task_index"]
        seen = set()
        for candidate in task_index_candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            raw_value = episode.get(candidate, None)
            try:
                task_idx = int(raw_value)
            except Exception:
                continue
            task_text = str(self.task_map.get(task_idx, "")).strip()
            if task_text:
                return task_text.split("|")[0].strip()

        return ""

    def resolve_task_text(
        self,
        df: pd.DataFrame,
        step_index: int,
        fallback_episode: Optional[Dict[str, Any]] = None,
    ) -> str:
        if fallback_episode is not None:
            task_text = self.resolve_task_text_from_episode_metadata(fallback_episode)
            if task_text:
                return task_text

        row_index = min(max(int(step_index), 0), max(len(df) - 1, 0))

        task_text = self._resolve_task_index_from_dataframe(df, row_index)
        if task_text:
            return task_text.split("|")[0].strip()

        for candidate in ("task", "task_description", "language_instruction"):
            if candidate in df.columns and len(df) > 0:
                value = str(df[candidate].iloc[row_index]).strip()
                if value:
                    return value.split("|")[0].strip()

        return ""

def _normalize_episode_index(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except Exception:
        pass

    match = re.search(r"(\d+)$", text)
    if match:
        return int(match.group(1))

    raise ValueError(f"Unsupported episode_index value: {value!r}")


def _build_episode_filter(raw: str) -> Optional[set[int]]:
    raw = str(raw or "").strip()
    if not raw:
        return None
    return {_normalize_episode_index(item.strip()) for item in raw.split(",") if item.strip()}


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "lerobot").strip().lower()
    if normalized not in {"lerobot", "simple"}:
        raise ValueError(f"Unsupported mode: {mode!r}. Expected one of: lerobot, simple")
    return normalized


def _normalize_policy_backend(policy_backend: str) -> str:
    normalized = str(policy_backend or "local").strip().lower()
    aliases = {
        "local": "local",
        "model": "local",
        "websocket": "websocket",
        "ws": "websocket",
        "remote": "websocket",
    }
    resolved = aliases.get(normalized, "")
    if not resolved:
        raise ValueError(
            f"Unsupported policy_backend: {policy_backend!r}. Expected one of: local, websocket"
        )
    return resolved


def _connect_websocket_policy_client(args: InferenceArguments):
    try:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    except ImportError as exc:
        raise ImportError(
            "WebSocket policy backend requires deployment.model_server.tools.websocket_policy_client "
            "and its websockets dependency."
        ) from exc

    logging.info(
        "Connecting to WebSocket policy server at ws://%s:%s",
        args.policy_host,
        args.policy_port,
    )
    client = WebsocketClientPolicy(
        host=str(args.policy_host),
        port=int(args.policy_port),
        api_key=(str(args.policy_api_key).strip() or None),
        ping_interval=args.policy_ping_interval,
        ping_timeout=args.policy_ping_timeout,
    )
    metadata = client.get_server_metadata()
    logging.info(
        "Connected to policy server metadata=%s",
        json.dumps(metadata, ensure_ascii=False, default=str),
    )
    client.init_device(_resolve_device(args.device))
    return client


def _build_policy_session_id(session_prefix: str, episode_index: int) -> str:
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session_prefix or "eval")).strip("._-") or "eval"
    return f"{safe_prefix}-episode-{int(episode_index):06d}"


def _normalize_lookup_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


VIDEO_FILE_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _resolve_single_simple_episode_index(raw: str) -> int:
    indices = _build_episode_filter(raw)
    if not indices:
        raise ValueError("simple mode without --simple_video_paths requires exactly one --episode_indices value")
    if len(indices) != 1:
        raise ValueError(f"simple mode expects exactly one episode index, got {sorted(indices)}")
    return next(iter(indices))


def _filter_simple_video_files_by_episode(all_video_files: Sequence[Path], episode_index: int) -> List[Path]:
    strict_patterns = [
        re.compile(rf"episode[_-]?{episode_index:08d}(?:\D|$)"),
        re.compile(rf"episode[_-]?{episode_index:06d}(?:\D|$)"),
        re.compile(rf"ep[_-]?{episode_index:08d}(?:\D|$)"),
        re.compile(rf"ep[_-]?{episode_index:06d}(?:\D|$)"),
    ]
    fallback_patterns = [
        re.compile(rf"(?:^|\D){episode_index:08d}(?:\D|$)"),
        re.compile(rf"(?:^|\D){episode_index:06d}(?:\D|$)"),
        re.compile(rf"(?:^|\D){episode_index}(?:\D|$)"),
    ]

    exact_matches: List[Path] = []
    fuzzy_matches: List[Path] = []
    for video_file in all_video_files:
        raw_path = str(video_file).lower()
        if any(pattern.search(raw_path) for pattern in strict_patterns):
            exact_matches.append(video_file)
            continue
        if any(pattern.search(raw_path) for pattern in fallback_patterns):
            fuzzy_matches.append(video_file)

    exact_matches = sorted(set(exact_matches))
    fuzzy_matches = sorted(set(fuzzy_matches))
    if exact_matches:
        return exact_matches
    if fuzzy_matches:
        return fuzzy_matches
    return []


def _discover_simple_video_paths(dataset_path: str, video_keys: Sequence[str], episode_index: Optional[int] = None) -> Dict[str, str]:
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"simple mode dataset_path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"simple mode dataset_path must be a directory: {root}")

    all_video_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_FILE_EXTENSIONS
    )
    if not all_video_files:
        raise FileNotFoundError(f"No video files were found under simple mode dataset_path: {root}")

    candidate_video_files = list(all_video_files)
    if episode_index is not None:
        candidate_video_files = _filter_simple_video_files_by_episode(all_video_files, episode_index)
        if not candidate_video_files:
            raise FileNotFoundError(
                f"Could not find any video files for simple mode episode index {episode_index} under {root}"
            )

    resolved: Dict[str, str] = {}
    for video_key in video_keys:
        candidates = [video_key, video_key.replace(".", "_"), video_key.split(".")[-1]]
        normalized_candidates = []
        for candidate in candidates:
            normalized = _normalize_lookup_token(candidate)
            if normalized and normalized not in normalized_candidates:
                normalized_candidates.append(normalized)

        exact_matches = []
        fuzzy_matches = []
        for video_file in candidate_video_files:
            path_tokens = [_normalize_lookup_token(part) for part in video_file.parts]
            stem_token = _normalize_lookup_token(video_file.stem)
            name_token = _normalize_lookup_token(video_file.name)
            joined_path = "".join(path_tokens)
            for candidate in normalized_candidates:
                if stem_token == candidate or name_token == candidate or candidate in path_tokens:
                    exact_matches.append(video_file)
                    break
                if candidate in stem_token or candidate in name_token or candidate in joined_path:
                    fuzzy_matches.append(video_file)
                    break

        exact_matches = sorted(set(exact_matches))
        fuzzy_matches = sorted(set(fuzzy_matches))
        if len(exact_matches) == 1:
            resolved[video_key] = str(exact_matches[0])
            continue
        if len(exact_matches) > 1:
            raise ValueError(
                f"Ambiguous exact video matches for key {video_key!r}: "
                + ", ".join(str(path) for path in exact_matches[:10])
            )
        if len(fuzzy_matches) == 1:
            resolved[video_key] = str(fuzzy_matches[0])
            continue
        if len(fuzzy_matches) > 1:
            raise ValueError(
                f"Ambiguous fuzzy video matches for key {video_key!r}: "
                + ", ".join(str(path) for path in fuzzy_matches[:10])
            )
        raise FileNotFoundError(f"Could not find a video file for key {video_key!r} under {root}")

    return resolved


def _resolve_simple_video_paths(args: InferenceArguments, video_keys: Sequence[str]) -> Dict[str, str]:
    items = [item.strip() for item in str(args.simple_video_paths or "").split(",") if item.strip()]
    if items:
        if len(items) != len(video_keys):
            raise ValueError(f"Expected {len(video_keys)} simple video paths to match video_keys, got {len(items)}")
        return {key: path_value for key, path_value in zip(video_keys, items)}

    if not args.dataset_path:
        raise ValueError("simple mode requires --dataset_path when --simple_video_paths is not provided")
    episode_index = _resolve_single_simple_episode_index(args.episode_indices)
    return _discover_simple_video_paths(args.dataset_path, video_keys, episode_index=episode_index)


def _inspect_video_timestamps(video_path: str, fallback_fps: float) -> Tuple[float, List[float]]:
    container = _open_video_container(video_path)
    try:
        video_stream = container.streams.video[0]
        fps = None
        try:
            if video_stream.average_rate is not None:
                fps = float(video_stream.average_rate)
        except Exception:
            fps = None
        if fps is None or fps <= 0:
            fps = float(fallback_fps) if fallback_fps and fallback_fps > 0 else 30.0

        num_frames = int(video_stream.frames or 0)
        if num_frames <= 0 and video_stream.duration is not None and video_stream.time_base is not None:
            try:
                duration_sec = float(video_stream.duration * video_stream.time_base)
                num_frames = max(1, int(round(duration_sec * fps)))
            except Exception:
                num_frames = 0
        if num_frames <= 0:
            timestamps = []
            for decoded_frame in container.decode(video_stream):
                current_ts = _decoded_frame_timestamp_sec(decoded_frame, video_stream.time_base)
                if current_ts is None:
                    current_ts = len(timestamps) / fps
                timestamps.append(float(current_ts))
            if not timestamps:
                raise ValueError(f"Could not infer timestamps from video: {video_path}")
            return fps, timestamps

        return fps, [idx / fps for idx in range(num_frames)]
    finally:
        container.close()


def _build_sample_indices(
    timestamps: Sequence[float],
    sample_interval: int,
    sample_interval_sec: float,
    start_index: int,
    include_last: bool,
) -> List[int]:
    total = len(timestamps)
    if total <= 0 or start_index >= total:
        return []

    start_index = max(0, int(start_index))
    if sample_interval_sec and sample_interval_sec > 0:
        indices = [start_index]
        last_ts = float(timestamps[start_index])
        for idx in range(start_index + 1, total):
            current_ts = float(timestamps[idx])
            if current_ts - last_ts >= sample_interval_sec:
                indices.append(idx)
                last_ts = current_ts
    else:
        interval = max(1, int(sample_interval))
        indices = list(range(start_index, total, interval))

    if include_last and total > 0 and (not indices or indices[-1] != total - 1):
        indices.append(total - 1)

    deduped: List[int] = []
    for idx in indices:
        if not deduped or deduped[-1] != idx:
            deduped.append(idx)
    return deduped


def _format_view_label(image_idx: int, num_images: int, video_keys: Sequence[str]) -> str:
    video_key = video_keys[image_idx] if image_idx < len(video_keys) else ""
    normalized_key = str(video_key).strip().lower()

    if "head" in normalized_key or "high" in normalized_key:
        return "head view"
    if "front" in normalized_key:
        return "front view"
    if "left" in normalized_key and any(token in normalized_key for token in ("hand", "wrist", "arm")):
        return "left wrist view"
    if "right" in normalized_key and any(token in normalized_key for token in ("hand", "wrist", "arm")):
        return "right wrist view"
    if any(token in normalized_key for token in ("hand", "wrist", "arm")):
        return "wrist view"
    if normalized_key:
        return f"{normalized_key.split('.')[-1].replace('_', ' ')} view"
    return "current view" if num_images == 1 else f"camera view {image_idx + 1}"



def _build_current_observation_prompt_lines(video_keys: Sequence[str]) -> List[str]:
    num_views = len(video_keys)
    if num_views <= 0:
        raise ValueError("video_keys must not be empty")
    if num_views == 1:
        view_label = _format_view_label(0, num_views, video_keys)
        return [f"Current Observation Image ({view_label}):"]
    prompt_lines = ["Current Observation Images (in order):"]
    for image_idx in range(num_views):
        view_label = _format_view_label(image_idx, num_views, video_keys)
        prompt_lines.append(f"Image {image_idx + 1} ({view_label}):")
    return prompt_lines


def _collect_observation_images(
    video_paths: Dict[str, str],
    video_keys: Sequence[str],
    current_timestamp: float,
    args: InferenceArguments,
    decode_with_frame_index: bool = False,
    current_step_index: Optional[int] = None,
) -> List[Image.Image]:
    if decode_with_frame_index and current_step_index is not None:
        return [_decode_frame_image_by_index(video_paths[key], current_step_index) for key in video_keys]
    return [_decode_frame_image(video_paths[key], current_timestamp) for key in video_keys]


def _get_current_observation_image_index(video_keys: Sequence[str], head_view_index: int) -> int:
    return min(max(0, int(head_view_index)), max(0, len(video_keys) - 1))


def _resolve_head_view_index(video_keys: Sequence[str]) -> int:
    for idx, video_key in enumerate(video_keys):
        normalized_key = str(video_key).strip().lower()
        if "head" in normalized_key or "cam_high" in normalized_key:
            return idx
    raise ValueError(f"video_keys must contain a head-view key, got: {list(video_keys)}")


def _save_sampled_head_image(
    image: Image.Image,
    visual_video_dir: Path,
    episode_index: int,
    sample_id: int,
    step_index: int,
    pred_subtask: str,
    next_memory: str,
) -> str:
    frame = image.convert("RGB").copy()
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    overlay_text = "\n".join(
        [
            _wrap_overlay_text("pred_subtask", pred_subtask),
            _wrap_overlay_text("next_memory", next_memory),
        ]
    )
    margin = 12
    line_spacing = 6
    bbox = draw.multiline_textbbox((margin, margin), overlay_text, font=font, spacing=line_spacing)
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    draw.rectangle(
        [margin - 6, margin - 6, margin + box_width + 6, margin + box_height + 6],
        fill=(0, 0, 0),
    )
    draw.multiline_text((margin, margin), overlay_text, fill=(255, 255, 255), font=font, spacing=line_spacing)

    image_dir = visual_video_dir / f"episode_{episode_index:06d}_sampled_head_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"sample_{sample_id:06d}_step_{step_index:06d}.jpg"
    frame.save(image_path, format="JPEG", quality=95)
    return str(image_path)


def _prepare_visual_video_dir(output_path: Path, args: InferenceArguments) -> Optional[Path]:
    if not args.save_visual_video:
        return None

    if args.visual_video_dir:
        video_dir = Path(args.visual_video_dir).expanduser().resolve()
    else:
        video_dir = output_path.parent / f"{output_path.stem}_visual_videos"

    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


def _sanitize_predicted_memory(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    trailing_observation_pattern = re.compile(
        r"(?is)(?:[,;:\-\s]*)?(?:current\s+observation\s+images?(?:\s*\([^)]*\))?\s*[:：]?|image\s*\d+\s*\([^)]*\)\s*[:：]?)"
    )
    match = trailing_observation_pattern.search(text)
    if match is not None:
        text = text[: match.start()].rstrip(" \t\n,;:-")

    return text.strip()


def _resolve_visual_video_fps(args: InferenceArguments, dataset_fps: float) -> float:
    if args.visual_video_fps is not None:
        return max(1.0, float(args.visual_video_fps))
    if args.sample_interval_sec and args.sample_interval_sec > 0:
        return max(1.0, 1.0 / float(args.sample_interval_sec))
    return max(1.0, float(dataset_fps) / max(1, int(args.sample_interval)))


def _wrap_overlay_text(prefix: str, value: str, width: int = 72) -> str:
    value = str(value or "").strip() or "<empty>"
    wrapped = textwrap.wrap(value, width=width) or ["<empty>"]
    lines = [f"{prefix}: {wrapped[0]}"]
    for line in wrapped[1:]:
        lines.append(" " * (len(prefix) + 2) + line)
    return "\n".join(lines)


def _render_visual_frame(
    image: Image.Image,
    input_memory: str,
    pred_subtask: str,
    next_memory: str,
    sample_id: int,
    step_index: int,
    timestamp: float,
) -> Image.Image:
    frame = image.convert("RGB").copy()
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()

    overlay_lines = [
        f"sample_id: {sample_id}",
        f"step_index: {step_index}",
        f"timestamp: {timestamp:.3f}",
        _wrap_overlay_text("input_memory", input_memory),
        _wrap_overlay_text("pred_subtask", pred_subtask),
        _wrap_overlay_text("next_memory", next_memory),
    ]
    overlay_text = "\n".join(overlay_lines)

    margin = 12
    line_spacing = 6
    bbox = draw.multiline_textbbox((margin, margin), overlay_text, font=font, spacing=line_spacing)
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    draw.rectangle(
        [margin - 6, margin - 6, margin + box_width + 6, margin + box_height + 6],
        fill=(0, 0, 0),
    )
    draw.multiline_text((margin, margin), overlay_text, fill=(255, 255, 255), font=font, spacing=line_spacing)
    return frame


def _open_visual_video_writer(video_path: Path, fps: float, width: int, height: int):
    if av is None:
        raise ImportError("pyav is required for video writing.")

    width = int(width)
    height = int(height)
    fps = float(fps)
    target_bit_rate = max(8_000_000, int(width * height * max(fps, 1.0) * 0.25))

    container = av.open(str(video_path), mode="w")
    stream = None
    last_error = None
    for codec_name in ("libx264", "h264", "mpeg4"):
        try:
            stream = container.add_stream(codec_name, rate=Fraction(fps).limit_denominator(1000))
            if codec_name in {"libx264", "h264"}:
                stream.options = {
                    "crf": "12",
                    "preset": "slow",
                    "profile": "high",
                }
            stream.bit_rate = target_bit_rate
            logging.info(
                "Opened visualization video writer with codec: %s, bit_rate=%s, resolution=%sx%s, fps=%.6f",
                codec_name,
                target_bit_rate,
                width,
                height,
                fps,
            )
            break
        except Exception as exc:
            last_error = exc

    if stream is None:
        container.close()
        raise RuntimeError(f"Failed to open visualization video writer: {last_error}")

    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    return container, stream


def _write_visual_video_frame(container, stream, image: Image.Image) -> None:
    frame = av.VideoFrame.from_image(image.convert("RGB"))
    for packet in stream.encode(frame):
        container.mux(packet)


def _close_visual_video_writer(container, stream) -> None:
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _open_video_container(video_path: str):
    if av is None:
        raise ImportError("pyav is required for video decoding.")
    if _is_ceph_path(video_path):
        return av.open(io.BytesIO(fileio.get(video_path)))
    return av.open(video_path)


def _decoded_frame_to_image(decoded_frame) -> Image.Image:
    return Image.fromarray(decoded_frame.to_ndarray(format="rgb24")).convert("RGB")


def _decoded_frame_timestamp_sec(decoded_frame, time_base) -> Optional[float]:
    if getattr(decoded_frame, "time", None) is not None:
        return float(decoded_frame.time)
    if decoded_frame.pts is not None and time_base is not None:
        return float(decoded_frame.pts * time_base)
    return None


def _decode_frame_image_by_index(video_path: str, frame_index: int) -> Image.Image:
    frame_index = max(int(frame_index), 0)
    container = _open_video_container(video_path)
    try:
        video_stream = container.streams.video[0]
        for current_index, decoded_frame in enumerate(container.decode(video_stream)):
            if current_index == frame_index:
                return _decoded_frame_to_image(decoded_frame)
    finally:
        container.close()

    raise IndexError(f"Failed to decode frame index {frame_index} from {video_path}")


def _decode_frame_image(video_path: str, timestamp_sec: float) -> Image.Image:
    target_timestamp_sec = max(float(timestamp_sec), 0.0)

    container = _open_video_container(video_path)
    try:
        video_stream = container.streams.video[0]
        time_base = video_stream.time_base
        duration = video_stream.duration

        if time_base is not None and duration is not None:
            max_timestamp_sec = max(0.0, float(duration * time_base))
            target_timestamp_sec = min(target_timestamp_sec, max_timestamp_sec)

        if time_base is not None:
            target_pts = int(target_timestamp_sec / float(time_base))
            best_image = None
            best_distance = None
            try:
                container.seek(max(target_pts, 0), stream=video_stream, backward=True, any_frame=False)
                for decoded_count, decoded_frame in enumerate(container.decode(video_stream), start=1):
                    current_timestamp_sec = _decoded_frame_timestamp_sec(decoded_frame, time_base)
                    current_image = _decoded_frame_to_image(decoded_frame)

                    if current_timestamp_sec is None:
                        if best_image is None:
                            best_image = current_image
                        if decoded_count >= 256:
                            break
                        continue

                    distance = abs(current_timestamp_sec - target_timestamp_sec)
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_image = current_image

                    if current_timestamp_sec >= target_timestamp_sec or decoded_count >= 256:
                        break

                if best_image is not None:
                    return best_image
            except Exception:
                pass
    finally:
        container.close()

    container = _open_video_container(video_path)
    try:
        video_stream = container.streams.video[0]
        time_base = video_stream.time_base
        best_image = None
        best_distance = None
        last_image = None

        for decoded_frame in container.decode(video_stream):
            current_image = _decoded_frame_to_image(decoded_frame)
            last_image = current_image
            current_timestamp_sec = _decoded_frame_timestamp_sec(decoded_frame, time_base)
            if current_timestamp_sec is None:
                continue

            distance = abs(current_timestamp_sec - target_timestamp_sec)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_image = current_image

            if current_timestamp_sec >= target_timestamp_sec and best_image is not None:
                return best_image

        if best_image is not None:
            return best_image
        if last_image is not None:
            return last_image
    finally:
        container.close()

    raise IndexError(f"Failed to decode timestamp {target_timestamp_sec} from {video_path}")


def _normalize_ordered_subtask_plan(ordered_subtask_plan: Any) -> List[str]:
    if isinstance(ordered_subtask_plan, str):
        normalized = [item.strip() for item in ordered_subtask_plan.split("|") if item.strip()]
        if normalized:
            return normalized
        normalized = [item.strip() for item in ordered_subtask_plan.splitlines() if item.strip()]
        if normalized:
            return normalized
    elif isinstance(ordered_subtask_plan, (list, tuple)):
        normalized = [str(item).strip() for item in ordered_subtask_plan if str(item).strip()]
        if normalized:
            return normalized
    return []


def _format_detailed_global_task_instruction(ordered_subtask_plan: Sequence[str]) -> str:
    lines = [str(action_text).strip() for action_text in ordered_subtask_plan if str(action_text).strip()]
    if not lines:
        return ""
    if len(lines) == 1 and "|" not in lines[0]:
        return lines[0]
    return "\n".join(f"{idx + 1}. {action_text}" for idx, action_text in enumerate(lines))


def _build_messages(
    task_text: str,
    input_memory: str,
    current_images: Sequence[Image.Image],
    video_keys: Sequence[str],
    ordered_subtask_plan: Optional[Sequence[str]] = None,
    detailed_task: Optional[str] = None,
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []

    header_lines: List[str] = []
    if task_text:
        header_lines.append(f"Global Task Goal: {task_text}")
    if input_memory:
        header_lines.append(f"Input Language Memory: {input_memory}")
    if ordered_subtask_plan:
        subtask_plan_text = _format_detailed_global_task_instruction(ordered_subtask_plan)
        if subtask_plan_text:
            header_lines.append(f"Subtask List:\n{subtask_plan_text}")
    if detailed_task:
        detailed_task_text = str(detailed_task or "").strip()
        if detailed_task_text:
            header_lines.append(f"Detailed Global Task Instruction: {detailed_task_text}")
    header_lines.append("Candidate Atomic Skills: [" + ", ".join(ROBOT_ATOMIC_SKILLS) + "]")
    header_text = "\n".join(header_lines).strip()

    observation_prompt_lines = _build_current_observation_prompt_lines(video_keys)
    if len(observation_prompt_lines) == 1 and len(current_images) == 1:
        observation_text = observation_prompt_lines[0]
        if header_text:
            observation_text = f"{header_text}\n{observation_text}"
        user_content.append({"type": "text", "text": observation_text})
        user_content.append({"type": "image", "image": current_images[0]})
    else:
        observation_text = observation_prompt_lines[0]
        if header_text:
            observation_text = f"{header_text}\n{observation_text}"
        user_content.append({"type": "text", "text": observation_text})
        for image_idx, image in enumerate(current_images):
            view_label = _format_view_label(image_idx, len(current_images), video_keys)
            user_content.append({"type": "text", "text": f"Image {image_idx + 1} ({view_label}):"})
            user_content.append({"type": "image", "image": image})

    prediction_instruction = (
        'Choose exactly one atomic skill from the candidate list when the task is in progress; '
        'if the task is already completed, set current_skill to null and current_subtask to task_completed. '
        'Predict the subtask and active language memory that should be active now. '
        'Return JSON only with keys "current_skill", "current_subtask", and "active_language_memory".'
    )
    user_content.append({"type": "text", "text": prediction_instruction})

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": user_content},
    ]


def _move_batch_to_device(batch: Dict[str, Any], device: str, model_dtype: torch.dtype) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.dtype.is_floating_point:
                moved[key] = value.to(device=device, dtype=model_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    candidates: List[str] = [text]
    candidates.extend(match.strip() for match in re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE))

    def _iter_candidates(raw: str):
        depth = 0
        in_string = False
        escaped = False
        start = -1
        for idx, ch in enumerate(raw):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield raw[start : idx + 1].strip()

    candidates.extend(_iter_candidates(text))

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_field_fallback(text: str, key: str) -> str:
    patterns = [
        rf'"{re.escape(key)}"\s*:\s*"([^"]+)"',
        rf"'{re.escape(key)}'\s*:\s*'([^']+)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def _parse_prediction(text: str) -> Dict[str, Any]:
    prediction = _extract_json_block(text) or {}
    if "current_subtask" not in prediction:
        fallback = _extract_field_fallback(text, "current_subtask")
        if fallback:
            prediction["current_subtask"] = fallback
    if "active_language_memory" not in prediction:
        fallback = _extract_field_fallback(text, "active_language_memory")
        if fallback:
            prediction["active_language_memory"] = fallback
    return prediction


def _get_episode_detailed_task_text_and_source(
    episode: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    if not isinstance(episode, dict):
        return "", ""

    for key in (
        "detailed_task_instruction",
        "detailed_global_task_instruction",
        "global_task_instruction",
        "task_instruction",
    ):
        value = episode.get(key, "")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        if value:
            return value, key
    return "", ""


def _get_completion_memory_text(task_text: str) -> str:
    if task_text:
        return f"The task '{task_text}' has been completed."
    return "The current task has been completed."


def _get_action_config_input_memory(
    action_config: Sequence[Dict[str, Any]],
    subtask_id: int,
) -> Optional[str]:
    if subtask_id < 0 or subtask_id >= len(action_config):
        return None

    subtask = action_config[subtask_id]
    for key in ("language_memory", "active_language_memory", "updated_language_memory"):
        value = subtask.get(key, "")
        if value:
            return value
    return None


def _get_action_config_active_memory(
    action_config: Sequence[Dict[str, Any]],
    subtask_id: int,
    task_text: str,
) -> Optional[str]:
    if subtask_id < 0 or subtask_id > len(action_config):
        return None

    if subtask_id == len(action_config):
        return _get_completion_memory_text(task_text)

    subtask = action_config[subtask_id]
    value = subtask.get("language_memory", "")
    return value or None


def _resolve_target_subtask_id_for_step(
    action_config: Sequence[Dict[str, Any]],
    step_index: int,
    fps: float,
    args: InferenceArguments,
) -> Optional[Tuple[int, int, str]]:
    if not action_config:
        return None

    frame_id = int(step_index)
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        return None

    tail_frames = max(1, math.ceil(float(args.transition_tail_sec) * fps))
    head_frames = max(1, math.ceil(float(args.transition_head_sec) * fps))
    last_tail_frames = max(1, math.ceil(float(args.last_tail_sec) * fps))
    ignore_frames = max(0, math.ceil(float(args.ignore_boundary_sec) * fps))

    for subtask_id, subtask in enumerate(action_config):
        try:
            start = int(subtask.get("start_frame", 0))
            end = int(subtask.get("end_frame", start))
        except Exception:
            continue
        if end <= start:
            continue

        is_first = subtask_id == 0
        is_last = subtask_id == len(action_config) - 1

        if not is_first:
            prev_subtask = action_config[subtask_id - 1]
            prev_start = int(prev_subtask.get("start_frame", 0))
            transition_start = max(start - tail_frames, prev_start)
            transition_end = min(end, start + head_frames)
            if transition_start <= frame_id < transition_end:
                return subtask_id, subtask_id - 1, "transition_dense"
            stable_start = transition_end
        else:
            stable_start = start

        stable_end = max(stable_start, end - tail_frames) if not is_last else max(stable_start, end - last_tail_frames)
        uniform_start = stable_start + (ignore_frames if not is_first else 0)
        uniform_end = stable_end - ignore_frames
        if uniform_end > uniform_start and uniform_start <= frame_id < uniform_end:
            return subtask_id, subtask_id, "uniform"

        if is_last:
            tail_start = stable_end
            if tail_start <= frame_id < end:
                return subtask_id + 1, subtask_id, "final_tail_dense"

    return None


def _build_closed_loop_eval_sample(
    *,
    episode: Optional[Dict[str, Any]],
    episode_index: int,
    step_index: int,
    task_text: str,
    input_memory: str,
    video_keys: Sequence[str],
    fps: float,
    args: InferenceArguments,
) -> Optional[_JudgeSubtaskEvalSample]:
    if not isinstance(episode, dict):
        return None

    action_config = episode.get("action_config", [])
    if not isinstance(action_config, list) or not action_config:
        return None

    target_info = _resolve_target_subtask_id_for_step(action_config, step_index, fps, args)
    if target_info is None:
        return None
    target_subtask_id, input_memory_subtask_id, sample_kind = target_info

    gt_current_subtask,gt_current_skill = _get_eval_subtask_action_text(
        action_config,
        target_subtask_id
    )
    gt_active_language_memory = _get_action_config_active_memory(
        action_config,
        target_subtask_id,
        task_text
    )
    if not gt_current_subtask or not gt_active_language_memory:
        return None

    detailed_task_text, detailed_task_source = _get_episode_detailed_task_text_and_source(
        episode,
    )
    task_id = str(episode.get("task_id", "") or "")

    return _JudgeSubtaskEvalSample(
        sample_kind=sample_kind,
        dataset_id=0,
        episode_pos=0,
        task_id=task_id,
        episode_index=int(episode_index),
        frame_id=int(step_index),
        target_subtask_id=int(target_subtask_id),
        input_memory_subtask_id=int(input_memory_subtask_id),
        task_text=str(task_text or "").strip(),
        detailed_task_text=detailed_task_text,
        detailed_task_source=detailed_task_source,
        input_language_memory=str(input_memory or "").strip(),
        gt_current_subtask=gt_current_subtask,
        gt_active_language_memory=gt_active_language_memory,
        gt_current_skill=gt_current_skill,
        video_keys=list(video_keys),
    )


def _score_closed_loop_prediction(
    *,
    sample: _JudgeSubtaskEvalSample,
    pred_text: str,
    pred_obj: Optional[Dict[str, Any]],
    judge_model,
    judge_tokenizer,
    judge_device: str,
    judge_max_new_tokens: int,
) -> Dict[str, Any]:
    prediction_json = dict(pred_obj) if isinstance(pred_obj, dict) else {}
    pred_subtask = _extract_eval_current_subtask_text(pred_text, prediction_json)
    pred_memory_raw = prediction_json.get("active_language_memory", "")
    pred_memory = _clean_eval_memory_text(pred_memory_raw)
    if pred_memory != str(pred_memory_raw or ""):
        prediction_json["active_language_memory"] = pred_memory

    judge_prompt = _build_shared_judge_prompt(sample, pred_text)
    judge_obj, judge_raw = _run_shared_judge(
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        prompt=judge_prompt,
        device=judge_device,
        max_new_tokens=judge_max_new_tokens,
    )

    char_exact_subtask_match = str(pred_subtask or "").strip() == str(sample.gt_current_subtask or "").strip()
    char_exact_memory_match = str(pred_memory or "").strip() == str(sample.gt_active_language_memory or "").strip()
    char_match_override_applied = char_exact_subtask_match or char_exact_memory_match
    if char_exact_subtask_match:
        judge_obj["subtask_score"] = 5.0
    if char_exact_memory_match:
        judge_obj["memory_score"] = 5.0
    if char_match_override_applied:
        final_subtask_score = float(judge_obj.get("subtask_score", 0.0) or 0.0)
        final_memory_score = float(judge_obj.get("memory_score", 0.0) or 0.0)
        final_total_score = final_subtask_score + final_memory_score
        judge_obj["total_score"] = final_total_score
        if final_total_score >= 9.999:
            judge_obj["verdict"] = "correct"
        elif final_total_score <= 0.0:
            judge_obj["verdict"] = "wrong"
        else:
            judge_obj["verdict"] = "partial"
        judge_obj["reason"] = "char_match_override"

    normalized_subtask_match = _normalize_eval_text(pred_subtask) == _normalize_eval_text(sample.gt_current_subtask)
    normalized_memory_match = _normalize_eval_text(pred_memory) == _normalize_eval_text(sample.gt_active_language_memory)
    return {
        "gt": {
            "current_subtask": sample.gt_current_subtask,
            "active_language_memory": sample.gt_active_language_memory,
        },
        "prediction_json": prediction_json,
        "pred_current_subtask": pred_subtask,
        "pred_active_language_memory": pred_memory,
        "judge": judge_obj,
        "judge_raw": judge_raw,
        "char_exact_subtask_match": char_exact_subtask_match,
        "char_exact_memory_match": char_exact_memory_match,
        "char_match_override_applied": char_match_override_applied,
        "normalized_subtask_match": normalized_subtask_match,
        "normalized_memory_match": normalized_memory_match,
        "target_subtask_id": sample.target_subtask_id,
        "input_memory_subtask_id": sample.input_memory_subtask_id,
        "sample_kind": sample.sample_kind,
        "detailed_task_source": sample.detailed_task_source,
    }


def _build_task_score_summary(episode_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_task: Dict[str, Dict[str, Any]] = {}
    overall = {
        "episode_count": 0,
        "evaluated_episode_count": 0,
        "sample_count": 0,
        "evaluated_sample_count": 0,
        "skipped_content_policy_count": 0,
        "total_subtask_score": 0.0,
        "total_memory_score": 0.0,
        "total_score": 0.0,
    }

    for item in episode_summaries:
        overall["episode_count"] += 1
        eval_summary = item.get("closed_loop_eval")
        if not isinstance(eval_summary, dict):
            continue

        task_key = str(item.get("task_text", "") or "<empty_task>").strip() or "<empty_task>"
        bucket = per_task.setdefault(
            task_key,
            {
                "task_text": str(item.get("task_text", "") or "").strip(),
                "episode_count": 0,
                "sample_count": 0,
                "skipped_content_policy_count": 0,
                "total_subtask_score": 0.0,
                "total_memory_score": 0.0,
                "total_score": 0.0,
            },
        )

        episode_eval_count = int(eval_summary.get("count", 0) or 0)
        bucket["episode_count"] += 1
        bucket["sample_count"] += episode_eval_count
        bucket["skipped_content_policy_count"] += int(item.get("skipped_content_policy_count", 0) or 0)
        bucket["total_subtask_score"] += float(eval_summary.get("total_subtask_score", 0.0) or 0.0)
        bucket["total_memory_score"] += float(eval_summary.get("total_memory_score", 0.0) or 0.0)
        bucket["total_score"] += float(eval_summary.get("total_score", 0.0) or 0.0)

        overall["evaluated_episode_count"] += 1
        overall["sample_count"] += int(item.get("num_samples", 0) or 0)
        overall["evaluated_sample_count"] += episode_eval_count
        overall["skipped_content_policy_count"] += int(item.get("skipped_content_policy_count", 0) or 0)
        overall["total_subtask_score"] += float(eval_summary.get("total_subtask_score", 0.0) or 0.0)
        overall["total_memory_score"] += float(eval_summary.get("total_memory_score", 0.0) or 0.0)
        overall["total_score"] += float(eval_summary.get("total_score", 0.0) or 0.0)

    per_task_summary: List[Dict[str, Any]] = []
    for _, bucket in sorted(per_task.items(), key=lambda item: item[0]):
        evaluated_count = max(int(bucket["sample_count"]), 1)
        per_task_summary.append(
            {
                **bucket,
                "avg_subtask_score": float(bucket["total_subtask_score"]) / evaluated_count,
                "avg_memory_score": float(bucket["total_memory_score"]) / evaluated_count,
                "avg_total_score": float(bucket["total_score"]) / evaluated_count,
            }
        )

    evaluated_samples = max(int(overall["evaluated_sample_count"]), 1) if overall["evaluated_sample_count"] else 0
    overall_summary = {
        **overall,
        "avg_subtask_score": (float(overall["total_subtask_score"]) / evaluated_samples) if evaluated_samples else None,
        "avg_memory_score": (float(overall["total_memory_score"]) / evaluated_samples) if evaluated_samples else None,
        "avg_total_score": (float(overall["total_score"]) / evaluated_samples) if evaluated_samples else None,
    }
    return {
        "overall": overall_summary,
        "per_task": per_task_summary,
    }


def _stable_int_from_text(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _select_eval_episodes_per_task(
    episodes: Sequence[Dict[str, Any]],
    reader: LeRobotEpisodeReader,
    seed: int,
    num_eval_episodes_per_task: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for episode in episodes:
        task_text = reader.resolve_task_text_from_episode_metadata(episode)
        task_key = task_text or str(episode.get("task_id", "") or "<empty_task>").strip() or "<empty_task>"
        grouped.setdefault(task_key, []).append(episode)

    selected: List[Dict[str, Any]] = []
    for task_key in sorted(grouped.keys()):
        task_episodes = sorted(
            grouped[task_key],
            key=lambda item: _normalize_episode_index(item.get("episode_index", -1)),
        )
        if num_eval_episodes_per_task > 0 and len(task_episodes) > num_eval_episodes_per_task:
            task_seed = int(seed) ^ _stable_int_from_text(task_key)
            rng = random.Random(task_seed)
            picked_indices = sorted(rng.sample(range(len(task_episodes)), num_eval_episodes_per_task))
            chosen = [task_episodes[idx] for idx in picked_indices]
        else:
            chosen = task_episodes
        selected.extend(chosen)

    return selected


def _split_episodes_for_rank(
    episodes: Sequence[Dict[str, Any]],
    rank: int,
    world_size: int,
) -> List[Dict[str, Any]]:
    if world_size <= 1:
        return list(episodes)
    return [episode for idx, episode in enumerate(episodes) if idx % world_size == rank]


def _merge_jsonl_shards(shard_paths: Sequence[Path], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fout:
        for shard_path in shard_paths:
            if not shard_path.exists():
                continue
            with shard_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)


def _build_episode_summaries_from_prediction_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    per_episode: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            episode_index = int(record.get("episode_index", -1))
        except Exception:
            continue
        per_episode.setdefault(episode_index, []).append(record)

    summaries: List[Dict[str, Any]] = []
    for episode_index in sorted(per_episode.keys()):
        episode_records = per_episode[episode_index]
        episode_records.sort(key=lambda item: int(item.get("sample_id", 0) or 0))
        first = episode_records[0]
        last = episode_records[-1]

        eval_records = []
        skipped_content_policy_count = 0
        for item in episode_records:
            if str(item.get("skip_reason", "") or "") == "content_policy_violation":
                skipped_content_policy_count += 1
            closed_loop_eval = item.get("closed_loop_eval")
            if isinstance(closed_loop_eval, dict) and isinstance(closed_loop_eval.get("judge"), dict):
                eval_records.append(closed_loop_eval)

        result = {
            "episode_index": episode_index,
            "task_text": str(first.get("task_text", "") or "").strip(),
            "num_samples": len(episode_records),
            "final_memory": str(last.get("next_input_language_memory", "") or "").strip(),
            "visual_video_path": last.get("visual_video_path") or first.get("visual_video_path"),
            "sampled_head_image_dir": None,
            "sample_indices": [int(item.get("step_index", 0) or 0) for item in episode_records],
            "skipped_content_policy_count": skipped_content_policy_count,
        }

        sampled_head_image_path = last.get("sampled_head_image_path") or first.get("sampled_head_image_path")
        if sampled_head_image_path:
            result["sampled_head_image_dir"] = str(Path(str(sampled_head_image_path)).parent)

        eval_count = len(eval_records)
        if eval_count > 0:
            total_subtask_score = sum(float(item["judge"].get("subtask_score", 0.0) or 0.0) for item in eval_records)
            total_memory_score = sum(float(item["judge"].get("memory_score", 0.0) or 0.0) for item in eval_records)
            total_score = sum(float(item["judge"].get("total_score", 0.0) or 0.0) for item in eval_records)
            subtask_exact = sum(1 for item in eval_records if bool(item.get("normalized_subtask_match", False)))
            memory_exact = sum(1 for item in eval_records if bool(item.get("normalized_memory_match", False)))
            result["closed_loop_eval"] = {
                "count": eval_count,
                "total_subtask_score": total_subtask_score,
                "total_memory_score": total_memory_score,
                "total_score": total_score,
                "avg_subtask_score": total_subtask_score / eval_count,
                "avg_memory_score": total_memory_score / eval_count,
                "avg_total_score": total_score / eval_count,
                "subtask_exact_rate": subtask_exact / eval_count,
                "memory_exact_rate": memory_exact / eval_count,
            }

        summaries.append(result)
    return summaries

@staticmethod
def _format_subtask_list_item(subtask_id, action_text):
    return f"{subtask_id + 1}. {action_text}"

def _build_subtask_list_text(episode):
    action_config = episode.get("action_config", []) if isinstance(episode, dict) else []
    subtask_lines = ["Subtask List:"]
    for subtask_id, subtask in enumerate(action_config):
        action_text = str(subtask.get("action_text", "")).strip()
        if not action_text:
            action_text = str(subtask.get("skill", "")).strip()
        if not action_text:
            continue
        subtask_lines.append(_format_subtask_list_item(subtask_id, action_text))
    if len(subtask_lines) == 1:
        subtask_lines.append("1. task_completed")
    return "\n".join(subtask_lines)

def _build_episode_summaries_from_prediction_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return _build_episode_summaries_from_prediction_records(records)


def _process_episode(
    episode_index: int,
    timestamps: Sequence[float],
    sample_indices: Sequence[int],
    video_paths: Dict[str, str],
    video_keys: Sequence[str],
    initial_task_text: str,
    task_text_resolver: Callable[[int, str], str],
    args: InferenceArguments,
    model,
    processor,
    tokenizer,
    device: str,
    fout,
    visual_video_dir: Optional[Path],
    visual_video_fps: float,
    decode_with_frame_index: bool = False,
    episode_metadata: Optional[Dict[str, Any]] = None,
    judge_model=None,
    judge_tokenizer=None,
    judge_device: Optional[str] = None,
    policy_session_prefix: str = "eval",
) -> Dict[str, Any]:
    previous_output_memory = ""
    task_text = initial_task_text
    sample_count = 0
    eval_count = 0
    eval_total_subtask_score = 0.0
    eval_total_memory_score = 0.0
    eval_total_score = 0.0
    eval_subtask_exact = 0
    eval_memory_exact = 0
    skipped_content_policy_count = 0
    visual_video_path = None
    visual_video_container = None
    visual_video_stream = None
    head_view_index = _resolve_head_view_index(video_keys)
    sampled_head_image_dir = None
    current_head_image_index = _get_current_observation_image_index(video_keys, head_view_index)
    episode_detailed_task_text = ""
    if episode_metadata is not None:
        episode_detailed_task_text, _ = _get_episode_detailed_task_text_and_source(
            episode_metadata,
        )
    fallback_detailed_task_text = str(getattr(args, "detailed_task", None) or "").strip()
    effective_detailed_task_text = episode_detailed_task_text or fallback_detailed_task_text
    policy_session_id = _build_policy_session_id(policy_session_prefix, episode_index)

    if args.use_subtask_list_instruction:
        ordered_subtask_plan = _build_subtask_list_text(episode_metadata)

    if visual_video_dir is not None:
        visual_video_path = visual_video_dir / f"episode_{episode_index:06d}.mp4"
        sampled_head_image_dir = visual_video_dir / f"episode_{episode_index:06d}_sampled_head_images"

    try:
        for sample_id, step_index in enumerate(sample_indices):
            timestamp = float(timestamps[step_index])
            input_memory = args.initial_memory.strip() if sample_id == 0 else previous_output_memory
            task_text = task_text_resolver(step_index, task_text) or task_text
            current_images: List[Image.Image] = []

            try:
                current_images = _collect_observation_images(
                    video_paths=video_paths,
                    video_keys=video_keys,
                    current_timestamp=timestamp,
                    args=args,
                    decode_with_frame_index=decode_with_frame_index,
                    current_step_index=step_index,
                )
                pred_text, pred_obj = _run_single_sample(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    device=device,
                    args=args,
                    task_text=task_text,
                    input_memory=input_memory,
                    current_images=current_images,
                    video_keys=video_keys,
                    detailed_task=effective_detailed_task_text if args.use_detailed_instruction else None,
                    ordered_subtask_plan=ordered_subtask_plan if args.use_subtask_list_instruction else None,
                    sample_id=sample_id,
                    session_id=policy_session_id,
                    reset_memory=(sample_id == 0),
                )
                skip_reason = str(pred_obj.get("skip_reason", "") or "")
                if skip_reason == "content_policy_violation":
                    skipped_content_policy_count += 1
                    predicted_memory = ""
                    next_memory = previous_output_memory or input_memory
                    current_subtask = ""
                    error = f"skip:{skip_reason}"
                else:
                    predicted_memory = _sanitize_predicted_memory(str(pred_obj.get("active_language_memory", "")).strip())
                    next_memory = predicted_memory or previous_output_memory or input_memory
                    current_subtask = str(pred_obj.get("current_subtask", "")).strip()
                    error = None
            except Exception as exc:
                logging.exception("Inference failed for episode %s sample %s: %s", episode_index, sample_id, exc)
                pred_text = ""
                pred_obj = {}
                predicted_memory = ""
                next_memory = previous_output_memory or input_memory
                current_subtask = ""
                error = str(exc)

            sampled_head_image_path = None
            if sampled_head_image_dir is not None and current_images:
                try:
                    sampled_head_image_path = _save_sampled_head_image(
                        image=current_images[min(current_head_image_index, len(current_images) - 1)],
                        visual_video_dir=visual_video_dir,
                        episode_index=episode_index,
                        sample_id=sample_id,
                        step_index=step_index,
                        pred_subtask=current_subtask,
                        next_memory=next_memory,
                    )
                except Exception as exc:
                    logging.exception("Failed to save sampled head image for episode %s sample %s: %s", episode_index, sample_id, exc)

            visual_frame_written = False
            if visual_video_path is not None and current_images:
                try:
                    annotated_frame = _render_visual_frame(
                        image=current_images[min(current_head_image_index, len(current_images) - 1)],
                        input_memory=input_memory,
                        pred_subtask=current_subtask,
                        next_memory=next_memory,
                        sample_id=sample_id,
                        step_index=step_index,
                        timestamp=timestamp,
                    )
                    if visual_video_container is None or visual_video_stream is None:
                        visual_video_container, visual_video_stream = _open_visual_video_writer(
                            visual_video_path,
                            visual_video_fps,
                            annotated_frame.size[0],
                            annotated_frame.size[1],
                        )
                    _write_visual_video_frame(visual_video_container, visual_video_stream, annotated_frame)
                    visual_frame_written = True
                except Exception as exc:
                    logging.exception("Failed to write visualization frame for episode %s sample %s: %s", episode_index, sample_id, exc)

            closed_loop_eval = None
            if (
                str(pred_obj.get("skip_reason", "") or "") != "content_policy_violation"
                and args.enable_closed_loop_eval
                and judge_model is not None
                and judge_tokenizer is not None
                and judge_device
                and episode_metadata is not None
            ):
                try:
                    eval_sample = _build_closed_loop_eval_sample(
                        episode=episode_metadata,
                        episode_index=episode_index,
                        step_index=int(step_index),
                        task_text=task_text,
                        input_memory=input_memory,
                        video_keys=video_keys,
                        fps=float(episode_metadata.get("_dataset_fps", getattr(args, "video_fps", 0.0)) or 0.0),
                        args=args,
                    )
                    if eval_sample is not None:
                        closed_loop_eval = _score_closed_loop_prediction(
                            sample=eval_sample,
                            pred_text=pred_text,
                            pred_obj=pred_obj,
                            judge_model=judge_model,
                            judge_tokenizer=judge_tokenizer,
                            judge_device=judge_device,
                            judge_max_new_tokens=int(args.judge_max_new_tokens),
                        )
                        judge_obj = closed_loop_eval["judge"]
                        eval_count += 1
                        eval_total_subtask_score += float(judge_obj.get("subtask_score", 0.0) or 0.0)
                        eval_total_memory_score += float(judge_obj.get("memory_score", 0.0) or 0.0)
                        eval_total_score += float(judge_obj.get("total_score", 0.0) or 0.0)
                        if bool(closed_loop_eval.get("normalized_subtask_match", False)):
                            eval_subtask_exact += 1
                        if bool(closed_loop_eval.get("normalized_memory_match", False)):
                            eval_memory_exact += 1
                except Exception as exc:
                    logging.exception(
                        "Closed-loop evaluation failed for episode %s sample %s: %s",
                        episode_index,
                        sample_id,
                        exc,
                    )
                    closed_loop_eval = {"error": str(exc)}

            record = {
                "episode_index": episode_index,
                "sample_id": sample_id,
                "task_text": task_text,
                "video_keys": list(video_keys),
                "video_paths": video_paths,
                "sampled_head_image_path": sampled_head_image_path,
                "visual_video_path": (str(visual_video_path) if visual_video_path is not None else None),
                "visual_frame_written": visual_frame_written,
                "step_index": int(step_index),
                "timestamp": timestamp,
                "input_language_memory": input_memory,
                "pred_current_subtask": current_subtask,
                "pred_active_language_memory": predicted_memory,
                "next_input_language_memory": next_memory,
                "prediction": pred_obj,
                "raw_prediction": pred_text,
                "closed_loop_eval": closed_loop_eval,
                "skip_reason": str(pred_obj.get("skip_reason", "") or ""),
                "error": error,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            previous_output_memory = next_memory
            sample_count += 1
            if args.max_samples_per_episode > 0 and sample_count >= args.max_samples_per_episode:
                break
    finally:
        if visual_video_container is not None and visual_video_stream is not None:
            try:
                _close_visual_video_writer(visual_video_container, visual_video_stream)
            except Exception as exc:
                logging.exception("Failed to finalize visualization video for episode %s: %s", episode_index, exc)

    result = {
        "episode_index": episode_index,
        "task_text": task_text,
        "num_samples": sample_count,
        "final_memory": previous_output_memory,
        "visual_video_path": (str(visual_video_path) if visual_video_path is not None else None),
        "sampled_head_image_dir": (str(sampled_head_image_dir) if sampled_head_image_dir is not None else None),
        "sample_indices": [int(idx) for idx in sample_indices[:sample_count]],
        "skipped_content_policy_count": skipped_content_policy_count,
    }
    if eval_count > 0:
        result["closed_loop_eval"] = {
            "count": eval_count,
            "total_subtask_score": eval_total_subtask_score,
            "total_memory_score": eval_total_memory_score,
            "total_score": eval_total_score,
            "avg_subtask_score": eval_total_subtask_score / eval_count,
            "avg_memory_score": eval_total_memory_score / eval_count,
            "avg_total_score": eval_total_score / eval_count,
            "subtask_exact_rate": eval_subtask_exact / eval_count,
            "memory_exact_rate": eval_memory_exact / eval_count,
        }
    return result


def _convert_images_to_numpy(current_images: Sequence[Image.Image]) -> List[np.ndarray]:
    image_arrays: List[np.ndarray] = []
    for image in current_images:
        if isinstance(image, Image.Image):
            image_arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        else:
            image_arrays.append(np.asarray(image, dtype=np.uint8))
    return image_arrays


def _is_content_policy_violation_error(error: Any) -> bool:
    message = ""
    if isinstance(error, dict):
        message = str(error.get("message", "") or "")
    else:
        message = str(error or "")
    lowered = message.lower()
    return "content_policy_violation" in lowered or "content safety system" in lowered


def _run_single_sample_via_websocket(
    client,
    args: InferenceArguments,
    task_text: str,
    input_memory: str,
    current_images: Sequence[Image.Image],
    video_keys: Sequence[str],
    ordered_subtask_plan: Optional[Sequence[str]] = None,
    detailed_task: Optional[str] = None,
    sample_id: int = -1,
    session_id: str = "default",
    reset_memory: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    if ordered_subtask_plan is None:
        ordered_subtask_plan = _normalize_ordered_subtask_plan(getattr(args, "ordered_subtask_plan", None))
    payload: Dict[str, Any] = {
        "request_id": f"{session_id}-sample-{max(int(sample_id), 0):06d}",
        "session_id": session_id,
        "task_text": task_text,
        "initial_memory": input_memory,
        "reset_memory": bool(reset_memory),
        "batch_images": [_convert_images_to_numpy(current_images)],
        "video_keys": list(video_keys),
    }
    if ordered_subtask_plan:
        payload["ordered_subtask_plan"] = list(ordered_subtask_plan)

    detailed_task_text = str(
        detailed_task if detailed_task is not None else getattr(args, "detailed_task", None) or ""
    ).strip()
    if detailed_task_text:
        payload["detailed_task"] = detailed_task_text

    response = client.infer(payload)
    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected response type from policy server: {type(response)}")
    if not bool(response.get("ok", True)):
        error = response.get("error")
        if _is_content_policy_violation_error(error):
            return "", {"skip_reason": "content_policy_violation", "error": error}
        raise RuntimeError(f"Policy server inference failed: {error}")

    data = response.get("data", {})
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected policy server payload: {type(data)}")

    raw_prediction = str(data.get("raw_prediction", "") or "")
    pred_obj = _parse_prediction(raw_prediction) if raw_prediction else {}

    current_subtask = str(data.get("current_subtask", pred_obj.get("current_subtask", "")) or "").strip()
    predicted_memory = _sanitize_predicted_memory(
        str(
            data.get(
                "active_language_memory",
                data.get("output_memory", pred_obj.get("active_language_memory", "")),
            )
            or ""
        ).strip()
    )
    if current_subtask:
        pred_obj["current_subtask"] = current_subtask
    if predicted_memory:
        pred_obj["active_language_memory"] = predicted_memory

    if not raw_prediction:
        raw_prediction = json.dumps(
            {
                "current_subtask": current_subtask,
                "active_language_memory": predicted_memory,
            },
            ensure_ascii=False,
        )
    return raw_prediction, pred_obj


@torch.inference_mode()
def _run_single_sample(
    model,
    processor,
    tokenizer,
    device: str,
    args: InferenceArguments,
    task_text: str,
    input_memory: str,
    current_images: Sequence[Image.Image],
    video_keys: Sequence[str],
    ordered_subtask_plan: Optional[Sequence[str]] = None,
    detailed_task: Optional[str] = None,
    sample_id: int = -1,
    session_id: str = "default",
    reset_memory: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    policy_backend = _normalize_policy_backend(getattr(args, "policy_backend", "local"))
    if policy_backend == "websocket":
        return _run_single_sample_via_websocket(
            client=model,
            args=args,
            task_text=task_text,
            input_memory=input_memory,
            current_images=current_images,
            video_keys=video_keys,
            ordered_subtask_plan=ordered_subtask_plan,
            detailed_task=detailed_task,
            sample_id=sample_id,
            session_id=session_id,
            reset_memory=reset_memory,
        )

    if ordered_subtask_plan is None:
        ordered_subtask_plan = _normalize_ordered_subtask_plan(getattr(args, "ordered_subtask_plan", None))

    messages = _build_messages(
        task_text=task_text,
        input_memory=input_memory,
        current_images=current_images,
        video_keys=video_keys,
        ordered_subtask_plan=ordered_subtask_plan,
        detailed_task=detailed_task,
    )
    if sample_id == 0:
        logging.info("[SYS2][PROMPT] step=0 messages=%s", json.dumps(messages, ensure_ascii=False, default=str))
    model_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    model_inputs = _move_batch_to_device(model_inputs, device, model.dtype)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.do_sample:
        generate_kwargs["temperature"] = args.temperature

    output_ids = model.generate(**model_inputs, **generate_kwargs)
    prompt_len = model_inputs["input_ids"].shape[1]
    pred_ids = output_ids[:, prompt_len:]
    pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
    pred_obj = _parse_prediction(pred_text)
    return pred_text, pred_obj


def main() -> None:
    parser = transformers.HfArgumentParser((InferenceArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    rank, world_size, local_rank = _init_distributed()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_output_path = output_path if world_size == 1 else output_path.with_name(f"{output_path.stem}.rank{rank}{output_path.suffix}")
    summary_path = output_path.with_suffix(".summary.json")
    shard_summary_path = summary_path if world_size == 1 else summary_path.with_name(f"{summary_path.stem}.rank{rank}{summary_path.suffix}")
    task_summary_path = output_path.with_suffix(".task_summary.json")

    mode = _normalize_mode(args.mode)
    policy_backend = _normalize_policy_backend(args.policy_backend)
    video_keys = [key.strip() for key in args.video_keys.split(",") if key.strip()]
    if not video_keys:
        raise ValueError("--video_keys must not be empty")

    if mode == "lerobot" and not args.dataset_path:
        raise ValueError("lerobot mode requires --dataset_path")
    if mode == "simple" and not (args.task_text or "").strip():
        raise ValueError("simple mode requires --task_text")
    if mode == "simple" and not args.dataset_path and not (args.simple_video_paths or "").strip():
        raise ValueError("simple mode requires --dataset_path or --simple_video_paths")

    if mode == "lerobot":
        reader = LeRobotEpisodeReader(args.dataset_path, task_key=args.task_key)
        args.video_fps = float(reader.dataset_fps)
        logging.info("Resolved dataset fps from info.json: %.6f", args.video_fps)
    else:
        reader = None
        simple_video_paths = _resolve_simple_video_paths(args, video_keys)
        args.video_fps, _ = _inspect_video_timestamps(simple_video_paths[video_keys[0]], args.video_fps)
        logging.info("Resolved simple mode base video fps: %.6f", args.video_fps)

    if world_size > 1:
        args.device = _resolve_device_for_local_rank(args.device, local_rank)
        args.judge_device = _resolve_device_for_local_rank(args.judge_device, local_rank)
        logging.info("Distributed evaluation enabled: rank=%d world_size=%d local_rank=%d device=%s judge_device=%s", rank, world_size, local_rank, args.device, args.judge_device)

    device = _resolve_device(args.device)
    if policy_backend == "local":
        model, processor, tokenizer, device = _load_model_processor_tokenizer(args)
    else:
        model = _connect_websocket_policy_client(args)
        processor = None
        tokenizer = None
        logging.info(
            "Using remote WebSocket policy backend; skipping local model load from %s",
            args.model_name_or_path,
        )
    visual_video_dir = _prepare_visual_video_dir(output_path, args)
    visual_video_fps = _resolve_visual_video_fps(args, args.video_fps)
    logging.info("Resolved visualization video fps: %.6f", visual_video_fps)
    summary: List[Dict[str, Any]] = []
    judge_model = None
    judge_tokenizer = None
    judge_device = ""

    if args.enable_closed_loop_eval and mode == "lerobot":
        judge_device = _resolve_device(args.judge_device)
        logging.info("Loading closed-loop judge model from %s on %s", args.judge_model_path, judge_device)
        judge_model, judge_tokenizer = _load_shared_judge_model(args.judge_model_path, judge_device)
    elif args.enable_closed_loop_eval:
        logging.warning("Closed-loop evaluation is only available in lerobot mode with episode annotations; disabling it.")
        args.enable_closed_loop_eval = False

    with open(shard_output_path, "w", encoding="utf-8") as fout:
        if mode == "lerobot":
            episode_filter = _build_episode_filter(args.episode_indices)
            selected_episodes: List[Dict[str, Any]] = []
            for episode in reader.episodes:
                episode_index = _normalize_episode_index(episode["episode_index"])
                if episode_filter is not None and episode_index not in episode_filter:
                    continue
                selected_episodes.append(episode)
            if args.enable_closed_loop_eval:
                selected_episodes = _select_eval_episodes_per_task(
                    episodes=selected_episodes,
                    reader=reader,
                    seed=int(args.seed),
                    num_eval_episodes_per_task=int(args.num_eval_episodes_per_task),
                )
            if args.max_episodes > 0:
                selected_episodes = selected_episodes[: args.max_episodes]
            selected_episodes = _split_episodes_for_rank(selected_episodes, rank=rank, world_size=world_size)

            logging.info("Rank %d processing %d episodes after per-task sampling", rank, len(selected_episodes))
            episode_iter = tqdm(selected_episodes, desc=f"episodes-rank{rank}", disable=(rank != 0))
            for episode in episode_iter:
                episode_index = _normalize_episode_index(episode["episode_index"])
                try:
                    df = reader.load_episode_dataframe(episode_index)
                except Exception as exc:
                    logging.exception("Failed to load parquet for episode %s: %s", episode_index, exc)
                    continue

                if len(df) == 0:
                    logging.warning("Skipping episode %s because it is empty", episode_index)
                    continue

                if "timestamp" in df.columns:
                    timestamps = [float(ts) for ts in df["timestamp"].tolist()]
                else:
                    logging.warning(
                        "Episode %s is missing 'timestamp'; constructing timestamps from dataset fps %.6f",
                        episode_index,
                        reader.dataset_fps,
                    )
                    timestamps = [float(idx) / float(reader.dataset_fps) for idx in range(len(df))]

                sample_indices = _build_sample_indices(
                    timestamps=timestamps,
                    sample_interval=args.sample_interval,
                    sample_interval_sec=args.sample_interval_sec,
                    start_index=args.start_index,
                    include_last=args.include_last,
                )
                if not sample_indices:
                    logging.warning("Skipping episode %s because no sample indices were constructed", episode_index)
                    continue

                initial_task_text = (args.task_text or "").strip() or reader.resolve_task_text(df, sample_indices[0], fallback_episode=episode)
                video_paths = {key: reader.get_video_path(episode_index, key) for key in video_keys}

                summary.append(
                    _process_episode(
                        episode_index=episode_index,
                        timestamps=timestamps,
                        sample_indices=sample_indices,
                        video_paths=video_paths,
                        video_keys=video_keys,
                        initial_task_text=initial_task_text,
                        task_text_resolver=lambda step_index, fallback_task_text, _df=df, _episode=episode: (args.task_text or "").strip()
                        or reader.resolve_task_text(_df, step_index, fallback_episode=_episode)
                        or fallback_task_text,
                        args=args,
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        device=device,
                        fout=fout,
                        visual_video_dir=visual_video_dir,
                        visual_video_fps=visual_video_fps,
                        decode_with_frame_index=True,
                        episode_metadata=episode,
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        judge_device=judge_device,
                        policy_session_prefix=f"rank{rank}-{output_path.stem}",
                    )
                )
        else:
            video_paths = _resolve_simple_video_paths(args, video_keys)
            simple_fps, timestamps = _inspect_video_timestamps(video_paths[video_keys[0]], args.video_fps)
            args.video_fps = float(simple_fps)
            sample_indices = _build_sample_indices(
                timestamps=timestamps,
                sample_interval=args.sample_interval,
                sample_interval_sec=args.sample_interval_sec,
                start_index=args.start_index,
                include_last=args.include_last,
            )
            if not sample_indices:
                raise ValueError("simple mode constructed no sample indices")

            summary.append(
                _process_episode(
                    episode_index=0,
                    timestamps=timestamps,
                    sample_indices=sample_indices,
                    video_paths=video_paths,
                    video_keys=video_keys,
                    initial_task_text=(args.task_text or "").strip(),
                    task_text_resolver=lambda step_index, fallback_task_text: (args.task_text or "").strip() or fallback_task_text,
                    args=args,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    device=device,
                    fout=fout,
                    visual_video_dir=visual_video_dir,
                    visual_video_fps=visual_video_fps,
                    decode_with_frame_index=True,
                    policy_session_prefix=f"rank{rank}-{output_path.stem}",
                )
            )

    if args.save_episode_summary and world_size == 1:
        with open(shard_summary_path, "w", encoding="utf-8") as fout:
            json.dump(summary, fout, ensure_ascii=False, indent=2)
        logging.info("Saved shard summary to %s", shard_summary_path)

    if world_size > 1 and rank == 0:
        prediction_shards = [output_path.with_name(f"{output_path.stem}.rank{i}{output_path.suffix}") for i in range(world_size)]
        _wait_for_files(prediction_shards, timeout_sec=args.shard_sync_timeout_sec)

        _merge_jsonl_shards(prediction_shards, output_path)
        logging.info("Saved merged predictions to %s", output_path)

        need_summary = args.save_episode_summary or (args.enable_closed_loop_eval and args.save_task_summary)
        merged_summary: List[Dict[str, Any]] = []
        if need_summary:
            merged_summary = _build_episode_summaries_from_prediction_jsonl(output_path)
        if args.save_episode_summary:
            with open(summary_path, "w", encoding="utf-8") as fout:
                json.dump(merged_summary, fout, ensure_ascii=False, indent=2)
            logging.info("Saved merged summary to %s", summary_path)
        if args.enable_closed_loop_eval and args.save_task_summary:
            task_summary = _build_task_score_summary(merged_summary)
            with open(task_summary_path, "w", encoding="utf-8") as fout:
                json.dump(task_summary, fout, ensure_ascii=False, indent=2)
            logging.info("Saved merged task summary to %s", task_summary_path)
    elif world_size == 1:
        if args.enable_closed_loop_eval and args.save_task_summary:
            task_summary = _build_task_score_summary(summary)
            with open(task_summary_path, "w", encoding="utf-8") as fout:
                json.dump(task_summary, fout, ensure_ascii=False, indent=2)
            logging.info("Saved task summary to %s", task_summary_path)
        logging.info("Saved predictions to %s", output_path)

    if policy_backend == "websocket" and hasattr(model, "close"):
        try:
            model.close()
        except Exception as exc:
            logging.warning("Failed to close policy client cleanly: %s", exc)

def start_debugpy_once() -> None:
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    port = int(os.getenv("DEBUGPY_PORT", "10092"))
    debugpy.listen(("0.0.0.0", port))
    print(f"Rank 0 waiting for debugger attach on port {port}...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    if os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        start_debugpy_once()
    main()
