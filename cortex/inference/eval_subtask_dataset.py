import bisect
import hashlib
import math
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from mmengine import fileio
from torch.utils.data import Dataset

try:
    import av
except ImportError:
    av = None

from cortex.inference.eval_utils import (
    rank0_print,
    read_json,
    read_jsonl,
)

def _load_annotation(path: str) -> List[Dict[str, Any]]:
    if str(path).lower().endswith(".jsonl"):
        return read_jsonl(path)
    data = read_json(path)
    if isinstance(data, dict):
        for key in ("episodes", "data", "annotations"):
            if isinstance(data.get(key), list):
                return data[key]
    if not isinstance(data, list):
        raise ValueError(f"Annotation file must contain a list of episodes: {path}")
    return data


def _get_dataset_sampling_seed(data_args, data: Dict[str, Any]) -> int:
    sampling_seed = int(getattr(data_args, "dataset_sampling_seed", getattr(data_args, "seed", 42)))
    rng_key = f"{sampling_seed}|{data.get('dataset_name', '')}|{data.get('annotation_path', '')}"
    digest = hashlib.sha256(rng_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)


def _build_sampled_indices(total: int, sampling_rate: float, data_args, data: Dict[str, Any]) -> Optional[List[int]]:
    sampling_rate = float(sampling_rate)
    if total == 0:
        return []
    if math.isclose(sampling_rate, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return None
    if sampling_rate < 0:
        raise ValueError(f"sampling_rate must be non-negative, got {sampling_rate}")

    rng = random.Random(_get_dataset_sampling_seed(data_args, data))
    if sampling_rate < 1.0:
        return sorted(rng.sample(range(total), int(total * sampling_rate)))

    full_repeats = int(math.floor(sampling_rate + 1e-12))
    sampled_indices = list(range(total)) * full_repeats
    extra = int(total * max(0.0, sampling_rate - full_repeats))
    if extra > 0:
        sampled_indices.extend(sorted(rng.sample(range(total), extra)))
    return sampled_indices


def _load_dataset_config_file(path: str) -> List[Dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict):
        if isinstance(data.get("datasets"), list):
            data = data["datasets"]
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Dataset config must be a JSON object/list: {path}")
    return [dict(item) for item in data]


def _parse_inline_dataset_config(item: str) -> Optional[Dict[str, Any]]:
    if "=" not in item:
        return None
    cfg: Dict[str, Any] = {}
    for part in item.split(";"):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Inline dataset config item must be key=value: {part}")
        key, value = part.split("=", 1)
        cfg[key.strip()] = value.strip()
    if "annotation_path" not in cfg:
        raise ValueError("Inline dataset config requires annotation_path=...")
    cfg.setdefault("data_path", str(Path(cfg["annotation_path"]).parent))
    return cfg


def resolve_eval_dataset_configs(dataset_source: str, dataset_config_path: str = "") -> List[Dict[str, Any]]:
    if dataset_config_path:
        return _load_dataset_config_file(dataset_config_path)

    source = str(dataset_source or "").strip()
    if not source:
        raise ValueError("Please provide --eval_dataset or --eval_dataset_config.")

    path = Path(source)
    if path.exists() and path.is_file():
        return _load_dataset_config_file(source)

    configs = []
    inline_items = [item.strip() for item in source.split(",") if item.strip()]
    if inline_items and all("=" in item for item in inline_items):
        for item in inline_items:
            parsed = _parse_inline_dataset_config(item)
            if parsed:
                configs.append(parsed)
        return configs

    raise ValueError(
        "Could not resolve eval dataset configs from --eval_dataset. "
        "Pass --eval_dataset_config as a JSON file, or use inline entries like "
        "annotation_path=/path/episodes.jsonl;data_path=/path/videos;video_keys=observation.images.head."
    )


class EvalSubtaskDataset(Dataset):
    """Evaluation-only subtask dataset, decoupled from supervised training datasets."""

    def __init__(self, tokenizer=None, processor=None, data_args=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_args = data_args
        self.dataset_blocks: List[Dict[str, Any]] = []
        self.dataset_cum_sizes: List[int] = []
        self.running_total = 0

        dataset_source = getattr(data_args, "dataset_use", "") or getattr(data_args, "eval_dataset", "")
        dataset_config_path = getattr(data_args, "eval_dataset_config", "")
        for data in resolve_eval_dataset_configs(dataset_source, dataset_config_path):
            self._append_dataset_block(data)

        if self.running_total <= 0:
            raise ValueError("No evaluation samples constructed for EvalSubtaskDataset.")
        rank0_print(f"Total evaluation samples: {self.running_total}")

    def __len__(self):
        return self.running_total

    def __getitem__(self, index):
        raise NotImplementedError(
            "EvalSubtaskDataset stores metadata for checkpoint auto-eval. "
            "Direct tokenized DataLoader inference is not supported by this open-source dataset."
        )

    def _append_dataset_block(self, data: Dict[str, Any]) -> None:
        episodes = _load_annotation(data["annotation_path"])
        sample_interleave = int(data.get("sample_interleave", getattr(self.data_args, "sample_interleave", 8)))
        dense_sample_step = int(data.get("dense_sample_step", 6))
        final_tail_sample_step = int(data.get("final_tail_sample_step", 3))
        ignore_boundary_sec = float(data.get("ignore_boundary_sec", 0.0))
        transition_tail_sec = float(data.get("transition_tail_sec", getattr(self.data_args, "transition_tail_sec", 0.5)))
        transition_head_sec = float(data.get("transition_head_sec", getattr(self.data_args, "transition_head_sec", 0.5)))
        last_tail_sec = float(data.get("last_tail_sec", getattr(self.data_args, "last_tail_sec", 1.0)))
        video_keys = [key.strip() for key in str(data.get("video_keys", getattr(self.data_args, "video_key", ""))).split(",") if key.strip()]
        if not video_keys:
            video_keys = ["observation.images.head"]

        sampling_blocks = []
        cum_counts = []
        total = 0
        sampling_stats = {"uniform": 0, "transition_dense": 0, "final_tail_dense": 0}
        data_path = str(data["data_path"]).rstrip("/")

        for episode_pos, episode in enumerate(episodes):
            action_config = episode.get("action_config", [])
            if not action_config:
                continue
            fps = float(self._load_dataset_fps(os.path.join(data_path, str(episode.get("task_id", ""))), episode))
            episode["_dataset_fps"] = fps
            tail_frames = max(1, math.ceil(transition_tail_sec * fps))
            head_frames = max(1, math.ceil(transition_head_sec * fps))
            last_tail_frames = max(1, math.ceil(last_tail_sec * fps))
            ignore_frames = max(0, math.ceil(ignore_boundary_sec * fps))

            for subtask_id, subtask in enumerate(action_config):
                start = int(subtask["start_frame"])
                end = int(subtask["end_frame"])
                if end <= start:
                    continue

                is_first = subtask_id == 0
                is_last = subtask_id == len(action_config) - 1
                if not is_first:
                    transition_start = max(start - tail_frames, int(action_config[subtask_id - 1]["start_frame"]))
                    transition_end = min(end, start + head_frames)
                    total = self._add_block(
                        sampling_blocks, cum_counts, total, episode_pos, subtask_id, subtask_id - 1,
                        transition_start, transition_end, step=dense_sample_step,
                        block_kind="transition_dense", sampling_stats=sampling_stats,
                    )
                    stable_start = transition_end
                else:
                    stable_start = start

                stable_end = max(stable_start, end - (last_tail_frames if is_last else tail_frames))
                total = self._add_block(
                    sampling_blocks, cum_counts, total, episode_pos, subtask_id, subtask_id,
                    stable_start, stable_end,
                    sample_min_frame=stable_start + (ignore_frames if not is_first else 0),
                    sample_max_frame=stable_end - ignore_frames,
                    step=sample_interleave, block_kind="uniform", sampling_stats=sampling_stats,
                )
                if is_last:
                    total = self._add_block(
                        sampling_blocks, cum_counts, total, episode_pos, subtask_id + 1, subtask_id,
                        stable_end, end, step=final_tail_sample_step,
                        block_kind="final_tail_dense", sampling_stats=sampling_stats,
                    )

        sampled_indices = _build_sampled_indices(total, float(data.get("sampling_rate", 1.0)), self.data_args, data)
        num_samples = len(sampled_indices) if sampled_indices is not None else total
        self.dataset_blocks.append(
            {
                "episodes": episodes,
                "sampling_blocks": sampling_blocks,
                "cum_counts": cum_counts,
                "sampled_indices": sampled_indices,
                "num_samples": num_samples,
                "data_path": data_path,
                "dataset_name": data.get("dataset_name", ""),
                "sampling_stats": sampling_stats,
                "video_keys": video_keys,
                "transition_tail_sec": transition_tail_sec,
                "transition_head_sec": transition_head_sec,
                "last_tail_sec": last_tail_sec,
            }
        )
        self.running_total += num_samples
        self.dataset_cum_sizes.append(self.running_total)

    def _add_block(self, sampling_blocks, cum_counts, total, episode_pos, target_subtask_id, input_memory_subtask_id, frame_start, frame_end, sample_min_frame=None, sample_max_frame=None, step=None, block_kind=None, sampling_stats=None):
        frame_start = int(frame_start)
        frame_end = int(frame_end)
        step = int(step)
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        if frame_end <= frame_start:
            return total
        sample_min_frame = frame_start if sample_min_frame is None else max(frame_start, int(sample_min_frame))
        sample_max_frame = frame_end if sample_max_frame is None else min(frame_end, int(sample_max_frame))
        if sample_max_frame <= sample_min_frame:
            return total
        full_num_samples = math.ceil((frame_end - frame_start) / step)
        first_sample_idx = max(0, min(math.ceil((sample_min_frame - frame_start) / step), full_num_samples))
        last_sample_exclusive = max(first_sample_idx, min(math.ceil((sample_max_frame - frame_start) / step), full_num_samples))
        num_samples = last_sample_exclusive - first_sample_idx
        if num_samples <= 0:
            return total
        sampling_blocks.append((episode_pos, target_subtask_id, input_memory_subtask_id, frame_start + first_sample_idx * step, step, num_samples, block_kind))
        if sampling_stats is not None:
            sampling_stats[block_kind] = sampling_stats.get(block_kind, 0) + num_samples
        total += num_samples
        cum_counts.append(total)
        return total

    def _load_dataset_fps(self, data_path, episode):
        if "fps" in episode:
            return float(episode["fps"])
        candidates = [
            os.path.join(str(data_path).rstrip("/"), "meta", "info.json"),
            os.path.join(str(data_path).rstrip("/"), "info.json"),
            os.path.join(os.path.dirname(str(data_path).rstrip("/")), "meta", "info.json"),
            os.path.join(os.path.dirname(str(data_path).rstrip("/")), "info.json"),
        ]
        for path in candidates:
            try:
                info = read_json(path)
            except Exception:
                continue
            if "fps" in info:
                return float(info["fps"])
            for meta in (info.get("features", {}) or {}).values():
                if "video_info" in meta and "video.fps" in meta["video_info"]:
                    return float(meta["video_info"]["video.fps"])
                if "info" in meta and "video.fps" in meta["info"]:
                    return float(meta["info"]["video.fps"])
        raise FileNotFoundError(f"Cannot resolve fps for {data_path}; add fps to each episode or provide info.json.")

    def _get_input_language_memory(self, action_config, subtask_id):
        if subtask_id < 0 or subtask_id >= len(action_config):
            return None
        for key in ("language_memory", "active_language_memory", "updated_language_memory"):
            value = str(action_config[subtask_id].get(key, "")).strip()
            if value:
                return value
        return None

    def _get_task_text(self, episode):
        tasks = episode.get("tasks", "")
        if isinstance(tasks, (list, tuple)):
            tasks = tasks[0] if tasks else ""
        tasks = str(tasks).strip().split("|")[0].strip()
        return tasks

    def _get_detailed_task_text(self, episode):
        value, _ = self._get_detailed_task_text_and_source(episode)
        return value

    def _get_detailed_task_text_and_source(self, episode):
        for key in ("detailed_task_instruction", "detailed_global_task_instruction", "global_task_instruction", "task_instruction"):
            value = episode.get(key, "")
            if isinstance(value, (list, tuple)):
                value = value[0] if value else ""
            value = str(value).strip()
            if value:
                return value, key
        return "", ""

    def _get_completion_text(self, task_text):
        return f"The task '{task_text}' has been completed." if task_text else "The current task has been completed."

    def _get_active_language_memory(self, action_config, subtask_id, task_text):
        if subtask_id < 0 or subtask_id > len(action_config):
            return None
        if subtask_id == len(action_config):
            return self._get_completion_text(task_text)
        value = str(action_config[subtask_id].get("language_memory", "")).strip()
        return value or None

    def _build_video_files(self, block, episode):
        video_path = episode.get("video_path")
        if not video_path:
            raise ValueError(f"Missing video_path for task_id={episode.get('task_id')} episode_index={episode.get('episode_index')}")
        return [os.path.join(block["data_path"], video_path).replace("{video_key}", key) for key in block["video_keys"]]

    @staticmethod
    def _decoded_frame_to_image(decoded_frame):
        return Image.fromarray(decoded_frame.to_ndarray(format="rgb24")).convert("RGB")

    @staticmethod
    def _decoded_frame_timestamp_sec(decoded_frame, time_base):
        if getattr(decoded_frame, "time", None) is not None:
            return float(decoded_frame.time)
        if decoded_frame.pts is not None and time_base is not None:
            return float(decoded_frame.pts * time_base)
        return None

    def _open_video_container(self, video_file):
        if av is None:
            raise ImportError("pyav is required to decode video bytes.")
        return av.open(BytesIO(fileio.get(video_file)))

    def _decode_frame_image(self, video_file, timestamp_sec):
        target_timestamp_sec = max(float(timestamp_sec), 0.0)
        container = self._open_video_container(video_file)
        try:
            video_stream = container.streams.video[0]
            time_base = video_stream.time_base
            if time_base is not None:
                target_pts = int(target_timestamp_sec / float(time_base))
                best_image = None
                best_distance = None
                container.seek(max(target_pts, 0), stream=video_stream, backward=True, any_frame=False)
                for decoded_count, decoded_frame in enumerate(container.decode(video_stream), start=1):
                    current_image = self._decoded_frame_to_image(decoded_frame)
                    current_timestamp_sec = self._decoded_frame_timestamp_sec(decoded_frame, time_base)
                    if current_timestamp_sec is None:
                        best_image = best_image or current_image
                    else:
                        distance = abs(current_timestamp_sec - target_timestamp_sec)
                        if best_distance is None or distance < best_distance:
                            best_distance = distance
                            best_image = current_image
                        if current_timestamp_sec >= target_timestamp_sec:
                            break
                    if decoded_count >= 256:
                        break
                if best_image is not None:
                    return best_image
        finally:
            container.close()
        raise IndexError(f"Unable to decode timestamp {target_timestamp_sec} from {video_file}")

    def resolve_sample_metadata(self, i: int) -> Dict[str, Any]:
        dataset_id = bisect.bisect_right(self.dataset_cum_sizes, i)
        dataset_prev = 0 if dataset_id == 0 else self.dataset_cum_sizes[dataset_id - 1]
        local_idx = i - dataset_prev
        block = self.dataset_blocks[dataset_id]
        sample_id = block["sampled_indices"][local_idx] if block["sampled_indices"] is not None else local_idx
        seg_id = bisect.bisect_right(block["cum_counts"], sample_id)
        seg_prev = 0 if seg_id == 0 else block["cum_counts"][seg_id - 1]
        local_offset = sample_id - seg_prev
        episode_pos, target_subtask_id, input_memory_subtask_id, start_frame, step, _, block_type = block["sampling_blocks"][seg_id]
        cur_frameid = start_frame + local_offset * step
        episode = block["episodes"][episode_pos]
        return {
            "dataset_id": dataset_id,
            "block": block,
            "block_type": block_type,
            "episode_pos": episode_pos,
            "episode": episode,
            "target_subtask_id": target_subtask_id,
            "input_memory_subtask_id": input_memory_subtask_id,
            "cur_frameid": cur_frameid,
            "cur_timestamp": cur_frameid / float(episode["_dataset_fps"]),
        }
