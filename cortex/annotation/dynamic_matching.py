#!/usr/bin/env python3
"""Few-shot subtask boundary refinement for LeRobot-style datasets.

This tool fits ordered subtask feature models from a small number of manually
corrected episodes, then segments additional episodes with dynamic programming.
It writes the same JSON format used by ``annotator_server.py`` and
keeps protected manual annotations untouched by default.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ANNOTATION_DIR = Path("annotations/manual")
DEFAULT_VISUALIZATION_DIR = Path("exp/cortex/annotation/visualizations")
DEFAULT_SUMMARY_DIR = Path("exp/cortex/annotation/summaries")

DEFAULT_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
DEFAULT_STATE_COLUMNS = (
    "states.left_joint.position",
    "states.left_gripper.position",
    "states.right_joint.position",
    "states.right_gripper.position",
    "actions.left_joint.position",
    "actions.left_gripper.position",
    "actions.right_joint.position",
    "actions.right_gripper.position",
)
EPISODE_RE = re.compile(r"episode_(\d+)\.(?:json|mp4|parquet)$")


@dataclass
class ManualLabel:
    episode_index: int
    path: Path
    length: int
    tasks: list[str]
    action_config: list[dict[str, Any]]


@dataclass
class EpisodeFeatures:
    episode_index: int
    length: int
    sample_frames: np.ndarray
    state_features: np.ndarray
    visual_features: np.ndarray
    combined_motion_full: np.ndarray
    state_motion_full: np.ndarray
    visual_motion_full: np.ndarray


@dataclass
class FittedModel:
    template_key: str
    source_episode_indices: list[int]
    num_subtasks: int
    tasks_fallback: list[str]
    action_texts: list[str]
    skills: list[str]
    start_ratio_median: np.ndarray
    boundary_ratio_median: np.ndarray
    segment_ratio_median: np.ndarray
    segment_ratio_min: np.ndarray
    state_mean: np.ndarray
    state_var: np.ndarray
    state_global_mean: np.ndarray
    state_global_std: np.ndarray
    visual_mean: np.ndarray
    visual_var: np.ndarray
    visual_global_mean: np.ndarray
    visual_global_std: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of a LeRobot-style dataset containing meta/, data/, and videos/.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing manual episode_XXXXXX.json files and refined outputs.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=DEFAULT_VISUALIZATION_DIR,
        help="Directory for optional boundary visualization images.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
        help="Directory for cached features and refinement summaries.",
    )
    parser.add_argument("--manual-episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--protect-episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-new", type=int, default=5, help="Annotate first N unannotated episodes. Use 0 for all.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--image-size", type=int, nargs=2, default=(64, 48), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--state-weight", type=float, default=0.45)
    parser.add_argument("--visual-weight", type=float, default=0.55)
    parser.add_argument("--duration-weight", type=float, default=0.85)
    parser.add_argument("--boundary-motion-weight", type=float, default=0.25)
    parser.add_argument("--min-segment-frames", type=int, default=30)
    parser.add_argument("--candidate-keep", type=int, default=45)
    parser.add_argument("--boundary-window-ratio", type=float, default=0.55)
    parser.add_argument("--refine-window", type=int, default=55)
    parser.add_argument(
        "--template-length-weight",
        type=float,
        default=2.0,
        help="Weight for trajectory-length similarity when selecting a variable-subtask template.",
    )
    parser.add_argument("--write-visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-recompute-cache", action="store_true")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="Video camera folders under videos/chunk-000 used for visual features.",
    )
    parser.add_argument(
        "--preview-camera",
        default=DEFAULT_CAMERAS[0],
        help="Camera folder used for visualization thumbnails.",
    )
    parser.add_argument(
        "--state-columns",
        nargs="+",
        default=list(DEFAULT_STATE_COLUMNS),
        help="Parquet state/action columns used for robot feature extraction.",
    )
    return parser.parse_args()


def episode_id_from_path(path: Path | str) -> int:
    match = EPISODE_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_episode_meta(dataset_root: Path) -> dict[int, dict[str, Any]]:
    meta_path = dataset_root / "meta" / "episodes.jsonl"
    rows: dict[int, dict[str, Any]] = {}
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[int(obj["episode_index"])] = obj
    return rows


def list_episode_indices(dataset_root: Path) -> list[int]:
    data_dir = dataset_root / "data" / "chunk-000"
    return sorted(episode_id_from_path(path) for path in data_dir.glob("episode_*.parquet"))


def annotation_path(annotation_dir: Path, episode_index: int) -> Path:
    return annotation_dir / f"episode_{episode_index:06d}.json"


def video_path(dataset_root: Path, camera: str, episode_index: int) -> Path:
    return dataset_root / "videos" / "chunk-000" / camera / f"episode_{episode_index:06d}.mp4"


def parquet_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def load_manual_label(annotation_dir: Path, episode_index: int) -> ManualLabel:
    path = annotation_path(annotation_dir, episode_index)
    obj = load_json(path)
    action_config = obj.get("action_config", [])
    if not isinstance(action_config, list) or not action_config:
        raise ValueError(f"Manual annotation has no action_config: {path}")
    for i, segment in enumerate(action_config):
        expected_start = 0 if i == 0 else int(action_config[i - 1]["end_frame"])
        if int(segment["start_frame"]) != expected_start:
            raise ValueError(f"Manual annotation is not contiguous at segment {i}: {path}")
    if int(action_config[-1]["end_frame"]) != int(obj["length"]):
        raise ValueError(f"Manual annotation does not end at length: {path}")
    return ManualLabel(
        episode_index=episode_index,
        path=path,
        length=int(obj["length"]),
        tasks=[str(x) for x in obj.get("tasks", [])],
        action_config=action_config,
    )


def read_column_matrix(table: pq.Table, column_name: str) -> np.ndarray:
    values = table[column_name].to_pylist()
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def temporal_context(base: np.ndarray, offsets: Sequence[int]) -> np.ndarray:
    t_len = base.shape[0]
    pieces = []
    base_indices = np.arange(t_len)
    for offset in offsets:
        idx = np.clip(base_indices + int(offset), 0, t_len - 1)
        pieces.append(base[idx])
    return np.concatenate(pieces, axis=1).astype(np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if window <= 1 or values.size == 0:
        return values
    window = int(window)
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def robust_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    q05, q95 = np.percentile(values, [5, 95])
    scale = max(float(q95 - q05), 1e-6)
    return np.clip((values - q05) / scale, 0.0, 1.0).astype(np.float32)


def load_state_base(
    dataset_root: Path,
    episode_index: int,
    state_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    path = parquet_path(dataset_root, episode_index)
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    columns = [name for name in state_columns if name in available]
    if not columns:
        raise RuntimeError(f"No robot state/action columns found in {path}")
    table = pq.read_table(path, columns=columns)
    matrices = [read_column_matrix(table, name) for name in columns]
    state = np.concatenate(matrices, axis=1).astype(np.float32)
    d1 = np.vstack([np.zeros((1, state.shape[1]), dtype=np.float32), np.diff(state, axis=0)])
    d2 = np.vstack([np.zeros((1, d1.shape[1]), dtype=np.float32), np.diff(d1, axis=0)])
    state_motion = moving_average(np.linalg.norm(d1, axis=1), window=17)
    base = np.concatenate([state, d1, d2], axis=1)
    features = temporal_context(base, offsets=(-20, -10, -5, 0, 5, 10, 20))
    return features, state_motion


def frame_feature(frame_bgr: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    grid = rgb.reshape(4, height // 4, 4, width // 4, 3).mean(axis=(1, 3)).reshape(-1)
    mean = rgb.mean(axis=(0, 1))
    std = rgb.std(axis=(0, 1))
    center = rgb[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4].mean(axis=(0, 1))
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    hist, _ = np.histogram(gray, bins=10, range=(0.0, 1.0))
    hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 40, 100)
    edge_density = np.asarray([edges.mean() / 255.0], dtype=np.float32)
    return np.concatenate([grid, mean, std, center, hist, edge_density], axis=0).astype(np.float32)


def visual_cache_path(
    summary_dir: Path,
    episode_index: int,
    sample_stride: int,
    image_size: tuple[int, int],
    cameras: Sequence[str],
) -> Path:
    width, height = image_size
    camera_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", "_".join(cameras)).strip("-") or "cameras"
    camera_tag = camera_tag[:96]
    return (
        summary_dir
        / "_cache"
        / f"episode_{episode_index:06d}_visual_{camera_tag}_s{sample_stride}_w{width}_h{height}.npz"
    )


def load_visual_base(
    dataset_root: Path,
    summary_dir: Path,
    episode_index: int,
    sample_stride: int,
    image_size: tuple[int, int],
    cameras: Sequence[str],
    force_recompute: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    cache_path = visual_cache_path(summary_dir, episode_index, sample_stride, image_size, cameras)
    if cache_path.exists() and not force_recompute:
        cached = np.load(cache_path)
        return (
            cached["sample_frames"].astype(np.int32),
            cached["visual_features"].astype(np.float32),
            cached["visual_motion_full"].astype(np.float32),
            int(cached["length"]),
        )

    sample_frames: list[int] = []
    raw_features: list[np.ndarray] = []
    visual_motion_full: list[float] = []
    previous_gray: list[np.ndarray | None] = [None for _ in cameras]
    frame_index = 0
    with ExitStack() as stack:
        containers = []
        for camera in cameras:
            path = video_path(dataset_root, camera, episode_index)
            if not path.is_file():
                raise RuntimeError(f"Video not found: {path}")
            try:
                containers.append(stack.enter_context(av.open(str(path))))
            except av.error.FFmpegError as exc:
                raise RuntimeError(f"Failed to open video: {path}") from exc

        frame_iterators = [container.decode(video=0) for container in containers]
        for decoded_frames in zip(*frame_iterators):
            frames = [frame.to_ndarray(format="bgr24") for frame in decoded_frames]
            frame_motions = []
            for camera_index, frame in enumerate(frames):
                small = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if previous_gray[camera_index] is None:
                    frame_motions.append(0.0)
                else:
                    frame_motions.append(float(np.mean(cv2.absdiff(gray, previous_gray[camera_index]))))
                previous_gray[camera_index] = gray
            visual_motion_full.append(float(np.mean(frame_motions)))
            if frame_index % sample_stride == 0:
                per_camera = [frame_feature(frame, image_size=image_size) for frame in frames]
                full_feature = np.concatenate(per_camera, axis=0)
                sample_frames.append(frame_index)
                raw_features.append(full_feature)
            frame_index += 1

    if not raw_features:
        raise RuntimeError(f"No visual features decoded for episode {episode_index}")
    sample_frames_arr = np.asarray(sample_frames, dtype=np.int32)
    raw = np.stack(raw_features, axis=0).astype(np.float32)
    d1 = np.vstack([np.zeros((1, raw.shape[1]), dtype=np.float32), np.diff(raw, axis=0)])
    d2 = np.vstack([np.zeros((1, d1.shape[1]), dtype=np.float32), np.diff(d1, axis=0)])
    context_base = np.concatenate([raw, d1, d2], axis=1)
    visual_features = temporal_context(context_base, offsets=(-2, -1, 0, 1, 2))
    visual_motion = moving_average(np.asarray(visual_motion_full, dtype=np.float32), window=17)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sample_frames=sample_frames_arr,
        visual_features=visual_features.astype(np.float32),
        visual_motion_full=visual_motion.astype(np.float32),
        length=np.asarray(frame_index, dtype=np.int32),
    )
    return sample_frames_arr, visual_features.astype(np.float32), visual_motion.astype(np.float32), frame_index


def load_episode_features(
    dataset_root: Path,
    summary_dir: Path,
    episode_index: int,
    sample_stride: int,
    image_size: tuple[int, int],
    cameras: Sequence[str],
    state_columns: Sequence[str],
    state_weight: float,
    visual_weight: float,
    force_recompute_cache: bool,
) -> EpisodeFeatures:
    state_features_full, state_motion_full = load_state_base(
        dataset_root=dataset_root,
        episode_index=episode_index,
        state_columns=state_columns,
    )
    sample_frames, visual_features, visual_motion_full, video_length = load_visual_base(
        dataset_root=dataset_root,
        summary_dir=summary_dir,
        episode_index=episode_index,
        sample_stride=sample_stride,
        image_size=image_size,
        cameras=cameras,
        force_recompute=force_recompute_cache,
    )
    length = min(state_features_full.shape[0], video_length)
    keep = sample_frames < length
    sample_frames = sample_frames[keep]
    visual_features = visual_features[keep]
    state_features = state_features_full[sample_frames]
    state_motion = state_motion_full[:length]
    visual_motion = visual_motion_full[:length]
    combined_motion = (
        float(state_weight) * robust_01(state_motion)
        + float(visual_weight) * robust_01(visual_motion)
    )
    return EpisodeFeatures(
        episode_index=episode_index,
        length=length,
        sample_frames=sample_frames,
        state_features=state_features.astype(np.float32),
        visual_features=visual_features.astype(np.float32),
        combined_motion_full=combined_motion.astype(np.float32),
        state_motion_full=state_motion.astype(np.float32),
        visual_motion_full=visual_motion.astype(np.float32),
    )


def normalize_train_test(train_parts: list[np.ndarray], test: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    train_all = np.concatenate(train_parts, axis=0)
    mean = train_all.mean(axis=0)
    std = train_all.std(axis=0)
    std = np.maximum(std, 1e-4)
    return [(x - mean) / std for x in train_parts], (test - mean) / std, mean, std


def fit_modality_gaussian(parts_by_subtask: list[list[np.ndarray]], dim: int) -> tuple[np.ndarray, np.ndarray]:
    means = []
    vars_ = []
    for subtask_index, chunks in enumerate(parts_by_subtask):
        if not chunks:
            means.append(np.zeros((dim,), dtype=np.float32))
            vars_.append(np.ones((dim,), dtype=np.float32))
            continue
        cat = np.concatenate(chunks, axis=0).astype(np.float32)
        means.append(cat.mean(axis=0))
        vars_.append(np.maximum(cat.var(axis=0), 1e-4))
    return np.stack(means, axis=0).astype(np.float32), np.stack(vars_, axis=0).astype(np.float32)


def fit_model(
    manual_labels: list[ManualLabel],
    feature_by_episode: dict[int, EpisodeFeatures],
) -> FittedModel:
    num_subtasks = len(manual_labels[0].action_config)
    if any(len(label.action_config) != num_subtasks for label in manual_labels):
        raise ValueError("Manual labels have different numbers of subtasks")

    state_train = [feature_by_episode[label.episode_index].state_features for label in manual_labels]
    visual_train = [feature_by_episode[label.episode_index].visual_features for label in manual_labels]
    state_norm_parts, _, state_global_mean, state_global_std = normalize_train_test(state_train, state_train[0])
    visual_norm_parts, _, visual_global_mean, visual_global_std = normalize_train_test(visual_train, visual_train[0])

    state_by_subtask: list[list[np.ndarray]] = [[] for _ in range(num_subtasks)]
    visual_by_subtask: list[list[np.ndarray]] = [[] for _ in range(num_subtasks)]
    start_ratios: list[list[float]] = [[] for _ in range(num_subtasks)]
    segment_ratios: list[list[float]] = [[] for _ in range(num_subtasks)]

    for label_index, label in enumerate(manual_labels):
        feat = feature_by_episode[label.episode_index]
        sample_frames = feat.sample_frames
        for subtask_index, segment in enumerate(label.action_config):
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            mask = (sample_frames >= start) & (sample_frames < end)
            if int(mask.sum()) < 2:
                nearest = np.argmin(np.abs(sample_frames - (start + end) // 2))
                mask[nearest] = True
            state_by_subtask[subtask_index].append(state_norm_parts[label_index][mask])
            visual_by_subtask[subtask_index].append(visual_norm_parts[label_index][mask])
            start_ratios[subtask_index].append(start / float(label.length))
            segment_ratios[subtask_index].append((end - start) / float(label.length))

    state_mean, state_var = fit_modality_gaussian(state_by_subtask, dim=state_norm_parts[0].shape[1])
    visual_mean, visual_var = fit_modality_gaussian(visual_by_subtask, dim=visual_norm_parts[0].shape[1])

    def most_common_nonempty(values: Sequence[str]) -> str:
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        return Counter(cleaned).most_common(1)[0][0] if cleaned else ""

    tasks_fallback = next((label.tasks for label in manual_labels if label.tasks), [])
    action_texts = [
        most_common_nonempty(
            str(label.action_config[index].get("action_text", "")) for label in manual_labels
        )
        for index in range(num_subtasks)
    ]
    skills = [
        most_common_nonempty(
            str(label.action_config[index].get("skill", "")) for label in manual_labels
        )
        for index in range(num_subtasks)
    ]
    start_ratio_median = np.asarray([np.median(values) for values in start_ratios], dtype=np.float32)
    segment_ratio_median = np.asarray([np.median(values) for values in segment_ratios], dtype=np.float32)
    segment_ratio_min = np.asarray([np.min(values) for values in segment_ratios], dtype=np.float32)
    boundaries = []
    for subtask_index in range(1, num_subtasks):
        starts = [label.action_config[subtask_index]["start_frame"] / float(label.length) for label in manual_labels]
        boundaries.append(float(np.median(starts)))

    return FittedModel(
        template_key=f"{num_subtasks}-subtasks",
        source_episode_indices=[label.episode_index for label in manual_labels],
        num_subtasks=num_subtasks,
        tasks_fallback=tasks_fallback,
        action_texts=action_texts,
        skills=skills,
        start_ratio_median=start_ratio_median,
        boundary_ratio_median=np.asarray(boundaries, dtype=np.float32),
        segment_ratio_median=segment_ratio_median,
        segment_ratio_min=segment_ratio_min,
        state_mean=state_mean,
        state_var=state_var,
        state_global_mean=state_global_mean.astype(np.float32),
        state_global_std=state_global_std.astype(np.float32),
        visual_mean=visual_mean,
        visual_var=visual_var,
        visual_global_mean=visual_global_mean.astype(np.float32),
        visual_global_std=visual_global_std.astype(np.float32),
    )


def fit_models(
    manual_labels: list[ManualLabel],
    feature_by_episode: dict[int, EpisodeFeatures],
) -> dict[int, FittedModel]:
    grouped: dict[int, list[ManualLabel]] = defaultdict(list)
    for label in manual_labels:
        grouped[len(label.action_config)].append(label)
    return {
        num_subtasks: fit_model(labels, feature_by_episode)
        for num_subtasks, labels in sorted(grouped.items())
    }


def episode_descriptor(features: EpisodeFeatures) -> np.ndarray:
    pieces = [np.asarray([math.log1p(features.length)], dtype=np.float32)]
    for sequence in (features.state_features, features.visual_features):
        pieces.extend([sequence.mean(axis=0), sequence.std(axis=0)])
    return np.concatenate(pieces, axis=0).astype(np.float32)


def match_template_models(
    manual_labels: list[ManualLabel],
    feature_by_episode: dict[int, EpisodeFeatures],
    target_episode_indices: Sequence[int],
    models: dict[int, FittedModel],
    length_weight: float,
) -> dict[int, tuple[FittedModel, int, float]]:
    manual_indices = [label.episode_index for label in manual_labels]
    all_indices = [*manual_indices, *[int(index) for index in target_episode_indices]]
    descriptors = np.stack(
        [episode_descriptor(feature_by_episode[index]) for index in all_indices], axis=0
    )
    descriptor_mean = descriptors.mean(axis=0, keepdims=True)
    descriptor_std = np.maximum(descriptors.std(axis=0, keepdims=True), 1e-4)
    normalized = (descriptors - descriptor_mean) / descriptor_std
    normalized_by_episode = {
        episode_index: normalized[position]
        for position, episode_index in enumerate(all_indices)
    }
    label_by_episode = {label.episode_index: label for label in manual_labels}
    descriptor_dim = max(1, normalized.shape[1])

    assignments: dict[int, tuple[FittedModel, int, float]] = {}
    for target_index in target_episode_indices:
        target_features = feature_by_episode[int(target_index)]
        candidates = []
        for manual_index in manual_indices:
            manual_features = feature_by_episode[manual_index]
            descriptor_distance = float(
                np.linalg.norm(
                    normalized_by_episode[int(target_index)] - normalized_by_episode[manual_index]
                )
                / math.sqrt(descriptor_dim)
            )
            length_distance = abs(
                math.log(max(target_features.length, 1) / max(manual_features.length, 1))
            )
            score = descriptor_distance + float(length_weight) * length_distance
            candidates.append((score, manual_index))
        score, matched_manual_index = min(candidates)
        num_subtasks = len(label_by_episode[matched_manual_index].action_config)
        assignments[int(target_index)] = (
            models[num_subtasks],
            matched_manual_index,
            float(score),
        )
    return assignments


def gaussian_cost(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - mean[None, :, :]
    cost = 0.5 * ((diff * diff) / var[None, :, :] + np.log(var[None, :, :]))
    return cost.mean(axis=2).astype(np.float32)


def normalize_cost(cost: np.ndarray) -> np.ndarray:
    centered = cost - cost.min(axis=1, keepdims=True)
    scale = np.percentile(centered, 90, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    return (centered / scale).astype(np.float32)


def episode_costs(features: EpisodeFeatures, model: FittedModel, state_weight: float, visual_weight: float) -> np.ndarray:
    state = (features.state_features - model.state_global_mean) / model.state_global_std
    visual = (features.visual_features - model.visual_global_mean) / model.visual_global_std
    state_cost = normalize_cost(gaussian_cost(state, model.state_mean, model.state_var))
    visual_cost = normalize_cost(gaussian_cost(visual, model.visual_mean, model.visual_var))
    return (float(state_weight) * state_cost + float(visual_weight) * visual_cost).astype(np.float32)


def local_minima_indices(values: np.ndarray, lo: int, hi: int) -> list[int]:
    lo = max(1, lo)
    hi = min(len(values) - 2, hi)
    if lo > hi:
        return []
    segment = values[lo : hi + 1]
    indices: list[int] = []
    for offset in range(1, len(segment) - 1):
        if segment[offset] <= segment[offset - 1] and segment[offset] <= segment[offset + 1]:
            indices.append(lo + offset)
    return indices


def candidate_boundaries(
    features: EpisodeFeatures,
    model: FittedModel,
    candidate_keep: int,
    boundary_window_ratio: float,
) -> list[list[int]]:
    candidates_all: list[list[int]] = []
    motion = features.combined_motion_full
    length = features.length
    sample_frames = features.sample_frames
    for boundary_index, ratio in enumerate(model.boundary_ratio_median):
        expected = int(round(float(ratio) * length))
        left_seg = float(model.segment_ratio_median[boundary_index]) * length
        right_seg = float(model.segment_ratio_median[boundary_index + 1]) * length
        window = int(max(65, min(460, boundary_window_ratio * max(left_seg, right_seg))))
        lo = max(1, expected - window)
        hi = min(length - 1, expected + window)
        in_window = sample_frames[(sample_frames >= lo) & (sample_frames <= hi)]
        if in_window.size == 0:
            in_window = np.asarray([expected], dtype=np.int32)
        minima = local_minima_indices(motion, lo=lo, hi=hi)
        pooled = set(int(x) for x in in_window.tolist())
        pooled.add(expected)
        pooled.update(int(x) for x in minima)
        ranked = sorted(
            pooled,
            key=lambda frame: (
                float(motion[min(max(frame, 0), len(motion) - 1)]),
                abs(frame - expected),
            ),
        )
        keep = sorted(ranked[:candidate_keep])
        if expected not in keep:
            keep.append(expected)
        candidates_all.append(sorted(set(max(1, min(length - 1, int(x))) for x in keep)))
    return candidates_all


def segment_sample_slice(sample_frames: np.ndarray, start_frame: int, end_frame: int) -> tuple[int, int]:
    a = int(np.searchsorted(sample_frames, start_frame, side="left"))
    b = int(np.searchsorted(sample_frames, end_frame, side="left"))
    if b <= a:
        nearest = int(np.searchsorted(sample_frames, (start_frame + end_frame) // 2, side="left"))
        nearest = min(max(nearest, 0), len(sample_frames) - 1)
        return nearest, nearest + 1
    return a, b


def solve_boundaries(
    features: EpisodeFeatures,
    cost: np.ndarray,
    model: FittedModel,
    duration_weight: float,
    boundary_motion_weight: float,
    min_segment_frames: int,
    candidate_keep: int,
    boundary_window_ratio: float,
    refine_window: int,
) -> tuple[list[int], list[float], list[int]]:
    candidates = candidate_boundaries(features, model, candidate_keep, boundary_window_ratio)
    num_boundaries = len(candidates)
    num_subtasks = model.num_subtasks
    length = features.length
    sample_frames = features.sample_frames
    prefix = np.vstack([np.zeros((1, num_subtasks), dtype=np.float64), np.cumsum(cost.astype(np.float64), axis=0)])
    expected_lengths = np.maximum(model.segment_ratio_median * length, float(min_segment_frames))
    min_lengths = np.maximum(model.segment_ratio_min * length * 0.45, float(min_segment_frames))

    def interval_cost(subtask_index: int, start: int, end: int) -> float:
        if end <= start:
            return math.inf
        a, b = segment_sample_slice(sample_frames, start, end)
        count = max(1, b - a)
        emission = float((prefix[b, subtask_index] - prefix[a, subtask_index]) / count)
        duration = end - start
        expected = float(expected_lengths[subtask_index])
        duration_cost = ((duration - expected) / max(expected, 1.0)) ** 2
        return emission + float(duration_weight) * duration_cost

    dp: list[np.ndarray] = []
    back: list[np.ndarray] = []

    first = np.full((len(candidates[0]),), np.inf, dtype=np.float64)
    for i, boundary in enumerate(candidates[0]):
        if boundary < min_lengths[0]:
            continue
        motion_cost = float(features.combined_motion_full[boundary]) * float(boundary_motion_weight)
        first[i] = interval_cost(0, 0, boundary) + motion_cost
    dp.append(first)
    back.append(np.full((len(candidates[0]),), -1, dtype=np.int32))

    for boundary_index in range(1, num_boundaries):
        current = np.full((len(candidates[boundary_index]),), np.inf, dtype=np.float64)
        current_back = np.full((len(candidates[boundary_index]),), -1, dtype=np.int32)
        for cur_i, cur_boundary in enumerate(candidates[boundary_index]):
            best_cost = math.inf
            best_prev = -1
            for prev_i, prev_boundary in enumerate(candidates[boundary_index - 1]):
                if cur_boundary - prev_boundary < min_lengths[boundary_index]:
                    continue
                value = (
                    float(dp[boundary_index - 1][prev_i])
                    + interval_cost(boundary_index, prev_boundary, cur_boundary)
                    + float(features.combined_motion_full[cur_boundary]) * float(boundary_motion_weight)
                )
                if value < best_cost:
                    best_cost = value
                    best_prev = prev_i
            current[cur_i] = best_cost
            current_back[cur_i] = best_prev
        dp.append(current)
        back.append(current_back)

    best_final = math.inf
    best_idx = -1
    last_boundary_list = candidates[-1]
    for prev_i, prev_boundary in enumerate(last_boundary_list):
        if length - prev_boundary < min_lengths[-1]:
            continue
        value = float(dp[-1][prev_i]) + interval_cost(num_subtasks - 1, prev_boundary, length)
        if value < best_final:
            best_final = value
            best_idx = prev_i
    if best_idx < 0 or not math.isfinite(best_final):
        raise RuntimeError("Failed to solve subtask boundaries")

    boundary_choices = [0] * num_boundaries
    boundary_choices[-1] = best_idx
    for boundary_index in range(num_boundaries - 1, 0, -1):
        boundary_choices[boundary_index - 1] = int(back[boundary_index][boundary_choices[boundary_index]])
    boundaries = [int(candidates[i][boundary_choices[i]]) for i in range(num_boundaries)]
    boundaries = refine_boundaries(
        features=features,
        boundaries=boundaries,
        model=model,
        min_segment_frames=min_segment_frames,
        refine_window=refine_window,
    )
    confidences = boundary_confidences(features, boundaries, model)
    expected = [int(round(float(ratio) * length)) for ratio in model.boundary_ratio_median]
    return boundaries, confidences, expected


def refine_boundaries(
    features: EpisodeFeatures,
    boundaries: Sequence[int],
    model: FittedModel,
    min_segment_frames: int,
    refine_window: int,
) -> list[int]:
    motion = features.combined_motion_full
    length = features.length
    refined: list[int] = []
    original = list(int(x) for x in boundaries)
    for i, boundary in enumerate(original):
        left_limit = min_segment_frames if i == 0 else refined[-1] + min_segment_frames
        right_limit = length - min_segment_frames if i == len(original) - 1 else original[i + 1] - min_segment_frames
        lo = max(left_limit, boundary - refine_window)
        hi = min(right_limit, boundary + refine_window)
        if lo > hi:
            refined.append(boundary)
            continue
        expected = int(round(float(model.boundary_ratio_median[i]) * length))
        best = min(
            range(lo, hi + 1),
            key=lambda frame: float(motion[frame]) + 0.08 * abs(frame - expected) / max(1, hi - lo),
        )
        refined.append(int(best))
    return refined


def boundary_confidences(features: EpisodeFeatures, boundaries: Sequence[int], model: FittedModel) -> list[float]:
    motion = features.combined_motion_full
    length = features.length
    confidences = []
    for i, boundary in enumerate(boundaries):
        lo = max(0, boundary - 90)
        hi = min(len(motion), boundary + 91)
        local = motion[lo:hi]
        if local.size == 0:
            pause_conf = 0.0
        else:
            local_min = float(local.min())
            local_p80 = float(np.percentile(local, 80))
            pause_conf = (local_p80 - float(motion[boundary])) / max(local_p80 - local_min, 1e-6)
        expected = int(round(float(model.boundary_ratio_median[i]) * length))
        left_seg = float(model.segment_ratio_median[i]) * length
        right_seg = float(model.segment_ratio_median[i + 1]) * length
        window = max(80.0, 0.55 * max(left_seg, right_seg))
        timing_conf = math.exp(-abs(boundary - expected) / window)
        confidences.append(round(float(np.clip(0.55 * pause_conf + 0.45 * timing_conf, 0.0, 1.0)), 4))
    return confidences


def build_annotation(
    episode_index: int,
    length: int,
    boundaries: Sequence[int],
    model: FittedModel,
    tasks: list[str],
) -> dict[str, Any]:
    starts = [0, *[int(x) for x in boundaries]]
    ends = [*[int(x) for x in boundaries], int(length)]
    action_config = []
    for subtask_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        action_config.append(
            {
                "seg_id": subtask_index,
                "start_frame": int(start),
                "end_frame": int(end),
                "action_text": model.action_texts[subtask_index] if subtask_index < len(model.action_texts) else "",
                "skill": model.skills[subtask_index] if subtask_index < len(model.skills) else "",
            }
        )
    return {
        "episode_index": int(episode_index),
        "tasks": tasks if tasks else model.tasks_fallback,
        "length": int(length),
        "action_config": action_config,
    }


def grab_frames(video: Path, frame_indices: Sequence[int], size: tuple[int, int]) -> dict[int, Image.Image]:
    targets = {int(frame_index) for frame_index in frame_indices if int(frame_index) >= 0}
    if not targets:
        return {}
    images: dict[int, Image.Image] = {}
    try:
        with av.open(str(video)) as container:
            for frame_index, decoded_frame in enumerate(container.decode(video=0)):
                if frame_index in targets:
                    frame = decoded_frame.to_ndarray(format="bgr24")
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images[frame_index] = Image.fromarray(rgb)
                    if len(images) == len(targets):
                        break
                if frame_index > max(targets):
                    break
    except av.error.FFmpegError as exc:
        raise RuntimeError(f"Failed to decode video for visualization: {video}") from exc
    return images


def draw_motion_chart(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    size: tuple[int, int],
    motion: np.ndarray,
    boundaries: Sequence[int],
    expected: Sequence[int],
) -> None:
    x0, y0 = origin
    width, height = size
    draw.rectangle([x0, y0, x0 + width, y0 + height], outline=(180, 186, 197), fill=(248, 250, 252))
    if motion.size:
        sampled_x = np.linspace(0, len(motion) - 1, num=min(width, len(motion))).astype(np.int32)
        values = robust_01(motion[sampled_x])
        points = [
            (x0 + i, y0 + height - int(float(value) * (height - 8)) - 4)
            for i, value in enumerate(values)
        ]
        if len(points) > 1:
            draw.line(points, fill=(15, 118, 110), width=2)
    length = max(1, len(motion))
    for frame in expected:
        x = x0 + int((int(frame) / length) * width)
        draw.line([x, y0, x, y0 + height], fill=(148, 163, 184), width=1)
    for frame in boundaries:
        x = x0 + int((int(frame) / length) * width)
        draw.line([x, y0, x, y0 + height], fill=(185, 28, 28), width=2)


def create_visualization(
    dataset_root: Path,
    visualization_dir: Path,
    features: EpisodeFeatures,
    annotation: dict[str, Any],
    boundaries: Sequence[int],
    expected: Sequence[int],
    confidences: Sequence[float],
    preview_camera: str,
) -> Path:
    visualization_dir.mkdir(parents=True, exist_ok=True)
    output_path = visualization_dir / f"episode_{features.episode_index:06d}.jpg"
    width = 1680
    height = 1060
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = (
        f"episode_{features.episode_index:06d}  length={features.length}  "
        f"subtasks={len(annotation['action_config'])}  mean_conf={np.mean(confidences) if confidences else 0:.3f}"
    )
    draw.text((28, 20), title, fill=(17, 24, 39), font=font)

    timeline_x, timeline_y, timeline_w, timeline_h = 28, 58, width - 56, 72
    draw.rectangle(
        [timeline_x, timeline_y, timeline_x + timeline_w, timeline_y + timeline_h],
        outline=(180, 186, 197),
        fill=(248, 250, 252),
    )
    palette = [
        (15, 118, 110),
        (37, 99, 235),
        (124, 58, 237),
        (202, 138, 4),
        (220, 38, 38),
        (2, 132, 199),
        (22, 163, 74),
    ]
    for i, segment in enumerate(annotation["action_config"]):
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        x1 = timeline_x + int(start / features.length * timeline_w)
        x2 = timeline_x + int(end / features.length * timeline_w)
        color = palette[i % len(palette)]
        draw.rectangle([x1, timeline_y + 8, max(x2, x1 + 2), timeline_y + timeline_h - 8], fill=color)
        draw.text((x1 + 3, timeline_y + 26), str(i), fill=(255, 255, 255), font=font)
    for i, boundary in enumerate(boundaries):
        x = timeline_x + int(int(boundary) / features.length * timeline_w)
        draw.line([x, timeline_y, x, timeline_y + timeline_h], fill=(0, 0, 0), width=2)
        draw.text((x + 3, timeline_y + timeline_h - 18), f"{boundary}:{confidences[i]:.2f}", fill=(17, 24, 39), font=font)

    draw_motion_chart(
        draw=draw,
        origin=(28, 160),
        size=(width - 56, 190),
        motion=features.combined_motion_full,
        boundaries=boundaries,
        expected=expected,
    )
    draw.text((28, 354), "green=state+visual motion, gray=timing prior, red=predicted boundary", fill=(75, 85, 99), font=font)

    thumb_w, thumb_h = 210, 150
    cols = 7
    start_y = 390
    midpoint_frames = [
        (int(seg["start_frame"]) + int(seg["end_frame"]) - 1) // 2
        for seg in annotation["action_config"]
    ]
    frames = grab_frames(
        video_path(dataset_root, preview_camera, features.episode_index),
        midpoint_frames,
        size=(thumb_w, thumb_h),
    )
    for i, segment in enumerate(annotation["action_config"]):
        row = i // cols
        col = i % cols
        x = 28 + col * 235
        y = start_y + row * 300
        frame = midpoint_frames[i]
        if frame in frames:
            image.paste(frames[frame], (x, y))
        draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(30, 41, 59), width=2)
        draw.text((x, y + thumb_h + 8), f"{i:02d}: {segment['start_frame']}-{segment['end_frame']}", fill=(17, 24, 39), font=font)
        draw.text((x, y + thumb_h + 26), f"mid={frame}", fill=(75, 85, 99), font=font)
    image.save(output_path, quality=92)
    return output_path


def validate_annotation(annotation: dict[str, Any]) -> None:
    length = int(annotation["length"])
    action_config = annotation["action_config"]
    if not action_config:
        raise ValueError("empty action_config")
    expected_start = 0
    for i, segment in enumerate(action_config):
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start != expected_start:
            raise ValueError(f"segment {i} starts at {start}, expected {expected_start}")
        if end <= start:
            raise ValueError(f"segment {i} has non-positive duration")
        expected_start = end
    if expected_start != length:
        raise ValueError(f"last end_frame {expected_start} != length {length}")


def write_annotation(path: Path, annotation: dict[str, Any], dry_run: bool) -> None:
    validate_annotation(annotation)
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(annotation, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def select_episodes(
    dataset_root: Path,
    annotation_dir: Path,
    requested: Sequence[int] | None,
    max_new: int,
    skip_existing: bool,
    overwrite: bool,
    protect_episodes: set[int],
) -> list[int]:
    all_episodes = list_episode_indices(dataset_root)
    if requested is None:
        candidates = all_episodes
    else:
        candidates = sorted(set(int(x) for x in requested))

    selected: list[int] = []
    for episode_index in candidates:
        if episode_index in protect_episodes:
            continue
        path = annotation_path(annotation_dir, episode_index)
        if path.exists() and (skip_existing or not overwrite):
            continue
        selected.append(episode_index)
        if requested is None and max_new > 0 and len(selected) >= max_new:
            break
    return selected


def write_summary(summary_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "boundary_refinement_summary.json"
    csv_path = summary_dir / "boundary_refinement_summary.csv"
    existing: list[dict[str, Any]] = []
    if json_path.exists():
        existing = json.loads(json_path.read_text())
    by_episode = {int(row["episode_index"]): row for row in existing}
    for row in rows:
        by_episode[int(row["episode_index"])] = row
    merged = [by_episode[key] for key in sorted(by_episode)]
    json_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode_index",
                "length",
                "template_key",
                "num_subtasks",
                "matched_manual_episode",
                "template_match_score",
                "mean_confidence",
                "boundaries",
                "annotation_path",
                "visualization_path",
            ]
        )
        for row in merged:
            writer.writerow(
                [
                    row["episode_index"],
                    row["length"],
                    row.get("template_key", ""),
                    row.get("num_subtasks", ""),
                    row.get("matched_manual_episode", ""),
                    row.get("template_match_score", ""),
                    row["mean_confidence"],
                    json.dumps(row["boundaries"]),
                    row["annotation_path"],
                    row["visualization_path"],
                ]
            )


def portable_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def main() -> int:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    annotation_dir = args.annotation_dir.resolve()
    visualization_dir = args.visualization_dir.resolve()
    summary_dir = args.summary_dir.resolve()
    cameras = [str(camera).strip() for camera in args.cameras if str(camera).strip()]
    state_columns = [str(column).strip() for column in args.state_columns if str(column).strip()]
    if not cameras:
        raise SystemExit("--cameras must contain at least one camera folder")
    if not state_columns:
        raise SystemExit("--state-columns must contain at least one parquet column")
    if len(args.image_size) != 2 or args.image_size[0] % 4 != 0 or args.image_size[1] % 4 != 0:
        raise SystemExit("--image-size WIDTH HEIGHT must contain values divisible by 4")
    protect_episodes = set(int(x) for x in args.protect_episodes)

    manual_labels = [load_manual_label(annotation_dir, episode_index) for episode_index in args.manual_episodes]
    meta = read_episode_meta(dataset_root)
    print(f"[info] loaded manual labels: {[label.episode_index for label in manual_labels]}", flush=True)

    feature_by_episode: dict[int, EpisodeFeatures] = {}
    for label in manual_labels:
        print(f"[fit] loading features for manual episode {label.episode_index:06d}", flush=True)
        feature_by_episode[label.episode_index] = load_episode_features(
            dataset_root=dataset_root,
            summary_dir=summary_dir,
            episode_index=label.episode_index,
            sample_stride=args.sample_stride,
            image_size=tuple(args.image_size),
            cameras=cameras,
            state_columns=state_columns,
            state_weight=args.state_weight,
            visual_weight=args.visual_weight,
            force_recompute_cache=args.force_recompute_cache,
        )

    models = fit_models(manual_labels, feature_by_episode)
    print(
        "[info] fitted templates: "
        + ", ".join(
            f"{model.template_key} from {model.source_episode_indices}"
            for model in models.values()
        ),
        flush=True,
    )
    episodes = select_episodes(
        dataset_root=dataset_root,
        annotation_dir=annotation_dir,
        requested=args.episodes,
        max_new=args.max_new,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
        protect_episodes=protect_episodes,
    )
    print(f"[info] selected episodes: {episodes}", flush=True)
    if not episodes:
        print("[info] nothing to annotate", flush=True)
        return 0

    for episode_index in episodes:
        print(f"[load] episode {episode_index:06d}", flush=True)
        feature_by_episode[episode_index] = load_episode_features(
            dataset_root=dataset_root,
            summary_dir=summary_dir,
            episode_index=episode_index,
            sample_stride=args.sample_stride,
            image_size=tuple(args.image_size),
            cameras=cameras,
            state_columns=state_columns,
            state_weight=args.state_weight,
            visual_weight=args.visual_weight,
            force_recompute_cache=args.force_recompute_cache,
        )
    template_assignments = match_template_models(
        manual_labels=manual_labels,
        feature_by_episode=feature_by_episode,
        target_episode_indices=episodes,
        models=models,
        length_weight=args.template_length_weight,
    )

    rows: list[dict[str, Any]] = []
    for episode_index in episodes:
        out_path = annotation_path(annotation_dir, episode_index)
        if episode_index in protect_episodes:
            print(f"[skip] protected episode {episode_index:06d}", flush=True)
            continue
        if out_path.exists() and not args.overwrite:
            print(f"[skip] existing annotation {out_path}", flush=True)
            continue

        print(f"[run] episode {episode_index:06d}", flush=True)
        features = feature_by_episode[episode_index]
        model, matched_manual_episode, template_match_score = template_assignments[episode_index]
        print(
            f"[match] episode {episode_index:06d} -> manual {matched_manual_episode:06d} "
            f"template={model.template_key} score={template_match_score:.4f}",
            flush=True,
        )
        cost = episode_costs(features, model, state_weight=args.state_weight, visual_weight=args.visual_weight)
        boundaries, confidences, expected = solve_boundaries(
            features=features,
            cost=cost,
            model=model,
            duration_weight=args.duration_weight,
            boundary_motion_weight=args.boundary_motion_weight,
            min_segment_frames=args.min_segment_frames,
            candidate_keep=args.candidate_keep,
            boundary_window_ratio=args.boundary_window_ratio,
            refine_window=args.refine_window,
        )
        tasks = [str(x) for x in meta.get(episode_index, {}).get("tasks", [])]
        annotation = build_annotation(
            episode_index=episode_index,
            length=features.length,
            boundaries=boundaries,
            model=model,
            tasks=tasks,
        )
        write_annotation(out_path, annotation, dry_run=args.dry_run)
        viz_path = ""
        if args.write_visualizations:
            viz_path = str(
                create_visualization(
                    dataset_root=dataset_root,
                    visualization_dir=visualization_dir,
                    features=features,
                    annotation=annotation,
                    boundaries=boundaries,
                    expected=expected,
                    confidences=confidences,
                    preview_camera=args.preview_camera,
                )
            )
        row = {
            "episode_index": episode_index,
            "length": features.length,
            "template_key": model.template_key,
            "num_subtasks": model.num_subtasks,
            "matched_manual_episode": matched_manual_episode,
            "template_match_score": round(template_match_score, 6),
            "boundaries": [int(x) for x in boundaries],
            "expected_boundaries": [int(x) for x in expected],
            "boundary_confidence": confidences,
            "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "annotation_path": portable_path(out_path),
            "visualization_path": portable_path(viz_path) if viz_path else "",
            "dry_run": bool(args.dry_run),
        }
        rows.append(row)
        print(
            f"[done] episode {episode_index:06d} mean_conf={row['mean_confidence']} "
            f"annotation={out_path}",
            flush=True,
        )

    if rows and not args.dry_run:
        write_summary(summary_dir, rows)
    print(json.dumps({"processed": rows}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
