"""VLM-based initial subtask annotation for robot videos."""

from openai import OpenAI, APIConnectionError, APITimeoutError, BadRequestError
import base64
import cv2
import httpx
import json
import re
import os
import ast
import logging
import math
import sys
import shutil
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from mmengine import fileio
import av
import io

import numpy as np

try:
    from petrel_client.client import Client
except ImportError:
    Client = None


def _build_file_client():
    config_path = os.environ.get("PETREL_CONF")
    if Client is None or not config_path:
        return None
    try:
        return Client(os.path.expanduser(config_path), enable_mc=False)
    except Exception:
        return None


file_client = _build_file_client()

# Core VLM endpoint configuration. Override these with OPENAI_*, VLLM_*, or
# NEWAPI_* environment variables when using a local vLLM/OpenAI-compatible
# server.
_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_API_KEY = "EMPTY"


# ======================== 核心更新1：拓展英文原子技能列表 ========================
# Extended robotic atomic skills (English only, open for expansion)
ROBOT_ATOMIC_SKILLS = [
    "Pick",          # 抓取
    "Place",         # 放置
    "Plug",          # 插入/插线
    "Press",         # 按下
    "Push",          # 推
    "Pull",          # 拉
    "Move",          # 只有底盘移动才算Move，手臂移动不算
    "Fold",          # 折叠
    "Wipe",          # 擦拭
    "Close",         # 关闭
    "Open",          # 开启
    "Pour",          # 倾倒
    "Cut",           # 切割
    "Screw",         # 拧动
    "Handover",
    "Sweep",
] 

_FRAME_FILE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
CEPH_PATH_PATTERN = re.compile(r"^(?:[^/]+:)?s3://")
CEPH_HEAD_VIDEO_MARKER = "/observation.images.head_rgb/"
CEPH_WRIST_VIEW_NAMES = ("left_wrist_rgb", "right_wrist_rgb")

# 环境变量解析函数
def _env_first(names: list[str], default: str) -> str:
    for name in names:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    return default

def _split_urls(raw: str | None, default: str) -> list[str]:
    if not raw:
        return [default]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or [default]

# 基础配置初始化
BASE_URL = _env_first(["NEWAPI_BASE_URL", "OPENAI_BASE_URL", "VLLM_BASE_URL"], _DEFAULT_BASE_URL)
BASE_URLS = _split_urls(_env_first(["NEWAPI_BASE_URLS", "OPENAI_BASE_URLS", "VLLM_BASE_URLS"], ""), BASE_URL)
MODEL_NAME = _env_first(["NEWAPI_MODEL", "OPENAI_MODEL", "VLLM_MODEL"], _DEFAULT_MODEL)
API_KEY = _env_first(["NEWAPI_API_KEY", "OPENAI_API_KEY", "VLLM_API_KEY"], _API_KEY)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[\033[34m%(asctime)s\033[0m] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

_SUBTASK_SUBJECT_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:robotic?\s+arm|robot\s+arm|robotic?\s+manipulator|manipulator|robot\s+base|mobile\s+base|base|chassis|robot|arm|gripper|end\s+effector)\s+",
    re.IGNORECASE,
)

_BASE_MOVE_KEYWORDS = (
    "base",
    "chassis",
    "mobile base",
    "mobile platform",
    "platform",
    "wheeled",
    "drive",
    "drives",
    "driving",
    "navigate",
    "navigates",
    "navigating",
    "reposition the robot base",
    "repositions the robot base",
    "moves the robot base",
    "moves the mobile base",
)

_ARM_ONLY_MOVE_KEYWORDS = (
    "robot arm",
    "arm",
    "gripper",
    "end effector",
    "reach",
    "reaches",
    "reaching",
    "move to select",
    "moves to select",
    "move toward",
    "moves toward",
    "move towards",
    "moves towards",
    "move above",
    "moves above",
    "align",
    "aligned",
    "hover",
    "hovers",
    "hovering",
    "idle",
    "wait",
    "waiting",
    "remain",
    "remains",
)

_MERGE_NEXT_AUXILIARY_PREFIXES = (
    "approach ",
    "approaches ",
    "reach ",
    "reaches ",
    "move to ",
    "moves to ",
    "move toward ",
    "moves toward ",
    "move towards ",
    "moves towards ",
    "move near ",
    "moves near ",
    "move above ",
    "moves above ",
    "move around ",
    "moves around ",
    "move into position",
    "moves into position",
    "move into alignment",
    "moves into alignment",
    "hover ",
    "hovers ",
)

_MERGE_PREV_AUXILIARY_PREFIXES = (
    "remain idle",
    "remains idle",
    "remain above",
    "remains above",
    "stay idle",
    "stays idle",
    "stay above",
    "stays above",
    "wait ",
    "waits ",
    "waiting ",
    "pause ",
    "pauses ",
    "hold position",
    "holds position",
)

_DRAWER_LIKE_TARGET_KEYWORDS = {
    "drawer": ("drawer",),
    "tray": ("pull-out tray", "pull out tray", "sliding tray", "slide-out tray", "slide out tray"),
    "shelf": ("pull-out shelf", "pull out shelf", "sliding shelf", "slide-out shelf", "slide out shelf"),
}


def _normalize_subtask_text(text: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip().rstrip(".")
    normalized = _SUBTASK_SUBJECT_PREFIX_RE.sub("", normalized)
    if normalized:
        normalized = normalized[0].lower() + normalized[1:]
    return normalized


def _is_valid_base_move_subtask(text: str) -> bool:
    normalized = text.lower()
    if any(keyword in normalized for keyword in _BASE_MOVE_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _ARM_ONLY_MOVE_KEYWORDS):
        return False
    # Be conservative: only explicit base/chassis motion qualifies as Move.
    return False


def _classify_auxiliary_subtask(text: str) -> Optional[str]:
    normalized = text.lower().strip()
    if not normalized:
        return None
    if any(normalized.startswith(prefix) for prefix in _MERGE_PREV_AUXILIARY_PREFIXES):
        return "merge_prev"
    if " idle " in f" {normalized} ":
        return "merge_prev"
    if any(normalized.startswith(prefix) for prefix in _MERGE_NEXT_AUXILIARY_PREFIXES):
        return "merge_next"
    return None


def _get_drawer_like_target_type(text: str) -> Optional[str]:
    normalized = text.lower()
    for target_type, keywords in _DRAWER_LIKE_TARGET_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return target_type
    return None


def _is_drawer_like_subtask(text: str) -> bool:
    return _get_drawer_like_target_type(text) is not None


def _normalize_drawer_like_skill_and_subtask(primitive_skill: str, substask: str) -> tuple[str, str]:
    if not _is_drawer_like_subtask(substask):
        return primitive_skill, substask

    normalized_skill = primitive_skill
    normalized_subtask = substask

    if primitive_skill == "Open":
        normalized_skill = "Pull"
        normalized_subtask = re.sub(
            r"^(?:open|opens|opening)\s+",
            "pulls open ",
            normalized_subtask,
            count=1,
            flags=re.IGNORECASE,
        )
    elif primitive_skill == "Close":
        normalized_skill = "Push"
        if re.match(r"^(?:close|closes|closing)\s+", normalized_subtask, flags=re.IGNORECASE):
            remainder = re.sub(
                r"^(?:close|closes|closing)\s+",
                "",
                normalized_subtask,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
            normalized_subtask = f"pushes {remainder} closed" if remainder else "pushes the drawer closed"

    return normalized_skill, normalized_subtask


def _merge_drawer_like_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    merged_segments: List[Dict[str, Any]] = [dict(segments[0])]

    for seg in segments[1:]:
        current = dict(seg)
        previous = merged_segments[-1]
        previous_target = _get_drawer_like_target_type(previous["substask"])
        current_target = _get_drawer_like_target_type(current["substask"])

        should_merge = (
            previous["primitive_skill"] == current["primitive_skill"]
            and previous["primitive_skill"] in {"Pull", "Push"}
            and previous_target is not None
            and previous_target == current_target
        )

        if should_merge:
            previous["end_frame"] = max(previous["end_frame"], current["end_frame"])
            previous["check_frame_id"] = max(previous.get("check_frame_id", previous["end_frame"]), current.get("check_frame_id", current["end_frame"]))
            if current["primitive_skill"] == "Pull" and re.match(r"^pull", current["substask"], flags=re.IGNORECASE):
                previous["substask"] = current["substask"]
            elif current["primitive_skill"] == "Push" and re.match(r"^push", current["substask"], flags=re.IGNORECASE):
                previous["substask"] = current["substask"]
            logger.warning(
                f"Merged drawer-like {current['primitive_skill']} segments into one continuous segment: "
                f"'{previous['substask']}' [{previous['start_frame']}, {previous['end_frame']}]"
            )
            continue

        merged_segments.append(current)

    return merged_segments


def _add_frame_id_watermark(frame: np.ndarray, frame_id: int) -> np.ndarray:
    annotated = frame.copy()
    label = f"frame_id: {frame_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.75, min(1.1, frame.shape[1] / 1100))
    thickness = max(2, int(round(font_scale * 2)))
    padding = max(10, int(round(font_scale * 12)))

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    box_x1 = padding
    box_y1 = padding
    box_x2 = box_x1 + text_w + padding * 2
    box_y2 = box_y1 + text_h + baseline + padding * 2

    overlay = annotated.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    text_org = (box_x1 + padding, box_y1 + padding + text_h)
    cv2.putText(
        annotated,
        label,
        text_org,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def _list_frame_files(frame_dir: str) -> List[str]:
    frame_files = [
        f for f in os.listdir(frame_dir)
        if os.path.splitext(f)[1].lower() in _FRAME_FILE_EXTENSIONS
    ]
    frame_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
    return frame_files


def _default_check_frames_dir(output_json_path: str) -> str:
    output_dir = os.path.dirname(output_json_path) or "."
    output_name = os.path.splitext(os.path.basename(output_json_path))[0]
    return os.path.join(output_dir, f"{output_name}_check_frames")


def _default_subtask_clips_dir(output_json_path: str) -> str:
    output_dir = os.path.dirname(output_json_path) or "."
    output_name = os.path.splitext(os.path.basename(output_json_path))[0]
    return os.path.join(output_dir, f"{output_name}_subtask_clips")


def _sanitize_subtask_filename(text: str, max_length: int = 120) -> str:
    normalized = _normalize_subtask_text(text).lower()
    sanitized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if not sanitized:
        sanitized = "subtask"
    return sanitized[:max_length].rstrip("_") or "subtask"


def _ensure_even_frame_size(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target_width = width if width % 2 == 0 else width - 1
    target_height = height if height % 2 == 0 else height - 1
    if target_width <= 0 or target_height <= 0:
        return frame
    if target_width == width and target_height == height:
        return frame
    return frame[:target_height, :target_width]


def _get_ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def _open_h264_writer(clip_path: str, fps: float, width: int, height: int) -> subprocess.Popen[bytes]:
    ffmpeg_path = _get_ffmpeg_path()
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found in PATH, cannot export H.264 clips")

    safe_fps = fps if fps and fps > 0 else 30.0
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{safe_fps}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        clip_path,
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _finalize_h264_writer(process: subprocess.Popen[bytes]) -> tuple[bool, str]:
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()

    stderr_text = ""
    if process.stderr is not None:
        stderr_text = process.stderr.read().decode("utf-8", errors="replace").strip()
        process.stderr.close()

    return_code = process.wait()
    return return_code == 0, stderr_text


def export_model_input_video(
    frames: List[np.ndarray],
    output_video_path: str,
    fps: float,
) -> Optional[str]:
    if not frames:
        logger.warning(f"No rendered frames available for model-input video export: {output_video_path}")
        return None

    output_dir = os.path.dirname(output_video_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    normalized_frames = []
    reference_frame = _ensure_even_frame_size(frames[0])
    target_height, target_width = reference_frame.shape[:2]
    normalized_frames.append(reference_frame)

    for frame in frames[1:]:
        normalized = _ensure_even_frame_size(frame)
        if normalized.shape[:2] != (target_height, target_width):
            normalized = cv2.resize(
                normalized,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        normalized_frames.append(normalized)

    writer = _open_h264_writer(
        output_video_path,
        fps if fps and fps > 0 else 1.0,
        target_width,
        target_height,
    )

    try:
        for frame in normalized_frames:
            if writer.stdin is None:
                raise RuntimeError("ffmpeg stdin is unavailable")
            writer.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError, RuntimeError) as exc:
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        raise RuntimeError(f"Failed to export model-input video to {output_video_path}: {exc}") from exc
    finally:
        ok, ffmpeg_error = _finalize_h264_writer(writer)

    if not ok:
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        raise RuntimeError(
            f"ffmpeg failed while exporting model-input video to {output_video_path}: {ffmpeg_error or 'unknown error'}"
        )

    logger.info(f"Saved model-input stitched video to: {output_video_path}")
    return output_video_path


def _read_video_bytes(video_path: str) -> bytes:
    try:
        video_bytes = fileio.get(video_path)
    except Exception as exc:
        raise ValueError(f"Failed to read video bytes from path: {video_path}") from exc

    if not video_bytes:
        raise ValueError(f"Empty video content: {video_path}")
    return video_bytes


def _is_ceph_path(path: str) -> bool:
    return isinstance(path, str) and CEPH_PATH_PATTERN.match(path) is not None


def _normalize_ceph_video_view_mode(mode: Optional[str]) -> str:
    if mode is None:
        return "head"

    normalized_mode = str(mode).strip().lower()
    mode_aliases = {
        "head": "head",
        "head_only": "head",
        "single": "head",
        "head_wrists": "head_wrists",
        "head+wrists": "head_wrists",
        "multiview": "head_wrists",
        "multi": "head_wrists",
    }
    if normalized_mode not in mode_aliases:
        raise ValueError(
            f"Unsupported ceph_video_view_mode={mode!r}. Supported values: head, head_wrists."
        )
    return mode_aliases[normalized_mode]


def _resolve_ceph_video_paths(video_path: str, ceph_video_view_mode: Optional[str]) -> List[str]:
    mode = _normalize_ceph_video_view_mode(ceph_video_view_mode)
    if mode != "head_wrists":
        return [video_path]

    if _is_ceph_path(video_path) and CEPH_HEAD_VIDEO_MARKER in video_path:
        resolved_paths = [video_path]
        for view_name in CEPH_WRIST_VIEW_NAMES:
            resolved_paths.append(
                video_path.replace(
                    CEPH_HEAD_VIDEO_MARKER,
                    f"/observation.images.{view_name}/",
                    1,
                )
            )
        return resolved_paths

    path = Path(video_path)
    camera_dir = path.parent
    camera_root = camera_dir.parent
    filename = path.name
    camera_name = camera_dir.name

    local_camera_mappings = {
        "observation.images.cam_high": (
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ),
        "observation.images.head_rgb": (
            "observation.images.left_wrist_rgb",
            "observation.images.right_wrist_rgb",
        ),
        "images.rgb.head": (
            "images.rgb.hand_left",
            "images.rgb.hand_right",
        ),
    }
    wrist_camera_names = local_camera_mappings.get(camera_name)

    if wrist_camera_names is None and camera_root.exists():
        sibling_names = [candidate.name for candidate in camera_root.iterdir() if candidate.is_dir()]
        left_candidates = [
            name
            for name in sibling_names
            if "left" in name.lower() and "wrist" in name.lower()
        ]
        right_candidates = [
            name
            for name in sibling_names
            if "right" in name.lower() and "wrist" in name.lower()
        ]
        wrist_camera_names = (
            sorted(left_candidates)[0] if left_candidates else "",
            sorted(right_candidates)[0] if right_candidates else "",
        )

    resolved_paths = [video_path]
    if wrist_camera_names:
        for wrist_camera_name in wrist_camera_names:
            if not wrist_camera_name:
                continue
            wrist_path = camera_root / wrist_camera_name / filename
            if wrist_path.exists():
                resolved_paths.append(str(wrist_path))
            else:
                logger.warning(f"Optional wrist view is missing: {wrist_path}")
    return resolved_paths


def _build_video_bytes_payload(
    video_path: str,
    ceph_video_view_mode: Optional[str],
    primary_video_bytes: bytes | None = None,
) -> bytes | Dict[str, bytes]:
    resolved_paths = _resolve_ceph_video_paths(video_path, ceph_video_view_mode)
    if len(resolved_paths) == 1:
        return primary_video_bytes if primary_video_bytes is not None else _read_video_bytes(video_path)

    payload: Dict[str, bytes] = {
        resolved_paths[0]: primary_video_bytes if primary_video_bytes is not None else _read_video_bytes(resolved_paths[0])
    }
    for extra_video_path in resolved_paths[1:]:
        try:
            payload[extra_video_path] = _read_video_bytes(extra_video_path)
        except Exception as exc:
            logger.warning(f"Failed to load optional wrist view {extra_video_path}, falling back to available views only: {exc}")
    return payload


def _get_video_bytes_for_path(video_path: str, video_bytes: bytes | Dict[str, bytes] | None = None) -> bytes:
    if isinstance(video_bytes, dict):
        if video_path in video_bytes:
            return video_bytes[video_path]
        raise ValueError(f"Missing cached video bytes for path: {video_path}")
    if video_bytes is not None:
        return video_bytes
    return _read_video_bytes(video_path)


def _probe_video_metadata(video_path: str, video_bytes: bytes | Dict[str, bytes] | None = None) -> tuple[float, int]:
    video_bytes = _get_video_bytes_for_path(video_path, video_bytes)

    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream found in file: {video_path}")

            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            total_frames = int(stream.frames or 0)

            if total_frames <= 0 and stream.duration and stream.time_base and stream.average_rate:
                total_frames = int(float(stream.duration * stream.time_base * stream.average_rate))

            if total_frames <= 0:
                logger.warning(
                    f"PyAV could not determine total frame count for {video_path}, counting frames via full decode"
                )
                total_frames = sum(1 for _ in container.decode(video=0))
    except av.error.FFmpegError as exc:
        raise ValueError(f"Cannot open video file with PyAV: {video_path}") from exc

    if total_frames <= 0:
        raise ValueError(f"No decodable frames found in video: {video_path}")

    return fps, total_frames


def _scale_frame_indices(
    canonical_indices: List[int],
    canonical_total_frames: int,
    target_total_frames: int,
) -> List[int]:
    if target_total_frames <= 0:
        raise ValueError("target_total_frames must be positive")

    if canonical_total_frames <= 1 or target_total_frames == 1:
        return [0 for _ in canonical_indices]

    if canonical_total_frames == target_total_frames:
        return [min(max(int(index), 0), target_total_frames - 1) for index in canonical_indices]

    scaled = []
    for index in canonical_indices:
        mapped = round(int(index) * (target_total_frames - 1) / (canonical_total_frames - 1))
        scaled.append(min(max(mapped, 0), target_total_frames - 1))
    return scaled


def _decode_frames_by_indices(
    video_path: str,
    frame_indices: List[int],
    video_bytes: bytes | Dict[str, bytes] | None = None,
) -> Dict[int, np.ndarray]:
    requested_indices = sorted({int(frame_id) for frame_id in frame_indices if int(frame_id) >= 0})
    if not requested_indices:
        return {}

    decoded_frames: Dict[int, np.ndarray] = {}
    video_bytes = _get_video_bytes_for_path(video_path, video_bytes)

    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            if not container.streams.video:
                return {}

            target_indices = set(requested_indices)
            end_index = requested_indices[-1]
            for current_frame_id, frame in enumerate(container.decode(video=0)):
                if current_frame_id > end_index:
                    break
                if current_frame_id in target_indices:
                    decoded_frames[current_frame_id] = frame.to_ndarray(format="bgr24")
                    if len(decoded_frames) >= len(target_indices):
                        break
    except (ValueError, av.error.FFmpegError):
        return {}

    return decoded_frames


def _overlay_view_label(frame: np.ndarray, label: str) -> np.ndarray:
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(0.95, frame.shape[1] / 520))
    thickness = 2
    padding = max(8, int(round(font_scale * 10)))

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    box_x1 = padding
    box_y1 = padding
    box_x2 = box_x1 + text_width + padding * 2
    box_y2 = box_y1 + text_height + baseline + padding * 2

    overlay = annotated.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    text_org = (box_x1 + padding, box_y1 + padding + text_height)
    cv2.putText(
        annotated,
        label,
        text_org,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def _concat_frames_horizontally(frames: List[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("frames must not be empty")

    target_height = max(frame.shape[0] for frame in frames)
    total_width = sum(frame.shape[1] for frame in frames)
    canvas = np.full((target_height, total_width, 3), 255, dtype=frames[0].dtype)

    offset_x = 0
    for frame in frames:
        height, width = frame.shape[:2]
        canvas[:height, offset_x:offset_x + width] = frame
        offset_x += width
    return canvas


def _concat_frames_vertically(frames: List[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("frames must not be empty")

    target_width = max(frame.shape[1] for frame in frames)
    total_height = sum(frame.shape[0] for frame in frames)
    canvas = np.full((total_height, target_width, 3), 255, dtype=frames[0].dtype)

    offset_y = 0
    for frame in frames:
        height, width = frame.shape[:2]
        offset_x = (target_width - width) // 2
        canvas[offset_y:offset_y + height, offset_x:offset_x + width] = frame
        offset_y += height
    return canvas


def _resize_frame_to_width(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        raise ValueError(f"Invalid target_width: {target_width}")

    src_height, src_width = frame.shape[:2]
    if src_width <= 0 or src_height <= 0:
        raise ValueError("Input frame has invalid size")

    if src_width == target_width:
        return frame

    scale = target_width / src_width
    target_height = max(1, int(round(src_height * scale)))
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def _resize_frame_to_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        raise ValueError(f"Invalid max_width: {max_width}")

    current_width = frame.shape[1]
    if current_width <= max_width:
        return frame
    return _resize_frame_to_width(frame, max_width)


def _compose_multiview_frame(
    head_frame: np.ndarray,
    left_wrist_frame: Optional[np.ndarray],
    right_wrist_frame: Optional[np.ndarray],
) -> np.ndarray:
    top_panel = _overlay_view_label(head_frame, "head")

    wrist_views: List[np.ndarray] = []
    if left_wrist_frame is not None:
        wrist_views.append(_overlay_view_label(left_wrist_frame, "left wrist"))
    if right_wrist_frame is not None:
        wrist_views.append(_overlay_view_label(right_wrist_frame, "right wrist"))

    if not wrist_views:
        return top_panel

    bottom_panel = _concat_frames_horizontally(wrist_views)
    bottom_panel = _resize_frame_to_width(bottom_panel, top_panel.shape[1])
    return _concat_frames_vertically([top_panel, bottom_panel])


def _load_frames_for_annotation(
    video_path: str,
    frame_indices: List[int],
    video_bytes: bytes | Dict[str, bytes] | None = None,
    ceph_video_view_mode: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    canonical_indices = [int(frame_id) for frame_id in frame_indices if int(frame_id) >= 0]
    if not canonical_indices:
        return {}

    resolved_paths = _resolve_ceph_video_paths(video_path, ceph_video_view_mode)
    if len(resolved_paths) == 1:
        return _decode_frames_by_indices(video_path, canonical_indices, video_bytes=video_bytes)

    head_fps, head_total_frames = _probe_video_metadata(video_path, video_bytes=video_bytes)
    del head_fps

    head_frames = _decode_frames_by_indices(video_path, canonical_indices, video_bytes=video_bytes)
    if not head_frames:
        return {}

    per_view_frames: Dict[str, Dict[int, np.ndarray]] = {video_path: head_frames}

    if isinstance(video_bytes, dict):
        for extra_video_path in resolved_paths[1:]:
            if extra_video_path not in video_bytes:
                logger.warning(f"Optional wrist view is missing for {extra_video_path}, skipping this view")
                continue

            _, extra_total_frames = _probe_video_metadata(extra_video_path, video_bytes=video_bytes)
            mapped_indices = _scale_frame_indices(canonical_indices, head_total_frames, extra_total_frames)
            decoded_extra_frames = _decode_frames_by_indices(
                extra_video_path,
                mapped_indices,
                video_bytes=video_bytes,
            )
            if not decoded_extra_frames:
                logger.warning(f"Failed to decode optional wrist view {extra_video_path}, skipping this view")
                continue

            aligned_frames: Dict[int, np.ndarray] = {}
            for canonical_index, mapped_index in zip(canonical_indices, mapped_indices):
                frame = decoded_extra_frames.get(mapped_index)
                if frame is not None:
                    aligned_frames[canonical_index] = frame

            if aligned_frames:
                per_view_frames[extra_video_path] = aligned_frames

    stitched_frames: Dict[int, np.ndarray] = {}
    left_wrist_path = resolved_paths[1] if len(resolved_paths) > 1 else None
    right_wrist_path = resolved_paths[2] if len(resolved_paths) > 2 else None
    for canonical_index in canonical_indices:
        head_frame = head_frames.get(canonical_index)
        if head_frame is None:
            continue

        stitched_frames[canonical_index] = _compose_multiview_frame(
            head_frame=head_frame,
            left_wrist_frame=(
                per_view_frames.get(left_wrist_path, {}).get(canonical_index)
                if left_wrist_path is not None
                else None
            ),
            right_wrist_frame=(
                per_view_frames.get(right_wrist_path, {}).get(canonical_index)
                if right_wrist_path is not None
                else None
            ),
        )

    return stitched_frames


def _load_frame_by_id(
    video_path: str,
    frame_id: int,
    video_bytes: bytes | Dict[str, bytes] | None = None,
    ceph_video_view_mode: Optional[str] = None,
) -> Optional[np.ndarray]:
    if frame_id < 0:
        return None

    frame_map = _load_frames_for_annotation(
        video_path,
        [frame_id],
        video_bytes=video_bytes,
        ceph_video_view_mode=ceph_video_view_mode,
    )
    return frame_map.get(frame_id)


def _render_check_frame(
    frame: np.ndarray,
    seg_id: int,
    frame_id: int,
    primitive_skill: str,
    substask: str,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    lines = [
        f"seg_id: {seg_id}",
        f"frame_id: {frame_id}",
        f"segment_range: [{start_frame}, {end_frame}]",
        f"skill: {primitive_skill}",
    ]
    wrap_width = max(28, min(90, frame.shape[1] // 14))
    lines.extend(textwrap.wrap(f"subtask: {substask}", width=wrap_width) or [f"subtask: {substask}"])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(0.85, frame.shape[1] / 1600))
    thickness = 2
    margin = 18
    line_height = max(26, int(30 * font_scale))
    panel_height = margin * 2 + line_height * len(lines)

    canvas = cv2.copyMakeBorder(
        frame,
        panel_height,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], panel_height), (20, 20, 20), thickness=-1)

    y = margin + line_height - 8
    for line in lines:
        cv2.putText(canvas, line, (margin, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height

    return canvas


def export_subtask_check_frames(
    video_path: str,
    result: Dict[str, Any],
    output_json_path: str,
    check_frames_dir: Optional[str] = None,
    video_bytes: bytes | Dict[str, bytes] | None = None,
    ceph_video_view_mode: Optional[str] = None,
) -> str:
    output_dir = check_frames_dir or _default_check_frames_dir(output_json_path)
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "sample_id": result.get("sample_id"),
        "nframes": result.get("nframes"),
        "segments": [],
    }

    target_frame_ids = {
        int(seg.get("check_frame_id", int(seg["end_frame"])))
        for seg in result.get("segments", [])
        if int(seg.get("check_frame_id", int(seg["end_frame"]))) >= 0
    }
    video_bytes = video_bytes if video_bytes is not None else _read_video_bytes(video_path)
    decoded_frames = _load_frames_for_annotation(
        video_path,
        sorted(target_frame_ids),
        video_bytes=video_bytes,
        ceph_video_view_mode=ceph_video_view_mode,
    )

    for seg in result.get("segments", []):
        seg_id = int(seg["seg_id"])
        start_frame = int(seg["start_frame"])
        end_frame = int(seg["end_frame"])
        frame_id = int(seg.get("check_frame_id", end_frame))
        primitive_skill = str(seg["primitive_skill"]).strip()
        substask = _normalize_subtask_text(seg["substask"])

        frame = decoded_frames.get(frame_id)
        if frame is None:
            logger.warning(f"Failed to load check frame {frame_id} for segment {seg_id}")
            continue

        annotated = _render_check_frame(
            frame=frame,
            seg_id=seg_id,
            frame_id=frame_id,
            primitive_skill=primitive_skill,
            substask=substask,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        image_name = f"seg_{seg_id:03d}_check_{frame_id:06d}.jpg"
        image_path = os.path.join(output_dir, image_name)
        if not cv2.imwrite(image_path, annotated):
            logger.warning(f"Failed to write check frame image: {image_path}")
            continue

        manifest["segments"].append({
            "seg_id": seg_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_id": frame_id,
            "primitive_skill": primitive_skill,
            "substask": substask,
            "image_path": image_path,
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved {len(manifest['segments'])} subtask check frames to: {output_dir}")
    return output_dir


def export_subtask_clips(
    video_path: str,
    result: Dict[str, Any],
    output_json_path: str,
    fps: float,
    clips_dir: Optional[str] = None,
    video_bytes: bytes | Dict[str, bytes] | None = None,
) -> str:
    output_dir = clips_dir or _default_subtask_clips_dir(output_json_path)
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "sample_id": result.get("sample_id"),
        "fps": fps,
        "nframes": result.get("nframes"),
        "video_codec": "h264",
        "segments": [],
    }
    video_bytes = _get_video_bytes_for_path(video_path, video_bytes)
    segment_states = []

    for seg in result.get("segments", []):
        seg_id = int(seg["seg_id"])
        start_frame = int(seg["start_frame"])
        end_frame = int(seg["end_frame"])
        primitive_skill = str(seg["primitive_skill"]).strip()
        substask = _normalize_subtask_text(seg["substask"])
        clip_stub = _sanitize_subtask_filename(substask)
        clip_name = f"seg_{seg_id:03d}_{clip_stub}.mp4"
        clip_path = os.path.join(output_dir, clip_name)

        segment_states.append({
            "seg_id": seg_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "primitive_skill": primitive_skill,
            "substask": substask,
            "clip_path": clip_path,
            "writer": None,
            "written_frames": 0,
            "write_error": None,
        })

    def _close_segment_writer(state: Dict[str, Any]) -> None:
        writer = state.get("writer")
        if writer is None:
            return
        ok, ffmpeg_error = _finalize_h264_writer(writer)
        state["writer"] = None
        if not ok:
            if ffmpeg_error:
                logger.warning(f"ffmpeg failed for seg {state['seg_id']}: {ffmpeg_error}")
            state["write_error"] = state["write_error"] or ffmpeg_error or (
                f"ffmpeg exited with non-zero status for seg {state['seg_id']}"
            )

    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            if not container.streams.video:
                logger.warning(f"No video stream found for clip export: {video_path}")
            else:
                seg_idx = 0
                for frame_idx, frame in enumerate(container.decode(video=0)):
                    while seg_idx < len(segment_states) and frame_idx > segment_states[seg_idx]["end_frame"]:
                        _close_segment_writer(segment_states[seg_idx])
                        seg_idx += 1

                    if seg_idx >= len(segment_states):
                        break

                    current_state = segment_states[seg_idx]
                    if frame_idx < current_state["start_frame"]:
                        continue

                    frame_bgr = _ensure_even_frame_size(frame.to_ndarray(format="bgr24"))
                    if current_state["writer"] is None:
                        height, width = frame_bgr.shape[:2]
                        current_state["writer"] = _open_h264_writer(current_state["clip_path"], fps, width, height)
                    try:
                        if current_state["writer"].stdin is None:
                            raise RuntimeError("ffmpeg stdin is unavailable")
                        current_state["writer"].stdin.write(frame_bgr.tobytes())
                    except (BrokenPipeError, OSError, RuntimeError) as exc:
                        current_state["write_error"] = (
                            f"Failed to write H.264 clip for seg {current_state['seg_id']}: {exc}"
                        )
                        logger.warning(current_state["write_error"])
                        _close_segment_writer(current_state)
                        seg_idx += 1
                        continue
                    current_state["written_frames"] += 1
    except av.error.FFmpegError as exc:
        logger.warning(f"Cannot open video file with PyAV for clip export: {video_path} | {exc}")
        for state in segment_states:
            state["write_error"] = state["write_error"] or str(exc)

    for state in segment_states:
        _close_segment_writer(state)
        if state["written_frames"] == 0 or state["write_error"]:
            if os.path.exists(state["clip_path"]):
                os.remove(state["clip_path"])
            if state["written_frames"] == 0 and not state["write_error"]:
                logger.warning(f"Failed to export clip for seg {state['seg_id']}: no frames written")
            continue

        manifest["segments"].append({
            "seg_id": state["seg_id"],
            "start_frame": state["start_frame"],
            "end_frame": state["end_frame"],
            "primitive_skill": state["primitive_skill"],
            "substask": state["substask"],
            "clip_path": state["clip_path"],
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved {len(manifest['segments'])} subtask clips to: {output_dir}")
    return output_dir


def _format_output_text(text: Any) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    normalized = normalized[0].upper() + normalized[1:]
    if normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def _format_output_fps(fps: Any) -> int | float:
    try:
        numeric_fps = float(fps)
    except (TypeError, ValueError):
        return fps
    if numeric_fps.is_integer():
        return int(numeric_fps)
    return round(numeric_fps, 3)


def _parse_episode_index(sample_id: Any) -> int | str:
    text = str(sample_id).strip()
    match = re.search(r"episode[_-](\d+)", text)
    if match:
        return int(match.group(1))
    try:
        return int(text)
    except ValueError:
        return text


def _build_output_json(result: Dict[str, Any]) -> Dict[str, Any]:
    task_text = _format_output_text(result.get("task"))
    length = int(result.get("nframes", 0))
    action_config = []

    for seg in result.get("segments", []):
        start_frame = int(seg["start_frame"])
        # The VLM prompt asks for inclusive frame ranges. Downstream Cortex
        # annotation tools use half-open ranges, so convert on write.
        end_frame = min(int(seg["end_frame"]) + 1, length)
        action_config.append({
            "seg_id": int(seg["seg_id"]),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "action_text": _format_output_text(seg.get("substask")),
            "skill": str(seg.get("primitive_skill", "")).strip(),
        })

    return {
        "episode_index": _parse_episode_index(result.get("sample_id", "")),
        "length": length,
        "fps": _format_output_fps(result.get("fps", 30)),
        "tasks": [task_text] if task_text else [],
        "action_config": action_config,
    }

# HTTP client configuration. Multimodal requests can take several minutes on
# shared or proxied model endpoints.
try:
    _API_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "300"))
except ValueError as exc:
    raise ValueError("OPENAI_TIMEOUT_SECONDS must be a number") from exc
if _API_TIMEOUT_SECONDS <= 0:
    raise ValueError("OPENAI_TIMEOUT_SECONDS must be greater than zero")

_HTTP_CLIENT = httpx.Client(
    trust_env=True,
    timeout=httpx.Timeout(_API_TIMEOUT_SECONDS, connect=min(30.0, _API_TIMEOUT_SECONDS)),
)

# 客户端管理
def get_clients(base_urls: list[str] | None = None, api_key: str | None = None) -> list[OpenAI]:
    urls = base_urls or BASE_URLS
    return [
        OpenAI(
            base_url=u,
            api_key=api_key or API_KEY,
            http_client=_HTTP_CLIENT,
            max_retries=0,
        )
        for u in urls
    ]

clients = get_clients()
default_client = clients[0]

# 异常处理工具函数（复用原有逻辑）
def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    seen = set()
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain

def _is_timeout_exception(exc: BaseException) -> bool:
    timeout_types = (APITimeoutError, httpx.ReadTimeout, httpx.TimeoutException, TimeoutError)
    for err in _iter_exception_chain(exc):
        if isinstance(err, timeout_types):
            return True
        if "timed out" in str(err).lower():
            return True
    return False


def _format_exception_chain(exc: BaseException) -> str:
    return " -> ".join(
        f"{type(error).__name__}: {error}" for error in _iter_exception_chain(exc)
    )

# 重试逻辑
def _api_create_with_retry(client, max_retries: int = 3, **kwargs):
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(** kwargs)
        except (APIConnectionError, APITimeoutError, ConnectionError) as e:
            last_err = e
            error_details = _format_exception_chain(e)
            if attempt < max_retries - 1:
                wait = 1 * (attempt + 1)  # 指数退避
                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{max_retries}): "
                    f"{error_details}; retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"API request failed after {max_retries} attempts: {error_details}"
                )
                raise
    raise last_err


def _extract_chat_response_text(response: Any) -> str:
    """Extract assistant text from OpenAI SDK and compatible gateway responses."""
    value = response

    # Some OpenAI-compatible gateways JSON-encode the entire response twice.
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("The model endpoint returned an empty string")
        if stripped.lower().startswith(("<!doctype html", "<html")):
            raise ValueError(
                "The model endpoint returned HTML instead of a chat-completion response. "
                "Check OPENAI_BASE_URL; OpenAI-compatible APIs usually require a /v1 suffix."
            )
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        if isinstance(decoded, str):
            return decoded.strip()
        if isinstance(decoded, dict) and "choices" not in decoded:
            return stripped
        value = decoded

    if hasattr(value, "choices"):
        choices = value.choices
    elif isinstance(value, dict):
        choices = value.get("choices")
    else:
        raise TypeError(
            "Unsupported model response type: "
            f"{type(value).__name__}; expected an OpenAI ChatCompletion, dict, or string"
        )

    if not choices:
        raise ValueError("The model endpoint returned no choices")

    first_choice = choices[0]
    message = (
        first_choice.get("message")
        if isinstance(first_choice, dict)
        else getattr(first_choice, "message", None)
    )
    if message is None:
        raise ValueError("The first model choice has no message")

    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    )
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                text_parts.append(str(part.get("text", "")))
            elif getattr(part, "type", None) in {"text", "output_text"}:
                text_parts.append(str(getattr(part, "text", "")))
        text = "\n".join(text_parts).strip()
    else:
        raise TypeError(
            "Unsupported assistant content type: "
            f"{type(content).__name__}; expected text content"
        )

    if not text:
        raise ValueError("The model endpoint returned empty assistant content")
    return text

def extract_video_frames(
    video_path: str,
    target_height: int = 336,
    max_sample_frames: int = 200,
    video_bytes: bytes | Dict[str, bytes] | None = None,
    ceph_video_view_mode: Optional[str] = None,
) -> tuple[List[str], float, int, List[np.ndarray], float]:
    """
    Extract video frames and convert to base64 encoding (optimized for robotic video sampling)
    :param video_path: Path to video file
    :param target_height: Resized target height
    :param max_sample_frames: Maximum number of sampled frames
    :return: (base64-encoded frames, fps, total_frames, rendered frames, sampled fps)
    """
    frames = []
    rendered_frames: List[np.ndarray] = []
    video_bytes = video_bytes if video_bytes is not None else _read_video_bytes(video_path)
    preserve_multiview_resolution = len(_resolve_ceph_video_paths(video_path, ceph_video_view_mode)) > 1

    try:
        fps, total_frames = _probe_video_metadata(video_path, video_bytes=video_bytes)

        step = max(1, math.ceil(total_frames / max_sample_frames))
        sampled_indices = list(range(0, total_frames, step))[:max_sample_frames]
        sampled_fps = (fps / step) if fps and fps > 0 else 1.0

        logger.info(
            f"Video info | Total frames: {total_frames} | FPS: {fps} | Step: {step}"
        )

        decoded_frames = _load_frames_for_annotation(
            video_path,
            sampled_indices,
            video_bytes=video_bytes,
            ceph_video_view_mode=ceph_video_view_mode,
        )
        for frame_id in sampled_indices:
            frame_bgr = decoded_frames.get(frame_id)
            if frame_bgr is None:
                logger.warning(f"Failed to decode sampled frame {frame_id} for {video_path}")
                continue

            if preserve_multiview_resolution:
                # Keep the stitched layout aspect ratio, but cap the final model-input width.
                frame_for_model = _resize_frame_to_max_width(frame_bgr, 720)
                frame_with_watermark = _add_frame_id_watermark(frame_for_model, frame_id)
            else:
                h, w = frame_bgr.shape[:2]
                scale = target_height / h
                new_w = int(w * scale)
                frame_resized = cv2.resize(frame_bgr, (new_w, target_height), interpolation=cv2.INTER_AREA)
                frame_with_watermark = _add_frame_id_watermark(frame_resized, frame_id)
            rendered_frames.append(frame_with_watermark)
            _, buf = cv2.imencode(".jpg", frame_with_watermark)
            frames.append(base64.b64encode(buf).decode("utf-8"))

        logger.info(f"Extracting frames from video | Sampled: {len(frames)}")
    except av.error.FFmpegError as exc:
        raise ValueError(f"Cannot open video file with PyAV: {video_path}") from exc

    return frames, fps, total_frames, rendered_frames, sampled_fps

# ======================== 核心更新2：全英文Prompt（强制输出英文） ========================
def build_robot_subtask_prompt(
    sample_id: str,
    total_frames: int,
    fps: float,
    task_instruction: Optional[str] = None,
    atomic_skills: List[str] = ROBOT_ATOMIC_SKILLS,
    stitched_multiview_input: bool = False,
) -> str:
    """
    Build English prompt for robotic video subtask annotation (VLA-S2 model task planning)
    """
    skills_list = ", ".join(atomic_skills)
    task_instruction = (task_instruction or "").strip()
    task_instruction_block = ""
    input_layout_block = ""
    if task_instruction:
        task_instruction_block = f"""
- task_instruction: {task_instruction}

Additional instruction handling:
- The provided task_instruction is high-priority guidance from the user.
- Follow it carefully if it does not conflict with the hard segmentation rules below.
"""
    if stitched_multiview_input:
        input_layout_block = """

Multi-view image layout:
- Each input image is a stitched multi-view frame rather than a single camera image.
- The head view stays on the top row.
- The available wrist views keep their original aspect ratios, are stitched horizontally into a single bottom row, and then that bottom row is resized in width to match the head-view width.
- When both wrist views are present, left wrist is on the lower left and right wrist is on the lower right.
- View labels are overlaid inside each panel: "head", "left wrist", and "right wrist".
- Use all visible views jointly when judging contact, grasp state, and key frame boundaries.
"""
    return f"""
You are a high-precision robotic manipulation video analyst.

Analyze the full video and output a JSON plan of atomic subtasks.

Input:
- sample_id: {sample_id}
- total_frames: {total_frames}
- fps: {fps}
- allowed_atomic_skills: [{skills_list}]
{task_instruction_block}
{input_layout_block}

Segmentation rules:
1. Segment the WHOLE video into temporally ordered subtasks that cover every frame exactly once.
2. The first segment must start at frame 0.
3. The last segment must end at frame {total_frames - 1}.
4. For every adjacent pair of segments, next.start_frame MUST equal previous.end_frame + 1.
5. No gaps. No overlaps. No missing frames. No duplicated frames.
6. Only split when the dominant atomic skill changes. Do not over-segment tiny transitions.
7. Each segment must contain exactly one atomic skill from the allowed list.
8. Composite behaviors must be decomposed into separate atomic-skill segments.
9. "Move" refers ONLY to robot base/chassis movement. Output "Move" only when the whole robot base physically drives/translates to another location.
10. Arm-only motion, reaching, approaching, retracting, lifting, lowering, hovering, aligning, or repositioning for a manipulation action does NOT count as "Move".
11. If the arm merely moves in order to pick/place/press/pull/push/open/close/etc., those frames must be merged into that neighboring manipulation subtask instead of being segmented out.
12. Idle, waiting, transition, repositioning, or non-operational frames must be absorbed into the nearest meaningful neighboring action segment instead of creating a separate static/idle segment.
13. There is NO "Static" skill. Never output "Static" or any skill outside the allowed list.
14. If the whole video shows only one meaningful action, output exactly one segment covering all frames.
15. Never output preparation-only or idle-only segments such as "approaches the red chili pepper", "moves toward the cucumber", or "remains idle above the shopping cart".
16. Merge preparation-only frames into the following key action segment, and merge ending idle/waiting frames into the preceding key action segment.
17. Pick and Place are mandatory key subtasks whenever grasp state changes. If the gripper changes from NOT holding an object/tool to firmly holding/lifting it, create a "Pick" segment. If the gripper releases a held object/tool onto a surface, into a container, or into a fixture, create a "Place" segment.
18. NEVER merge Pick or Place into higher-level skills such as Sweep, Wipe, Open, Close, Pour, Move, Push, or Pull. Even if the pick/place duration is short, it must remain an explicit segment.
19. If a tool is picked up before wiping/sweeping, the sequence must be decomposed as "Pick tool" -> "Wipe/Sweep" -> "Place tool". Do not collapse the whole sequence into only "Wipe" or only "Sweep".
20. If an object is picked up before another downstream action, preserve that pick/place boundary explicitly. Example: "Pick cup" -> "Pour water" -> "Place cup", not a single merged "Pour water" segment.
21. When unsure between a high-level semantic task and Pick/Place, prefer preserving Pick/Place as separate segments. Under-segmenting Pick/Place is worse than slightly shorter atomic segments.
22. A Place segment is usually preceded earlier by a Pick segment for the same object/tool. If the video clearly shows grasping before release, do not omit that Pick segment.
23. For drawers and other sliding pull-out parts, opening should usually be labeled as "Pull", and closing should usually be labeled as "Push". Do NOT label a drawer-opening motion as "Open" if the mechanism is mainly being pulled outward.
24. If the robot first loosens a drawer slightly and then continues pulling it farther open, treat the whole continuous sequence as ONE "Pull" segment. Do not split it into "Open" followed by "Pull".
25. For drawer opening, merge the initial crack-open/unlatch phase and the later pull-out phase into the same subtask, because both belong to the same opening process.
26. Reserve "Open" and "Close" mainly for hinged or articulated parts such as doors, lids, covers, caps, or flaps where the state change itself is the main action rather than a drawer-like pulling/pushing motion.
27. Do not add subjects such as "robot" or "person" unless you can accurately distinguish whether the operation is performed by the left or the right hand/arm. If the hands cannot be distinguished, omit the subject.

Description rules:
- "task" is a short English description of the overall goal.
- "substask" must be a short verb phrase plus object, with NO subject.
- Do NOT write subjects such as "robot arm", "the robot", "the manipulator", or "gripper".
- Good examples: "picks up the yellow corn cob", "places the yellow corn cob into the plastic bag", "presses the red button", "drives to the shelf with the mobile base".
- Good decomposition examples: "picks up the sponge" + "wipes the table surface" + "places the sponge on the tray"; "picks up the broom" + "sweeps debris into the dustpan" + "places the broom down".
- Good drawer examples: "pulls open the drawer", "pushes the drawer closed".
- Bad examples: "Robot arm picks up the yellow corn cob", "Robot arm moves to select the cucumber", "approaches the red chili pepper", "remains idle above the table", "wipes the table with the sponge" when the video clearly shows first picking up the sponge and later placing it down, "opens the drawer slightly" + "pulls the drawer farther open" as two separate segments.
- Use concise, literal English.

Return JSON only in this exact schema:
{{
  "sample_id": "{sample_id}",
  "nframes": {total_frames},
  "fps": {fps},
  "task": "Overall task description in English",
  "segments": [
    {{
      "seg_id": 0,
      "start_frame": 0,
      "end_frame": 0,
      "substask": "Subtask description in English",
      "primitive_skill": "One allowed atomic skill"
    }}
  ]
}}

Output constraints:
- Return valid JSON only, with no markdown and no extra text.
- All descriptions must be in English.
- Every primitive_skill must exactly match one item in allowed_atomic_skills.
- Ensure segments are sorted by time and seg_id is sequential starting from 0.
    """

# ======================== 核心更新3：适配新格式的解析函数 ========================
def parse_robot_subtask_response(
    response_text: str,
    total_frames: int
) -> Dict[str, Any]:
    """
    Parse robotic subtask annotation results from model response (validate new format)
    :param response_text: Raw model response text
    :param total_frames: Total video frames (for boundary check)
    :return: Validated annotation result dict
    """
    try:
        # Extract JSON part (handle extra text)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No valid JSON structure found in response")
        
        json_str = response_text[json_start:json_end]
        result = json.loads(json_str)
        
        # 1. Validate required top-level fields
        required_fields = ["sample_id", "nframes", "fps", "task", "segments"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required top-level field: {field}")
        
        if total_frames <= 0:
            raise ValueError(f"Invalid total_frames: {total_frames}")

        # 2. Validate segments structure
        segments = result["segments"]
        if not isinstance(segments, list) or len(segments) == 0:
            raise ValueError("Segments must be a non-empty list")
        
        # 3. Validate and normalize segments
        normalized_segments = []
        
        for idx, seg in enumerate(segments):
            # Validate segment required fields
            seg_required = ["seg_id", "start_frame", "end_frame", "substask", "primitive_skill"]
            for field in seg_required:
                if field not in seg:
                    raise ValueError(f"Segment {idx} missing required field: {field}")
            
            # Type validation
            if not all(isinstance(seg[field], int) for field in ["seg_id", "start_frame", "end_frame"]):
                raise ValueError(f"Segment {idx} 'seg_id', 'start_frame', 'end_frame' must be integers")
            
            # Clamp frame boundaries
            start_frame = max(0, seg["start_frame"])
            end_frame = min(total_frames - 1, seg["end_frame"])
            if end_frame < start_frame:
                end_frame = start_frame
            
            # Validate atomic skill and drop invalid/static segments
            substask = _normalize_subtask_text(seg["substask"])

            primitive_skill = str(seg["primitive_skill"]).strip()
            primitive_skill, substask = _normalize_drawer_like_skill_and_subtask(primitive_skill, substask)
            auxiliary_merge_direction = _classify_auxiliary_subtask(substask)

            if primitive_skill not in ROBOT_ATOMIC_SKILLS:
                if auxiliary_merge_direction == "merge_prev" and normalized_segments:
                    normalized_segments[-1]["end_frame"] = max(normalized_segments[-1]["end_frame"], end_frame)
                    logger.warning(
                        f"Segment {idx} skill '{primitive_skill}' is invalid and auxiliary, merged into previous segment"
                    )
                else:
                    logger.warning(
                        f"Segment {idx} skill '{primitive_skill}' not in standard list, dropping this segment"
                    )
                continue

            if primitive_skill == "Move" and not _is_valid_base_move_subtask(substask):
                move_merge_direction = auxiliary_merge_direction or "merge_next"
                if move_merge_direction == "merge_prev" and normalized_segments:
                    normalized_segments[-1]["end_frame"] = max(normalized_segments[-1]["end_frame"], end_frame)
                    logger.warning(
                        f"Segment {idx} Move '{substask}' is arm-only/idle motion, merged into previous segment"
                    )
                else:
                    logger.warning(
                        f"Segment {idx} Move '{substask}' is not explicit base/chassis motion, merging into next segment"
                    )
                continue

            if auxiliary_merge_direction == "merge_prev":
                if normalized_segments:
                    normalized_segments[-1]["end_frame"] = max(normalized_segments[-1]["end_frame"], end_frame)
                    logger.warning(
                        f"Segment {idx} substask '{substask}' is auxiliary/idle, merged into previous segment"
                    )
                else:
                    logger.warning(
                        f"Segment {idx} substask '{substask}' is auxiliary/idle with no previous segment, merging into next segment"
                    )
                continue

            if auxiliary_merge_direction == "merge_next":
                logger.warning(
                    f"Segment {idx} substask '{substask}' is preparatory motion, merging into next segment"
                )
                continue

            normalized_segments.append({
                "original_seg_id": seg["seg_id"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "check_frame_id": end_frame,
                "substask": substask or f"{primitive_skill} action",
                "primitive_skill": primitive_skill
            })

        if not normalized_segments:
            raise ValueError("No valid segments with supported atomic skills found in model response")

        normalized_segments.sort(
            key=lambda seg: (seg["start_frame"], seg["end_frame"], seg["original_seg_id"])
        )
        normalized_segments = _merge_drawer_like_segments(normalized_segments)

        if len(normalized_segments) > total_frames:
            logger.warning(
                f"Model returned {len(normalized_segments)} segments for {total_frames} frames, truncating extra segments"
            )
            normalized_segments = normalized_segments[:total_frames]

        # 4. Rebuild segments so they are continuous, gap-free, and cover [0, total_frames - 1]
        fixed_segments = []
        cursor = 0
        segment_count = len(normalized_segments)

        for idx, seg in enumerate(normalized_segments):
            remaining_after = segment_count - idx - 1
            max_end = total_frames - 1 - remaining_after

            if cursor > total_frames - 1:
                logger.warning("Frame coverage already reached the end of the video, dropping extra segments")
                break

            if idx == segment_count - 1:
                end_frame = total_frames - 1
            else:
                end_frame = min(max(seg["end_frame"], cursor), max_end)

            fixed_segments.append({
                "seg_id": len(fixed_segments),
                "start_frame": cursor,
                "end_frame": end_frame,
                "check_frame_id": min(max(int(seg.get("check_frame_id", seg["end_frame"])), cursor), end_frame),
                "substask": seg["substask"],
                "primitive_skill": seg["primitive_skill"]
            })
            cursor = end_frame + 1

        if not fixed_segments:
            raise ValueError("No valid segments remained after normalization")

        # 5. Update result with fixed segments
        result["segments"] = fixed_segments
        result["nframes"] = total_frames  # Ensure consistency
        
        logger.info(f"Successfully parsed {len(fixed_segments)} valid subtasks")
        return result
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Result parsing failed: {str(e)}")
        raise

# ======================== 核心更新4：适配新格式的主函数 ========================
def extract_robot_subtasks(
    video_path: str,
    output_json_path: str,
    sample_id: str,  # New: Video unique ID
    max_sample_frames: int = 200,
    target_height: int = 336,
    task_instruction: Optional[str] = None,
    client: Optional[OpenAI] = None,
    check_frames_dir: Optional[str] = None,
    subtask_clips_dir: Optional[str] = None,
    ceph_video_view_mode: Optional[str] = None,
    model_input_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract robotic video subtask annotations and save to JSON file (VLA-S2 optimized)
    :param video_path: Path to video file
    :param output_json_path: Output JSON file path
    :param sample_id: Unique ID for the video (e.g., "episode_000000 (6)")
    :param max_sample_frames: Maximum number of sampled frames
    :param target_height: Frame resizing height
    :param task_instruction: Optional extra user instruction to inject into the prompt
    :param client: OpenAI client instance
    :param check_frames_dir: Directory for annotated representative check images
    :param subtask_clips_dir: Directory for exported subtask clips
    :return: Annotated result dict
    """
    if client is None:
        client = default_client
    
    # 1. Extract video frames
    logger.info(f"Starting video processing: {video_path} (Sample ID: {sample_id})")
    video_bytes = _build_video_bytes_payload(
        video_path,
        ceph_video_view_mode=ceph_video_view_mode,
    )
    frames, fps, total_frames, rendered_input_frames, sampled_input_fps = extract_video_frames(
        video_path,
        target_height,
        max_sample_frames,
        video_bytes=video_bytes,
        ceph_video_view_mode=ceph_video_view_mode,
    )
    if not frames:
        raise ValueError("No video frames extracted (empty frame list)")

    if model_input_video_path:
        export_model_input_video(
            rendered_input_frames,
            model_input_video_path,
            sampled_input_fps,
        )
    
    # 2. Get video basic info
    
    
    # 3. Build English prompt (critical update)
    stitched_multiview_input = len(_resolve_ceph_video_paths(video_path, ceph_video_view_mode)) > 1
    prompt = build_robot_subtask_prompt(
        sample_id,
        total_frames,
        fps,
        task_instruction=task_instruction,
        stitched_multiview_input=stitched_multiview_input,
    )
    
    # 4. Build API request messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                # Add video frames (base64 encoded)
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}} for frame in frames]
            ]
        }
    ]
    
    # 5. Call LLM API (force English output)
    encoded_payload_bytes = sum(len(frame) for frame in frames)
    logger.info(
        "Calling LLM for robotic subtask annotation (English only) | "
        f"Images: {len(frames)} | Encoded image payload: "
        f"{encoded_payload_bytes / (1024 * 1024):.1f} MiB"
    )
    response = _api_create_with_retry(
        client,
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,  # Low temperature for stable results
        max_tokens=4096,
        # Force English output
        top_p=1.0
    )
    
    # 6. Parse and validate response
    response_text = _extract_chat_response_text(response)
    logger.info(f"Model response (first 200 chars): {response_text[:200]}...")
    result = parse_robot_subtask_response(response_text, total_frames)
    
    # 7. Save standardized JSON file (UTF-8 encoding for English)
    output_dir = os.path.dirname(output_json_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    saved_result = _build_output_json(result)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(saved_result, f, ensure_ascii=False, indent=4)

    export_subtask_check_frames(
        video_path=video_path,
        result=result,
        output_json_path=output_json_path,
        check_frames_dir=check_frames_dir,
        video_bytes=video_bytes,
        ceph_video_view_mode=ceph_video_view_mode,
    )
    export_subtask_clips(
        video_path=video_path,
        result=result,
        output_json_path=output_json_path,
        fps=fps,
        clips_dir=subtask_clips_dir,
        video_bytes=video_bytes,
    )
    
    logger.info(f"Annotation results saved to: {output_json_path}")
    return saved_result


def _require_petrel_file_client() -> None:
    if file_client is None:
        raise RuntimeError("petrel_client is not available, but batch ceph listing was requested")


def _ensure_full_ceph_path(path: str, bucket_prefix: Optional[str] = None) -> str:
    path = path.strip()
    if path.startswith("s3://") or re.match(r"^[^:]+:s3://", path):
        return path
    prefix = bucket_prefix
    if prefix is None:
        prefix = os.environ.get("CORTEX_OBJECT_STORAGE_PREFIX", "")
    if not prefix:
        return path
    return prefix.rstrip("/") + "/" + path.lstrip("/")


def _append_jsonl(jsonl_path: str, item: Dict[str, Any]) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _episode_stem(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path.rstrip("/")))[0]


def _process_task_videos(
    task_name: str,
    videos_list: List[str],
    output_root: str,
    max_sample_frames: int,
    task_instruction: Optional[str] = None,
    ceph_video_view_mode: Optional[str] = None,
) -> None:
    videos_list = [video_path for video_path in videos_list if not video_path.endswith("/")]
    task_dir = os.path.join(output_root, task_name)
    episodes_dir = os.path.join(task_dir, "episodes")
    check_frames_root = os.path.join(task_dir, "check_frames")
    clips_root = os.path.join(task_dir, "clips")
    model_input_videos_root = os.path.join(task_dir, "model_input_videos")
    episodes_jsonl_path = os.path.join(task_dir, "episodes.jsonl")

    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(episodes_dir, exist_ok=True)
    os.makedirs(check_frames_root, exist_ok=True)
    os.makedirs(clips_root, exist_ok=True)
    os.makedirs(model_input_videos_root, exist_ok=True)

    with open(episodes_jsonl_path, "w", encoding="utf-8"):
        pass

    for index, raw_video_path in enumerate(videos_list, start=1):
        video_path = _ensure_full_ceph_path(raw_video_path)
        sample_id = video_path
        episode_name = _episode_stem(video_path)

        output_json_path = os.path.join(episodes_dir, f"{episode_name}.json")
        check_frames_dir = os.path.join(check_frames_root, episode_name)
        subtask_clips_dir = os.path.join(clips_root, episode_name)
        model_input_video_path = os.path.join(model_input_videos_root, f"{episode_name}.mp4")

        logger.info(f"[{task_name}] Processing episode {index}/{len(videos_list)}: {video_path}")
        episode_infos = extract_robot_subtasks(
            video_path=video_path,
            output_json_path=output_json_path,
            sample_id=sample_id,
            max_sample_frames=max_sample_frames,
            task_instruction=task_instruction,
            check_frames_dir=check_frames_dir,
            subtask_clips_dir=subtask_clips_dir,
            ceph_video_view_mode=ceph_video_view_mode,
            model_input_video_path=model_input_video_path,
        )
        _append_jsonl(episodes_jsonl_path, episode_infos)
        break

    logger.info(f"[{task_name}] Saved episode infos to: {episodes_jsonl_path}")


def batch_process_ceph_tasks(
    ceph_root: str,
    output_root: str,
    max_sample_frames: int,
    ceph_video_view_mode: Optional[str] = None,
) -> None:
    _require_petrel_file_client()
    os.makedirs(output_root, exist_ok=True)

    contents = sorted(file_client.list(ceph_root))
    for content in contents:
        if not content.endswith("/"):
            continue

        task_name = os.path.basename(content.rstrip("/"))
        task_instruction = re.findall(r'[A-Za-z]+', task_name)
        task_instruction = ' '.join(word.lower() for word in task_instruction)
        
        videos_path = _ensure_full_ceph_path(
            content.rstrip("/") + "/videos/chunk-000/observation.images.head_rgb/"
        )
        
        videos_list = sorted(file_client.list(videos_path))
        logger.info(f"Processing task '{task_name}' with {len(videos_list)} entries from: {videos_path}")
        _process_task_videos(
            task_name=task_name,
            videos_list=videos_list,
            output_root=output_root,
            max_sample_frames=max_sample_frames,
            task_instruction=task_instruction,
            ceph_video_view_mode=ceph_video_view_mode,
        )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLA Robotic Video Subtask Annotation Tool (English Only)")
    parser.add_argument("--video_path", help="Optional single video path (local path or ceph path)")
    parser.add_argument("--output_path", default="annotations/vlm/episode_000000.json", help="Output root directory for batch mode, or JSON path for single-video mode")
    parser.add_argument("--sample_id", help="Unique video ID for single-video mode; defaults to video_path")
    parser.add_argument("--ceph_path", help="Optional object-storage root containing task folders for batch mode")
    parser.add_argument("--task_instruction", help="Optional extra instruction injected into the prompt")
    parser.add_argument("--max_sample_frames", type=int, default=200, help="Maximum sampled frames (default: 300)")
    parser.add_argument(
        "--ceph_video_view_mode",
        default="head_wrists",
        choices=["head", "head_wrists"],
        help="Video input mode: head only, or stitched layout with head on top and left/right wrists on the bottom row",
    )
    parser.add_argument("--check_frames_dir", default="exp/cortex/annotation/check_frames", help="Directory to save annotated representative check images")
    parser.add_argument("--subtask_clips_dir", default="exp/cortex/annotation/subtask_clips", help="Directory to save exported subtask clips")
    parser.add_argument("--base_url", help="API base URL (override default)")
    parser.add_argument("--model", help="Model name or served model path (override default)")
    parser.add_argument("--api_key", help="API key for the OpenAI-compatible endpoint")
    
    args = parser.parse_args()
    
    if args.model:
        global MODEL_NAME
        MODEL_NAME = args.model

    # Override endpoint config if specified.
    if args.base_url or args.api_key:
        global clients, default_client
        clients = get_clients([args.base_url] if args.base_url else None, api_key=args.api_key)
        default_client = clients[0]

    if args.video_path:
        sample_id = args.sample_id or args.video_path
        if args.sample_id is None:
            logger.info(f"No sample_id provided, using video_path as sample_id: {sample_id}")

        try:
            extract_robot_subtasks(
                video_path=args.video_path,
                output_json_path=args.output_path,
                sample_id=sample_id,
                max_sample_frames=args.max_sample_frames,
                task_instruction=args.task_instruction,
                check_frames_dir=args.check_frames_dir,
                subtask_clips_dir=args.subtask_clips_dir,
                ceph_video_view_mode=args.ceph_video_view_mode,
            )
            logger.info("Single-video annotation completed successfully!")
        except Exception as e:
            logger.error(f"Annotation failed: {str(e)}")
            sys.exit(1)
        return

    if not args.ceph_path:
        parser.error("provide --video_path for single-video mode, or --ceph_path for batch mode")

    if args.check_frames_dir or args.subtask_clips_dir:
        logger.warning("check_frames_dir and subtask_clips_dir are ignored in batch mode; task-specific directories are created automatically")

    try:
        batch_process_ceph_tasks(
            ceph_root=args.ceph_path,
            output_root=args.output_path,
            max_sample_frames=args.max_sample_frames,
            ceph_video_view_mode=args.ceph_video_view_mode,
        )
        logger.info("Batch annotation completed successfully!")
    except Exception as e:
        logger.error(f"Batch annotation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
