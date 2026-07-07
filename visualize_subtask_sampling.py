#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import posixpath
import sys
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import av
from PIL import Image, ImageDraw, ImageFont

def _resolve_repo_root() -> Path:
    """Resolve repo root robustly even if this script is moved."""
    cur = Path(__file__).resolve().parent
    candidates = [cur] + list(cur.parents)
    for p in candidates:
        if (p / "cortex" / "dataloader" / "qwenvl_llavajson" / "qwen_data_config.py").exists():
            return p
    raise FileNotFoundError("Cannot resolve repo root containing cortex/dataloader/qwenvl_llavajson/qwen_data_config.py")


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_data_list_from_config():
    cfg_path = REPO_ROOT / "cortex" / "dataloader" / "qwenvl_llavajson" / "qwen_data_config.py"
    spec = importlib.util.spec_from_file_location("qwen_data_config", str(cfg_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config module from {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "data_list"):
        raise AttributeError(f"data_list not found in {cfg_path}")
    return module.data_list


try:
    from mmengine import fileio as mm_fileio
except Exception:
    mm_fileio = None

try:
    from petrel_client.client import Client
    file_client = Client("~/petreloss.conf", enable_mc=False)
except Exception:
    file_client = None


def is_remote_path(path: str) -> bool:
    return "://" in str(path)


def normalize_remote_path_for_petrel(path: str) -> str:
    path = str(path)
    marker = "s3://"
    idx = path.find(marker)
    if idx >= 0:
        return path[idx:]
    return path


def join_path(base: str, *parts: str) -> str:
    parts = tuple(p for p in parts if p)
    if not parts:
        return base
    if is_remote_path(base):
        out = str(base).rstrip("/")
        for p in parts:
            out = posixpath.join(out, str(p).lstrip("/"))
        return out
    return os.path.join(base, *parts)


def path_exists(path: str) -> bool:
    path = str(path)
    if is_remote_path(path):
        if mm_fileio is not None:
            return bool(mm_fileio.exists(path))
        if file_client is not None:
            try:
                _ = file_client.get(normalize_remote_path_for_petrel(path))
                return True
            except Exception:
                return False
        raise RuntimeError("mmengine or petrel_client is required for remote paths.")
    return os.path.exists(path)


def read_path_text(path: str) -> str:
    path = str(path)
    if is_remote_path(path):
        if mm_fileio is not None:
            if hasattr(mm_fileio, "get_text"):
                return str(mm_fileio.get_text(path, encoding="utf-8"))
            payload = mm_fileio.get(path)
            return payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
        if file_client is not None:
            payload = file_client.get(normalize_remote_path_for_petrel(path))
            if isinstance(payload, (bytes, bytearray)):
                return bytes(payload).decode("utf-8")
            return str(payload)
        raise RuntimeError("mmengine or petrel_client is required for remote paths.")
    return Path(path).read_text(encoding="utf-8")


def read_path_bytes(path: str) -> bytes:
    path = str(path)
    if is_remote_path(path):
        if mm_fileio is not None:
            payload = mm_fileio.get(path)
            if not isinstance(payload, (bytes, bytearray)):
                raise TypeError(f"Expected bytes for remote file {path}, got {type(payload).__name__}")
            return bytes(payload)
        if file_client is not None:
            payload = file_client.get(normalize_remote_path_for_petrel(path))
            if not isinstance(payload, (bytes, bytearray)):
                raise TypeError(f"Expected bytes for remote file {path}, got {type(payload).__name__}")
            return bytes(payload)
        raise RuntimeError("mmengine or petrel_client is required for remote paths.")
    return Path(path).read_bytes()


@dataclass
class SamplingBlock:
    block_kind: str
    subtask_id: int
    frame_start: int
    frame_end: int
    sampled_frame_start: int
    step: int
    num_samples: int


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    text = read_path_text(path).strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array in {path}")
        return payload
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def resolve_fps(data_path: str, task_id: str) -> float:
    candidates = [
        join_path(data_path, task_id, "meta", "info.json"),
        join_path(data_path, task_id, "info.json"),
        join_path(data_path, "meta", "info.json"),
        join_path(data_path, "info.json"),
    ]
    info = None
    for path in candidates:
        try:
            info = json.loads(read_path_text(path))
            break
        except Exception:
            continue
    if info is None:
        raise FileNotFoundError(f"Cannot find info.json from candidates: {candidates}")

    if "fps" in info:
        return float(info["fps"])

    features = info.get("features", {})
    if isinstance(features, dict):
        for meta in features.values():
            if isinstance(meta, dict):
                vinfo = meta.get("video_info")
                if isinstance(vinfo, dict) and "video.fps" in vinfo:
                    return float(vinfo["video.fps"])
                iinfo = meta.get("info")
                if isinstance(iinfo, dict) and "video.fps" in iinfo:
                    return float(iinfo["video.fps"])

    raise ValueError("Cannot parse fps from info.json")


def choose_main_video_key(video_keys_csv: str) -> str:
    keys = [k.strip() for k in video_keys_csv.split(",") if k.strip()]
    if not keys:
        raise ValueError("video_keys is empty")
    for k in keys:
        lk = k.lower()
        if "head" in lk or "front" in lk:
            return k
    return keys[0]


def add_block(
    blocks: List[SamplingBlock],
    sampled_frames: Dict[int, List[SamplingBlock]],
    block_kind: str,
    subtask_id: int,
    frame_start: int,
    frame_end: int,
    step: int,
    sample_min_frame: Optional[int] = None,
    sample_max_frame: Optional[int] = None,
) -> None:
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    step = int(step)
    if step <= 0 or frame_end <= frame_start:
        return

    if sample_min_frame is None:
        sample_min_frame = frame_start
    if sample_max_frame is None:
        sample_max_frame = frame_end

    sample_min_frame = max(frame_start, int(sample_min_frame))
    sample_max_frame = min(frame_end, int(sample_max_frame))
    if sample_max_frame <= sample_min_frame:
        return

    full_num_samples = math.ceil((frame_end - frame_start) / step)
    first_sample_idx = math.ceil((sample_min_frame - frame_start) / step)
    last_sample_exclusive = math.ceil((sample_max_frame - frame_start) / step)

    first_sample_idx = max(0, min(first_sample_idx, full_num_samples))
    last_sample_exclusive = max(first_sample_idx, min(last_sample_exclusive, full_num_samples))
    num_samples = last_sample_exclusive - first_sample_idx
    if num_samples <= 0:
        return

    sampled_frame_start = frame_start + first_sample_idx * step

    block = SamplingBlock(
        block_kind=block_kind,
        subtask_id=subtask_id,
        frame_start=frame_start,
        frame_end=frame_end,
        sampled_frame_start=sampled_frame_start,
        step=step,
        num_samples=num_samples,
    )
    blocks.append(block)

    for j in range(num_samples):
        frame_id = sampled_frame_start + j * step
        sampled_frames.setdefault(frame_id, []).append(block)


def build_sampling_blocks(
    episode: Dict[str, Any],
    fps: float,
    sample_interleave: int,
    dense_sample_step: int,
    final_tail_sample_step: int,
    ignore_boundary_sec: float,
    transition_tail_sec: float,
    transition_head_sec: float,
    last_tail_sec: float,
) -> Tuple[List[SamplingBlock], Dict[int, List[SamplingBlock]]]:
    action_config = episode.get("action_config", [])
    blocks: List[SamplingBlock] = []
    sampled_frames: Dict[int, List[SamplingBlock]] = {}

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
            prev_start = int(action_config[subtask_id - 1]["start_frame"])
            transition_start = max(start - tail_frames, prev_start)
            transition_end = min(end, start + head_frames)
            add_block(
                blocks,
                sampled_frames,
                block_kind="transition_dense",
                subtask_id=subtask_id,
                frame_start=transition_start,
                frame_end=transition_end,
                step=dense_sample_step,
            )
            stable_start = transition_end
        else:
            stable_start = start

        stable_end = max(stable_start, end - tail_frames) if not is_last else max(stable_start, end - last_tail_frames)
        add_block(
            blocks,
            sampled_frames,
            block_kind="uniform",
            subtask_id=subtask_id,
            frame_start=stable_start,
            frame_end=stable_end,
            step=sample_interleave,
            sample_min_frame=stable_start + (ignore_frames if not is_first else 0),
            sample_max_frame=stable_end - ignore_frames,
        )

        if is_last:
            add_block(
                blocks,
                sampled_frames,
                block_kind="final_tail_dense",
                subtask_id=subtask_id + 1,
                frame_start=stable_end,
                frame_end=end,
                step=final_tail_sample_step,
            )

    return blocks, sampled_frames


def find_episode(episodes: Sequence[Dict[str, Any]], task_id: Optional[str], episode_index: Optional[int]) -> Dict[str, Any]:
    for ep in episodes:
        if task_id is not None and str(ep.get("task_id")) != task_id:
            continue
        if episode_index is not None and int(ep.get("episode_index", -1)) != int(episode_index):
            continue
        return ep
    examples = ", ".join(
        f"{ep.get('task_id')}:{ep.get('episode_index')}" for ep in episodes[:8]
    )
    raise ValueError(
        f"Cannot find episode with task_id={task_id}, episode_index={episode_index}. "
        f"First available examples: {examples}"
    )


def get_subtask_at_frame(action_config: Sequence[Dict[str, Any]], frame_idx: int) -> Tuple[int, str]:
    for i, st in enumerate(action_config):
        s = int(st.get("start_frame", 0))
        e = int(st.get("end_frame", 0))
        if s <= frame_idx < e:
            text = str(st.get("action_text") or st.get("skill") or "").strip()
            return i, text if text else f"subtask_{i}"
    return len(action_config), "task_completed"


def block_kind_for_frame(blocks: Sequence[SamplingBlock], frame_idx: int) -> str:
    # priority: transition > final_tail > uniform > outside
    winner = "outside"
    for b in blocks:
        if b.frame_start <= frame_idx < b.frame_end:
            if b.block_kind == "transition_dense":
                return "transition_dense"
            if b.block_kind == "final_tail_dense":
                winner = "final_tail_dense"
            elif winner == "outside":
                winner = "uniform"
    return winner


def color_for_kind(kind: str) -> Tuple[int, int, int]:
    if kind == "transition_dense":
        return (255, 146, 43)
    if kind == "final_tail_dense":
        return (219, 82, 77)
    if kind == "uniform":
        return (80, 184, 90)
    return (120, 120, 120)


def _load_font(font_size: int) -> ImageFont.ImageFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def resize_frame_keep_aspect(
    frame_rgb,
    resize_long_side: Optional[int],
    resize_max_width: Optional[int],
    resize_min_width: Optional[int],
):
    if (resize_long_side is None or int(resize_long_side) <= 0) and (
        resize_max_width is None or int(resize_max_width) <= 0
    ) and (resize_min_width is None or int(resize_min_width) <= 0):
        return frame_rgb
    img = Image.fromarray(frame_rgb)
    w, h = img.size
    scale = 1.0

    if resize_long_side is not None and int(resize_long_side) > 0:
        long_side = max(w, h)
        target_long = int(resize_long_side)
        if long_side > target_long:
            scale = min(scale, target_long / float(long_side))

    if resize_max_width is not None and int(resize_max_width) > 0:
        target_w = int(resize_max_width)
        if w > target_w:
            scale = min(scale, target_w / float(w))

    # Optional upsample path for low-resolution source videos.
    if resize_min_width is not None and int(resize_min_width) > 0:
        target_min_w = int(resize_min_width)
        if w < target_min_w:
            scale = max(scale, target_min_w / float(w))

    if abs(scale - 1.0) < 1e-6:
        return frame_rgb

    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    # yuv420p prefers even dimensions.
    if new_w % 2 == 1:
        new_w -= 1
    if new_h % 2 == 1:
        new_h -= 1
    new_w = max(2, new_w)
    new_h = max(2, new_h)
    return img.resize((new_w, new_h), Image.Resampling.BICUBIC)


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, stroke_width: int) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return max(1, box[2] - box[0]), max(1, box[3] - box[1])


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, stroke_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        cand = f"{cur} {w}"
        tw, _ = _text_bbox(draw, cand, font, stroke_width)
        if tw <= max_width:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _resolve_font_size(font_size: int, w: int, h: int) -> int:
    if font_size > 0:
        return font_size
    auto_size = int(w * 0.065)
    return max(34, min(96, auto_size))


def _pick_text_anchor(
    w: int,
    text_h: int,
    pad: int,
    text_position: str,
    max_text_width: int,
) -> Tuple[int, int]:
    # Stable anchor: never switch side frame-by-frame.
    left_x = pad
    right_x = max(pad, w - max_text_width - pad)
    top_y = pad

    if text_position == "top-right":
        return right_x, top_y
    return left_x, top_y


def draw_overlay(
    frame_rgb,
    frame_idx: int,
    total_frames: int,
    fps: float,
    episode: Dict[str, Any],
    action_config: Sequence[Dict[str, Any]],
    blocks: Sequence[SamplingBlock],
    sampled_frames: Dict[int, List[SamplingBlock]],
    font_size: int,
    text_position: str,
) -> Image.Image:
    if isinstance(frame_rgb, Image.Image):
        img = frame_rgb
    else:
        img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font_size = _resolve_font_size(font_size, w, h)
    font = _load_font(font_size)
    small_font = _load_font(max(16, int(font_size * 0.62)))
    stroke_w = max(2, int(font_size * 0.1))
    pad = max(12, int(font_size * 0.4))

    subtask_id, subtask_text = get_subtask_at_frame(action_config, frame_idx)
    kind = block_kind_for_frame(blocks, frame_idx)
    is_key = frame_idx in sampled_frames

    # timeline first (used to bound text placement)
    tl_margin = max(20, int(w * 0.02))
    tl_h = max(26, int(h * 0.05))
    tl_y1 = h - max(24, int(h * 0.04))
    tl_y0 = tl_y1 - tl_h
    x0, x1 = tl_margin, w - tl_margin

    goal = episode.get("tasks", "")
    if isinstance(goal, list):
        goal = goal[0] if goal else ""
    goal = str(goal).strip()

    t = frame_idx / fps if fps > 0 else 0.0
    raw_items: List[Tuple[str, str]] = [
        ("meta", f"task_id={episode.get('task_id')}  episode_index={episode.get('episode_index')}  frame={frame_idx}/{max(total_frames-1,0)}  t={t:.2f}s"),
        ("subtask", f"subtask[{subtask_id}]: {subtask_text}"),
        ("meta", f"phase={kind}  sampled_keyframe={'YES' if is_key else 'no'}"),
        ("goal", f"goal: {goal}"),
    ]
    max_text_width = int(w * 0.65)
    lines: List[Tuple[str, str]] = []
    for line_kind, line in raw_items:
        for wrapped in _wrap_text(draw, line, font=font, max_width=max_text_width, stroke_width=stroke_w):
            lines.append((line_kind, wrapped))

    line_h = _text_bbox(draw, "Ag", font, stroke_w)[1]
    line_gap = max(4, int(line_h * 0.22))
    text_h = len(lines) * line_h + max(0, len(lines) - 1) * line_gap
    text_w = 0
    for _, line in lines:
        lw, _ = _text_bbox(draw, line, font, stroke_w)
        text_w = max(text_w, lw)
    text_x, text_y = _pick_text_anchor(
        w=w,
        text_h=text_h,
        pad=pad,
        text_position=text_position,
        max_text_width=max_text_width,
    )

    y = text_y
    for line_kind, line in lines:
        lw, _ = _text_bbox(draw, line, font, stroke_w)
        if line_kind == "subtask":
            # Highlight subtask text line.
            bx0 = max(0, text_x - 6)
            by0 = max(0, y - 4)
            bx1 = min(w, text_x + lw + 8)
            by1 = min(h, y + line_h + 4)
            draw.rectangle([bx0, by0, bx1, by1], fill=(255, 232, 76), outline=(255, 198, 0), width=2)
            txt_fill = (20, 20, 20)
            txt_stroke = (255, 255, 255)
        else:
            txt_fill = (255, 255, 255)
            txt_stroke = (0, 0, 0)
        draw.text(
            (text_x, y),
            line,
            fill=txt_fill,
            font=font,
            stroke_width=stroke_w,
            stroke_fill=txt_stroke,
        )
        y += line_h + line_gap

    draw.line([(x0, tl_y0), (x1, tl_y0)], fill=(255, 255, 255), width=2)
    draw.line([(x0, tl_y1), (x1, tl_y1)], fill=(255, 255, 255), width=2)

    # subtask segments
    span = max(total_frames - 1, 1)
    palette = [(84, 182, 255), (180, 135, 255), (115, 224, 130), (255, 187, 90), (255, 122, 122), (130, 220, 220)]
    for i, st in enumerate(action_config):
        s = max(0, int(st.get("start_frame", 0)))
        e = min(total_frames - 1, int(st.get("end_frame", 0)))
        if e <= s:
            continue
        sx = x0 + int((s / span) * (x1 - x0))
        ex = x0 + int((e / span) * (x1 - x0))
        color = palette[i % len(palette)]
        draw.rectangle([sx, tl_y0 + 3, max(sx + 1, ex), tl_y1 - 3], fill=color)

    # transition/final windows as overlay bands
    for b in blocks:
        if b.block_kind not in {"transition_dense", "final_tail_dense"}:
            continue
        sx = x0 + int((max(0, b.frame_start) / span) * (x1 - x0))
        ex = x0 + int((min(total_frames - 1, b.frame_end) / span) * (x1 - x0))
        c = color_for_kind(b.block_kind)
        band_h = max(4, (tl_y1 - tl_y0) // 4)
        draw.rectangle([sx, tl_y0 + band_h, max(sx + 1, ex), tl_y0 + band_h * 2], fill=c)

    # sampled keyframe ticks
    for fidx in sampled_frames.keys():
        if fidx < 0 or fidx >= total_frames:
            continue
        x = x0 + int((fidx / span) * (x1 - x0))
        draw.line([(x, tl_y0 - 8), (x, tl_y0 - 2)], fill=(255, 255, 255), width=1)

    # current frame cursor + pulse when sampled
    cx = x0 + int((frame_idx / span) * (x1 - x0))
    draw.line([(cx, tl_y0 - 10), (cx, tl_y1 + 10)], fill=(255, 255, 255), width=3)
    if is_key:
        pulse_c = color_for_kind(kind)
        r = max(6, int(font_size * 0.22))
        draw.ellipse([cx - r, tl_y0 - (r * 2 + 6), cx + r, tl_y0 - 6], outline=pulse_c, width=3)

    # small legend
    legend = [
        ("uniform", color_for_kind("uniform")),
        ("transition", color_for_kind("transition_dense")),
        ("final_tail", color_for_kind("final_tail_dense")),
    ]
    lx = x0
    ly = tl_y1 + 8
    for name, c in legend:
        draw.rectangle([lx, ly, lx + 12, ly + 12], fill=c)
        draw.text(
            (lx + 16, ly - 3),
            name,
            fill=(245, 245, 245),
            font=small_font,
            stroke_width=max(1, stroke_w - 1),
            stroke_fill=(0, 0, 0),
        )
        lx += max(100, int(font_size * 3.8))

    return img


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Visualize subtask frame sampling with pyav")
    p.add_argument("--dataset-name", default="behavior_subtask_train", help="name in qwen_data_config.py")
    p.add_argument("--task-id", default=None, help="e.g. task-0000")
    p.add_argument("--episode-index", type=int, default=None, help="e.g. 20")
    p.add_argument("--annotation-path", default=None, help="override annotation path")
    p.add_argument("--data-path", default=None, help="override data root")
    p.add_argument("--main-video-key", default=None, help="override main view key")
    p.add_argument("--fps", type=float, default=None, help="override fps and skip reading info.json")
    p.add_argument("--max-frames", type=int, default=-1, help="max decoded frames to render, -1 means all frames")
    p.add_argument("--font-size", type=int, default=0, help="overlay text font size, 0 means auto")
    p.add_argument(
        "--text-position",
        default="top-left",
        choices=["auto", "top-left", "top-right"],
        help="overlay text position",
    )
    p.add_argument("--resize-long-side", type=int, default=None, help="optional downsample long side, keep aspect ratio")
    p.add_argument("--resize-max-width", type=int, default=1280, help="optional max output width, keep aspect ratio")
    p.add_argument("--resize-min-width", type=int, default=None, help="optional min output width (upsample if source too small)")
    p.add_argument("--crf", type=int, default=16, help="x264 quality (lower is sharper, typical 14-23)")
    p.add_argument("--preset", default="medium", help="x264 preset, e.g. veryfast/medium/slow")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--output", default="./exp/visualize_subtask_sampling.mp4")
    return p


def fallback_fps_for_dataset(dataset_name: str) -> Optional[float]:
    name = str(dataset_name).lower()
    if "agibot_subtask" in name:
        return 30.0
    if "galaxea_subtask" in name:
        return 30.0
    if "behavior_subtask" in name:
        return 30.0
    return None


def main() -> None:
    args = build_argparser().parse_args()

    data_list = load_data_list_from_config()
    cfgs = data_list([args.dataset_name])
    if not cfgs:
        raise ValueError(f"dataset_name not found: {args.dataset_name}")
    cfg = cfgs[0]

    annotation_path = args.annotation_path or cfg["annotation_path"]
    data_path = args.data_path or cfg["data_path"]

    episodes = read_json_or_jsonl(annotation_path)
    episode = find_episode(episodes, task_id=args.task_id, episode_index=args.episode_index)

    task_id = str(episode.get("task_id", ""))
    if args.fps is not None:
        fps = float(args.fps)
    else:
        try:
            fps = resolve_fps(data_path, task_id)
        except Exception as e:
            fallback_fps = fallback_fps_for_dataset(args.dataset_name)
            if fallback_fps is None:
                raise
            print(
                f"[warn] failed to read fps from info.json ({e}); "
                f"fallback to dataset default fps={fallback_fps}"
            )
            fps = fallback_fps

    video_key = args.main_video_key or choose_main_video_key(cfg.get("video_keys", "observation.images.rgb.head"))
    video_rel = str(episode.get("video_path", ""))
    if not video_rel:
        raise ValueError("episode missing video_path")
    video_path = join_path(data_path, video_rel).replace("{video_key}", video_key)

    blocks, sampled_frames = build_sampling_blocks(
        episode=episode,
        fps=fps,
        sample_interleave=int(cfg.get("sample_interleave", 8)),
        dense_sample_step=int(cfg.get("dense_sample_step", 6)),
        final_tail_sample_step=int(cfg.get("final_tail_sample_step", 3)),
        ignore_boundary_sec=float(cfg.get("ignore_boundary_sec", 0.0)),
        transition_tail_sec=float(cfg.get("transition_tail_sec", 0.1)),
        transition_head_sec=float(cfg.get("transition_head_sec", 0.2)),
        last_tail_sec=float(cfg.get("last_tail_sec", 1.0)),
    )

    if is_remote_path(args.output):
        raise ValueError("--output must be a local path for now.")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if is_remote_path(video_path):
        in_container = av.open(BytesIO(read_path_bytes(video_path)))
    else:
        in_container = av.open(video_path)
    try:
        in_stream = in_container.streams.video[0]
        in_fps = float(in_stream.average_rate) if in_stream.average_rate else fps
        if not math.isfinite(in_fps) or in_fps <= 0:
            in_fps = fps

        total_frames = int(in_stream.frames) if in_stream.frames else int(episode.get("length", 0))
        if total_frames <= 0:
            total_frames = max(int(st.get("end_frame", 0)) for st in episode.get("action_config", []))
        total_frames = max(total_frames, 1)

        out_container = av.open(args.output, mode="w")
        try:
            out_stream = out_container.add_stream("libx264", rate=Fraction(str(in_fps)).limit_denominator(1000))
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {
                "crf": str(int(args.crf)),
                "preset": str(args.preset),
            }

            wrote = 0
            for frame_idx, frame in enumerate(in_container.decode(in_stream)):
                if frame_idx < args.start_frame:
                    continue
                if args.max_frames > 0 and wrote >= args.max_frames:
                    break

                rgb = frame.to_ndarray(format="rgb24")
                resized = resize_frame_keep_aspect(
                    rgb,
                    resize_long_side=args.resize_long_side,
                    resize_max_width=args.resize_max_width,
                    resize_min_width=args.resize_min_width,
                )
                overlay_img = draw_overlay(
                    frame_rgb=resized,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    fps=fps,
                    episode=episode,
                    action_config=episode.get("action_config", []),
                    blocks=blocks,
                    sampled_frames=sampled_frames,
                    font_size=int(args.font_size),
                    text_position=args.text_position,
                )

                out_frame = av.VideoFrame.from_image(overlay_img)
                if wrote == 0:
                    out_stream.width = out_frame.width
                    out_stream.height = out_frame.height

                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)
                wrote += 1

            for packet in out_stream.encode():
                out_container.mux(packet)
        finally:
            out_container.close()

    finally:
        in_container.close()

    sampled_count = len(sampled_frames)
    print(
        f"Saved visualization to {args.output}\n"
        f"dataset={args.dataset_name} task_id={episode.get('task_id')} episode_index={episode.get('episode_index')}\n"
        f"video={video_path}\n"
        f"fps={fps:.3f} sampled_keyframes={sampled_count} blocks={len(blocks)}"
    )


if __name__ == "__main__":
    main()
