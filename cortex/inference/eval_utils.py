import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mmengine import fileio
from PIL import Image

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
CEPH_PATH_PATTERN = re.compile(r"^(?:[^/]+:)?s3://")

ROBOT_ATOMIC_SKILLS = [
    "Pick", "PickAndPlace", "Place", "Remove", "Press", "Push", "Pull",
    "Navigate", "Fold", "Wipe", "Close", "Open", "Pour", "Cut", "Rotate",
    "Handover", "Sweep", "Stack", "Unstack", "Screw", "Unscrew", "Scan",
    "Aim", "Clamp", "Rinse", "Spread", "Release", "Retreat",
    "AdjustPosture", "Tie", "Strike", "Stir",
]

SYSTEM_MESSAGE = (
    'You are a robot program for high-level manipulation. '
    'Given the global task goal, the input language memory, an optional task-order cue, a list of candidate atomic skills, '
    'and the camera observations, '
    'first choose exactly one current atomic skill from the candidate skill list when the task is in progress, '
    'then predict the subtask the robot should currently be in and the language memory that should be active now. '
    'The optional task-order cue can be either a Detailed Global Task Instruction, a Subtask List, or absent; '
    'Detailed Global Task Instruction and Subtask List are mutually exclusive. '
    'When a Detailed Global Task Instruction is provided, it is ordered according to the subtask sequence that should be predicted. '
    'When a Subtask List is provided, use its item order as the subtask sequence and prefer the corresponding item text for current_subtask. '
    'When no task-order cue is provided, infer the current progress from the global task goal, input language memory, and observations. '
    'Use any provided task-order cue as a constraint when reasoning about progress, and do not skip ahead to later subtasks unless the observations and input language memory clearly indicate that the earlier subtasks have already been completed. '
    'If the observations show that the current subtask is still ongoing, keep the same subtask and keep the active language memory unchanged. '
    'If the observations show that the previous subtask has just been completed or the robot has already entered the next subtask, '
    'predict the next subtask and switch to the new active language memory that reflects the completed progress. '
    "If the task has already been completed, set current_subtask to task_completed and set current_skill to null. "
    'Return a JSON object only with keys "current_skill", "current_subtask", and "active_language_memory". '
    '"current_skill" must be exactly one skill copied from the candidate atomic skill list when the task is in progress, '
    'or null when the task is already completed. '
    '"active_language_memory" should be a concise semantic summary containing only task-relevant completed progress, '
    'without low-level visual details or speculation about future subtasks.'
)

_ROBOCEREBRA_TASK_INSTRUCTION_ARGUMENT_REPLACEMENTS = (
    (re.compile(r"(?i)(?<!brown box )\bchocolate\b"), "brown box chocolate"),
    (re.compile(r"(?i)(?<!red box )\bbutter\b"), "red box butter"),
    (re.compile(r"(?i)(?<!yellow box )\bcookies\b"), "yellow box cookies"),
)


def rank0_print(*args) -> None:
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
    except Exception:
        pass
    print(*args)


def is_ceph_path(path: Any) -> bool:
    return isinstance(path, str) and CEPH_PATH_PATTERN.match(path) is not None


def join_path(base_path: str, rel_path: Any) -> Any:
    if not isinstance(rel_path, str):
        return rel_path
    if is_ceph_path(rel_path) or os.path.isabs(rel_path):
        return rel_path
    if is_ceph_path(base_path):
        rel_path = rel_path[2:] if rel_path.startswith("./") else rel_path
        return f"{base_path.rstrip('/')}/{rel_path.lstrip('/')}"
    return str((Path(base_path) / rel_path).resolve())


def read_json(path: str) -> Any:
    return json.loads(fileio.get_text(path))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in fileio.get_text(path).splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows



def format_relative_observation_time_label(offset_sec: float) -> str:
    offset_sec = float(offset_sec)
    if abs(offset_sec) < 1e-9:
        return "t"
    if offset_sec < 0:
        return f"t-{abs(offset_sec):.1f}s"
    return f"t+{abs(offset_sec):.1f}s"


def format_absolute_observation_time_label(timestamp_sec: float) -> str:
    return f"{float(timestamp_sec):.1f}s"


def resize_frame_with_pad(frame: Image.Image, target_height: int, target_width: int) -> Image.Image:
    width, height = frame.size
    scale = min(float(target_width) / max(width, 1), float(target_height) / max(height, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = frame.resize((new_width, new_height), Image.BICUBIC)
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    canvas.paste(resized, (left, top))
    return canvas
