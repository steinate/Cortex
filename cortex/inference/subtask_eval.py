import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from cortex.inference.eval_subtask_dataset import EvalSubtaskDataset
from cortex.inference.eval_utils import (
    ROBOT_ATOMIC_SKILLS,
    SYSTEM_MESSAGE,
    rank0_print,
)
logger = logging.getLogger(__name__)


_ACTIVE_MEMORY_NOISE_PATTERN = re.compile(
    r"(?i)(?:"
    r"\bcurrent\s+observation\s+images?(?:\s*\([^)]*\))?\s*[:：]?\s*"
    r"|"
    r"\bimage\s*\d+\s*\([^)]*\)\s*[:：]?\s*"
    r")"
)

@dataclass
class SubtaskEvalSample:
    sample_kind: str  # subtask | transition | final_tail
    dataset_id: int
    episode_pos: int
    task_id: str
    episode_index: int
    frame_id: int
    target_subtask_id: int
    input_memory_subtask_id: int
    task_text: str
    detailed_task_text: str
    detailed_task_source: str
    input_language_memory: str
    gt_current_subtask: str
    gt_active_language_memory: str
    gt_current_skill: str
    video_keys: List[str]

@staticmethod
def _format_subtask_list_item(subtask_id, action_text):
    return f"{subtask_id + 1}. {action_text}"

def _build_subtask_list_text(episode):
    action_config = episode.get("action_config", [])
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

def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _exact_char_match(text_a: Any, text_b: Any) -> bool:
    return str(text_a or "").strip() == str(text_b or "").strip()


def _clean_active_language_memory_text(text: Any) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = _ACTIVE_MEMORY_NOISE_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n,;:-")
    return cleaned


def _contains_normalized_term(text: Any, terms: Sequence[str]) -> bool:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return False
    return any(_normalize_text(term) in normalized_text for term in terms if str(term or "").strip())



def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    candidates: List[str] = [text]

    code_block_matches = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    candidates.extend(match.strip() for match in code_block_matches if match and match.strip())

    def _iter_json_object_candidates(raw: str):
        in_string = False
        escaped = False
        depth = 0
        start = -1

        for i, ch in enumerate(raw):
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
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    snippet = raw[start : i + 1].strip()
                    if snippet:
                        yield snippet

    candidates.extend(_iter_json_object_candidates(text))

    seen = set()
    for snippet in candidates:
        if snippet in seen:
            continue
        seen.add(snippet)
        try:
            obj = json.loads(snippet)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj

    return None


def _extract_judge_fields_fallback(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    parsed: Dict[str, Any] = {}

    def _parse_score(key: str) -> Optional[float]:
        m = re.search(rf"{key}\s*[:=]\s*([-+]?[0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    subtask_score = _parse_score("subtask_score")
    memory_score = _parse_score("memory_score")
    total_score = _parse_score("total_score")

    if subtask_score is not None:
        parsed["subtask_score"] = subtask_score
    if memory_score is not None:
        parsed["memory_score"] = memory_score
    if total_score is not None:
        parsed["total_score"] = total_score

    verdict_match = re.search(r"verdict\s*[:=]\s*(correct|partial|wrong)", text, flags=re.IGNORECASE)
    if verdict_match:
        parsed["verdict"] = verdict_match.group(1).lower()

    reason_match = re.search(r"reason\s*[:=]\s*(.+)", text, flags=re.IGNORECASE)
    if reason_match:
        parsed["reason"] = reason_match.group(1).strip().strip('"')

    return parsed if parsed else None





def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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


def _connect_websocket_policy_client(
    host: str,
    port: int,
    api_key: Optional[str] = None,
    ping_interval: Optional[float] = None,
    ping_timeout: Optional[float] = None,
):
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    logger.info(
        "Connecting checkpoint auto-eval policy client to ws://%s:%s",
        host,
        port,
    )
    client = WebsocketClientPolicy(
        host=str(host),
        port=int(port),
        api_key=(str(api_key).strip() or None) if api_key is not None else None,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
    )
    metadata = client.get_server_metadata()
    logger.info("Connected policy server metadata=%s", json.dumps(metadata, ensure_ascii=False, default=str))
    return client


def _messages_to_websocket_payload(messages: Sequence[Dict[str, Any]]) -> tuple[str, List[Any]]:
    system_instruction = ""
    contents: List[Any] = []
    for message in messages:
        role = message.get("role", "")
        for item in message.get("content", []):
            item_type = item.get("type")
            if item_type == "text":
                text = str(item.get("text", ""))
                if role == "system":
                    system_instruction = text if not system_instruction else f"{system_instruction}\n{text}"
                else:
                    contents.append(text)
            elif item_type == "image":
                contents.append(item.get("image"))
    return system_instruction, contents


def _convert_images_to_numpy(images: Sequence[Any]) -> List[np.ndarray]:
    converted: List[np.ndarray] = []
    for image in images:
        if isinstance(image, Image.Image):
            converted.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        else:
            converted.append(np.asarray(image, dtype=np.uint8))
    return converted


def _is_content_policy_violation_error(error: Any) -> bool:
    message = ""
    if isinstance(error, dict):
        message = str(error.get("message", "") or "")
    else:
        message = str(error or "")
    lowered = message.lower()
    return "content_policy_violation" in lowered or "content safety system" in lowered


def _run_policy_generation_batch_via_websocket(
    client,
    samples: Sequence[SubtaskEvalSample],
    images_batch: Sequence[Sequence[Any]],
    use_detailed_instruction: bool = False,
    batch_ordered_subtask_plan: Optional[Sequence[Optional[Sequence[str]]]] = None,
    session_prefix: str = "ckpt-auto-eval",
) -> List[Dict[str, Any]]:
    if len(samples) != len(images_batch):
        raise ValueError(f"samples/images_batch size mismatch: {len(samples)} vs {len(images_batch)}")
    if not samples:
        return []
    if batch_ordered_subtask_plan is None:
        batch_ordered_subtask_plan = [None for _ in samples]

    pred_results: List[Dict[str, Any]] = []
    for offset, (sample, images, ordered_subtask_plan) in enumerate(
        zip(samples, images_batch, batch_ordered_subtask_plan)
    ):
        messages = _make_policy_messages(
            system_message=SYSTEM_MESSAGE,
            sample=sample,
            images=images,
            ordered_subtask_plan=ordered_subtask_plan,
        )
        _, _ = _messages_to_websocket_payload(messages[0])
        session_id = f"{session_prefix}-episode-{int(sample.episode_index):06d}"
        request_id = f"{session_id}-frame-{int(sample.frame_id):06d}-offset-{offset:03d}"
        payload: Dict[str, Any] = {
            "request_id": request_id,
            "session_id": session_id,
            "task_text": sample.task_text,
            "detailed_task": sample.detailed_task_text,
            "initial_memory": sample.input_language_memory,
            "reset_memory": True,
            "batch_images": [_convert_images_to_numpy(images)],
            "video_keys": list(sample.video_keys),
        }
        if ordered_subtask_plan:
            payload["ordered_subtask_plan"] = list(ordered_subtask_plan)

        response = client.infer(payload)
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected response type from policy server: {type(response)}")
        if not bool(response.get("ok", True)):
            error = response.get("error")
            if _is_content_policy_violation_error(error):
                pred_results.append(
                    {
                        "raw_prediction": "",
                        "prediction_json": {},
                        "skip_reason": "content_policy_violation",
                        "error": error,
                    }
                )
                continue
            raise RuntimeError(f"Policy server inference failed: {error}")

        data = response.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected response payload from policy server: {type(data)}")

        raw_prediction = str(data.get("raw_prediction", "") or "")
        if not raw_prediction:
            raw_prediction = json.dumps(
                {
                    "current_subtask": str(data.get("current_subtask", "") or ""),
                    "active_language_memory": str(
                        data.get("active_language_memory", data.get("output_memory", "")) or ""
                    ),
                },
                ensure_ascii=False,
            )
        pred_results.append(
            {
                "raw_prediction": raw_prediction,
                "prediction_json": _extract_json_block(raw_prediction) or {},
                "skip_reason": "",
                "error": None,
            }
        )

    return pred_results


def _resolve_policy_model_cls(base_model_path: str):
    lower = base_model_path.lower()
    base_name = Path(base_model_path.rstrip("/")).name.lower()

    if "qwen3" in lower and "a" in base_name:
        return Qwen3VLMoeForConditionalGeneration
    if "qwen3" in lower:
        return Qwen3VLForConditionalGeneration
    if "qwen2.5" in lower:
        return Qwen2_5_VLForConditionalGeneration
    return Qwen2VLForConditionalGeneration


def _rand_frame(rng: random.Random, start: int, end: int) -> Optional[int]:
    start = int(start)
    end = int(end)
    if end <= start:
        return None
    return rng.randrange(start, end)


def _get_subtask_action_text(action_cfg: Sequence[Dict[str, Any]], subtask_id: int) -> str:
    if subtask_id < 0 or subtask_id >= len(action_cfg):
        return "task_completed",None

    subtask = action_cfg[subtask_id]
    action_text = str(subtask.get("action_text", "")).strip()
    raw_skill = subtask.get("skill", None)
    skill = str(raw_skill).strip() if raw_skill is not None else None
    if action_text:
        return action_text,skill
    return "task_completed",None


def _format_observation_view_label(image_idx: int, num_images: int, video_keys: Sequence[str]) -> str:
    video_key = video_keys[image_idx] if image_idx < len(video_keys) else ""
    normalized_key = str(video_key).strip().lower()

    if "head" in normalized_key:
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


def _build_debug_overlay_text(
    sample: SubtaskEvalSample,
    pred_text: str,
    pred_obj: Optional[Dict[str, Any]] = None,
    judge_obj: Optional[Dict[str, Any]] = None,
) -> str:
    pred_obj = pred_obj if isinstance(pred_obj, dict) else {}
    judge_obj = judge_obj if isinstance(judge_obj, dict) else {}
    pred_subtask = _extract_current_subtask_text(pred_text, pred_obj)
    pred_memory = _clean_active_language_memory_text(pred_obj.get("active_language_memory", ""))
    subtask_score = judge_obj.get("subtask_score", "")
    memory_score = judge_obj.get("memory_score", "")
    total_score = judge_obj.get("total_score", "")
    return "\n".join(
        [
            f"Pred subtask: {pred_subtask or '<empty>'}",
            f"GT subtask: {sample.gt_current_subtask or '<empty>'}",
            f"Pred memory: {pred_memory or '<empty>'}",
            f"GT memory: {sample.gt_active_language_memory or '<empty>'}",
            f"Score(sub/mem/total): {subtask_score}/{memory_score}/{total_score}",
        ]
    )


def _render_prediction_overlay_on_image(image: Any, overlay_text: str):
    if image is None:
        return image

    try:
        rendered = image.copy() if hasattr(image, "copy") else image
        if getattr(rendered, "mode", None) != "RGB":
            rendered = rendered.convert("RGB")

        draw = ImageDraw.Draw(rendered)
        image_width = int(getattr(rendered, "width", 0) or 0)
        font_size = max(15, min(24, image_width // 40 if image_width > 0 else 15))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        except Exception:
            font = ImageFont.load_default()
        padding = max(8, font_size // 3)
        line_spacing = max(2, font_size // 6)
        max_width = max(40, image_width - 2 * padding)
        text_value = str(overlay_text or "") or "<empty>"

        wrapped_lines = []
        for raw_line in text_value.splitlines() or [text_value]:
            line = raw_line.strip() or " "
            current = ""
            for ch in line:
                trial = current + ch
                try:
                    bbox = draw.textbbox((0, 0), trial, font=font)
                    trial_width = bbox[2] - bbox[0]
                except Exception:
                    trial_width = len(trial) * max(8, font_size // 2)
                if current and trial_width > max_width:
                    wrapped_lines.append(current)
                    current = ch
                else:
                    current = trial
            wrapped_lines.append(current or " ")

        overlay_text = "\n".join(wrapped_lines)

        try:
            text_bbox = draw.multiline_textbbox((padding, padding), overlay_text, font=font, spacing=line_spacing)
            box_w = text_bbox[2] - text_bbox[0]
            box_h = text_bbox[3] - text_bbox[1]
        except Exception:
            lines = overlay_text.splitlines()
            box_w = min(max_width, max((len(line) for line in lines), default=0) * max(8, font_size // 2))
            box_h = max(1, len(lines)) * (font_size + line_spacing)

        box_pad = max(4, font_size // 4)
        draw.rectangle(
            [padding - box_pad, padding - box_pad, padding + box_w + box_pad, padding + box_h + box_pad],
            fill=(0, 0, 0),
        )
        draw.multiline_text(
            (padding, padding),
            overlay_text,
            fill=(255, 255, 255),
            font=font,
            spacing=line_spacing,
        )
        return rendered
    except Exception:
        return image


def _build_observation_prompt_lines(num_images: int, video_keys: Sequence[str]) -> List[str]:
    if num_images <= 0:
        raise ValueError("num_images must be positive when building observation prompts.")

    if num_images == 1:
        view_label = _format_observation_view_label(0, num_images, video_keys)
        return [f"Current Observation Image ({view_label}):"]

    prompt_lines = ["Current Observation Images (in order):"]
    for image_idx in range(num_images):
        view_label = _format_observation_view_label(image_idx, num_images, video_keys)
        prompt_lines.append(f"Image {image_idx + 1} ({view_label}):")
    return prompt_lines


def _normalize_max_samples_mode(max_samples_mode: str) -> str:
    mode = str(max_samples_mode or "global").strip().lower()
    if mode in {"global", "merged", "combined", "overall"}:
        return "global"
    if mode in {"per_dataset_equal", "per_dataset", "dataset_equal", "split_by_dataset"}:
        return "per_dataset_equal"
    raise ValueError(
        f"Unsupported max_samples_mode={max_samples_mode!r}. "
        "Expected one of: global, per_dataset_equal."
    )


def _apply_max_samples_policy(
    *,
    samples: List[SubtaskEvalSample],
    eval_dataset: EvalSubtaskDataset,
    rng: random.Random,
    max_samples: int,
    max_samples_mode: str,
) -> tuple[List[SubtaskEvalSample], str, Optional[int]]:
    normalized_mode = _normalize_max_samples_mode(max_samples_mode)
    if max_samples <= 0:
        return samples, normalized_mode, None

    if normalized_mode == "global":
        if len(samples) > max_samples:
            samples = rng.sample(samples, max_samples)
        return samples, normalized_mode, None

    dataset_count = max(1, len(eval_dataset.dataset_blocks))
    per_dataset_quota = max_samples // dataset_count
    if per_dataset_quota <= 0:
        logger.warning(
            "[checkpoint-auto-eval] max_samples_mode=per_dataset_equal but max_samples(%d) < num_datasets(%d); "
            "fallback to global sampling.",
            max_samples,
            dataset_count,
        )
        if len(samples) > max_samples:
            samples = rng.sample(samples, max_samples)
        return samples, "global", None

    grouped_samples: List[List[SubtaskEvalSample]] = [[] for _ in range(dataset_count)]
    for sample in samples:
        dataset_id = int(getattr(sample, "dataset_id", -1))
        if 0 <= dataset_id < dataset_count:
            grouped_samples[dataset_id].append(sample)

    balanced_samples: List[SubtaskEvalSample] = []
    for dataset_samples in grouped_samples:
        if len(dataset_samples) > per_dataset_quota:
            balanced_samples.extend(rng.sample(dataset_samples, per_dataset_quota))
        else:
            balanced_samples.extend(dataset_samples)

    rng.shuffle(balanced_samples)
    return balanced_samples, normalized_mode, per_dataset_quota


def build_random_subtask_samples(
    eval_dataset: EvalSubtaskDataset,
    seed: int,
    max_videos: int = 0,
    max_samples: int = 0,
    max_samples_mode: str = "global",
) -> List[SubtaskEvalSample]:
    rng = random.Random(seed)
    samples: List[SubtaskEvalSample] = []

    video_count = 0
    for dataset_id, block in enumerate(eval_dataset.dataset_blocks):
        episodes = block.get("episodes", [])
        video_keys = list(block.get("video_keys", ["observation.images.head"]))
        for episode_pos, episode in enumerate(episodes):
            if max_videos > 0 and video_count >= max_videos:
                break

            action_config = episode.get("action_config", [])
            if not action_config:
                continue

            task_id = str(episode.get("task_id", ""))
            fps = eval_dataset._load_dataset_fps(os.path.join(block["data_path"], task_id), episode)
            tail_frames = max(1, math.ceil(float(block.get("transition_tail_sec", 0.1)) * fps))
            head_frames = max(1, math.ceil(float(block.get("transition_head_sec", 0.2)) * fps))
            last_tail_frames = max(1, math.ceil(float(block.get("last_tail_sec", 1.0)) * fps))

            task_text = eval_dataset._get_task_text(episode)
            detailed_task_text = ""
            detailed_task_source = ""
            if hasattr(eval_dataset, "_get_detailed_task_text_and_source"):
                detailed_task_text, detailed_task_source = eval_dataset._get_detailed_task_text_and_source(episode)
            elif hasattr(eval_dataset, "_get_detailed_task_text"):
                detailed_task_text = eval_dataset._get_detailed_task_text(episode)
            episode_index = int(episode.get("episode_index", -1))

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
                    transition_frame = _rand_frame(rng, transition_start, transition_end)
                    if transition_frame is not None:
                        input_mem = eval_dataset._get_input_language_memory(action_config, subtask_id - 1)
                        active_mem = eval_dataset._get_active_language_memory(action_config, subtask_id, task_text)
                        gt_current_subtask,gt_current_skill=_get_subtask_action_text(action_config, subtask_id)
                        if input_mem and active_mem:
                            samples.append(
                                SubtaskEvalSample(
                                    sample_kind="transition",
                                    dataset_id=dataset_id,
                                    episode_pos=episode_pos,
                                    task_id=task_id,
                                    episode_index=episode_index,
                                    frame_id=transition_frame,
                                    target_subtask_id=subtask_id,
                                    input_memory_subtask_id=subtask_id - 1,
                                    task_text=task_text,
                                    detailed_task_text=detailed_task_text,
                                    detailed_task_source=detailed_task_source,
                                    input_language_memory=input_mem,
                                    gt_current_subtask=gt_current_subtask,
                                    gt_current_skill=gt_current_skill,
                                    gt_active_language_memory=active_mem,
                                    video_keys=video_keys,
                                )
                            )
                    stable_start = transition_end
                else:
                    stable_start = start

                if not is_last:
                    stable_end = max(stable_start, end - tail_frames)
                else:
                    stable_end = max(stable_start, end - last_tail_frames)

                stable_frame = _rand_frame(rng, stable_start, stable_end)
                if stable_frame is not None:
                    input_mem = eval_dataset._get_input_language_memory(action_config, subtask_id)
                    active_mem = eval_dataset._get_active_language_memory(action_config, subtask_id, task_text)
                    gt_current_subtask,gt_current_skill=_get_subtask_action_text(action_config, subtask_id)
                    if input_mem and active_mem:
                        samples.append(
                            SubtaskEvalSample(
                                sample_kind="subtask",
                                dataset_id=dataset_id,
                                episode_pos=episode_pos,
                                task_id=task_id,
                                episode_index=episode_index,
                                frame_id=stable_frame,
                                target_subtask_id=subtask_id,
                                input_memory_subtask_id=subtask_id,
                                task_text=task_text,
                                detailed_task_text=detailed_task_text,
                                detailed_task_source=detailed_task_source,
                                input_language_memory=input_mem,
                                gt_current_subtask=gt_current_subtask,
                                gt_current_skill=gt_current_skill,
                                gt_active_language_memory=active_mem,
                                video_keys=video_keys,
                            )
                        )

                if is_last:
                    tail_start = stable_end
                    tail_frame = _rand_frame(rng, tail_start, end)
                    if tail_frame is None and end > start:
                        tail_frame = end - 1
                    if tail_frame is not None:
                        input_mem = eval_dataset._get_input_language_memory(action_config, subtask_id)
                        active_mem = eval_dataset._get_active_language_memory(action_config, subtask_id + 1, task_text)
                        gt_current_subtask,gt_current_skill=_get_subtask_action_text(action_config, subtask_id + 1)
                        if input_mem and active_mem:
                            samples.append(
                                SubtaskEvalSample(
                                    sample_kind="final_tail",
                                    dataset_id=dataset_id,
                                    episode_pos=episode_pos,
                                    task_id=task_id,
                                    episode_index=episode_index,
                                    frame_id=tail_frame,
                                    target_subtask_id=subtask_id + 1,
                                    input_memory_subtask_id=subtask_id,
                                    task_text=task_text,
                                    detailed_task_text=detailed_task_text,
                                    detailed_task_source=detailed_task_source,
                                    input_language_memory=input_mem,
                                    gt_current_subtask=gt_current_subtask,
                                    gt_current_skill=gt_current_skill,
                                    gt_active_language_memory=active_mem,
                                    video_keys=video_keys,
                                )
                            )

            video_count += 1

        if max_videos > 0 and video_count >= max_videos:
            break

    samples, applied_mode, per_dataset_quota = _apply_max_samples_policy(
        samples=samples,
        eval_dataset=eval_dataset,
        rng=rng,
        max_samples=max_samples,
        max_samples_mode=max_samples_mode,
    )

    logger.info(
        "[checkpoint-auto-eval] sampled %d cases from %d videos (seed=%d, max_videos=%d, max_samples=%d, "
        "max_samples_mode=%s, per_dataset_quota=%s)",
        len(samples),
        video_count,
        seed,
        max_videos,
        max_samples,
        applied_mode,
        per_dataset_quota,
    )
    for sample_idx, sample in enumerate(samples):
        logger.info(
            "[checkpoint-auto-eval][sample %d] task_id=%s episode_index=%s frame_id=%s sample_kind=%s "
            "detailed_source=%s detailed_present=%s detailed_text=%r",
            sample_idx,
            sample.task_id,
            sample.episode_index,
            sample.frame_id,
            sample.sample_kind,
            sample.detailed_task_source or "<missing>",
            bool(sample.detailed_task_text),
            sample.detailed_task_text,
        )
    return samples


def _make_policy_messages(
    system_message: str,
    sample: SubtaskEvalSample,
    images: Sequence[Any],
    use_detailed_instruction: bool = False,
    ordered_subtask_plan: Optional[Sequence[str]] = None,
) -> List[List[Dict[str, Any]]]:
    content: List[Dict[str, Any]] = []

    header_lines = []
    if sample.task_text:
        header_lines.append(f"Global Task Goal: {sample.task_text}")
    if sample.input_language_memory:
        header_lines.append(f"Input Language Memory: {sample.input_language_memory}")
    if ordered_subtask_plan and not use_detailed_instruction:
        header_lines.append(ordered_subtask_plan)
    if use_detailed_instruction and sample.detailed_task_text and sample.detailed_task_text != sample.task_text and not ordered_subtask_plan:
        header_lines.append(f"Detailed Global Task Instruction: {sample.detailed_task_text}")

    header_lines.append(
        "Candidate Atomic Skills: [" + ", ".join(ROBOT_ATOMIC_SKILLS) + "]"
    )
    prompt_lines = _build_observation_prompt_lines(len(images), sample.video_keys)
    if header_lines:
        prompt_lines[0] = "\n".join(header_lines + [prompt_lines[0]])

    image_slot_idx = 0
    content.append({"type": "text", "text": prompt_lines[0]})
    if len(prompt_lines) == 1 and len(images) == 1:
        content.append({"type": "image", "image": images[0]})
        image_slot_idx = 1
    else:
        for line in prompt_lines[1:]:
            content.append({"type": "text", "text": line})
            stripped_line = str(line).strip()
            is_image_slot = stripped_line.startswith("Image ")
            if is_image_slot:
                if image_slot_idx >= len(images):
                    raise ValueError(
                        f"Observation prompt expects more images than provided: {image_slot_idx + 1} > {len(images)}"
                    )
                content.append({"type": "image", "image": images[image_slot_idx]})
                image_slot_idx += 1

    if image_slot_idx != len(images):
        raise ValueError(
            f"Unused images remain after prompt formatting: used {image_slot_idx}, provided {len(images)}"
        )
    content.append(
        {
        "type": "text",
        "text": (
            'Choose exactly one atomic skill from the candidate list when the task is in progress; '
            'if the task is already completed, set current_skill to null and current_subtask to task_completed. '
            'Predict the subtask and active language memory that should be active now. '
            'Return JSON only with keys "current_skill", "current_subtask", and "active_language_memory".'
        ),
    }
    )

    return [
        [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": content},
        ]
    ]


def _build_policy_inputs(
    processor,
    sample: SubtaskEvalSample,
    images: Sequence[Any],
    use_detailed_instruction: bool = False,
    ordered_subtask_plan: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    messages = _make_policy_messages(
        system_message=SYSTEM_MESSAGE,
        sample=sample,
        images=images,
        use_detailed_instruction=use_detailed_instruction,
        ordered_subtask_plan=ordered_subtask_plan,
    )

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list):
        text = text[0]
    return processor(text=text, images=list(images), padding=True, return_tensors="pt")


def _collate_policy_inputs(sample_inputs: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    if not sample_inputs:
        raise ValueError("sample_inputs must not be empty")

    input_ids = []
    attention_masks = []
    for item in sample_inputs:
        ids = item["input_ids"]
        if ids.dim() == 2:
            ids = ids[0]
        input_ids.append(ids)

        attn = item.get("attention_mask", None)
        if attn is None:
            attn = torch.ones_like(ids)
        elif attn.dim() == 2:
            attn = attn[0]
        attention_masks.append(attn)

    batch_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id,
        padding_side="left",
    )
    batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_masks,
        batch_first=True,
        padding_value=0,
        padding_side="left",
    )

    batch: Dict[str, torch.Tensor] = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
    }

    pixel_values = [item["pixel_values"] for item in sample_inputs if torch.is_tensor(item.get("pixel_values", None))]
    image_grid_thw = [item["image_grid_thw"] for item in sample_inputs if torch.is_tensor(item.get("image_grid_thw", None))]

    if pixel_values:
        batch["pixel_values"] = torch.cat(pixel_values, dim=0)
    if image_grid_thw:
        batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

    return batch


def _run_policy_generation_batch(
    model,
    processor,
    samples: Sequence[SubtaskEvalSample],
    images_batch: Sequence[Sequence[Any]],
    device: str,
    max_new_tokens: int,
    use_detailed_instruction: bool = False,
    batch_ordered_subtask_plan: Optional[Sequence[str]] = None,
) -> List[str]:
    if len(samples) != len(images_batch):
        raise ValueError(f"samples/images_batch size mismatch: {len(samples)} vs {len(images_batch)}")
    if not samples:
        return []
    if batch_ordered_subtask_plan is None:
        batch_ordered_subtask_plan = [None for _ in samples]

    sample_inputs = [
        _build_policy_inputs(
            processor=processor,
            sample=sample,
            images=images,
            use_detailed_instruction=use_detailed_instruction,
            ordered_subtask_plan=ordered_subtask_plan,
        )
        for sample, images, ordered_subtask_plan in zip(samples, images_batch, batch_ordered_subtask_plan)
    ]

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id if processor.tokenizer.eos_token_id is not None else 0

    inputs = _collate_policy_inputs(sample_inputs=sample_inputs, pad_token_id=int(pad_token_id))
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    gen_ids = output_ids[:, input_len:]
    return [processor.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in gen_ids]


def _build_shared_subtask_score_rules_prompt() -> str:
    return (
        "- First normalize wording before scoring:\n"
        "  * Pronouns are equivalent for the acting agent (e.g., I/me/my/you/your) and can be ignored when meaning is unchanged.\n"
        "  * If the subject is the robot, pronouns and explicit robot mentions are equivalent (this can still be exact match), and optional robot descriptors are non-essential.\n"
        "  * Treat hand, arm, and gripper as equivalent references to the manipulator.\n"
        "- Use only these discrete scores: 0, 0.4, 0.9, 1.\n"
        "- subtask_score rubric:\n"
        "  * 1: exact match with reference after applying the normalization above (agent-pronoun equivalence, pronoun/robot-mention equivalence when subject is robot, optional robot-descriptor ignore, hand/arm/gripper equivalence). Exact match MUST be 1.0 (never 0.9).\n"
        "  * 0.9: semantically equivalent overall but not exact after normalization (e.g., synonyms, different word order, minor non-critical attribute omission, or non-contradictory object-attribute addition/omission such as on the left vs. no mentioned, red vs. dark red, or other coarse/fine attribute granularity differences). These are allowed only when they do not introduce any factual error or contradiction.\n"
        "  * 0.4: describes essentially the same high-level thing but is incomplete or substantially rephrased without explicit contradiction. For multi-part subtasks, predicting only the earlier/front part while missing later parts should be 0.4.\n"
        "  * 0: any logical or factual error exists (wrong action/object/state/relation/attribute, missing required action/object, contradiction, or unsupported extra claim). For multi-part subtasks, predicting only a later part while missing the earlier/front part is a logical error => 0.\n"
    )



def _build_judge_prompt(sample: SubtaskEvalSample, pred_text: str) -> str:
    question = (
        f"Task Goal: {sample.task_text}\n"
        f"Input Language Memory: {sample.input_language_memory}"
    )
    ref_answer = json.dumps(
        {
            "current_subtask": sample.gt_current_subtask,
            "active_language_memory": sample.gt_active_language_memory,
        },
        ensure_ascii=False,
    )

    return (
        "You are a strict evaluator.\n"
        "Do NOT output any thinking process, analysis, explanation, markdown, or extra text.\n"
        "Output exactly one JSON object only.\n"
        "Allowed JSON keys: subtask_score, memory_score, total_score, verdict, reason.\n"
        "Scoring rules:\n"
        f"{_build_shared_subtask_score_rules_prompt()}"
        "- memory_score rubric:\n"
        "  * 1 and 0.9: same criteria as subtask_score.\n"
        "  * 0.4: memory is broadly related but substantially rephrased/coarse, with no contradiction and no omitted required subtask.\n"
        "  * 0: any logical/factual error, contradiction, unsupported extra claim, or omission of any required subtask. Omission => 0 (not 0.4).\n"
        "- total_score = subtask_score + memory_score.\n"
        "- verdict: correct if total_score==2, partial if 0<total_score<2, wrong if total_score==0.\n"
        "Keep reason concise (<= 40 words).\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "[Reference Answer]\n"
        f"{ref_answer}\n\n"
        "[Assistant Answer]\n"
        f"{pred_text}\n\n"
        "Now output JSON only."
    )


def _run_judge(
    judge_model,
    judge_tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> Tuple[Dict[str, Any], str]:
    system_prompt = (
        "You are a strict JSON-only judge. "
        "Never output thinking process, analysis, or markdown. "
        "Return exactly one JSON object and nothing else."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if getattr(judge_tokenizer, "chat_template", None):
        try:
            text = judge_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"

    judge_inputs = judge_tokenizer(text, return_tensors="pt")
    for key, value in judge_inputs.items():
        if torch.is_tensor(value):
            judge_inputs[key] = value.to(device)

    input_len = judge_inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out_ids = judge_model.generate(
            **judge_inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    raw = judge_tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()
    parsed = _extract_json_block(raw)
    if parsed is None:
        parsed = _extract_judge_fields_fallback(raw)

    if parsed is None:
        parsed = {
            "subtask_score": 0,
            "memory_score": 0,
            "total_score": 0,
            "verdict": "wrong",
            "reason": "judge_output_not_json",
        }

    if "subtask_score" not in parsed and "memory_score" not in parsed and "rating" in parsed:
        parsed["subtask_score"] = parsed.get("rating", 0)
        parsed["memory_score"] = parsed.get("rating", 0)

    try:
        subtask_score = float(parsed.get("subtask_score", 0))
    except Exception:
        subtask_score = 0.0
    try:
        memory_score = float(parsed.get("memory_score", 0))
    except Exception:
        memory_score = 0.0

    if 0.0 <= subtask_score <= 1.0 and 0.0 <= memory_score <= 1.0:
        subtask_score *= 5.0
        memory_score *= 5.0

    parsed["subtask_score"] = subtask_score
    parsed["memory_score"] = memory_score

    try:
        total_score = float(parsed.get("total_score", subtask_score + memory_score))
    except Exception:
        total_score = subtask_score + memory_score

    if 0.0 <= total_score <= 2.0:
        total_score *= 5.0
    if total_score <= 0:
        total_score = subtask_score + memory_score
    parsed["total_score"] = total_score

    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in {"correct", "partial", "wrong"}:
        if total_score >= 9.0:
            verdict = "correct"
        elif total_score >= 5.0:
            verdict = "partial"
        else:
            verdict = "wrong"
    parsed["verdict"] = verdict
    parsed["reason"] = str(parsed.get("reason", ""))
    return parsed, raw



def _extract_current_subtask_text(pred_text: str, pred_obj: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(pred_obj, dict):
        value = pred_obj.get("current_subtask", "")
        if value is not None and str(value).strip():
            return str(value).strip()

    parsed = _extract_json_block(pred_text)
    if isinstance(parsed, dict):
        value = parsed.get("current_subtask", "")
        if value is not None and str(value).strip():
            return str(value).strip()

    match = re.search(
        r"[\"']current_subtask[\"']\s*:\s*[\"']([^\"']+)[\"']",
        str(pred_text or ""),
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return ""



def _snap_subtask_score_4level(score: Any) -> float:
    allowed = [0.0, 0.4, 0.9, 1.0]
    try:
        value = float(score)
    except Exception:
        value = 0.0

    if 1.0 < value <= 5.0:
        value = value / 5.0
    elif 5.0 < value <= 10.0:
        value = value / 10.0

    return min(allowed, key=lambda x: abs(x - value))



def _build_subtask_compare_prompt(reference_subtask: str, candidate_subtask: str) -> str:
    return (
        "You are a strict evaluator for subtask matching.\n"
        "Do NOT output any thinking process, analysis, explanation, markdown, or extra text.\n"
        "Output exactly one JSON object only.\n"
        "Allowed JSON keys: subtask_score, reason.\n"
        "Scoring rules:\n"
        f"{_build_shared_subtask_score_rules_prompt()}"
        "Keep reason concise (<= 40 words).\n\n"
        "[Reference Subtask]\n"
        f"{reference_subtask}\n\n"
        "[Assistant Subtask]\n"
        f"{candidate_subtask}\n\n"
        "Now output JSON only."
    )


def _score_subtask_with_existing_judge(
    judge_model,
    judge_tokenizer,
    judge_device: str,
    judge_max_new_tokens: int,
    reference_subtask: str,
    candidate_subtask: str,
) -> float:
    if not _normalize_text(reference_subtask) or not _normalize_text(candidate_subtask):
        return 0.0

    prompt = _build_subtask_compare_prompt(reference_subtask=reference_subtask, candidate_subtask=candidate_subtask)
    judge_obj, _ = _run_judge(
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        prompt=prompt,
        device=judge_device,
        max_new_tokens=judge_max_new_tokens,
    )
    return _snap_subtask_score_4level(judge_obj.get("subtask_score", 0.0))


def _get_neighbor_subtask_text(action_cfg: Sequence[Dict[str, Any]], subtask_id: int) -> str:
    if subtask_id < 0:
        return ""
    if subtask_id == len(action_cfg):
        return "task_completed"
    if subtask_id > len(action_cfg):
        return ""
    gt_current_subtask,gt_current_skill = _get_subtask_action_text(action_cfg, subtask_id)
    return gt_current_subtask



def _load_policy_model(
    checkpoint_dir: str,
    base_model_path: str,
    device: str,
    attn_implementation: str,
):
    model_cls = _resolve_policy_model_cls(base_model_path)
    kwargs = {
        "attn_implementation": attn_implementation if device.startswith("cuda") else "sdpa",
        "low_cpu_mem_usage": True,
    }
    if device.startswith("cuda"):
        kwargs["dtype"] = torch.bfloat16

    model = model_cls.from_pretrained(checkpoint_dir, **kwargs)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model_path)
    return model, processor


def _load_judge_model(judge_model_path: str, device: str):
    torch_dtype = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        judge_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path, use_fast=False, trust_remote_code=True)
    return model, tokenizer


def run_checkpoint_auto_eval(
    *,
    checkpoint_dir: str,
    base_model_path: str,
    eval_dataset: EvalSubtaskDataset,
    judge_model_path: str,
    output_dir: str,
    sampling_seed: int,
    max_videos: int,
    max_samples: int,
    max_samples_mode: str,
    policy_device: str,
    judge_device: str,
    policy_max_new_tokens: int,
    judge_max_new_tokens: int,
    attn_implementation: str = "flash_attention_2",
    policy_batch_size: int = 1,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    enable_neighbor_subtask_compare: bool = False,
    neighbor_low_score_threshold: float = 0.5,
    use_detailed_instruction: bool = False,
    use_subtask_list: bool = False,
    policy_backend: str = "local",
    policy_host: str = "127.0.0.1",
    policy_port: int = 10094,
    policy_api_key: str = "",
    policy_ping_interval: Optional[float] = None,
    policy_ping_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    policy_backend = _normalize_policy_backend(policy_backend)
    policy_device = _resolve_device(policy_device)
    judge_device = _resolve_device(judge_device)

    dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
    if dist_ready:
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
    else:
        world_size = max(1, int(distributed_world_size))
        rank = int(distributed_rank)
        rank = min(max(rank, 0), world_size - 1)

    max_samples_mode = _normalize_max_samples_mode(max_samples_mode)

    samples = build_random_subtask_samples(
        eval_dataset=eval_dataset,
        seed=sampling_seed,
        max_videos=max_videos,
        max_samples=max_samples,
        max_samples_mode=max_samples_mode,
    )
    if not samples:
        raise RuntimeError("No evaluation samples were built from eval_dataset.")

    shard_items: List[Tuple[int, SubtaskEvalSample]] = [
        (idx, sample) for idx, sample in enumerate(samples) if idx % world_size == rank
    ]
    logger.info(
        "[checkpoint-auto-eval] rank %d/%d handles %d/%d samples",
        rank,
        world_size,
        len(shard_items),
        len(samples),
    )

    checkpoint_name = Path(checkpoint_dir).name
    result_dir = Path(output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    detail_path = result_dir / f"{checkpoint_name}_details.jsonl"
    summary_path = result_dir / f"{checkpoint_name}_summary.json"
    detail_rank_path = detail_path if world_size == 1 else result_dir / f"{checkpoint_name}_details.rank{rank}.jsonl"
    compare_detail_path = result_dir / f"{checkpoint_name}_details.neighbor_compare.jsonl"
    compare_summary_path = result_dir / f"{checkpoint_name}_details.neighbor_compare.summary.json"

    debug_head_dir = result_dir / "logged_head_images"
    debug_head_dir.mkdir(parents=True, exist_ok=True)

    policy_model = None
    policy_processor = None
    judge_model = None
    judge_tokenizer = None
    rank0_pbar = None
    detail_writer = None
    try:
        details: List[Dict[str, Any]] = []
        total_score = 0.0
        total_subtask = 0.0
        total_memory = 0.0
        subtask_exact = 0
        memory_exact = 0
        skipped_content_policy = 0
        sample_kind_stats: Dict[str, Dict[str, float]] = {}
        completed_indices: set[int] = set()
        cached_detail_needs_repair = False
        if detail_rank_path.exists():
            with detail_rank_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        cached_detail_needs_repair = True
                        logger.warning(
                            "[checkpoint-auto-eval] skip malformed cached detail line %d in %s",
                            line_no,
                            detail_rank_path,
                        )
                        continue
                    if not isinstance(item, dict):
                        cached_detail_needs_repair = True
                        logger.warning(
                            "[checkpoint-auto-eval] skip non-object cached detail line %d in %s",
                            line_no,
                            detail_rank_path,
                        )
                        continue
                    details.append(item)
                    try:
                        completed_indices.add(int(item.get("index", -1)))
                    except Exception:
                        pass

                    judge_obj = item.get("judge", {}) if isinstance(item.get("judge", {}), dict) else {}
                    total_subtask += float(judge_obj.get("subtask_score", 0.0) or 0.0)
                    total_memory += float(judge_obj.get("memory_score", 0.0) or 0.0)
                    total_score += float(judge_obj.get("total_score", 0.0) or 0.0)
                    if str(item.get("skip_reason", "") or "") == "content_policy_violation":
                        skipped_content_policy += 1
                        continue
                    if bool(item.get("char_exact_subtask_match", False)):
                        subtask_exact += 1
                    if bool(item.get("char_exact_memory_match", False)):
                        memory_exact += 1

                    sample_kind = str(item.get("sample_kind", "unknown") or "unknown")
                    kind_stat = sample_kind_stats.setdefault(
                        sample_kind,
                        {
                            "count": 0,
                            "total_subtask_score": 0.0,
                            "total_memory_score": 0.0,
                            "total_score": 0.0,
                            "subtask_exact": 0,
                            "memory_exact": 0,
                        },
                    )
                    kind_stat["count"] += 1
                    kind_stat["total_subtask_score"] += float(judge_obj.get("subtask_score", 0.0) or 0.0)
                    kind_stat["total_memory_score"] += float(judge_obj.get("memory_score", 0.0) or 0.0)
                    kind_stat["total_score"] += float(judge_obj.get("total_score", 0.0) or 0.0)
                    if bool(item.get("char_exact_subtask_match", False)):
                        kind_stat["subtask_exact"] += 1
                    if bool(item.get("char_exact_memory_match", False)):
                        kind_stat["memory_exact"] += 1

        if cached_detail_needs_repair:
            logger.warning(
                "[checkpoint-auto-eval] repairing cached detail jsonl before resume: %s",
                detail_rank_path,
            )
            with detail_rank_path.open("w", encoding="utf-8") as f:
                for item in details:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        if completed_indices:
            logger.info(
                "[checkpoint-auto-eval] rank %d resumes from %d cached samples in %s",
                rank,
                len(completed_indices),
                detail_rank_path,
            )

        shard_items = [(idx, sample) for idx, sample in shard_items if idx not in completed_indices]
        policy_batch_size = max(1, int(policy_batch_size))
        local_total = len(shard_items)
        local_processed = 0

        if local_total > 0:
            if policy_backend == "local":
                policy_model, policy_processor = _load_policy_model(
                    checkpoint_dir=checkpoint_dir,
                    base_model_path=base_model_path,
                    device=policy_device,
                    attn_implementation=attn_implementation,
                )
            else:
                policy_model = _connect_websocket_policy_client(
                    host=policy_host,
                    port=policy_port,
                    api_key=policy_api_key,
                    ping_interval=policy_ping_interval,
                    ping_timeout=policy_ping_timeout,
                )
            judge_model, judge_tokenizer = _load_judge_model(judge_model_path=judge_model_path, device=judge_device)
        else:
            logger.info(
                "[checkpoint-auto-eval] rank %d has no remaining samples after resume filtering; reuse cached shard only",
                rank,
            )

        detail_writer = detail_rank_path.open("a", encoding="utf-8")

        def _append_detail_record(record: Dict[str, Any]) -> None:
            details.append(record)
            detail_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            detail_writer.flush()

        if rank == 1 and local_total > 0:
            try:
                from tqdm.auto import tqdm

                rank0_pbar = tqdm(
                    total=local_total,
                    desc="[checkpoint-auto-eval][rank0]",
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:
                rank0_pbar = None
        logger.info(
            "[checkpoint-auto-eval] use_detailed_instruction=%s use_subtask_list=%s",
            bool(use_detailed_instruction),
            bool(use_subtask_list),
        )

        for batch_start in range(0, len(shard_items), policy_batch_size):
            batch_items = shard_items[batch_start : batch_start + policy_batch_size]
            batch_indices = [item[0] for item in batch_items]
            batch_samples = [item[1] for item in batch_items]
            batch_images: List[List[Any]] = []
            batch_ordered_subtask_plan: List[List[Any]] = []

            for sample in batch_samples:
                block = eval_dataset.dataset_blocks[sample.dataset_id]
                episode = block["episodes"][sample.episode_pos]
                fps = float(episode.get("_dataset_fps", 0.0) or 0.0)
                if not math.isfinite(fps) or fps <= 0:
                    task_id = str(episode.get("task_id", ""))
                    fps = float(eval_dataset._load_dataset_fps(os.path.join(block["data_path"], task_id), episode))
                    episode["_dataset_fps"] = fps
                frame_timestamp_sec = float(sample.frame_id) / fps

                if use_subtask_list:
                    ordered_subtask_plan = _build_subtask_list_text(episode)
                    batch_ordered_subtask_plan.append(ordered_subtask_plan)
                else:
                    batch_ordered_subtask_plan.append(None)
                video_files = eval_dataset._build_video_files(block, episode)
                images: List[Any] = [
                    eval_dataset._decode_frame_image(vf, frame_timestamp_sec)
                    for vf in video_files
                ]
                batch_images.append(images)

            if policy_backend == "local":
                pred_results = [
                    {
                        "raw_prediction": pred_text,
                        "prediction_json": _extract_json_block(pred_text) or {},
                        "skip_reason": "",
                        "error": None,
                    }
                    for pred_text in _run_policy_generation_batch(
                        model=policy_model,
                        processor=policy_processor,
                        samples=batch_samples,
                        images_batch=batch_images,
                        device=policy_device,
                        max_new_tokens=policy_max_new_tokens,
                        use_detailed_instruction=use_detailed_instruction,
                        batch_ordered_subtask_plan=batch_ordered_subtask_plan,
                    )
                ]
            else:
                pred_results = _run_policy_generation_batch_via_websocket(
                    client=policy_model,
                    samples=batch_samples,
                    images_batch=batch_images,
                    use_detailed_instruction=use_detailed_instruction,
                    batch_ordered_subtask_plan=batch_ordered_subtask_plan,
                    session_prefix=f"ckpt-auto-eval-rank{rank}-{checkpoint_name}",
                )

            for offset, (idx, sample, pred_result) in enumerate(zip(batch_indices, batch_samples, pred_results)):
                local_processed += 1
                if rank0_pbar is not None:
                    rank0_pbar.update(1)
                elif rank == 1 and (local_processed % 20 == 0 or local_processed == local_total):
                    logger.info(
                        "[checkpoint-auto-eval] %s rank0 local progress: %d/%d",
                        checkpoint_name,
                        local_processed,
                        local_total,
                    )

                pred_text = str(pred_result.get("raw_prediction", "") or "")
                pred_obj = pred_result.get("prediction_json", {}) if isinstance(pred_result.get("prediction_json", {}), dict) else {}
                skip_reason = str(pred_result.get("skip_reason", "") or "")
                if skip_reason == "content_policy_violation":
                    skipped_content_policy += 1
                    record = {
                        "index": idx,
                        "sample_kind": sample.sample_kind,
                        "dataset_id": sample.dataset_id,
                        "episode_pos": sample.episode_pos,
                        "task_id": sample.task_id,
                        "episode_index": sample.episode_index,
                        "frame_id": sample.frame_id,
                        "target_subtask_id": sample.target_subtask_id,
                        "input_memory_subtask_id": sample.input_memory_subtask_id,
                        "task_text": sample.task_text,
                        "detailed_task_text": sample.detailed_task_text,
                        "detailed_task_source": sample.detailed_task_source,
                        "input_language_memory": sample.input_language_memory,
                        "gt": {
                            "current_subtask": sample.gt_current_subtask,
                            "active_language_memory": sample.gt_active_language_memory,
                        },
                        "prediction_raw": pred_text,
                        "prediction_json": pred_obj,
                        "pred_current_subtask": "",
                        "pred_active_language_memory": "",
                        "judge": None,
                        "judge_raw": "",
                        "char_exact_subtask_match": False,
                        "char_exact_memory_match": False,
                        "char_match_override_applied": False,
                        "exact_match_bucket": "skipped_content_policy",
                        "skip_reason": skip_reason,
                        "error": pred_result.get("error"),
                    }
                    _append_detail_record(record)
                    continue

                judge_prompt = _build_judge_prompt(sample, pred_text)
                judge_obj, judge_raw = _run_judge(
                    judge_model=judge_model,
                    judge_tokenizer=judge_tokenizer,
                    prompt=judge_prompt,
                    device=judge_device,
                    max_new_tokens=judge_max_new_tokens,
                )

                pred_subtask = _extract_current_subtask_text(pred_text, pred_obj)
                pred_memory_raw = pred_obj.get("active_language_memory", "")
                pred_memory = _clean_active_language_memory_text(pred_memory_raw)
                if pred_memory != str(pred_memory_raw or ""):
                    pred_obj["active_language_memory"] = pred_memory
                char_exact_subtask_match = _exact_char_match(pred_subtask, sample.gt_current_subtask)
                char_exact_memory_match = _exact_char_match(pred_memory, sample.gt_active_language_memory)
                char_match_override_applied = char_exact_subtask_match or char_exact_memory_match
                if char_exact_subtask_match:
                    judge_obj["subtask_score"] = 5.0
                if char_exact_memory_match:
                    judge_obj["memory_score"] = 5.0
                if char_match_override_applied:
                    try:
                        final_subtask_score = float(judge_obj.get("subtask_score", 0.0))
                    except Exception:
                        final_subtask_score = 0.0
                    try:
                        final_memory_score = float(judge_obj.get("memory_score", 0.0))
                    except Exception:
                        final_memory_score = 0.0
                    final_total_score = final_subtask_score + final_memory_score
                    judge_obj["total_score"] = final_total_score
                    if final_total_score >= 9.999:
                        judge_obj["verdict"] = "correct"
                    elif final_total_score <= 0.0:
                        judge_obj["verdict"] = "wrong"
                    else:
                        judge_obj["verdict"] = "partial"
                    judge_obj["reason"] = "char_match_override"

                total_subtask += float(judge_obj.get("subtask_score", 0.0))
                total_memory += float(judge_obj.get("memory_score", 0.0))
                total_score += float(judge_obj.get("total_score", 0.0))

                subtask_match = _normalize_text(pred_subtask) == _normalize_text(sample.gt_current_subtask)
                memory_match = _normalize_text(pred_memory) == _normalize_text(sample.gt_active_language_memory)
                if subtask_match:
                    subtask_exact += 1
                if memory_match:
                    memory_exact += 1

                kind_stat = sample_kind_stats.setdefault(
                    sample.sample_kind,
                    {
                        "count": 0,
                        "total_subtask_score": 0.0,
                        "total_memory_score": 0.0,
                        "total_score": 0.0,
                        "subtask_exact": 0,
                        "memory_exact": 0,
                    },
                )
                kind_stat["count"] += 1
                kind_stat["total_subtask_score"] += float(judge_obj.get("subtask_score", 0.0))
                kind_stat["total_memory_score"] += float(judge_obj.get("memory_score", 0.0))
                kind_stat["total_score"] += float(judge_obj.get("total_score", 0.0))
                if subtask_match:
                    kind_stat["subtask_exact"] += 1
                if memory_match:
                    kind_stat["memory_exact"] += 1

                subtask_judge_score = _snap_subtask_score_4level(judge_obj.get("subtask_score", 0.0))
                neighbor_relation = "disabled"
                neighbor_prev_subtask = ""
                neighbor_next_subtask = ""
                neighbor_prev_score = None
                neighbor_next_score = None

                if enable_neighbor_subtask_compare:
                    neighbor_relation = "not_needed"
                    if subtask_judge_score < neighbor_low_score_threshold:
                        neighbor_relation = "unknown"
                        block = eval_dataset.dataset_blocks[sample.dataset_id]
                        episode = block["episodes"][sample.episode_pos]
                        action_config = episode.get("action_config", [])
                        if isinstance(action_config, list) and action_config:
                            neighbor_prev_subtask = _get_neighbor_subtask_text(action_config, sample.target_subtask_id - 1)
                            neighbor_next_subtask = _get_neighbor_subtask_text(action_config, sample.target_subtask_id + 1)

                            if pred_subtask:
                                if neighbor_prev_subtask:
                                    neighbor_prev_score = _score_subtask_with_existing_judge(
                                        judge_model=judge_model,
                                        judge_tokenizer=judge_tokenizer,
                                        judge_device=judge_device,
                                        judge_max_new_tokens=judge_max_new_tokens,
                                        reference_subtask=neighbor_prev_subtask,
                                        candidate_subtask=pred_subtask,
                                    )
                                if neighbor_next_subtask:
                                    neighbor_next_score = _score_subtask_with_existing_judge(
                                        judge_model=judge_model,
                                        judge_tokenizer=judge_tokenizer,
                                        judge_device=judge_device,
                                        judge_max_new_tokens=judge_max_new_tokens,
                                        reference_subtask=neighbor_next_subtask,
                                        candidate_subtask=pred_subtask,
                                    )

                                best_score = max(
                                    neighbor_prev_score if neighbor_prev_score is not None else -1.0,
                                    neighbor_next_score if neighbor_next_score is not None else -1.0,
                                )
                                if best_score < neighbor_low_score_threshold:
                                    neighbor_relation = "neither"
                                elif neighbor_prev_score is not None and neighbor_next_score is not None:
                                    if abs(neighbor_prev_score - neighbor_next_score) < 1e-6:
                                        neighbor_relation = "neither"
                                    else:
                                        neighbor_relation = "previous_subtask" if neighbor_prev_score > neighbor_next_score else "next_subtask"
                                elif neighbor_prev_score is not None:
                                    neighbor_relation = "previous_subtask"
                                elif neighbor_next_score is not None:
                                    neighbor_relation = "next_subtask"
                                else:
                                    neighbor_relation = "unknown"

                record = {
                    "index": idx,
                    "sample_kind": sample.sample_kind,
                    "dataset_id": sample.dataset_id,
                    "episode_pos": sample.episode_pos,
                    "task_id": sample.task_id,
                    "episode_index": sample.episode_index,
                    "frame_id": sample.frame_id,
                    "target_subtask_id": sample.target_subtask_id,
                    "input_memory_subtask_id": sample.input_memory_subtask_id,
                    "task_text": sample.task_text,
                    "detailed_task_text": sample.detailed_task_text,
                    "detailed_task_source": sample.detailed_task_source,
                    "input_language_memory": sample.input_language_memory,
                    "gt": {
                        "current_subtask": sample.gt_current_subtask,
                        "active_language_memory": sample.gt_active_language_memory,
                    },
                    "prediction_raw": pred_text,
                    "prediction_json": pred_obj,
                    "pred_current_subtask": pred_subtask,
                    "pred_active_language_memory": pred_memory,
                    "judge": judge_obj,
                    "judge_raw": judge_raw,
                    "char_exact_subtask_match": char_exact_subtask_match,
                    "char_exact_memory_match": char_exact_memory_match,
                    "char_match_override_applied": char_match_override_applied,
                    "skip_reason": "",
                }
                if enable_neighbor_subtask_compare:
                    record.update(
                        {
                            "pred_current_subtask": pred_subtask,
                            "ref_current_subtask": sample.gt_current_subtask,
                            "subtask_judge_score": subtask_judge_score,
                            "low_score_neighbor_relation": neighbor_relation,
                            "neighbor_previous_subtask": neighbor_prev_subtask,
                            "neighbor_next_subtask": neighbor_next_subtask,
                            "neighbor_prev_judge_score": neighbor_prev_score,
                            "neighbor_next_judge_score": neighbor_next_score,
                        }
                    )
                _append_detail_record(record)

                if rank == 1 and local_processed % 2 == 0:
                    head_image_path = debug_head_dir / f"{idx + 1}.jpg"
                    try:
                        sample_images = batch_images[offset] if offset < len(batch_images) else []
                        current_group_first_idx = 0
                        head_image = sample_images[current_group_first_idx] if current_group_first_idx < len(sample_images) else None
                        if head_image is not None and hasattr(head_image, "save"):
                            overlay_text = _build_debug_overlay_text(sample, pred_text, pred_obj, judge_obj)
                            rendered_head_image = _render_prediction_overlay_on_image(head_image, overlay_text)
                            rendered_head_image.save(head_image_path)
                        else:
                            logger.warning(
                                "[checkpoint-auto-eval] cannot save current head image for sample=%d: idx=%d total_images=%d type=%s",
                                idx + 1,
                                current_group_first_idx,
                                len(sample_images),
                                str(type(head_image)),
                            )
                    except Exception as e:
                        logger.warning(
                            "[checkpoint-auto-eval] failed to save current head image for sample=%d to %s: %s",
                            idx + 1,
                            str(head_image_path),
                            str(e),
                        )

                    print(
                        "[checkpoint-auto-eval] sample=%d/%d\nGT=%s\nPrediction=%s\nJudge=%s\nSavedHeadImage=%s" % (
                            idx + 1,
                            len(samples),
                            json.dumps(record["gt"], ensure_ascii=False),
                            pred_text,
                            json.dumps(judge_obj, ensure_ascii=False),
                            str(head_image_path),
                        )
                    )

        if rank0_pbar is not None:
            rank0_pbar.close()
            rank0_pbar = None

        if detail_writer is not None:
            detail_writer.close()
            detail_writer = None

        details.sort(key=lambda x: int(x.get("index", -1)))
        with detail_rank_path.open("w", encoding="utf-8") as f:
            for item in details:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        local_payload: Dict[str, Any] = {
            "rank": rank,
            "count": len(details),
            "total_subtask": float(total_subtask),
            "total_memory": float(total_memory),
            "total_score": float(total_score),
            "subtask_exact": int(subtask_exact),
            "memory_exact": int(memory_exact),
            "skipped_content_policy": int(skipped_content_policy),
            "sample_kind_stats": sample_kind_stats,
            "detail_path": str(detail_rank_path),
        }

        if dist_ready and world_size > 1:
            all_payloads: List[Dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore[assignment]
            torch.distributed.all_gather_object(all_payloads, local_payload)
        else:
            all_payloads = [local_payload]

        if rank != 0:
            if dist_ready and world_size > 1:
                torch.distributed.barrier()
            return {
                "checkpoint": checkpoint_name,
                "checkpoint_dir": str(checkpoint_dir),
                "rank": rank,
                "world_size": world_size,
                "local_sample_count": len(details),
                "detail_path": str(detail_rank_path),
                "sharded": world_size > 1,
            }

        merged_details: List[Dict[str, Any]] = []
        merged_total_subtask = 0.0
        merged_total_memory = 0.0
        merged_total_score = 0.0
        merged_subtask_exact = 0
        merged_memory_exact = 0
        merged_skipped_content_policy = 0
        merged_kind_stats: Dict[str, Dict[str, float]] = {}

        for payload in all_payloads:
            if not payload:
                continue

            merged_total_subtask += float(payload.get("total_subtask", 0.0))
            merged_total_memory += float(payload.get("total_memory", 0.0))
            merged_total_score += float(payload.get("total_score", 0.0))
            merged_subtask_exact += int(payload.get("subtask_exact", 0))
            merged_memory_exact += int(payload.get("memory_exact", 0))
            merged_skipped_content_policy += int(payload.get("skipped_content_policy", 0))

            payload_kind_stats = payload.get("sample_kind_stats", {}) or {}
            for sample_kind, stats in payload_kind_stats.items():
                acc = merged_kind_stats.setdefault(
                    sample_kind,
                    {
                        "count": 0,
                        "total_subtask_score": 0.0,
                        "total_memory_score": 0.0,
                        "total_score": 0.0,
                        "subtask_exact": 0,
                        "memory_exact": 0,
                    },
                )
                acc["count"] += int(stats.get("count", 0))
                acc["total_subtask_score"] += float(stats.get("total_subtask_score", 0.0))
                acc["total_memory_score"] += float(stats.get("total_memory_score", 0.0))
                acc["total_score"] += float(stats.get("total_score", 0.0))
                acc["subtask_exact"] += int(stats.get("subtask_exact", 0))
                acc["memory_exact"] += int(stats.get("memory_exact", 0))

            payload_detail_path = Path(str(payload.get("detail_path", "")))
            if payload_detail_path.exists():
                with payload_detail_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            merged_details.append(json.loads(line))

        merged_details.sort(key=lambda x: int(x.get("index", -1)))
        with detail_path.open("w", encoding="utf-8") as f:
            for item in merged_details:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        n = len(merged_details)
        if n <= 0:
            raise RuntimeError("No evaluated samples were collected across ranks.")
        evaluated_details = [
            item for item in merged_details
            if str(item.get("skip_reason", "") or "") != "content_policy_violation"
        ]
        evaluated_n = len(evaluated_details)
        if evaluated_n <= 0:
            raise RuntimeError("No non-skipped evaluated samples were collected across ranks.")

        summary_kind_stats: Dict[str, Dict[str, Any]] = {}

        compare_summary_obj: Optional[Dict[str, Any]] = None
        if enable_neighbor_subtask_compare:
            with compare_detail_path.open("w", encoding="utf-8") as f:
                for item in evaluated_details:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            relation_keys = ["previous_subtask", "next_subtask", "neither", "unknown"]
            failed_relation_counts: Dict[str, int] = {k: 0 for k in relation_keys}
            by_sample_kind: Dict[str, Dict[str, Any]] = {}
            analyzed = 0
            failed = 0

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
                        total=len(evaluated_details),
                        desc=f"[checkpoint-auto-eval][{checkpoint_name}] neighbor-compare",
                        dynamic_ncols=True,
                        leave=True,
                    )
                except Exception:
                    compare_pbar = None

                for item in evaluated_details:
                    if compare_pbar is not None:
                        compare_pbar.update(1)

                    if "subtask_judge_score" not in item:
                        continue

                    analyzed += 1
                    sample_kind = str(item.get("sample_kind", "unknown") or "unknown")
                    kind_stat = _ensure_kind_stat(sample_kind)
                    kind_stat["analyzed"] += 1

                    try:
                        score = float(item.get("subtask_judge_score", 0.0))
                    except Exception:
                        score = 0.0

                    if score < neighbor_low_score_threshold:
                        failed += 1
                        kind_stat["failed"] += 1
                        relation = str(item.get("low_score_neighbor_relation", "unknown") or "unknown")
                        if relation not in failed_relation_counts:
                            relation = "unknown"
                        failed_relation_counts[relation] += 1
                        kind_stat["failed_relation_counts"][relation] += 1
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

            compare_summary_obj = {
                "input_file": str(detail_path),
                "output_file": str(compare_detail_path),
                "low_score_threshold": neighbor_low_score_threshold,
                "total_records": evaluated_n,
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

            with compare_summary_path.open("w", encoding="utf-8") as f:
                json.dump(compare_summary_obj, f, ensure_ascii=False, indent=2)

            logger.info("[checkpoint-auto-eval] neighbor compare saved to %s", compare_detail_path)
            logger.info("[checkpoint-auto-eval] neighbor compare summary saved to %s", compare_summary_path)

        bucket_stats: Dict[str, Dict[str, Any]] = {
            "regular": {"count": 0, "subtask_exact": 0, "memory_exact": 0},
        }
        detailed_task_source_stats: Dict[str, int] = {}
        for item in merged_details:
            detailed_source = str(item.get("detailed_task_source", "") or "<missing>")
            detailed_task_source_stats[detailed_source] = detailed_task_source_stats.get(detailed_source, 0) + 1
            if str(item.get("skip_reason", "") or "") == "content_policy_violation":
                continue
            gt_obj = item.get("gt", {}) if isinstance(item.get("gt", {}), dict) else {}
            pred_obj = item.get("prediction_json", {}) if isinstance(item.get("prediction_json", {}), dict) else {}
            pred_raw = str(item.get("prediction_raw", "") or "")
            gt_subtask = str(gt_obj.get("current_subtask", "") or "")
            gt_memory = str(gt_obj.get("active_language_memory", "") or "")
            pred_subtask = str(item.get("pred_current_subtask", "") or "") or _extract_current_subtask_text(pred_raw, pred_obj)
            pred_memory = _clean_active_language_memory_text(
                item.get("pred_active_language_memory", pred_obj.get("active_language_memory", ""))
            )
            bucket_key = "regular"
            bucket_stats[bucket_key]["count"] += 1
            if _normalize_text(pred_subtask) == _normalize_text(gt_subtask):
                bucket_stats[bucket_key]["subtask_exact"] += 1
            if _normalize_text(pred_memory) == _normalize_text(gt_memory):
                bucket_stats[bucket_key]["memory_exact"] += 1

        def _safe_ratio(numerator: int, denominator: int) -> float:
            if denominator <= 0:
                return 0.0
            return float(numerator) / float(denominator)

        regular_count = int(bucket_stats["regular"]["count"])
        regular_subtask_exact = int(bucket_stats["regular"]["subtask_exact"])
        regular_memory_exact = int(bucket_stats["regular"]["memory_exact"])

        filtered_kind_acc: Dict[str, Dict[str, Any]] = {}
        for item in merged_details:
            if str(item.get("exact_match_bucket", "") or "") != "regular":
                continue
            sample_kind = str(item.get("sample_kind", "unknown") or "unknown")
            acc = filtered_kind_acc.setdefault(
                sample_kind,
                {
                    "count": 0,
                    "total_subtask_score": 0.0,
                    "total_memory_score": 0.0,
                    "total_score": 0.0,
                    "subtask_exact": 0,
                    "memory_exact": 0,
                },
            )
            judge_obj = item.get("judge", {}) if isinstance(item.get("judge", {}), dict) else {}
            gt_obj = item.get("gt", {}) if isinstance(item.get("gt", {}), dict) else {}
            pred_obj = item.get("prediction_json", {}) if isinstance(item.get("prediction_json", {}), dict) else {}
            pred_raw = str(item.get("prediction_raw", "") or "")
            gt_subtask = str(gt_obj.get("current_subtask", "") or "")
            gt_memory = str(gt_obj.get("active_language_memory", "") or "")
            pred_subtask = str(item.get("pred_current_subtask", "") or "") or _extract_current_subtask_text(pred_raw, pred_obj)
            pred_memory = _clean_active_language_memory_text(
                item.get("pred_active_language_memory", pred_obj.get("active_language_memory", ""))
            )
            acc["count"] += 1
            acc["total_subtask_score"] += float(judge_obj.get("subtask_score", 0.0))
            acc["total_memory_score"] += float(judge_obj.get("memory_score", 0.0))
            acc["total_score"] += float(judge_obj.get("total_score", 0.0))
            if _normalize_text(pred_subtask) == _normalize_text(gt_subtask):
                acc["subtask_exact"] += 1
            if _normalize_text(pred_memory) == _normalize_text(gt_memory):
                acc["memory_exact"] += 1

        summary_kind_stats = {}
        for sample_kind, stats in sorted(filtered_kind_acc.items()):
            count = int(stats.get("count", 0))
            if count <= 0:
                continue
            summary_kind_stats[sample_kind] = {
                "count": count,
                "ratio": _safe_ratio(count, regular_count),
                "avg_subtask_score": float(stats.get("total_subtask_score", 0.0)) / count,
                "avg_memory_score": float(stats.get("total_memory_score", 0.0)) / count,
                "avg_total_score": float(stats.get("total_score", 0.0)) / count,
                "subtask_exact_match": _safe_ratio(int(stats.get("subtask_exact", 0)), count),
                "memory_exact_match": _safe_ratio(int(stats.get("memory_exact", 0)), count),
            }


        summary = {
            "checkpoint": checkpoint_name,
            "checkpoint_dir": str(checkpoint_dir),
            "sample_count": n,
            "evaluated_sample_count": evaluated_n,
            "skipped_content_policy_count": merged_skipped_content_policy,
            "sampling_seed": sampling_seed,
            "max_videos": max_videos,
            "max_samples": max_samples,
            "max_samples_mode": max_samples_mode,
            "policy_batch_size": policy_batch_size,
            "world_size": world_size,
            "avg_subtask_score": merged_total_subtask / evaluated_n,
            "avg_memory_score": merged_total_memory / evaluated_n,
            "avg_total_score": merged_total_score / evaluated_n,
            "subtask_exact_match": _safe_ratio(regular_subtask_exact, regular_count),
            "memory_exact_match": _safe_ratio(regular_memory_exact, regular_count),
            "exact_match_sample_count": regular_count,
            "excluded_from_main_exact_match_sample_count": evaluated_n - regular_count,
            "all_samples_subtask_exact_match": merged_subtask_exact / evaluated_n,
            "all_samples_memory_exact_match": merged_memory_exact / evaluated_n,
            "sample_kind_stats": summary_kind_stats,
            "detailed_task_source_stats": {
                key: detailed_task_source_stats[key] for key in sorted(detailed_task_source_stats.keys())
            },
            "detail_path": str(detail_path),
        }
        if compare_summary_obj is not None:
            summary["neighbor_compare_path"] = str(compare_detail_path)
            summary["neighbor_compare_summary_path"] = str(compare_summary_path)

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("[checkpoint-auto-eval] summary saved to %s", summary_path)

        if dist_ready and world_size > 1:
            torch.distributed.barrier()
        return summary
    finally:
        if detail_writer is not None:
            detail_writer.close()
        if policy_backend == "websocket" and hasattr(policy_model, "close"):
            try:
                policy_model.close()
            except Exception:
                logger.exception("failed to close websocket policy client")
        if rank0_pbar is not None:
            rank0_pbar.close()
        if policy_model is not None:
            del policy_model
        if judge_model is not None:
            del judge_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



def run_neighbor_compare_from_detail_file(
    *,
    detail_file: str,
    eval_dataset: EvalSubtaskDataset,
    judge_model_path: str,
    judge_device: str,
    judge_max_new_tokens: int,
    low_score_threshold: float = 0.5,
    output_file: Optional[str] = None,
    summary_file: Optional[str] = None,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> Dict[str, Any]:
    detail_path = Path(detail_file)
    if not detail_path.exists():
        raise FileNotFoundError(f"detail file not found: {detail_path}")

    output_path = Path(output_file) if output_file else detail_path.with_name(detail_path.stem + ".neighbor_compare.jsonl")
    summary_path = Path(summary_file) if summary_file else output_path.with_name(output_path.stem + ".summary.json")

    dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
    if dist_ready:
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
    else:
        world_size = max(1, int(distributed_world_size))
        rank = int(distributed_rank)
        rank = min(max(rank, 0), world_size - 1)

    shard_output_path = (
        output_path
        if world_size == 1
        else output_path.parent / f"{output_path.stem}.rank{rank}{output_path.suffix}"
    )

    judge_device = _resolve_device(judge_device)
    judge_model = None
    compare_pbar = None
    try:
        judge_model, judge_tokenizer = _load_judge_model(judge_model_path=judge_model_path, device=judge_device)

        relation_keys = ["previous_subtask", "next_subtask", "neither", "unknown"]
        local_failed_relation_counts: Dict[str, int] = {k: 0 for k in relation_keys}
        local_by_sample_kind: Dict[str, Dict[str, Any]] = {}

        local_total_records = 0
        local_shard_records = 0
        local_analyzed = 0
        local_failed = 0

        def _safe_int(v: Any, default: int = -1) -> int:
            try:
                return int(v)
            except Exception:
                return default

        def _ensure_kind_stat(kind: str) -> Dict[str, Any]:
            key = str(kind or "unknown")
            if key not in local_by_sample_kind:
                local_by_sample_kind[key] = {
                    "analyzed": 0,
                    "failed": 0,
                    "failed_relation_counts": {k: 0 for k in relation_keys},
                }
            return local_by_sample_kind[key]

        def _ratio(a: int, b: int) -> float:
            if b <= 0:
                return 0.0
            return float(a) / float(b)

        fallback_samples_by_idx: Dict[int, SubtaskEvalSample] = {}
        try:
            summary_candidate: Optional[Path] = None
            if detail_path.stem.endswith("_details"):
                prefix = detail_path.stem[: -len("_details")]
                cand = detail_path.with_name(prefix + "_summary.json")
                if cand.exists():
                    summary_candidate = cand

            if summary_candidate is None:
                cand = detail_path.with_name(detail_path.stem + ".summary.json")
                if cand.exists():
                    summary_candidate = cand

            if summary_candidate is not None and summary_candidate.exists():
                with summary_candidate.open("r", encoding="utf-8") as f:
                    summary_meta = json.load(f)
                sampling_seed = _safe_int(summary_meta.get("sampling_seed", 42), 42)
                sampling_max_videos = _safe_int(summary_meta.get("max_videos", 0), 0)
                sampling_max_samples = _safe_int(summary_meta.get("max_samples", 0), 0)
                sampling_max_samples_mode = str(summary_meta.get("max_samples_mode", "global") or "global")
                fallback_samples = build_random_subtask_samples(
                    eval_dataset=eval_dataset,
                    seed=sampling_seed,
                    max_videos=sampling_max_videos,
                    max_samples=sampling_max_samples,
                    max_samples_mode=sampling_max_samples_mode,
                )
                fallback_samples_by_idx = {i: s for i, s in enumerate(fallback_samples)}
                if rank == 1:
                    logger.info(
                        "[detail-compare] loaded sampling meta from %s (seed=%s, max_videos=%s, max_samples=%s, max_samples_mode=%s)",
                        summary_candidate,
                        sampling_seed,
                        sampling_max_videos,
                        sampling_max_samples,
                        sampling_max_samples_mode,
                    )
        except Exception:
            if rank == 1:
                logger.exception("[detail-compare] failed to build fallback sample map from summary metadata")

        if rank == 1 and not fallback_samples_by_idx:
            logger.warning(
                "[detail-compare] no fallback sample map. If detail file lacks dataset_id/episode_pos/target_subtask_id, "
                "please keep companion *_summary.json in the same directory."
            )

        if rank == 1:
            try:
                from tqdm.auto import tqdm

                compare_pbar = tqdm(
                    total=None,
                    desc=f"[detail-compare][rank{rank}]",
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:
                compare_pbar = None

        with detail_path.open("r", encoding="utf-8") as fin, shard_output_path.open("w", encoding="utf-8") as fout:
            for line_idx, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue

                local_total_records += 1
                if line_idx % world_size != rank:
                    continue

                local_shard_records += 1
                if compare_pbar is not None:
                    compare_pbar.update(1)

                record = json.loads(line)
                sample_kind = str(record.get("sample_kind", "unknown") or "unknown")
                dataset_id = _safe_int(record.get("dataset_id", -1), -1)
                episode_pos = _safe_int(record.get("episode_pos", -1), -1)
                target_subtask_id = _safe_int(record.get("target_subtask_id", -1), -1)

                pred_obj = record.get("prediction_json", {})
                if not isinstance(pred_obj, dict):
                    pred_obj = {}
                pred_text = str(record.get("prediction_raw", record.get("prediction", "")) or "")
                pred_subtask = _extract_current_subtask_text(pred_text, pred_obj)

                ref_subtask = ""
                gt = record.get("gt", {})
                if isinstance(gt, dict):
                    ref_subtask = str(gt.get("current_subtask", "") or "")

                record_index = _safe_int(record.get("index", line_idx), line_idx)
                if (dataset_id < 0 or episode_pos < 0 or target_subtask_id < 0) and fallback_samples_by_idx:
                    fallback_sample = fallback_samples_by_idx.get(record_index)
                    if fallback_sample is None and line_idx >= 0:
                        fallback_sample = fallback_samples_by_idx.get(line_idx)
                    if fallback_sample is not None:
                        if dataset_id < 0:
                            dataset_id = int(fallback_sample.dataset_id)
                        if episode_pos < 0:
                            episode_pos = int(fallback_sample.episode_pos)
                        if target_subtask_id < 0:
                            target_subtask_id = int(fallback_sample.target_subtask_id)
                        if sample_kind == "unknown":
                            sample_kind = str(fallback_sample.sample_kind or "unknown")
                        if not ref_subtask:
                            ref_subtask = str(fallback_sample.gt_current_subtask or "")

                prev_subtask = ""
                next_subtask = ""
                if 0 <= dataset_id < len(eval_dataset.dataset_blocks):
                    episodes = eval_dataset.dataset_blocks[dataset_id].get("episodes", [])
                    if 0 <= episode_pos < len(episodes):
                        action_config = episodes[episode_pos].get("action_config", [])
                        if isinstance(action_config, list) and action_config:
                            if not ref_subtask and target_subtask_id >= 0:
                                ref_subtask,ref_skill = _get_subtask_action_text(action_config, target_subtask_id, eval_dataset.enable_instruction_augment)
                            prev_subtask = _get_neighbor_subtask_text(action_config, target_subtask_id - 1, eval_dataset.enable_instruction_augment)
                            next_subtask = _get_neighbor_subtask_text(action_config, target_subtask_id + 1, eval_dataset.enable_instruction_augment)

                if not pred_subtask and not ref_subtask:
                    record["__line_idx"] = line_idx
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                existing_score = record.get("subtask_judge_score", None)
                if existing_score is None:
                    judge_obj = record.get("judge", {})
                    if isinstance(judge_obj, dict):
                        existing_score = judge_obj.get("subtask_score", None)

                if existing_score is not None:
                    subtask_score = _snap_subtask_score_4level(existing_score)
                else:
                    subtask_score = _score_subtask_with_existing_judge(
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        judge_device=judge_device,
                        judge_max_new_tokens=judge_max_new_tokens,
                        reference_subtask=ref_subtask,
                        candidate_subtask=pred_subtask,
                    )

                relation = "not_needed"
                prev_score = None
                next_score = None
                if subtask_score < low_score_threshold:
                    relation = "unknown"
                    if pred_subtask:
                        if prev_subtask:
                            prev_score = _score_subtask_with_existing_judge(
                                judge_model=judge_model,
                                judge_tokenizer=judge_tokenizer,
                                judge_device=judge_device,
                                judge_max_new_tokens=judge_max_new_tokens,
                                reference_subtask=prev_subtask,
                                candidate_subtask=pred_subtask,
                            )
                        if next_subtask:
                            next_score = _score_subtask_with_existing_judge(
                                judge_model=judge_model,
                                judge_tokenizer=judge_tokenizer,
                                judge_device=judge_device,
                                judge_max_new_tokens=judge_max_new_tokens,
                                reference_subtask=next_subtask,
                                candidate_subtask=pred_subtask,
                            )

                        best = max(prev_score if prev_score is not None else -1.0, next_score if next_score is not None else -1.0)
                        if best < low_score_threshold:
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

                record.update(
                    {
                        "sample_kind": sample_kind,
                        "pred_current_subtask": pred_subtask,
                        "ref_current_subtask": ref_subtask,
                        "subtask_judge_score": subtask_score,
                        "low_score_neighbor_relation": relation,
                        "neighbor_previous_subtask": prev_subtask,
                        "neighbor_next_subtask": next_subtask,
                        "neighbor_prev_judge_score": prev_score,
                        "neighbor_next_judge_score": next_score,
                        "__line_idx": line_idx,
                    }
                )

                local_analyzed += 1
                kind_stat = _ensure_kind_stat(sample_kind)
                kind_stat["analyzed"] += 1
                if subtask_score < low_score_threshold:
                    local_failed += 1
                    kind_stat["failed"] += 1
                    rel = relation if relation in local_failed_relation_counts else "unknown"
                    local_failed_relation_counts[rel] += 1
                    kind_stat["failed_relation_counts"][rel] += 1

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        local_payload: Dict[str, Any] = {
            "rank": rank,
            "world_size": world_size,
            "total_records": local_total_records,
            "shard_records": local_shard_records,
            "analyzed_records": local_analyzed,
            "failed_records": local_failed,
            "failed_relation_counts": local_failed_relation_counts,
            "by_sample_kind": local_by_sample_kind,
            "shard_output_path": str(shard_output_path),
        }

        if dist_ready and world_size > 1:
            all_payloads: List[Dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore[assignment]
            torch.distributed.all_gather_object(all_payloads, local_payload)
        else:
            all_payloads = [local_payload]

        if rank != 0:
            if dist_ready and world_size > 1:
                torch.distributed.barrier()
            return {
                "input_file": str(detail_path),
                "output_file": str(shard_output_path),
                "summary_file": str(summary_path),
                "rank": rank,
                "world_size": world_size,
                "sharded": world_size > 1,
            }

        # rank0 merge shards
        merged_records: List[Dict[str, Any]] = []
        total_records = 0
        analyzed_records = 0
        failed_records = 0
        failed_relation_counts: Dict[str, int] = {k: 0 for k in relation_keys}
        by_sample_kind: Dict[str, Dict[str, Any]] = {}

        def _ensure_merged_kind(kind: str) -> Dict[str, Any]:
            key = str(kind or "unknown")
            if key not in by_sample_kind:
                by_sample_kind[key] = {
                    "analyzed": 0,
                    "failed": 0,
                    "failed_relation_counts": {k: 0 for k in relation_keys},
                }
            return by_sample_kind[key]

        def _ratio(a: int, b: int) -> float:
            if b <= 0:
                return 0.0
            return float(a) / float(b)

        for payload in all_payloads:
            if not payload:
                continue
            total_records = max(total_records, int(payload.get("total_records", 0)))
            analyzed_records += int(payload.get("analyzed_records", 0))
            failed_records += int(payload.get("failed_records", 0))

            prc = payload.get("failed_relation_counts", {}) or {}
            for k in relation_keys:
                failed_relation_counts[k] += int(prc.get(k, 0))

            pks = payload.get("by_sample_kind", {}) or {}
            for sample_kind, stat in pks.items():
                acc = _ensure_merged_kind(sample_kind)
                acc["analyzed"] += int(stat.get("analyzed", 0))
                acc["failed"] += int(stat.get("failed", 0))
                rc = stat.get("failed_relation_counts", {}) or {}
                for k in relation_keys:
                    acc["failed_relation_counts"][k] += int(rc.get(k, 0))

            shard_path = Path(str(payload.get("shard_output_path", "")))
            if shard_path.exists():
                with shard_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            merged_records.append(json.loads(line))

        merged_records.sort(key=lambda x: int(x.get("__line_idx", -1)))
        with output_path.open("w", encoding="utf-8") as fout:
            for rec in merged_records:
                rec.pop("__line_idx", None)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for kind, stat in by_sample_kind.items():
            analyzed_count = int(stat.get("analyzed", 0))
            failed_count = int(stat.get("failed", 0))
            stat["failed_ratio"] = _ratio(failed_count, analyzed_count)
            rc = dict(stat.get("failed_relation_counts", {}))
            stat["failed_relation_ratio"] = {k: _ratio(int(rc.get(k, 0)), failed_count) for k in relation_keys}

        summary = {
            "input_file": str(detail_path),
            "output_file": str(output_path),
            "summary_file": str(summary_path),
            "low_score_threshold": low_score_threshold,
            "total_records": total_records,
            "analyzed_records": analyzed_records,
            "failed_records": failed_records,
            "failed_ratio": _ratio(failed_records, analyzed_records),
            "failed_relation_counts": failed_relation_counts,
            "failed_relation_ratio": {k: _ratio(int(failed_relation_counts.get(k, 0)), failed_records) for k in relation_keys},
            "by_sample_kind": {k: by_sample_kind[k] for k in sorted(by_sample_kind.keys())},
            "world_size": world_size,
            "sharded": world_size > 1,
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("[detail-compare] saved details: %s", output_path)
        logger.info("[detail-compare] saved summary: %s", summary_path)

        if dist_ready and world_size > 1:
            torch.distributed.barrier()
        return summary
    finally:
        if compare_pbar is not None:
            compare_pbar.close()
        if judge_model is not None:
            del judge_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
