# Copyright 2025 cortex community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import argparse
import atexit
import json
import logging
import os
import socket
import threading
import time
from fractions import Fraction
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

try:
    import av
except ImportError:
    av = None

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from cortex.inference.episode_level_eval import (
    InferenceArguments,
    _load_model_processor_tokenizer,
    _run_single_sample,
    _sanitize_predicted_memory,
)


class DebugTraceRecorder:
    def __init__(self, enabled: bool = False, debug_dir: str = "") -> None:
        self.enabled = bool(enabled)
        self.debug_dir = Path(debug_dir or "./exp/cortex/inference_sys2/server_debug_trace").expanduser().resolve()
        self._lock = threading.Lock()
        self._session_state: Dict[str, Dict[str, Any]] = {}
        self._warned_no_av = False
        if self.enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            atexit.register(self.close)

    @staticmethod
    def _sanitize_session_id(session_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(session_id or "default"))
        return safe or "default"

    def _get_session_dir(self, session_id: str) -> Dict[str, Any]:
        safe_session_id = self._sanitize_session_id(session_id)
        session_dir = self.debug_dir / safe_session_id
        frames_dir = session_dir / "frames"
        session_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        return {
            "session_dir": session_dir,
            "frames_dir": frames_dir,
            "jsonl_path": session_dir / "predictions.jsonl",
            "video_path": session_dir / "preview.mp4",
        }

    @staticmethod
    def _annotate_head_image(image: Image.Image, current_subtask: str, output_memory: str, step_id: int) -> Image.Image:
        frame = image.convert("RGB").copy()
        draw = ImageDraw.Draw(frame)
        font = ImageFont.load_default()
        overlay_text = f"step_id: {step_id}\ncurrent_subtask: {current_subtask or '<empty>'}\nupdated_memory: {output_memory or '<empty>'}"
        margin = 12
        spacing = 6
        bbox = draw.multiline_textbbox((margin, margin), overlay_text, font=font, spacing=spacing)
        draw.rectangle([margin - 6, margin - 6, bbox[2] + 6, bbox[3] + 6], fill=(0, 0, 0))
        draw.multiline_text((margin, margin), overlay_text, fill=(255, 255, 255), font=font, spacing=spacing)
        return frame

    def _open_video_writer(self, video_path: Path, width: int, height: int):
        if av is None:
            if not self._warned_no_av:
                logging.warning("pyav is unavailable; preview.mp4 will not be written")
                self._warned_no_av = True
            return None, None

        width = int(width)
        height = int(height)
        fps = 1.0
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
                    "Opened preview video writer with codec: %s, bit_rate=%s, resolution=%sx%s, fps=%.6f",
                    codec_name,
                    target_bit_rate,
                    width,
                    height,
                    fps,
                )
                break
            except Exception as exc:
                last_error = exc
                stream = None

        if stream is None:
            container.close()
            logging.warning("Failed to open preview video writer for %s: %s", video_path, last_error)
            return None, None

        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        return container, stream

    @staticmethod
    def _write_video_frame(container, stream, image: Image.Image) -> None:
        frame = av.VideoFrame.from_image(image.convert("RGB"))
        for packet in stream.encode(frame):
            container.mux(packet)

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            for state in self._session_state.values():
                container = state.get("video_container")
                stream = state.get("video_stream")
                if container is None or stream is None:
                    continue
                try:
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()
                except Exception as exc:
                    logging.warning("Failed to finalize preview video: %s", exc)
                state["video_container"] = None
                state["video_stream"] = None

    def close_session(self, session_id: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            state = self._session_state.get(session_id)
            if not state:
                return
            container = state.get("video_container")
            stream = state.get("video_stream")
            if container is None or stream is None:
                return
            try:
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                logging.info("Finalized preview video for session %s", session_id)
            except Exception as exc:
                logging.warning("Failed to finalize preview video for session %s: %s", session_id, exc)
            state["video_container"] = None
            state["video_stream"] = None

    def record(
        self,
        session_id: str,
        task_text: str,
        input_memory: str,
        output_memory: str,
        current_subtask: str,
        head_image: Optional[Image.Image],
        raw_prediction: str,
    ) -> None:
        if not self.enabled or head_image is None:
            return

        with self._lock:
            session_files = self._get_session_dir(session_id)
            state = self._session_state.setdefault(
                session_id,
                {
                    "step_id": 0,
                    "video_container": None,
                    "video_stream": None,
                    **session_files,
                },
            )
            state["step_id"] += 1
            step_id = int(state["step_id"])

            frame_name = f"{step_id:06d}.jpg"
            frame_path = Path(state["frames_dir"]) / frame_name
            head_rgb = head_image.convert("RGB")
            annotated = self._annotate_head_image(
                head_rgb,
                current_subtask=current_subtask,
                output_memory=output_memory,
                step_id=step_id,
            )
            annotated.save(frame_path, format="JPEG", quality=95)
            if state["video_container"] is None and state["video_stream"] is None:
                container, stream = self._open_video_writer(Path(state["video_path"]), annotated.size[0], annotated.size[1])
                state["video_container"] = container
                state["video_stream"] = stream
            if state["video_container"] is not None and state["video_stream"] is not None:
                try:
                    self._write_video_frame(state["video_container"], state["video_stream"], annotated)
                except Exception as exc:
                    logging.warning("Failed to append preview frame for session %s: %s", session_id, exc)

            record = {
                "session_id": session_id,
                "step_id": step_id,
                "timestamp": time.time(),
                "task_text": task_text,
                "input_memory": input_memory,
                "output_memory": output_memory,
                "current_subtask": current_subtask,
                "head_image_path": str(frame_path),
                "preview_video_path": str(state["video_path"]),
                "raw_prediction": raw_prediction,
            }
            with open(state["jsonl_path"], "a", encoding="utf-8") as fout:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")


class Sys2PolicyAdapter:
    """Adapter that exposes Sys2 text prediction through predict_action(...).

    Compatible with WebsocketPolicyServer, which calls policy.predict_action(**payload).
    """

    def __init__(
        self,
        infer_args: InferenceArguments,
        default_video_keys: Sequence[str],
        expose_memory: bool = False,
        trace_recorder: Optional[DebugTraceRecorder] = None,
    ):
        self.args = infer_args
        self.model, self.processor, self.tokenizer, self.device = _load_model_processor_tokenizer(infer_args)
        self.default_video_keys = [str(k).strip() for k in default_video_keys if str(k).strip()]
        self.expose_memory = bool(expose_memory)
        self.trace_recorder = trace_recorder or DebugTraceRecorder(enabled=False)
        self.default_ordered_subtask_plan = self._normalize_ordered_subtask_plan(
            getattr(infer_args, "ordered_subtask_plan", None),
            [],
        )

        self._state_lock = threading.Lock()
        self._session_state: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _normalize_task_text(task_text: Any, instructions: Any, default_task_text: str) -> str:
        if isinstance(task_text, str) and task_text.strip():
            return task_text.strip()
        if isinstance(instructions, str) and instructions.strip():
            return instructions.strip()
        if isinstance(instructions, (list, tuple)) and instructions:
            candidate = str(instructions[0]).strip()
            if candidate:
                return candidate
        return str(default_task_text or "").strip()

    @staticmethod
    def _normalize_video_keys(video_keys: Any, fallback_keys: Sequence[str], num_images: int) -> List[str]:
        if isinstance(video_keys, str):
            keys = [k.strip() for k in video_keys.split(",") if k.strip()]
        elif isinstance(video_keys, (list, tuple)):
            keys = [str(k).strip() for k in video_keys if str(k).strip()]
        else:
            keys = []

        if not keys:
            keys = list(fallback_keys)

        if len(keys) < num_images:
            keys = list(keys) + [f"camera_{idx}" for idx in range(len(keys), num_images)]
        elif len(keys) > num_images:
            keys = list(keys[:num_images])

        return keys

    @staticmethod
    def _normalize_ordered_subtask_plan(ordered_subtask_plan: Any, default_plan: Sequence[str]) -> List[str]:
        if isinstance(ordered_subtask_plan, str):
            normalized = [item.strip() for item in ordered_subtask_plan.split("|") if item.strip()]
            if normalized:
                return normalized
            normalized = [item.strip() for item in ordered_subtask_plan.split("\n") if item.strip()]
            if normalized:
                return normalized
        elif isinstance(ordered_subtask_plan, (list, tuple)):
            normalized = [str(item).strip() for item in ordered_subtask_plan if str(item).strip()]
            if normalized:
                return normalized

        return [str(item).strip() for item in default_plan if str(item).strip()]

    @staticmethod
    def _select_real_images(batch_images: Any, images_mask: Any) -> List[Any]:
        if not isinstance(batch_images, list) or not batch_images:
            raise ValueError("batch_images must be a non-empty list with batch dimension")

        images = batch_images[0]
        if not isinstance(images, list) or not images:
            raise ValueError("batch_images[0] must be a non-empty list of images")

        if isinstance(images_mask, list) and images_mask:
            sample_mask = images_mask[0]
            if isinstance(sample_mask, list) and len(sample_mask) == len(images):
                filtered = [img for img, keep in zip(images, sample_mask) if bool(keep)]
                if filtered:
                    return filtered

        return images

    def _get_or_init_memory(self, session_id: str, reset_memory: bool, memory_init: str) -> str:
        with self._state_lock:
            if reset_memory or session_id not in self._session_state:
                self._session_state[session_id] = {
                    "memory": memory_init,
                    "last_subtask": "",
                    "updated_at": time.time(),
                }
            state = self._session_state[session_id]
            return str(state.get("memory", memory_init))

    def _update_state(self, session_id: str, next_memory: str, current_subtask: str) -> None:
        with self._state_lock:
            self._session_state[session_id] = {
                "memory": next_memory,
                "last_subtask": current_subtask,
                "updated_at": time.time(),
            }

    def on_disconnect(self, session_ids: Sequence[str]) -> None:
        for session_id in session_ids:
            self.trace_recorder.close_session(str(session_id))

    def predict_action(self, batch_images, instructions=None, **kwargs):
        session_id = str(kwargs.get("session_id") or kwargs.get("request_id") or "default")
        reset_memory = bool(kwargs.get("reset_memory", False) or kwargs.get("reset", False))
        memory_init = str(kwargs.get("initial_memory") or self.args.initial_memory or "").strip()

        def _summarize_images(images: Sequence[Any]) -> List[List[int]]:
            shapes: List[List[int]] = []
            for img in images:
                try:
                    shapes.append(list(getattr(img, "shape", np.asarray(img).shape)))
                except Exception:
                    shapes.append([])
            return shapes

        task_text = self._normalize_task_text(
            task_text=kwargs.get("task_text"),
            instructions=instructions,
            default_task_text=getattr(self.args, "task_text", "") or "",
        )
        if not task_text:
            raise ValueError("Missing task text. Provide payload.task_text or payload.instructions[0].")

        real_images = self._select_real_images(batch_images=batch_images, images_mask=kwargs.get("images_mask"))
        video_keys = self._normalize_video_keys(kwargs.get("video_keys"), self.default_video_keys, len(real_images))
        ordered_subtask_plan = self._normalize_ordered_subtask_plan(
            kwargs.get("ordered_subtask_plan"),
            self.default_ordered_subtask_plan,
        )
        detailed_task = str(kwargs.get("detailed_task") or getattr(self.args, "detailed_task", "") or "").strip()

        input_memory = self._get_or_init_memory(
            session_id=session_id,
            reset_memory=reset_memory,
            memory_init=memory_init,
        )

        logging.info(
            "[SYS2][INPUT] session_id=%s reset_memory=%s task_text=%s detailed_task=%s input_memory=%s video_keys=%s image_shapes=%s ordered_subtask_plan=%s",
            session_id,
            reset_memory,
            task_text,
            detailed_task,
            input_memory,
            video_keys,
            _summarize_images(real_images),
            ordered_subtask_plan,
        )

        pred_text, pred_obj = _run_single_sample(
            model=self.model,
            processor=self.processor,
            tokenizer=self.tokenizer,
            device=self.device,
            args=self.args,
            task_text=task_text,
            input_memory=input_memory,
            current_images=real_images,
            video_keys=video_keys,
            ordered_subtask_plan=ordered_subtask_plan,
            detailed_task=detailed_task,
            sample_id=0 if reset_memory else -1,
        )

        current_subtask = str(pred_obj.get("current_subtask", "")).strip()
        predicted_memory = _sanitize_predicted_memory(str(pred_obj.get("active_language_memory", "")).strip())
        next_memory = predicted_memory or input_memory

        self._update_state(
            session_id=session_id,
            next_memory=next_memory,
            current_subtask=current_subtask,
        )

        head_image = real_images[0] if real_images else None
        self.trace_recorder.record(
            session_id=session_id,
            task_text=task_text,
            input_memory=input_memory,
            output_memory=next_memory,
            current_subtask=current_subtask,
            head_image=head_image,
            raw_prediction=pred_text,
        )

        logging.info(
            "[SYS2][OUTPUT] session_id=%s current_subtask=%s output_memory=%s active_language_memory=%s",
            session_id,
            current_subtask,
            next_memory,
            next_memory,
        )

        result = {
            "current_subtask": current_subtask,
            "session_id": session_id,
            "input_memory": input_memory,
            "output_memory": next_memory,
        }
        if self.expose_memory:
            result["active_language_memory"] = next_memory
            result["raw_prediction"] = pred_text
        return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Sys2 model via existing WebSocket policy protocol")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str, default="")
    parser.add_argument("--processor_name_or_path", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10094)

    parser.add_argument(
        "--video_keys",
        type=str,
        default="observation.head_image,observation.wrist_image",
        help="Fallback view-key order when payload omits video_keys.",
    )
    parser.add_argument("--task_text", type=str, default="")
    parser.add_argument("--initial_memory", type=str, default="This is the first subtask, and no subtasks have been completed yet.")
    parser.add_argument("--ordered_subtask_plan", type=str, default="", help="Fallback ordered subtask plan. Use | to separate subtasks.")

    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--expose_memory", action="store_true", help="Include memory/raw_prediction in server response")
    parser.add_argument("--save_debug_trace", action="store_true", help="Save head frames, jsonl predictions and preview mp4 per session")
    parser.add_argument("--debug_dir", type=str, default="./exp/cortex/inference_sys2/server_debug_trace")

    return parser


def _to_infer_args(ns: argparse.Namespace) -> InferenceArguments:
    allowed = {f.name for f in fields(InferenceArguments)}
    kwargs = {k: v for k, v in vars(ns).items() if k in allowed}

    kwargs.setdefault("mode", "simple")
    kwargs.setdefault("output_path", "./exp/cortex/inference_sys2/server_dummy.jsonl")
    kwargs.setdefault("save_visual_video", False)
    kwargs.setdefault("save_episode_summary", False)

    kwargs["model_name_or_path"] = ns.model_name_or_path
    kwargs["video_keys"] = ns.video_keys

    for key in ("base_model_name_or_path", "processor_name_or_path", "cache_dir"):
        if not kwargs.get(key):
            kwargs[key] = None

    infer_args = InferenceArguments(**kwargs)
    setattr(infer_args, "deploy", True)
    setattr(infer_args, "ordered_subtask_plan", ns.ordered_subtask_plan)
    return infer_args


def _safe_get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        return str(socket.gethostbyname(hostname))
    except Exception:
        pass

    return "unknown"


def start_debugpy_once() -> None:
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10091))
    print("Waiting for VSCode attach on 0.0.0.0:10091 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    if os.getenv("DEBUG", ""):
        start_debugpy_once()

    logging.basicConfig(level=logging.INFO, force=True)

    infer_args = _to_infer_args(args)
    default_video_keys = [k.strip() for k in args.video_keys.split(",") if k.strip()]
    trace_recorder = DebugTraceRecorder(enabled=args.save_debug_trace, debug_dir=args.debug_dir)
    policy = Sys2PolicyAdapter(
        infer_args=infer_args,
        default_video_keys=default_video_keys,
        expose_memory=args.expose_memory,
        trace_recorder=trace_recorder,
    )

    local_ip = _safe_get_local_ip()
    logging.info("Creating Sys2 server (bind=%s:%s, local_ip=%s)", args.host, args.port, local_ip)
    if args.save_debug_trace:
        logging.info("Debug trace enabled: %s", trace_recorder.debug_dir)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "env": "sys2_env",
            "model_name_or_path": args.model_name_or_path,
            "memory_owner": "server",
            "response_fields": ["current_subtask"],
            "save_debug_trace": bool(args.save_debug_trace),
        },
    )
    logging.info("sys2 server running ...")
    server.serve_forever()


if __name__ == "__main__":
    main()
