import argparse
import base64
import io
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import socket as socket_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from deployment.model_server.server_sys2 import DebugTraceRecorder
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from cortex.inference.episode_level_eval import (
    _build_messages,
    _parse_prediction,
    _sanitize_predicted_memory,
)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class GeminiSys2PolicyAdapter:
    """Gemini-backed Sys2 planner using the existing WebSocket policy protocol."""

    def __init__(
        self,
        model_name: str,
        api_key_env: str,
        base_url: str,
        default_video_keys: Sequence[str],
        initial_memory: str,
        ordered_subtask_plan: str = "",
        task_text: str = "",
        detailed_task: str = "",
        enable_detailed_task: bool = False,
        expose_memory: bool = False,
        trace_recorder: Optional[DebugTraceRecorder] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int = 256,
        max_retries: int = 2,
        request_timeout: float = 120.0,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        if not self.base_url and (genai is None or types is None):
            raise ImportError(
                "Missing Gemini SDK. Install it with `pip install google-genai` "
                "in the Python environment used to launch server_sys2_gemini.py."
            )

        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise ValueError(f"Missing Gemini API key. Set environment variable {api_key_env}.")

        self.client = None if self.base_url else genai.Client(api_key=api_key)
        self.api_key = api_key
        self.model_name = model_name
        self.default_video_keys = [str(k).strip() for k in default_video_keys if str(k).strip()]
        self.initial_memory = str(initial_memory or "")
        self.default_ordered_subtask_plan = self._normalize_ordered_subtask_plan(ordered_subtask_plan, [])
        self.default_task_text = str(task_text or "").strip()
        self.default_detailed_task = str(detailed_task or "").strip()
        self.enable_detailed_task = bool(enable_detailed_task)
        self.expose_memory = bool(expose_memory)
        self.trace_recorder = trace_recorder or DebugTraceRecorder(enabled=False)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = max(0, int(max_retries))
        self.request_timeout = float(request_timeout)

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
            normalized = [item.strip() for item in ordered_subtask_plan.splitlines() if item.strip()]
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
            return str(self._session_state[session_id].get("memory", memory_init))

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

    @staticmethod
    def _messages_to_gemini(messages: List[Dict[str, Any]]) -> tuple[str, List[Any]]:
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

    def _call_gemini(self, system_instruction: str, contents: List[Any]) -> tuple[str, Optional[str]]:
        if self.base_url:
            return self._call_openai_compatible_api(system_instruction=system_instruction, contents=contents)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return str(getattr(response, "text", "") or ""), None

    @staticmethod
    def _image_to_data_url(image: Any) -> str:
        if not hasattr(image, "save"):
            raise TypeError(f"Expected PIL-like image for OpenAI-compatible API, got {type(image)}")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _call_openai_compatible_api(self, system_instruction: str, contents: List[Any]) -> tuple[str, Optional[str]]:
        user_content: List[Dict[str, Any]] = []
        for item in contents:
            if isinstance(item, str):
                if item:
                    user_content.append({"type": "text", "text": item})
            else:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_to_data_url(item)},
                    }
                )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                response_status = getattr(response, "status", None)
                response_content_type = response.headers.get("Content-Type", "")
                response_body = response.read().decode("utf-8", errors="replace")
            try:
                response_obj = json.loads(response_body)
            except json.JSONDecodeError as exc:
                response_preview = response_body[:500].replace("\n", "\\n")
                raise RuntimeError(
                    "OpenAI-compatible API returned non-JSON response "
                    f"status={response_status} content_type={response_content_type!r} "
                    f"body_prefix={response_preview!r}"
                ) from exc
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible API HTTP {exc.code}: {error_body}") from exc

        choices = response_obj.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenAI-compatible API returned no choices: {response_obj}")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message") or {}
        content = message.get("content", "")
        logging.info("[SYS2][GEMINI][API] finish_reason=%s content_type=%s", finish_reason, type(content).__name__)
        if isinstance(content, list):
            text_parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
            text = "\n".join(part for part in text_parts if part)
        else:
            text = str(content or "")
        if not text:
            logging.warning(
                "[SYS2][GEMINI][API] empty content response_summary=%s",
                json.dumps(
                    {
                        "id": response_obj.get("id"),
                        "model": response_obj.get("model"),
                        "usage": response_obj.get("usage"),
                        "finish_reason": finish_reason,
                        "message_keys": sorted(message.keys()) if isinstance(message, dict) else [],
                    },
                    ensure_ascii=False,
                ),
            )
        return text, str(finish_reason or "")

    def _predict_with_retries(self, system_instruction: str, contents: List[Any]) -> tuple[str, Dict[str, Any], Optional[str]]:
        last_text = ""
        last_obj: Dict[str, Any] = {}
        last_finish_reason: Optional[str] = None
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                logging.warning(
                    "[SYS2][GEMINI][RETRY] attempt=%s previous_finish_reason=%s previous_error=%s previous_text_prefix=%s",
                    attempt + 1,
                    last_finish_reason,
                    repr(last_error) if last_error else "",
                    last_text[:120].replace("\n", "\\n"),
                )
            try:
                pred_text, finish_reason = self._call_gemini(
                    system_instruction=system_instruction,
                    contents=contents,
                )
                last_error = None
            except (TimeoutError, socket_module.timeout, urllib.error.URLError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                continue
            pred_obj = _parse_prediction(pred_text)
            current_subtask = str(pred_obj.get("current_subtask", "")).strip()
            last_text = pred_text
            last_obj = pred_obj
            last_finish_reason = finish_reason
            if current_subtask and finish_reason != "length":
                break
        return last_text, last_obj, last_finish_reason

    def predict_action(self, **kwargs) -> Dict[str, Any]:
        session_id = str(kwargs.get("session_id") or kwargs.get("request_id") or "default")
        reset_memory = bool(kwargs.get("reset_memory", False))
        memory_init = str(kwargs.get("initial_memory") or self.initial_memory)
        task_text = self._normalize_task_text(
            task_text=kwargs.get("task_text"),
            instructions=kwargs.get("instructions"),
            default_task_text=self.default_task_text,
        )
        if not task_text:
            raise ValueError("Missing task text. Provide payload.task_text or payload.instructions[0].")

        real_images = self._select_real_images(kwargs.get("batch_images"), kwargs.get("images_mask"))
        video_keys = self._normalize_video_keys(kwargs.get("video_keys"), self.default_video_keys, len(real_images))
        ordered_subtask_plan = self._normalize_ordered_subtask_plan(
            kwargs.get("ordered_subtask_plan"),
            self.default_ordered_subtask_plan,
        )
        detailed_task = str(kwargs.get("detailed_task") or self.default_detailed_task or "").strip()
        if not detailed_task and self.enable_detailed_task:
            detailed_task = task_text
        input_memory = self._get_or_init_memory(
            session_id=session_id,
            reset_memory=reset_memory,
            memory_init=memory_init,
        )

        logging.info(
            "[SYS2][GEMINI][INPUT] session_id=%s reset_memory=%s task_text=%s detailed_task=%s input_memory=%s video_keys=%s ordered_subtask_plan=%s",
            session_id,
            reset_memory,
            task_text,
            detailed_task,
            input_memory,
            video_keys,
            ordered_subtask_plan,
        )

        messages = _build_messages(
            task_text=task_text,
            input_memory=input_memory,
            current_images=real_images,
            video_keys=video_keys,
            ordered_subtask_plan=ordered_subtask_plan,
            detailed_task=detailed_task,
        )
        system_instruction, contents = self._messages_to_gemini(messages)
        pred_text, pred_obj, finish_reason = self._predict_with_retries(
            system_instruction=system_instruction,
            contents=contents,
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
            "[SYS2][GEMINI][OUTPUT] session_id=%s current_subtask=%s output_memory=%s raw_prediction=%s",
            session_id,
            current_subtask,
            next_memory,
            pred_text,
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
            result["finish_reason"] = finish_reason
        return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Gemini Sys2 planner via existing WebSocket policy protocol")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10094)
    parser.add_argument("--gemini_model", type=str, default="gemini-3.1-pro-preview")
    parser.add_argument("--base_url", type=str, default=os.getenv("GEMINI_BASE_URL", ""))
    parser.add_argument("--api_key_env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--video_keys", type=str, default="observation.head_image,observation.wrist_image")
    parser.add_argument("--task_text", type=str, default="")
    parser.add_argument("--initial_memory", type=str, default="This is the first subtask, and no subtasks have been completed yet.")
    parser.add_argument("--ordered_subtask_plan", type=str, default="", help="Fallback ordered subtask plan. Use | to separate subtasks.")
    parser.add_argument("--detailed_task", type=str, default="")
    parser.add_argument("--enable_detailed_task", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--max_concurrent_requests", type=int, default=8)
    parser.add_argument("--ping_interval", type=float, default=None)
    parser.add_argument("--ping_timeout", type=float, default=None)
    parser.add_argument("--expose_memory", action="store_true", help="Include memory/raw_prediction in server response")
    parser.add_argument("--save_debug_trace", action="store_true", help="Save head frames, jsonl predictions and preview mp4 per session")
    parser.add_argument("--debug_dir", type=str, default="./exp/cortex/inference_sys2/gemini_server_debug_trace")
    return parser


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


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    default_video_keys = [k.strip() for k in args.video_keys.split(",") if k.strip()]
    trace_recorder = DebugTraceRecorder(enabled=args.save_debug_trace, debug_dir=args.debug_dir)
    policy = GeminiSys2PolicyAdapter(
        model_name=args.gemini_model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        default_video_keys=default_video_keys,
        initial_memory=args.initial_memory,
        ordered_subtask_plan=args.ordered_subtask_plan,
        task_text=args.task_text,
        detailed_task=args.detailed_task,
        enable_detailed_task=args.enable_detailed_task,
        expose_memory=args.expose_memory,
        trace_recorder=trace_recorder,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
    )

    local_ip = _safe_get_local_ip()
    logging.info("Creating Gemini Sys2 server (bind=%s:%s, local_ip=%s)", args.host, args.port, local_ip)
    if args.save_debug_trace:
        logging.info("Debug trace enabled: %s", trace_recorder.debug_dir)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        max_concurrent_requests=args.max_concurrent_requests,
        ping_interval=args.ping_interval,
        ping_timeout=args.ping_timeout,
        metadata={
            "env": "sys2_gemini_env",
            "model_name_or_path": args.gemini_model,
            "base_url": args.base_url,
            "memory_owner": "server",
            "response_fields": ["current_subtask"],
            "save_debug_trace": bool(args.save_debug_trace),
        },
    )
    logging.info("gemini sys2 server running ...")
    server.serve_forever()


if __name__ == "__main__":
    main()
