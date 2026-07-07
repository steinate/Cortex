import argparse
import logging
import os
import socket

from deployment.model_server.server_sys2 import DebugTraceRecorder
from deployment.model_server.server_sys2_gemini import GeminiSys2PolicyAdapter
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve GPT Sys2 planner via existing WebSocket policy protocol")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10094)
    parser.add_argument("--gpt_model", type=str, default="gpt-5")
    parser.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
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
    parser.add_argument("--debug_dir", type=str, default="./exp/cortex/inference_sys2/gpt_server_debug_trace")
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
        model_name=args.gpt_model,
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
    logging.info("Creating GPT Sys2 server (bind=%s:%s, local_ip=%s)", args.host, args.port, local_ip)
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
            "env": "sys2_gpt_env",
            "model_name_or_path": args.gpt_model,
            "base_url": args.base_url,
            "memory_owner": "server",
            "response_fields": ["current_subtask"],
            "save_debug_trace": bool(args.save_debug_trace),
        },
    )
    logging.info("gpt sys2 server running ...")
    server.serve_forever()


if __name__ == "__main__":
    main()
