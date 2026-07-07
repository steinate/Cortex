# Copyright 2025 cortex community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy
from . import image_tools
from PIL import Image


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        max_concurrent_requests: int = 8,
        ping_interval: float | None = None,
        ping_timeout: float | None = None,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._max_concurrent_requests = max(1, int(max_concurrent_requests))
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._inference_semaphore: asyncio.Semaphore | None = None
        self._executor: ThreadPoolExecutor | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self._inference_semaphore = asyncio.Semaphore(self._max_concurrent_requests)
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_concurrent_requests,
            thread_name_prefix="policy-server",
        )
        logging.info(
            "Starting WebsocketPolicyServer on %s:%s with max_concurrent_requests=%s ping_interval=%s ping_timeout=%s",
            self._host,
            self._port,
            self._max_concurrent_requests,
            self._ping_interval,
            self._ping_timeout,
        )
        try:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
            ) as server:
                await server.serve_forever()
        finally:
            executor = self._executor
            self._executor = None
            self._inference_semaphore = None
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        active_session_ids = set()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._collect_session_ids(msg, active_session_ids)
                ret = await self._route_message_async(msg)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                self._notify_disconnect(active_session_ids)
                break
            except Exception:
                self._notify_disconnect(active_session_ids)
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _route_message_async(self, msg: dict) -> dict:
        if msg.get("type", "infer") != "infer":
            return self._route_message(msg)

        semaphore = self._inference_semaphore
        executor = self._executor
        if semaphore is None or executor is None:
            return await asyncio.to_thread(self._route_message, msg)

        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, self._route_message, msg)

    def _collect_session_ids(self, msg: dict, active_session_ids: set[str]) -> None:
        payload = msg.get("payload", msg)
        if not isinstance(payload, dict):
            return
        session_id = payload.get("session_id") or payload.get("request_id")
        if session_id:
            active_session_ids.add(str(session_id))

    def _notify_disconnect(self, active_session_ids: set[str]) -> None:
        if not active_session_ids:
            return
        callback = getattr(self._policy, "on_disconnect", None)
        if callable(callback):
            try:
                callback(sorted(active_session_ids))
            except Exception:
                logging.exception("Policy on_disconnect callback failed")

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Always returns a dict containing:
            {
              "status": "ok" | "error",
              "ok": bool,
              "type": <str>,
              "request_id": <str>,
              ... (data | error)
            }
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        payload = msg.get("payload", msg)         # when no explicit payload, treat top-level as payload
        payload_req_id = payload.get("request_id") if isinstance(payload, dict) else None
        req_id = payload_req_id or msg.get("request_id", "default")
        mtype = msg.get("type", "infer")          # default = infer

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer
        elif mtype == "infer":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))}
                }
            try:
                pil_images = image_tools.to_pil_preserve(payload["batch_images"])
                MAX_IMAGE_NUM = 3
                bs, image_num = len(payload["batch_images"]), len(payload["batch_images"][0])
                print(f"bs {bs}, image num {image_num}")
                pad_images = []
                pad_masks = []
                for images in pil_images:
                    width, height = images[0].size
                    img_mode = images[0].mode
                    current_num = len(images)

                    pad_num = MAX_IMAGE_NUM - current_num
                    padded = list(images)
                    blank_img = Image.new(img_mode, (width, height), 0)
                    padded.extend([blank_img for _ in range(pad_num)])
                    padded_mask = [True] * current_num + [False] * pad_num

                    pad_images.append(padded)
                    pad_masks.append(padded_mask)
                payload["batch_images"] = pad_images
                payload["images_mask"] = pad_masks
                payload["run_eval"] = True
                ouput_dict = self._policy.predict_action(** payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                        # "traceback": traceback.format_exc(),
                    },
                }
            data = ouput_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
