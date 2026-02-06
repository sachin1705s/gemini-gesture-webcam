import json
import os
import time
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import asyncio
import websockets

from google import genai
from google.genai import types

MODEL_VISION = "gemini-2.5-flash"
MODEL_TEXT = "gemini-2.5-flash"

PROMPT_VISION = (
    "You are a real-time perception assistant. Look at the image and identify the user's action/gesture. "
    "The action may involve hands, face, or body posture. Return a short label and (optionally) a brief reason. "
    "Be concise. Examples of possible actions: waving hello, victory sign, thumbs up, namaste, salute, laughing. "
    "If uncertain, say 'uncertain'."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant that converts a user action/gesture label into a concise, safe, cinematic mid-stream "
    "prompt for Odyssey. Keep it short (1 sentence). Focus on the change/state, not a full scene reset."
)

SEND_INTERVAL = 1.0  # seconds
UPLOAD_LONG_EDGE = 480  # downscale to reduce latency
WS_HOST = "localhost"
WS_PORT = 8765
USE_LLM_REWRITE = True


class PromptController:
    def __init__(self):
        self.lock = threading.Lock()
        self.system_prompt = DEFAULT_SYSTEM_PROMPT

    def set_system_prompt(self, text: str):
        with self.lock:
            self.system_prompt = text.strip() or DEFAULT_SYSTEM_PROMPT

    def get_system_prompt(self) -> str:
        with self.lock:
            return self.system_prompt


class GeminiWorker:
    def __init__(self, client: genai.Client, prompt_controller: PromptController):
        self.client = client
        self.prompt_controller = prompt_controller
        self.lock = threading.Lock()
        self.pending = None  # (frame, sent_ts)
        self.last_result = (None, "waiting...", None, None, None)  # (frame, text, latency_ms, ts, rewritten)
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame, sent_ts):
        with self.lock:
            self.pending = (frame, sent_ts)

    def get_last(self):
        with self.lock:
            return self.last_result

    def stop(self):
        self.stop_flag = True
        self.thread.join(timeout=2)

    def _rewrite_prompt(self, label: str) -> str:
        if not USE_LLM_REWRITE:
            return label

        system_prompt = self.prompt_controller.get_system_prompt()
        user_prompt = (
            "Convert this gesture/action label into a short Odyssey mid-stream prompt. "
            "Return only the prompt sentence.\n\n"
            f"Label: {label}"
        )

        try:
            response = self.client.models.generate_content(
                model=MODEL_TEXT,
                contents=[system_prompt, user_prompt],
            )
            text = (response.text or "").strip()
            return text or label
        except Exception:
            return label

    def _run(self):
        while not self.stop_flag:
            item = None
            with self.lock:
                item = self.pending
                self.pending = None

            if item is None:
                time.sleep(0.01)
                continue

            frame, sent_ts = item

            h, w = frame.shape[:2]
            scale = UPLOAD_LONG_EDGE / max(h, w)
            if scale < 1.0:
                resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                resized = frame

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            image_bytes = buf.tobytes()

            try:
                response = self.client.models.generate_content(
                    model=MODEL_VISION,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        PROMPT_VISION,
                    ],
                )
                label = (response.text or "").strip() or "uncertain"
            except Exception as e:
                label = f"error: {e}"

            rewritten = None
            if label and not label.lower().startswith("error"):
                rewritten = self._rewrite_prompt(label)

            latency_ms = int((time.time() - sent_ts) * 1000)
            ts = int(time.time() * 1000)

            with self.lock:
                self.last_result = (frame, label, latency_ms, ts, rewritten)


class WebsocketBroadcaster:
    def __init__(self):
        self.clients = set()

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, message: str):
        if not self.clients:
            return
        await asyncio.gather(*[ws.send(message) for ws in list(self.clients)])


def run_prompt_window(controller: PromptController):
    root = tk.Tk()
    root.title("Odyssey Prompt Controller")
    root.geometry("520x260")

    ttk.Label(root, text="System prompt for Odyssey mid-stream rewriting:").pack(padx=12, pady=(12, 6), anchor="w")

    text = tk.Text(root, height=8, wrap="word")
    text.insert("1.0", controller.get_system_prompt())
    text.pack(fill="both", expand=True, padx=12)

    status = ttk.Label(root, text="")
    status.pack(padx=12, pady=(6, 0), anchor="w")

    def apply_prompt():
        controller.set_system_prompt(text.get("1.0", "end").strip())
        status.config(text="Saved")
        root.after(1000, lambda: status.config(text=""))

    ttk.Button(root, text="Save Prompt", command=apply_prompt).pack(padx=12, pady=10, anchor="e")

    root.mainloop()


async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")

    client = genai.Client(api_key=api_key)
    prompt_controller = PromptController()

    # Start prompt controller window in separate thread
    threading.Thread(target=run_prompt_window, args=(prompt_controller,), daemon=True).start()

    worker = GeminiWorker(client, prompt_controller)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    broadcaster = WebsocketBroadcaster()
    server = await websockets.serve(broadcaster.handler, WS_HOST, WS_PORT)

    print(f"WebSocket server running on ws://{WS_HOST}:{WS_PORT}")
    print("Press q in the window to quit.")

    last_sent = 0.0
    last_frame_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = now - last_frame_time
            if dt > 0:
                fps = 1.0 / dt
            last_frame_time = now

            if now - last_sent >= SEND_INTERVAL:
                last_sent = now
                worker.submit(frame.copy(), now)

            processed_frame, label, latency_ms, ts, rewritten = worker.get_last()

            payload = {
                "label": label,
                "rewritten": rewritten or label,
                "latency_ms": latency_ms,
                "ts": ts,
            }
            await broadcaster.broadcast(json.dumps(payload))

            # Left panel: live video
            left = frame

            # Right panel: last processed frame + label
            if processed_frame is None:
                right = frame.copy()
                overlay_text = "waiting..."
                overlay_rewrite = ""
            else:
                right = processed_frame.copy()
                overlay_text = label
                overlay_rewrite = rewritten or ""

            cv2.rectangle(right, (0, 0), (right.shape[1], 110), (0, 0, 0), -1)
            cv2.putText(
                right,
                f"Action: {overlay_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if overlay_rewrite:
                cv2.putText(
                    right,
                    f"Prompt: {overlay_rewrite[:80]}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                )
            if latency_ms is not None:
                cv2.putText(
                    right,
                    f"Latency: {latency_ms} ms",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            # Optional: FPS on left
            cv2.putText(
                left,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            side_by_side = cv2.hconcat([left, right])
            cv2.imshow("Gesture Recognition (Left: Live | Right: Processed)", side_by_side)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.001)
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
