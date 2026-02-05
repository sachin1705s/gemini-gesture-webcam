import os
import time
import threading

import cv2

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"

PROMPT = (
    "You are a real-time perception assistant. Look at the image and identify the user's action/gesture. "
    "The action may involve hands, face, or body posture. Return a short label and (optionally) a brief reason. "
    "Be concise. Examples of possible actions: waving hello, victory sign, thumbs up, namaste, salute, laughing. "
    "If uncertain, say 'uncertain'."
)

SEND_INTERVAL = 1.0  # seconds
UPLOAD_LONG_EDGE = 480  # downscale to reduce latency


class GeminiWorker:
    def __init__(self, client: genai.Client):
        self.client = client
        self.lock = threading.Lock()
        self.pending = None  # (frame, sent_ts)
        self.last_result = (None, "waiting...", None)  # (frame, text, latency_ms)
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

            # Prepare image for upload (downscale + JPEG)
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
                    model=MODEL,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        PROMPT,
                    ],
                )
                text = (response.text or "").strip() or "uncertain"
            except Exception as e:
                text = f"error: {e}"

            latency_ms = int((time.time() - sent_ts) * 1000)

            with self.lock:
                self.last_result = (frame, text, latency_ms)


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")

    client = genai.Client(api_key=api_key)
    worker = GeminiWorker(client)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_sent = 0.0
    last_frame_time = time.time()
    fps = 0.0

    print("Press q in the window to quit.")

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

        # Left panel: live video
        left = frame

        # Right panel: last processed frame + label
        processed_frame, last_text, latency_ms = worker.get_last()
        if processed_frame is None:
            right = frame.copy()
            overlay_text = "waiting..."
        else:
            right = processed_frame.copy()
            overlay_text = last_text

        cv2.rectangle(right, (0, 0), (right.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(
            right,
            f"Action: {overlay_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if latency_ms is not None:
            cv2.putText(
                right,
                f"Latency: {latency_ms} ms",
                (10, 60),
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

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
