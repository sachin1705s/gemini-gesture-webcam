# Gemini Gesture Webcam

Local webcam gesture/action recognition using Gemini Vision. Includes a live preview and a WebSocket broadcaster so other apps can consume the detected action labels.

## Files
- `gemini_gesture_webcam.py` – Live webcam + Gemini action label overlay
- `gemini_gesture_stream.py` – Live webcam + Gemini action label + WebSocket feed (`ws://localhost:8765`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install google-genai opencv-python websockets
```

Set your API key:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

## Run (local overlay)

```bash
python gemini_gesture_webcam.py
```

## Run (WebSocket feed + prompt controller)

```bash
python gemini_gesture_stream.py
```

A separate **Prompt Controller** window opens to set a system prompt that rewrites gesture labels into Odyssey mid‑stream prompts.

The WebSocket server runs at:

```
ws://localhost:8765
```

## Payload

Each WebSocket message includes both the raw label and the rewritten prompt:

```json
{"label":"waving hello","rewritten":"The subject is waving a greeting.","latency_ms":312,"ts":1738789999000}
```

## Notes
- Latency is shown in the overlay (ms).
- You can tune `SEND_INTERVAL` and `UPLOAD_LONG_EDGE` in the scripts for cost/latency tradeoffs.
