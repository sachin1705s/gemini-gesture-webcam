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
python -m pip install google-genai opencv-python
```

Set your API key:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

## Run (local overlay)

```bash
python gemini_gesture_webcam.py
```

## Run (WebSocket feed)

```bash
python gemini_gesture_stream.py
```

The WebSocket server runs at:

```
ws://localhost:8765
```

## Notes
- Latency is shown in the overlay (ms).
- You can tune `SEND_INTERVAL` and `UPLOAD_LONG_EDGE` in the scripts for cost/latency tradeoffs.
