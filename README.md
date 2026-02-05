# Gemini Gesture Webcam

Live webcam action recognition using Gemini Vision. Left panel shows the live feed, right panel shows the last processed frame with the Gemini label and round‑trip latency.

## Features
- Open‑ended action recognition (hands, face, body posture)
- Side‑by‑side view (live vs processed)
- Latency overlay in milliseconds
- Adjustable send interval and upload resolution

## Setup

Create a virtual environment and install dependencies:

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

## Run

```bash
python gemini_gesture_webcam.py
```

Press `q` to quit the window.

## Configuration

Open `gemini_gesture_webcam.py` and adjust:

- `SEND_INTERVAL` – seconds between requests (default `1.0`)
- `UPLOAD_LONG_EDGE` – downscale long edge (default `480`)
- `MODEL` – Gemini model id

## Notes
- Latency includes upload + model processing + response time.
- Lower `UPLOAD_LONG_EDGE` to reduce latency and bandwidth.
