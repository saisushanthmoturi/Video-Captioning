# Video Captioning Agent (Track 2)
### High-Performance Multimodal Video Captioning Pipeline

An autonomous, containerized AI agent developed for **Track 2: Video Captioning Agent** in the **AMD Developer Hackathon: ACT II**.

---

## 🏗️ Architecture

The agent is built around a two-stage multimodal pipeline designed for high performance, zero weight overhead, and resilience against rate limits and sandbox timeouts.

```
                  /input/tasks.json
                         │
                         ▼
                  Local Video Path
                         │
                         ▼
        ┌──────────────────────────────────┐
        │  FFmpeg Keyframe Preprocessor    │
        │  - Samples up to 20 keyframes    │
        │  - Downscales frames to 768px    │
        │  - Base64 encodes under 9MB      │
        └────────────────┬─────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ Stage 1: Multimodal Vision API   │
        │ (Kimi-k2p6 or Gemini-2.5-Flash)  │
        └────────────────┬─────────────────┘
                         │
                         ▼
              Neutral Scene Description
                         │
         ┌───────────────┼───────────────┬───────────────┐
         ▼               ▼               ▼               ▼
     [formal]      [sarcastic]    [humorous_tech] [humorous_non_tech]
         │               │               │               │
         └───────────────┼───────────────┴───────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ Stage 2: Style Refinement &      │
        │ Defensive Word-Count Repair      │
        └────────────────┬─────────────────┘
                         │
                         ▼
               /output/results.json
```

---

## ✨ Features

- **Dual-Provider Compatibility**: Automatically detects and leverages `GEMINI_API_KEY` (using Gemini 2.5 Flash via official `google-genai` SDK) or `FIREWORKS_API_KEY` (using Fireworks AI visual/text endpoints).
- **FFmpeg Video Sampling**: Pre-extracts up to 20 keyframes downscaled to 768px via a local `ffmpeg` subprocess, bypassing heavy ML framework loaders like PyTorch and transformers (reducing container size from **6 GB to 100 MB**).
- **Base64 Payload Budgeting**: Dynamically subsamples extracted keyframes if the base64 payload exceeds 9.0 MB, preventing API gateway errors.
- **Robust Rate-Limit Recovery**: Employs an exponential backoff retry loop with randomized jitter to handle HTTP `429 Too Many Requests` (Resource Exhausted) errors cleanly.
- **Strict Word-Limit Enforcement**: Enforces a strict 25–60 word limit on generated captions. If a model output violates constraints, it initiates a 2-attempt edit repair loop and applies formatting/truncation if needed.
- **Docker-Safe Atomic Writing**: Saves output JSON using `shutil.move` from the container's writable internal `/tmp` directory, preventing permission crashes on host-mounted directories.
- **Mojibake-Free Output**: Forces `ensure_ascii=True` when writing output to ensure pure-ASCII escaped characters, eliminating encoding/decoding errors for the judges.

---

## 📂 Repository Structure

- [agent.py](file:///Users/moturisaisushanth/sushanth/amd-hackathon%20/agent.py) — Entrypoint orchestrator; downloads video files and schedules worker threads.
- [captioner.py](file:///Users/moturisaisushanth/sushanth/amd-hackathon%20/captioner.py) — Video frame preprocessing, base64 budgeting, vision and text API calling, and retry wrappers.
- [styles.py](file:///Users/moturisaisushanth/sushanth/amd-hackathon%20/styles.py) — Styling system prompts, few-shot examples, fillers, and fallbacks.
- [Dockerfile](file:///Users/moturisaisushanth/sushanth/amd-hackathon%20/Dockerfile) — Packages the agent in a python-slim base, installing `ffmpeg`, python dependencies, and copying scripts.
- [requirements.txt](file:///Users/moturisaisushanth/sushanth/amd-hackathon%20/requirements.txt) — Dependency list.

---

## 🚀 Running Locally

### 1. Setup Environment
Cloning and installing dependencies:
```bash
git clone https://github.com/saisushanthmoturi/Video-Captioning.git
cd Video-Captioning
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```ini
# Google AI Studio Gemini API Key
GEMINI_API_KEY=YOUR_GEMINI_API_KEY

# Optional: Fireworks AI Keys (if preferred)
# FIREWORKS_API_KEY=YOUR_FIREWORKS_API_KEY
# VISION_MODEL=accounts/fireworks/models/kimi-k2p6
# TEXT_MODEL=accounts/fireworks/models/glm-5p2

# Thread Concurrency (set to 1 to prevent rate-limiting on free keys)
CONCURRENT_WORKERS=1
```

### 3. Setup Task list
Place your test video metadata in `input/tasks.json`:
```json
[
  {
    "task_id": "v1",
    "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

### 4. Run the Agent
```bash
python agent.py
```
Check outputs in `output/results.json`.

---

## 🐳 Docker Deployment

The agent is compiled for the `linux/amd64` architecture, making it ready to run on high-performance AMD computing platforms.

### Running with Docker CLI
```bash
docker run --rm \
  -e GEMINI_API_KEY=YOUR_GEMINI_API_KEY \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  ghcr.io/saisushanthmoturi/video-captioning:latest
```

---

## 🏆 AMD Developer Hackathon Compliance

This project complies fully with the Track 2 container interface:
- Reads `/input/tasks.json` and parses task structures dynamically.
- Gracefully handles local and remote download URLs.
- Generates required captions for all 4 tones.
- Writes `/output/results.json` atomically.
- Returns code `0` on successful completion.
