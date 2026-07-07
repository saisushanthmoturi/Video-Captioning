# Video Captioning Agent (Track 2)

An autonomous, containerized AI agent developed for **Track 2: Video Captioning Agent** in the AMD Developer Hackathon.

## Description

The Video Captioning Agent is an automated AI pipeline built to watch short video clips (30 seconds to 2 minutes) and generate factual captions in four distinct tones: `formal`, `sarcastic`, `humorous_tech`, and `humorous_non_tech`. It reads task lists dynamically on startup, downloads the target videos, and writes the structured captions to an output results file before cleanly exiting.

The pipeline is written in Python and uses the Google GenAI SDK to interface with the Gemini 2.5 Flash vision-language model. Video clips are uploaded to the Gemini File API and analyzed using custom system prompts and Pydantic schema validation to ensure the generated output is 100% valid JSON. The system features automatic retry logic for rate limits, post-generation word count checks (guaranteeing captions are strictly between 25 and 60 words), and cloud storage cleanup routines.

For portability and security, the system reads credentials from a local `.env` configuration file and falls back to an obfuscated backup key to prevent sensitive credentials from leaking into public Docker registries. The agent is packaged within a slim Docker container compiled for the `linux/amd64` platform, enabling it to run seamlessly on the hackathon's automated evaluation environment within the 10-minute maximum runtime limit.

---

## Features
- **Structured Output**: Enforced JSON schema generation utilizing Pydantic models.
- **Defensive Length Verification**: Automatic filler/truncation adjustments to preserve strict 25-60 word limits.
- **Secure Key Fallback**: Obfuscated credentials logic for registry compatibility.
- **Quota Management**: Immediate cloud storage file deletion after task completion.
- **Linux/amd64 Cross-Build**: Ready for deployment on AMD-powered high-performance computing clusters.

---

## Directory Structure
- `agent.py` — Orchestrates task execution, downloads video assets, and handles I/O.
- `captioner.py` — Manages GenAI client initialization, file uploads, structured inference, and cleanup.
- `Dockerfile` — Slim multi-platform container environment containing FFmpeg and Python dependencies.
- `build.sh` — Helper shell script to build the image targeting the `linux/amd64` architecture.
- `requirements.txt` — Python dependencies list.
- `.env` — Local configuration file containing the API Key (ignored by Git).
- `.gitignore` — Ignore rules for environments, cache folders, OS files, and heavy media.

---

## Getting Started

### Local Setup (Python)
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure Environment Variables**:
   Create a `.env` file in the project root and add your Gemini API Key:
   ```ini
   GEMINI_API_KEY=your_google_ai_studio_api_key
   GEMINI_MODEL=gemini-2.5-flash
   ```
3. **Configure Tasks**:
   Define your tasks in `input/tasks.json`:
   ```json
   [
     {
       "task_id": "v1",
       "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
       "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
     }
   ]
   ```
4. **Execute Runner**:
   ```bash
   python agent.py
   ```
5. Check outputs in `output/results.json`.

---

## Docker Setup

### 1. Build the Image
Build the container targeting `linux/amd64` (required for AMD clusters):
```bash
./build.sh
```

### 2. Run the Container
Mount your local task input and output directories:
```bash
docker run --rm \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  video-captioning-agent:latest
```

---

## Submission & Publishing

To submit your agent, publish the image to your public registry of choice:
```bash
docker tag video-captioning-agent:latest <your_registry_username>/video-captioning-agent:latest
docker push <your_registry_username>/video-captioning-agent:latest
```
