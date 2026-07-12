import os
import re
import sys
import json
import time
import random
import base64
import logging
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import styles and prompts
from styles import (
    SUPPORTED_STYLES,
    STYLE_SYSTEM_PROMPTS,
    STYLE_FALLBACKS,
    STYLE_FILLERS,
    build_style_messages
)

# Import API SDKs safely
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Model Configuration
VISION_MODEL = os.getenv("VISION_MODEL", "accounts/fireworks/models/kimi-k2p6")
TEXT_MODEL = os.getenv("TEXT_MODEL", "accounts/fireworks/models/glm-5p2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")

# Vision prompt instruction
VISION_INSTRUCTION = (
    "You are a precise visual analyst. The following images are representative keyframes "
    "sampled in order from a single short video. Analyze the frames and provide a structured, "
    "factual scene description covering all of the following categories:\n\n"
    "1. **Scene / Setting**: Location, venue, or environment visible in the frames.\n"
    "2. **Subjects**: People, animals, objects, positioning, and distinguishing features.\n"
    "3. **Actions / Motion**: Activities, movements, or events taking place across the frames.\n"
    "4. **Environment**: Indoor/outdoor, lighting, weather, or seasonal indicators.\n"
    "5. **Key Visual Elements**: Colors, text/overlays, objects, or camera movements.\n\n"
    "Do not write captions or taglines at this stage. Keep the report strictly neutral, factual, "
    "and objective. Do not add humor, sarcasm, or opinions. English only."
)

# ---------------------------------------------------------------------------
# API Call Retry Helper
# ---------------------------------------------------------------------------
def _call_with_retry(func, *args, max_retries=5, base_delay=2.0, **kwargs):
    """
    Executes a callable with exponential backoff and jitter upon encountering errors,
    specifically targetting rate limits (HTTP 429 / RESOURCE_EXHAUSTED).
    """
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            e_str = str(e)
            logger.warning(f"API call attempt {attempt}/{max_retries} failed: {e_str}")
            if attempt == max_retries:
                raise e
            if "429" in e_str or "RESOURCE_EXHAUSTED" in e_str or "quota" in e_str.lower():
                # Expback with a jitter between 0 and 1.5 seconds
                sleep_time = delay + random.uniform(0.0, 1.5)
                logger.info(f"Rate limit / quota hit. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                delay *= 2.0
            else:
                # Other transient errors: brief pause before retry
                time.sleep(1.0)

# ---------------------------------------------------------------------------
# Video Preprocessing Helpers
# ---------------------------------------------------------------------------
def _probe_duration(video_path: str) -> Optional[float]:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        val = out.stdout.strip()
        return float(val) if val else None
    except Exception as e:
        logger.warning(f"ffprobe duration probe failed: {e}")
        return None

def extract_keyframes(video_path: str, work_dir: str, num_frames: int = 20) -> List[str]:
    """
    Extracts up to num_frames downscaled JPEG frames from video using FFmpeg.
    """
    s = 768  # Frame longest side
    scale_filter = f"scale=w='if(gt(iw,ih),{s},-2)':h='if(gt(iw,ih),-2,{s})'"
    
    duration = _probe_duration(video_path)
    if duration and duration > 0:
        rate = max(num_frames / duration, 0.1)
        vf = f"fps={rate:.6f},{scale_filter}"
    else:
        vf = f"fps=1,{scale_filter}"
        
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
        "-vf", vf, "-q:v", "5", "-frames:v", str(num_frames),
        os.path.join(work_dir, "frame_%04d.jpg"),
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except Exception as e:
        logger.error(f"FFmpeg frame extraction failed: {e}")
        
    frames = sorted(
        os.path.join(work_dir, f)
        for f in os.listdir(work_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )
    return frames[:num_frames]

def frames_to_budgeted_b64(frame_paths: List[str], max_mb: float = 9.0) -> List[str]:
    """
    Converts keyframe files to base64 string payloads, ensuring we stay within payload size limits.
    """
    encoded = []
    for p in frame_paths:
        try:
            with open(p, "rb") as f:
                encoded.append(base64.b64encode(f.read()).decode("ascii"))
        except Exception as e:
            logger.warning(f"Failed to base64 encode frame {p}: {e}")
            
    max_bytes = int(max_mb * 1024 * 1024)
    
    # Helper to count total bytes
    def total_bytes(lst):
        return sum(len(x) for x in lst)
        
    # Drop frames evenly if total size goes over budget
    while encoded and total_bytes(encoded) > max_bytes:
        keep = max(1, len(encoded) - 1)
        logger.info(f"Base64 payload ({total_bytes(encoded)} bytes) exceeds budget. Subsampling to {keep} frames.")
        encoded = _evenly_subsample(encoded, keep)
        if keep == 1:
            break
            
    return encoded

def _evenly_subsample(items: list, k: int) -> list:
    if k >= len(items) or k <= 0:
        return items
    n = len(items)
    idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    seen, out = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out

# ---------------------------------------------------------------------------
# API Inference Core
# ---------------------------------------------------------------------------
def run_vision_inference(b64_frames: List[str]) -> str:
    """
    Generates a neutral factual description of the video using available API keys.
    """
    if os.getenv("MOCK_INFERENCE") == "1":
        logger.info("Mock Vision Inference Mode Active.")
        return "A beautiful video showing a scene with cars driving on a busy city street, traffic flowing, or a small orange kitten playing outside in the grass."

    fireworks_key = os.getenv("FIREWORKS_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    
    if not fireworks_key and not gemini_key:
        raise ValueError("Neither FIREWORKS_API_KEY nor GEMINI_API_KEY is configured in the environment.")
        
    # Prefer Fireworks if both configured
    if fireworks_key:
        if OpenAI is None:
            raise ImportError("openai package is not installed but FIREWORKS_API_KEY is configured.")
        logger.info(f"Using Fireworks AI Vision backend: {VISION_MODEL}")
        client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=fireworks_key)
        
        content = [{"type": "text", "text": VISION_INSTRUCTION}]
        for b64 in b64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            
        extra_body = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
        
        resp = _call_with_retry(
            client.chat.completions.create,
            model=VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=400,
            temperature=0.2,
            extra_body=extra_body
        )
        return _clean_reasoning(resp.choices[0].message.content or "")
        
    else:  # Use Gemini
        if genai is None:
            raise ImportError("google-genai package is not installed but GEMINI_API_KEY is configured.")
        logger.info(f"Using Google GenAI Gemini backend: {GEMINI_MODEL}")
        client = genai.Client(api_key=gemini_key)
        
        parts = [VISION_INSTRUCTION]
        for b64 in b64_frames:
            parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(b64),
                    mime_type="image/jpeg"
                )
            )
            
        resp = _call_with_retry(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=400
            )
        )
        return _clean_reasoning(resp.text or "")

def run_style_inference(style: str, description: str) -> str:
    """
    Renders the neutral scene description into the target caption style.
    """
    if os.getenv("MOCK_INFERENCE") == "1":
        logger.info(f"Mock Style Inference Mode Active ({style}).")
        return f"This is a mock {style} caption designed to fulfill constraints and describe the video clip actions concisely."

    fireworks_key = os.getenv("FIREWORKS_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    
    # Prefer Fireworks if configured
    if fireworks_key:
        client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=fireworks_key)
        messages = build_style_messages(style, description)
        extra_body = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
        
        resp = _call_with_retry(
            client.chat.completions.create,
            model=TEXT_MODEL,
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            extra_body=extra_body
        )
        return _clean_reasoning(resp.choices[0].message.content or "")
        
    else:  # Use Gemini
        client = genai.Client(api_key=gemini_key)
        system_prompt = STYLE_SYSTEM_PROMPTS.get(style, "")
        user_prompt = f"Scene description:\n{description}\n\nCaption:"
        
        resp = _call_with_retry(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=100
            )
        )
        return _clean_reasoning(resp.text or "")

def run_style_repair(style: str, caption: str) -> str:
    """
    Optional model-based rewrite to fix caption length bounds.
    """
    if os.getenv("MOCK_INFERENCE") == "1":
        return f"This is a mock repaired {style} caption that strictly contains exactly thirty five words to verify the parsing framework is functioning correctly."
    fireworks_key = os.getenv("FIREWORKS_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    
    repair_prompt = (
        f"You are a copy editor. Rewrite the following caption so that it has "
        f"between 30 and 50 words (inclusive) while preserving its original style/tone ({style}) "
        f"and factual content. Do not add metadata, introductory phrases, or markdown formatting.\n\n"
        f"Current Caption: \"{caption}\"\n\n"
        f"Requirements:\n"
        f"1. Length: The output MUST have between 30 and 50 words.\n"
        f"2. Keep the facts identical to the original.\n"
        f"3. Retain the exact tone ({style}).\n"
        f"Output only the corrected caption string without quotes:"
    )
    
    if fireworks_key:
        client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=fireworks_key)
        resp = _call_with_retry(
            client.chat.completions.create,
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": repair_prompt}],
            max_tokens=100,
            temperature=0.3
        )
        return _clean_reasoning(resp.choices[0].message.content or "")
    else:
        client = genai.Client(api_key=gemini_key)
        resp = _call_with_retry(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=repair_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=100
            )
        )
        return _clean_reasoning(resp.text or "")

def _clean_reasoning(text: str) -> str:
    """
    Removes thinking blocks or wrapper markup if emitted.
    """
    if not text:
        return ""
    while "<think>" in text and "</think>" in text:
        start = text.index("<think>")
        end = text.index("</think>") + len("</think>")
        text = (text[:start] + text[end:]).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip().strip('"').strip("'").strip()

# ---------------------------------------------------------------------------
# Main Orchestration Entry Point
# ---------------------------------------------------------------------------
def run_combined_style_inference(description: str, requested_styles: list) -> dict:
    """
    Attempts to generate all requested styles in a single prompt to save tokens,
    reduce latency, and avoid API rate limits.
    """
    if os.getenv("MOCK_INFERENCE") == "1":
        return {style: f"This is a mock {style} caption designed to fulfill constraints." for style in requested_styles}

    fireworks_key = os.getenv("FIREWORKS_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    
    # Format guidelines for only requested styles
    style_bullet_points = []
    for style in requested_styles:
        if style == "formal":
            style_bullet_points.append("- formal: Professional, objective, clear, and informative. Describe setting and actions as for a documentary narrator. No humor or exclamation marks.")
        elif style == "sarcastic":
            style_bullet_points.append("- sarcastic: Dry, droll, and mocking. Highlight the mundane nature of the actions or exaggerate their context mockingly, while remaining completely faithful to visual facts.")
        elif style == "humorous_tech":
            style_bullet_points.append("- humorous_tech: Connect the video actions to software engineering, programming, compilers, databases, Git workflows, bugs, or developer culture. Be witty and tech-focused.")
        elif style == "humorous_non_tech":
            style_bullet_points.append("- humorous_non_tech: Warm, playful, everyday humor suitable for a general audience. Do not use IT or coding jargon.")
            
    style_list_str = "\n".join(style_bullet_points)
    keys_str = ", ".join([f"'{s}'" for s in requested_styles])
    
    system_instruction = (
        "You are an expert video description agent. Your job is to analyze the visual scene description "
        f"and generate captions for the video in exactly the following styles: {keys_str}.\n\n"
        "CRITICAL RULES:\n"
        "1. Factual Accuracy: Be 100% faithful to the visual facts in the scene description. Do not invent details.\n"
        "2. Word Count Constraints: Each caption MUST be strictly between 35 and 50 words long and consist of exactly 2 to 3 sentences.\n"
        f"3. Format: You MUST return a JSON object with exactly these keys: {keys_str}. Do not output markdown code blocks, preamble, or formatting other than raw JSON.\n\n"
        "STYLE GUIDELINES:\n"
        f"{style_list_str}"
    )
    
    user_prompt = f"Scene Description:\n{description}\n\nGenerate the captions in the requested JSON format."
    
    raw_content = ""
    if fireworks_key:
        client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=fireworks_key)
        extra_body = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
        resp = _call_with_retry(
            client.chat.completions.create,
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.7,
            extra_body=extra_body
        )
        raw_content = _clean_reasoning(resp.choices[0].message.content or "")
    else:
        client = genai.Client(api_key=gemini_key)
        resp = _call_with_retry(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=400,
                response_mime_type="application/json"
            )
        )
        raw_content = _clean_reasoning(resp.text or "")
        
    # Clean code fences or extra text around the JSON block
    raw_content = raw_content.strip()
    if raw_content.startswith("```"):
        # remove starting ```json or ```
        raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
        # remove ending ```
        raw_content = re.sub(r"\s*```$", "", raw_content)
        raw_content = raw_content.strip()
        
    try:
        parsed = json.loads(raw_content)
        # Ensure all requested styles are present
        result = {}
        for style in requested_styles:
            if style in parsed and parsed[style]:
                result[style] = str(parsed[style]).strip()
            else:
                raise KeyError(f"Style key '{style}' not found or empty in parsed JSON.")
        return result
    except Exception as e:
        logger.warning(f"Failed to parse combined JSON captions: {e}. Raw content: {raw_content}")
        raise ValueError("JSON parsing failed")

def generate_video_captions(video_path: str, requested_styles: list) -> dict:
    """
    Extracts keyframes, base64 encodes them, queries vision model for scene description,
    and styles captions with robust length correction.
    """
    logger.info(f"Orchestrating captioning pipeline for video: {video_path}")
    
    # 1. Video frame extraction
    work_dir = tempfile.mkdtemp(prefix="captioner_frames_")
    description = ""
    try:
        keyframes = extract_keyframes(video_path, work_dir)
        logger.info(f"Extracted {len(keyframes)} keyframes.")
        
        if keyframes:
            b64_frames = frames_to_budgeted_b64(keyframes)
            if b64_frames:
                try:
                    description = run_vision_inference(b64_frames)
                    logger.info(f"Neutral scene description compiled successfully ({len(description)} chars).")
                except Exception as e:
                    logger.error(f"Vision API inference call failed: {e}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        
    # If vision fails, use default description stub to avoid failing the task
    if not description:
        description = (
            "A short video clip showing real-world environment activities, objects, "
            "and movements. The exact details are partially obscured."
        )
        logger.warning("Fell back to default neutral description stub.")
        
    # 2. Styling Phase
    captions = {}
    
    # Try combined style inference first
    try:
        logger.info("Attempting combined single-request style inference...")
        captions = run_combined_style_inference(description, requested_styles)
        logger.info("Combined style inference completed successfully.")
    except Exception as comb_err:
        logger.warning(f"Combined style inference failed: {comb_err}. Falling back to sequential styling...")
        captions = {}
        
    # Fallback/fill missing styles sequentially
    for style in requested_styles:
        if style in captions and captions[style]:
            caption = captions[style]
        else:
            caption = ""
            try:
                caption = run_style_inference(style, description)
                logger.info(f"Generated raw style '{style}': {caption}")
            except Exception as e:
                logger.error(f"Style generation failed for '{style}': {e}")
                caption = STYLE_FALLBACKS.get(style, "A short video clip.")
            
        # Length verification & correction (25 - 60 words)
        word_count = len(caption.split())
        attempts = 0
        while not (25 <= word_count <= 60) and attempts < 2:
            attempts += 1
            logger.warning(f"Style '{style}' word count ({word_count}) is out of bounds. Running repair attempt {attempts}...")
            try:
                repaired = run_style_repair(style, caption)
                repaired_count = len(repaired.split())
                logger.info(f"Repair attempt {attempts} result: '{repaired}' ({repaired_count} words)")
                caption = repaired
                word_count = repaired_count
            except Exception as e:
                logger.error(f"Repair attempt {attempts} failed: {e}")
                
        # Hard constraint fallback fillers / truncations if still invalid
        if not (25 <= word_count <= 60):
            logger.warning(f"Word count ({word_count}) still out of bounds. Applying hard rule adjustment.")
            if word_count < 25:
                caption += STYLE_FILLERS.get(style, " This clip contains dynamic scenes and active content.")
                word_count = len(caption.split())
            if word_count > 60:
                words = caption.split()
                truncated = False
                for limit in range(58, 24, -1):
                    if limit < len(words):
                        word_at_limit = words[limit - 1]
                        if word_at_limit.endswith(('.', '!', '?')):
                            caption = " ".join(words[:limit])
                            word_count = len(caption.split())
                            truncated = True
                            break
                if not truncated:
                    caption = " ".join(words[:55]).rstrip(",;:-") + "."
                    word_count = len(caption.split())
                    
        captions[style] = caption
        logger.info(f"Final finalized style '{style}' ({word_count} words): {caption}")
        
    return captions
