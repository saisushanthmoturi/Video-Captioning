import os
import json
import sys
import logging
import tempfile
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from captioner import generate_video_captions

# Setup logging (directing to stderr so stdout is kept entirely clean)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def download_video(url: str, dest_path: str):
    """
    Downloads a video clip from url to dest_path with progress bar.
    """
    logger.info(f"Downloading video from {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest_path, 'wb') as file, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path), file=sys.stderr
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)
    logger.info("Download completed.")

def process_single_task(task):
    """
    Worker function to process a single video captioning task.
    """
    task_id = task.get("task_id")
    video_url = task.get("video_url")
    styles = task.get("styles", ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"])
    
    logger.info(f"--- Starting Task: ID={task_id} ---")
    
    if not task_id or not video_url:
        logger.error("Skipping invalid task (missing task_id or video_url).")
        return {
            "task_id": task_id or "unknown",
            "captions": {style: "Invalid task parameters." for style in styles}
        }

    temp_video_path = f"temp_video_{task_id}.mp4"
    downloaded = False

    try:
        # 1. Download or locate video
        if video_url.startswith("http://") or video_url.startswith("https://"):
            download_video(video_url, temp_video_path)
            downloaded = True
            video_to_process = temp_video_path
        else:
            # Local video file for debugging
            if os.path.exists(video_url):
                video_to_process = video_url
            else:
                raise FileNotFoundError(f"Local video path {video_url} not found.")

        # 2. Generate captions using dual-API pipeline
        captions = generate_video_captions(video_to_process, styles)
        
        logger.info(f"Successfully captioned task {task_id}")
        return {
            "task_id": task_id,
            "captions": captions
        }

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        # Output fallback captions structure for this task on failure to preserve format
        from styles import STYLE_FALLBACKS
        return {
            "task_id": task_id,
            "captions": {style: STYLE_FALLBACKS.get(style, "A short video clip.") for style in styles}
        }

    finally:
        # Clean up downloaded video file
        if downloaded and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.info(f"Cleaned up temporary video file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {temp_video_path}: {e}")

def write_results_atomic(output_file: str, results: list):
    """
    Writes the results JSON file atomically to prevent partial writes and encoding issues.
    """
    output_dir = os.path.dirname(output_file) or "."
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # ensure_ascii=True escapes non-ASCII characters to keep encoding clean of mojibake
    payload = json.dumps(results, ensure_ascii=True, indent=2)
    # Write temp file in the same directory as final output for atomic single-mount rename
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_file)
        logger.info(f"Results successfully written to {output_file}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def main():
    # Load input and output paths from environment variables
    input_file = os.getenv("INPUT_PATH", "/input/tasks.json")
    output_file = os.getenv("OUTPUT_PATH", "/output/results.json")
    
    # Fallback to local files for testing if mount paths do not exist
    if not os.path.exists(input_file) and input_file == "/input/tasks.json":
        input_file = "./input/tasks.json"
        
    output_dir = os.path.dirname(output_file) or "."
    if not os.path.exists(output_dir) and output_file == "/output/results.json":
        output_file = "./output/results.json"

    logger.info(f"Starting Video Captioning Agent...")
    logger.info(f"Input file path: {input_file}")
    logger.info(f"Output file path: {output_file}")

    if not os.path.exists(input_file):
        logger.error(f"Input tasks file not found at {input_file}")
        sys.exit(1)

    try:
        with open(input_file, 'r') as f:
            tasks = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse input tasks JSON: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(tasks)} task(s).")
    
    # Process tasks concurrently using a ThreadPoolExecutor
    max_workers = int(os.environ.get("CONCURRENT_WORKERS", "3"))
    logger.info(f"Configuring concurrent execution with max_workers={max_workers}")
    
    from concurrent.futures import ThreadPoolExecutor
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_task, task) for task in tasks]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as fut_err:
                logger.error(f"Worker thread execution failed: {fut_err}")

    # Write output JSON atomically
    try:
        write_results_atomic(output_file, results)
    except Exception as e:
        logger.error(f"Failed to write results file: {e}")
        sys.exit(1)

    logger.info("Video Captioning Agent finished execution successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
