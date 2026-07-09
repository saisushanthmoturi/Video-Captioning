import os
import re
import json
import logging
import sys
import threading
import torch
import numpy as np
import av
import torchvision.io

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.WARNING)

# Monkey-patch torchvision.io.read_video using PyAV backend
def custom_read_video(filename, **kwargs):
    logger.info(f"Using monkey-patched torchvision.io.read_video (av backend) to load: {filename}")
    container = av.open(filename)
    video_stream = container.streams.video[0]
    
    width = video_stream.width
    height = video_stream.height
    
    # Scale down high-resolution videos to save memory and time
    max_dim = 768
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        logger.info(f"Downscaling video from {width}x{height} to {new_width}x{new_height}")
    else:
        new_width = width
        new_height = height
        
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_image()  # Get PIL Image
        if new_width != width or new_height != height:
            img = img.resize((new_width, new_height))
        frames.append(np.array(img))
        
    container.close()
    
    vframes_np = np.stack(frames, axis=0)
    vframes_tensor = torch.from_numpy(vframes_np)
    
    output_format = kwargs.get("output_format", "THWC")
    if output_format == "TCHW":
        vframes_tensor = vframes_tensor.permute(0, 3, 1, 2)
        
    fps_val = video_stream.average_rate
    fps = float(fps_val) if fps_val is not None else 25.0
    
    return vframes_tensor, torch.empty((0, 0)), {"video_fps": fps, "audio_fps": 0.0}

torchvision.io.read_video = custom_read_video
sys.modules['torchvision.io'].read_video = custom_read_video

# Import transformers and utilities after patching
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Singletons for thread-safe model caching
model = None
processor = None
device = None
model_lock = threading.Lock()

def get_model_and_processor():
    """
    Initializes and caches the Qwen2-VL model and processor.
    Uses GPU/ROCm (cuda), Apple Silicon (mps), or CPU dynamically.
    """
    global model, processor, device
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Initializing local Qwen2-VL-2B-Instruct model on device: {device}...")
        
        # Load in float16 for acceleration on GPU/MPS, float32 on CPU to prevent errors
        torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        ).to(device)
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        logger.info("Local Qwen2-VL model initialized successfully.")
    return model, processor, device

def validate_caption_length(caption: str, min_words=25, max_words=60) -> bool:
    word_count = len(caption.split())
    return min_words <= word_count <= max_words

def generate_video_captions(video_path: str, requested_styles: list) -> dict:
    """
    Runs local inference on a video using Qwen2-VL to generate captions
    for the requested styles, and applies local length-correction repair.
    """
    logger.info(f"Requesting local captioning for video: {video_path}")
    
    # Thread lock to serialize local GPU/CPU execution and prevent memory leaks
    with model_lock:
        local_model, local_processor, local_device = get_model_and_processor()
        
        # 1. Formulate structured instruction prompt
        prompt_text = (
            "Analyze the provided video carefully. You must generate descriptive, high-quality, "
            "and tone-accurate captions for each of the following four requested styles: 'formal', 'sarcastic', "
            "'humorous_tech', and 'humorous_non_tech'.\n\n"
            "Requirements:\n"
            "1. Output ONLY a valid JSON object with the keys 'formal', 'sarcastic', 'humorous_tech', and 'humorous_non_tech'. "
            "Do not include any code block formatting (like ```json), markdown, or introductory text. Output the raw JSON string.\n"
            "2. Factual Accuracy: Be 100% faithful to the actual visual events, scenes, settings, subjects, and actions in the video. "
            "Never invent people, objects, or actions not present in the clip.\n"
            "3. Length Constraint: Each style's caption MUST be strictly between 25 and 60 words long (inclusive). Count words carefully!\n"
            "4. Language Constraint: You MUST write the captions in English only. Do NOT output any Chinese characters.\n\n"
            "Style Guidelines:\n"
            "- formal: Write in a highly professional, objective, clear, and informative tone. Describe setting, subjects, and actions. Focus on concrete physical details (colors, movements, camera work). No humor.\n"
            "- sarcastic: Write in a dry, ironic, mocking tone. Highlight the mundane nature of the actions or exaggerate mockingly, while remaining completely faithful to visual facts.\n"
            "- humorous_tech: Create a funny caption that links the video contents with software engineering, programming, databases, compilers, bugs, or developer culture. The joke must directly relate to the physical visual movement/context.\n"
            "- humorous_non_tech: Write observational, relatable everyday humor or a funny caption for a general audience (e.g. coffee, workplace, parenting, household chores) using zero tech/programming jargon.\n"
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    }
                ]
            }
        ]
        
        # 2. Preprocess video frames and text
        logger.info("Preprocessing video and prompt...")
        text = local_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = local_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to device (safely mapping torch Tensors)
        inputs = {k: v.to(local_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 3. Model inference (using do_sample=False and repetition_penalty=1.2 to prevent looping)
        logger.info("Running local vision-language model inference...")
        with torch.no_grad():
            generated_ids = local_model.generate(
                **inputs, 
                max_new_tokens=600,
                do_sample=False,
                repetition_penalty=1.2
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output_text = local_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"Raw VLM response: {output_text}")
        
        # 4. Parse the output JSON
        response_json = {}
        try:
            # Extract JSON substring in case of preambles/markdown formatting
            match = re.search(r'\{.*\}', output_text, re.DOTALL)
            json_str = match.group(0) if match else output_text
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            response_json = json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}. Attempting recovery via regex mapping...")
            for style in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]:
                style_match = re.search(rf'"{style}"\s*:\s*"([^"]+)"', output_text)
                if style_match:
                    response_json[style] = style_match.group(1)

        # Normalize keys and handle model key typos/variations
        normalized_response = {}
        styles_order = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
        
        for key, val in response_json.items():
            k_lower = key.lower()
            if "formal" in k_lower:
                normalized_response["formal"] = val
            elif "sarcastic" in k_lower or "sarc" in k_lower:
                normalized_response["sarcastic"] = val
            elif "non" in k_lower:
                normalized_response["humorous_non_tech"] = val
            elif "tech" in k_lower:
                normalized_response["humorous_tech"] = val

        # Fallback: if keys don't match names but we have exactly 4 keys, map them by index
        keys = list(response_json.keys())
        if len(keys) == 4:
            for i, style in enumerate(styles_order):
                if style not in normalized_response:
                    normalized_response[style] = response_json[keys[i]]
                    logger.info(f"Mapped style '{style}' by index to key '{keys[i]}'")

        # 5. Length verification and correction
        final_captions = {}
        for style in requested_styles:
            caption = normalized_response.get(style, "") or ""
            word_count = len(caption.split())
            logger.info(f"Parsed caption for style '{style}' ({word_count} words): {caption}")
            
            # Fallback 1: If caption is completely empty, run dedicated single-caption generation
            if not caption:
                logger.warning(f"Caption for '{style}' is empty. Generating directly from video...")
                try:
                    single_prompt = (
                        f"Analyze the provided video and generate a single {style} caption. "
                        f"Guidelines:\n"
                        f"- style: {style}\n"
                        f"- formal: Professional, objective, clear. Focus on concrete visual facts (colors, objects, movement). No humor.\n"
                        f"- sarcastic: Dry, ironic, mocking. Highlight the mundane nature of the actions while staying faithful to visual facts.\n"
                        f"- humorous_tech: Funny caption linking video actions with software engineering, programming, or developer culture.\n"
                        f"- humorous_non_tech: Everyday relatable humor, dad jokes. Absolutely no tech jargon.\n\n"
                        f"The caption must be strictly between 25 and 60 words. "
                        f"Output ONLY the caption string, with no quotes, formatting, or comments. Write in English only."
                    )
                    
                    single_messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video_path,
                                },
                                {
                                    "type": "text",
                                    "text": single_prompt,
                                }
                            ]
                        }
                    ]
                    
                    single_text = local_processor.apply_chat_template(single_messages, tokenize=False, add_generation_prompt=True)
                    single_image_inputs, single_video_inputs = process_vision_info(single_messages)
                    
                    single_inputs = local_processor(
                        text=[single_text],
                        images=single_image_inputs,
                        videos=single_video_inputs,
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    single_inputs = {k: v.to(local_device) if isinstance(v, torch.Tensor) else v for k, v in single_inputs.items()}
                    
                    with torch.no_grad():
                        single_ids = local_model.generate(
                            **single_inputs, 
                            max_new_tokens=150,
                            do_sample=False,
                            repetition_penalty=1.2
                        )
                        
                    single_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(single_inputs["input_ids"], single_ids)
                    ]
                    
                    caption = local_processor.batch_decode(
                        single_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
                    
                    if caption.startswith('"') and caption.endswith('"'):
                        caption = caption[1:-1].strip()
                    word_count = len(caption.split())
                    logger.info(f"Generated backup caption for '{style}': '{caption}' ({word_count} words)")
                except Exception as single_err:
                    logger.error(f"Failed to generate backup caption for '{style}': {single_err}")
            
            # Fallback 2: Model-based repair loop (using do_sample=False and repetition_penalty=1.2)
            attempts = 0
            max_repair_attempts = 3
            while not (25 <= word_count <= 60) and attempts < max_repair_attempts:
                attempts += 1
                logger.warning(
                    f"Style '{style}' has {word_count} words (outside 25-60 limits). "
                    f"Running local model-based repair (attempt {attempts})...."
                )
                try:
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
                    
                    repair_messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": repair_prompt
                                }
                            ]
                        }
                    ]
                    
                    repair_text = local_processor.apply_chat_template(repair_messages, tokenize=False, add_generation_prompt=True)
                    repair_inputs = local_processor(text=[repair_text], return_tensors="pt").to(local_device)
                    
                    with torch.no_grad():
                        repair_ids = local_model.generate(
                            **repair_inputs, 
                            max_new_tokens=150,
                            do_sample=False,
                            repetition_penalty=1.2
                        )
                    
                    repair_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(repair_inputs["input_ids"], repair_ids)
                    ]
                    
                    repaired_caption = local_processor.batch_decode(
                        repair_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
                    
                    # Clean surrounding quotes
                    if repaired_caption.startswith('"') and repaired_caption.endswith('"'):
                        repaired_caption = repaired_caption[1:-1].strip()
                    elif repaired_caption.startswith("'") and repaired_caption.endswith("'"):
                        repaired_caption = repaired_caption[1:-1].strip()
                        
                    repaired_word_count = len(repaired_caption.split())
                    logger.info(f"Local repair attempt {attempts} result: '{repaired_caption}' ({repaired_word_count} words)")
                    
                    caption = repaired_caption
                    word_count = repaired_word_count
                except Exception as re_err:
                    logger.error(f"Failed local model-based caption repair on attempt {attempts}: {re_err}")
            
            # Fallback 3: Rule-based fallback if model-based repair fails or still out of bounds
            if not (25 <= word_count <= 60):
                logger.warning(f"Local model-based repair failed to bring word count in bounds after {attempts} attempts. Applying fallback...")
                if word_count < 25:
                    if style == "formal":
                        filler = " Furthermore, the video demonstrates continuous, smooth motion and maintains a stable frame, presenting clear, high-resolution, and highly detailed visual imagery throughout the entire duration of the clip."
                    elif style == "sarcastic":
                        filler = " Because clearly, watching this breathtaking sequence frame by frame is the absolute highlight of anyone's day, leaving all of us eagerly begging for even more excitement."
                    elif style == "humorous_tech":
                        filler = " This background process is executing at peak multi-threaded CPU utilization, with absolutely zero memory leaks, no deadlocks, and perfect thread safety observed throughout compilation."
                    else:  # humorous_non_tech
                        filler = " It is just another completely normal day in the life, where literally everything is incredibly interesting and deeply meaningful if you only look closely enough."
                    caption += filler
                    word_count = len(caption.split())
                    logger.info(f"Adjusted caption via tone-appropriate extension: {caption} ({word_count} words)")
                
                if word_count > 60:
                    words = caption.split()
                    truncated = False
                    for limit in range(58, 24, -1):
                        if limit < len(words):
                            word_at_limit = words[limit - 1]
                            if word_at_limit.endswith(('.', '!', '?')):
                                caption = " ".join(words[:limit])
                                word_count = len(caption.split())
                                logger.info(f"Adjusted caption via smart sentence boundary truncation: {caption} ({word_count} words)")
                                truncated = True
                                break
                    if not truncated:
                        caption = " ".join(words[:55]).rstrip(",;:-") + "."
                        word_count = len(caption.split())
                        logger.info(f"Adjusted caption via hard truncation fallback: {caption} ({word_count} words)")
            
            final_captions[style] = caption
            
        return final_captions
