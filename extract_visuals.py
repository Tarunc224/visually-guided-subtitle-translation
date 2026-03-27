import os
import sys
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  

# -------------------------
# Model configuration
# -------------------------
FASTVLM_MODEL_ID = "apple/FastVLM-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_TOKEN_INDEX = -200
GEN_MAX_LENGTH = 128
GEN_NUM_BEAMS = 4

FRAME_PROMPT = (
    "You are analyzing a single video frame. Focus only on what is visually present avoid assumptions or repetitive phrases. "
    "Provide a concise, factual analysis of the visual context without adding extra commentary or redundant descriptions.\n\n"
    "Tasks:\n"
    "1. Describe only the key visual elements (scene type, subjects, actions, setting, and mood).\n"
    "2. Limit the description strictly to 1–2 sentences (maximum 512 tokens total).\n\n"
    "Do not add anything beyond this format (no introductions, explanations, or extra text)."
)


def load_model(model_id: str):
    """Load FastVLM model and tokenizer."""
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def describe_frame(tokenizer, model, frame_rgb_array):
    """Run FastVLM on a single frame and return its textual description."""
    pil_image = Image.fromarray(frame_rgb_array.astype('uint8'), 'RGB')
    messages = [{"role": "user", "content": f"<|image|>\n{FRAME_PROMPT}"}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    pre, post = rendered.split("<|image|>", 1)
    pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    px = model.get_vision_tower().image_processor(images=pil_image, return_tensors="pt")["pixel_values"]
    px = px.to(model.device, dtype=model.dtype)

    with torch.no_grad():
        out = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=px,
            max_new_tokens=GEN_MAX_LENGTH,
            num_beams=GEN_NUM_BEAMS,
            do_sample=False,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    # Clean up
    if "assistant" in full:
        full = full.split("assistant")[-1].strip()
    full = full.replace("The image captured", "").strip()
    return full


def extract_descriptions(video_path, output_file, start_sec=0, end_sec=None, sample_rate=1):
    """Extract visual descriptions at sample_rate fps and save to output_file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(total_frames / fps)

    if end_sec is None:
        end_sec = duration_sec

    start_sec = max(0, start_sec)
    end_sec = min(duration_sec, end_sec)

    print(f"Video duration: {duration_sec}s, processing seconds {start_sec} to {end_sec} at {sample_rate} fps")

    tokenizer, model = load_model(FASTVLM_MODEL_ID)

    # Open output file in append mode if resuming
    mode = "a" if start_sec > 0 else "w"
    with open(output_file, mode, encoding="utf-8") as out_f:
        # Optionally use tqdm if available
        try:
            from tqdm import tqdm
            iterator = tqdm(range(start_sec, end_sec), desc="Extracting frames")
        except ImportError:
            iterator = range(start_sec, end_sec)

        for sec in iterator:
            frame_idx = int(sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                desc = describe_frame(tokenizer, model, frame_rgb)
            except Exception as e:
                print(f"Error at second {sec}: {e}", file=sys.stderr)
                desc = "Processing Error"

            out_f.write(f"[t={sec}s] {desc}\n")
            if sec % 50 == 0 and not isinstance(iterator, tqdm):
                print(f"Processed {sec}s / {end_sec}s")

    cap.release()
    print(f"Finished. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract visual descriptions from a movie using Apple FastVLM")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: <video_basename>_descriptions.log)")
    parser.add_argument("--start", type=int, default=0, help="Start second (resume)")
    parser.add_argument("--end", type=int, default=None, help="End second")
    parser.add_argument("--sample-rate", type=int, default=1, help="Frames per second (default: 1)")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{base}_descriptions.log"

    extract_descriptions(args.video, args.output, args.start, args.end, args.sample_rate)