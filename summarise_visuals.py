# # For Attribute Visual Context (5‑minute sliding window attributes)
# python summarise_visuals.py Titanic --lang bengali --method attr \
#     --visuals Titanic_descriptions.log \
#     --subtitles Titanic_aligned.csv \
#     --output Titanic_bengali_attr_context.csv

# # For Inter‑Chunk Visual Summarisation (gap‑based)
# python summarise_visuals.py Titanic --lang bengali --method gap \
#     --visuals Titanic_descriptions.log \
#     --subtitles Titanic_aligned.csv \
#     --output Titanic_bengali_gap_context.csvimport os

import argparse
import re
import pandas as pd
import torch
import csv
from transformers import pipeline
from tqdm import tqdm

# -------------------------
# Default configuration
# -------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
WINDOW_RADIUS = 150          # 2.5 minutes each side => 5 minute window
CACHE_STEP = 30              # group dialogues in 30-second blocks

def time_to_seconds(t_str):
    """Convert timeline string (HH:MM:SS,mmm --> HH:MM:SS,mmm) to start seconds."""
    try:
        start = str(t_str).split('-->')[0].split(',')[0].strip()
        parts = list(map(int, start.split(':')))
        if len(parts) == 3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
        return parts[0]*60 + parts[1]
    except:
        return 0

def get_attribute_context(descriptions, start_sec, end_sec, summarizer, target_lang):
    """Summarise a 5‑minute window into structured attributes."""
    if not descriptions:
        return "Ambient movie setting."
    sample = " ".join(descriptions[::6])   # stride sampling
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Identify these cinematic attributes to guide {target_lang} translation:
        [SETTING]: (e.g., Formal, Public, Intimate)
        [GENDER]: (Speaker/Listener gender)
        [RELATION]: (e.g., Stranger, Family, Hostile)
        [HONORIFIC]: (language‑specific, e.g., APNI/TUMI for Bengali)
        [SUMMARY]: (One sentence factual summary with emotional intent)
        Output ONLY these tags.<|eot_id|><|start_header_id|>user<|end_header_id|>
        Visual Data: {sample[:3000]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    outputs = summarizer(prompt, max_new_tokens=250, do_sample=False, pad_token_id=128001)
    res = outputs[0]["generated_text"].split("assistant")[-1].strip()
    return re.sub(r'\n', ' ', res)

def get_gap_context(descriptions, start_sec, end_sec, summarizer, target_lang):
    """Summarise the visual content between dialogue turns (gap)."""
    if not descriptions:
        return "No visual activity between dialogues."
    sample = " ".join(descriptions[::3])   # adjust sampling
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a movie analyzer. Summarize the following visual descriptions
        from {start_sec}s to {end_sec}s of the movie into 2-3 sentences.
        Focus ONLY on the current location and character actions.
        Do not use introductory filler.<|eot_id|><|start_header_id|>user<|end_header_id|>
        Visual Data: {sample[:2500]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    outputs = summarizer(prompt, max_new_tokens=80, do_sample=False, pad_token_id=128001)
    res = outputs[0]["generated_text"].split("assistant")[-1].strip()
    return res

def main():
    parser = argparse.ArgumentParser(description="Summarise visual descriptions for subtitle translation.")
    parser.add_argument("movie", help="Movie name (used for file naming)")
    parser.add_argument("--lang", default="bengali", help="Target language (bengali, hindi, telugu, tamil, kannada)")
    parser.add_argument("--method", choices=["attr", "gap"], default="attr",
                        help="Summarisation method: attr (5‑minute sliding window attributes) or gap (inter‑chunk summarisation)")
    parser.add_argument("--visuals", required=True, help="Path to visual descriptions text file (one per second, lines like '[t=Ns] description')")
    parser.add_argument("--subtitles", required=True, help="Path to aligned subtitle CSV (must contain 'en_dialogue' and '{lang}_target')")
    parser.add_argument("--output", default=None, help="Output CSV file (default: <movie>_<lang>_<method>_context.csv)")
    parser.add_argument("--window", type=int, default=WINDOW_RADIUS, help="Window radius in seconds (default: 150)")
    parser.add_argument("--cache", type=int, default=CACHE_STEP, help="Cache step in seconds (default: 30)")
    parser.add_argument("--device", default="auto", help="Device for summariser (cuda, cpu, auto)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.movie}_{args.lang}_{args.method}_context.csv"

    # Load visual descriptions
    with open(args.visuals, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    desc_by_sec = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("[t="):
            continue
        try:
            sec = int(line.split("[t=")[1].split("]")[0])
        except:
            continue
        desc = line.split("]")[-1].strip()
        desc_by_sec[sec] = desc

    # Load subtitles
    df = pd.read_csv(args.subtitles)
    src_col = "en_dialogue"
    tgt_col = f"{args.lang}_target"
    if src_col not in df.columns or tgt_col not in df.columns:
        raise ValueError(f"Subtitle CSV must contain '{src_col}' and '{tgt_col}' columns.")
    if 'timeline' not in df.columns:
        raise ValueError("Subtitle CSV must have a 'timeline' column.")
    df['start_sec'] = df['timeline'].apply(time_to_seconds)

    # Load summariser
    print(f"Loading summariser {MODEL_ID}...")
    summarizer = pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True},
        device_map=args.device
    )

    # Process dialogues with caching
    cache = {}
    results = []
    total = len(df)
    for idx, row in tqdm(df.iterrows(), total=total, desc="Processing dialogues"):
        current_sec = row['start_sec']
        start_win = max(0, current_sec - args.window)
        end_win = current_sec + args.window

        # Collect descriptions within the window
        window_descs = [desc_by_sec[t] for t in range(start_win, end_win+1) if t in desc_by_sec]

        cache_key = (current_sec // args.cache) * args.cache
        if cache_key not in cache:
            if args.method == "attr":
                summary = get_attribute_context(window_descs, start_win, end_win, summarizer, args.lang)
            else:
                summary = get_gap_context(window_descs, start_win, end_win, summarizer, args.lang)
            cache[cache_key] = summary

        active_context = cache[cache_key]
        model_input = f"[CONTEXT: {active_context}] {row[src_col]}"

        results.append({
            "timeline": row['timeline'],
            "visual_context": active_context,
            "model_input": model_input,
            "english_source": row[src_col],
            tgt_col: row[tgt_col]
        })

    # Save
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()