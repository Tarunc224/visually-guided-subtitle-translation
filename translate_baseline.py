# # Run baseline for a single movie
# python translate_baseline.py --input Titanic_aligned.csv --movie Titanic --langs bengali hindi telugu

# # Test mode (first 10 rows)
# python translate_baseline.py --input Titanic_aligned.csv --movie Titanic --test

import os
import argparse
import pandas as pd
import torch
import re
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def clean_translation(raw_text):
    """Remove English text and meta‑commentary, keep only target language."""
    if not raw_text:
        return " "
    # Isolate assistant response
    res = raw_text.split("<|im_start|>assistant")[-1].strip() if "<|im_start|>assistant" in raw_text else raw_text.strip()
    res = res.split("assistant\n")[-1].strip() if "assistant\n" in res else res.strip()
    # Remove common English headers
    res = re.sub(r'^(Bengali|Hindi|Telugu|Kannada|Tamil|Translation|Dialogue|Result|Translation to .*?)[:：\s]*', '', res, flags=re.IGNORECASE)
    # Remove meta‑commentary in brackets
    res = re.sub(r'[\(\[].*?[\)\]]', '', res)
    # Remove any remaining English letters
    res = re.sub(r'[a-zA-Z]', '', res)
    # Normalize whitespace
    res = re.sub(r'\s+', ' ', res).replace("\n", " ").replace("\r", " ").strip()
    return res if res else " "

def generate_baseline(en_dialogue, target_lang, tokenizer, model):
    """Return baseline (text‑only) translation."""
    prompt = f"""<|im_start|>system
Translate this from English to {target_lang}.
RULES:
 - DO NOT include explanations, or English text.
<|im_end|>
<|im_start|>user
[SOURCE]: "{en_dialogue}"
[TASK]: Translate to {target_lang} dialogue.
<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.1
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_translation(full_output)

def translate_baseline(movie_name, input_path, output_dir, languages, test_mode=False, max_rows=10):
    """Process one movie: generate baseline translations for specified languages."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    required = ['timeline', 'en_dialogue']
    # Also need the actual target columns for merging; they must be present.
    actual_cols = [f"{lang}_target" for lang in languages.keys()]
    for col in actual_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV missing required column '{col}'")
    if test_mode:
        df = df.head(max_rows)
    results = []
    total = len(df)
    for idx, row in tqdm(df.iterrows(), total=total, desc=f"{movie_name}"):
        record = {
            'timeline': row['timeline'],
            'en_dialogue': row['en_dialogue'],
        }
        for lang_code, lang_name in languages.items():
            pred = generate_baseline(row['en_dialogue'], lang_name, tokenizer, model)
            record[f'predicted_{lang_code}'] = pred
            record[f'actual_{lang_code}'] = row[f'{lang_code}_target']
        results.append(record)
    out_df = pd.DataFrame(results)
    suffix = "_test" if test_mode else ""
    out_file = os.path.join(output_dir, f"{movie_name}_baseline_predicted{suffix}.csv")
    out_df.to_csv(out_file, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
    print(f"Saved {movie_name} to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline subtitle translation with Qwen (text‑only)")
    parser.add_argument("--input", required=True, help="Input CSV file (must have timeline, en_dialogue, and {lang}_target columns)")
    parser.add_argument("--movie", required=True, help="Movie name (used for output naming)")
    parser.add_argument("--output_dir", default="./baseline_translations", help="Output directory")
    parser.add_argument("--langs", nargs="+", default=["bengali", "hindi", "telugu", "kannada", "tamil"],
                        help="List of languages to translate (bengali, hindi, telugu, kannada, tamil)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (only first MAX_ROWS rows)")
    parser.add_argument("--max_rows", type=int, default=10, help="Number of rows for test mode")
    args = parser.parse_args()

    lang_map = {
        "bengali": "Bengali",
        "hindi": "Hindi",
        "telugu": "Telugu",
        "kannada": "Kannada",
        "tamil": "Tamil"
    }
    languages = {code: lang_map[code] for code in args.langs if code in lang_map}

    # Load model once
    print("Loading Qwen model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    ).eval()

    translate_baseline(args.movie, args.input, args.output_dir, languages, args.test, args.max_rows)