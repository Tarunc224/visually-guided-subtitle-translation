# python translate_visual.py --input Titanic_attr_context.csv --movie Titanic --output_dir ./translations --langs bengali telugu --test

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
    # Isolate assistant response if tags leaked
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

def generate_translation(en_dialogue, visual_context, target_lang, tokenizer, model):
    """Return translated text for given source and context."""
    prompt = f"""<|im_start|>system
You are a cinematic multimodal translator specializing in English-to-{target_lang}.
Your goal is to provide a "grounded translation" where the choice of words depends on the visual scene.

RULES:
1. GENDER: Use the Visual Context to identify speaker/listener gender.
   - Example: If a woman is being addressed, use appropriate feminine terms.
2. HONORIFICS: Determine social hierarchy from the scene (Formal vs. Informal).
3. LOOSE MEANING: Prioritize emotional intent and natural {target_lang} flow.
4. Output ONLY the translated {target_lang} dialogue text. No names, no English.<|im_end|>
<|im_start|>user
[VISUAL CONTEXT]: {visual_context}
[ENGLISH SOURCE]: "{en_dialogue}"
[TASK]: Based on the visual scene, provide the most natural {target_lang} translation.<|im_end|>
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

def translate_movie(movie_name, input_path, output_dir, languages, test_mode=False, max_rows=10):
    """Process one movie: generate translations for all specified languages."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    required = ['timeline', 'visual_context', 'en_dialogue']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns {required} in {input_path}")
    if test_mode:
        df = df.head(max_rows)
    results = []
    total = len(df)
    for idx, row in tqdm(df.iterrows(), total=total, desc=f"{movie_name}"):
        record = {
            'timeline': row['timeline'],
            'visual_context': row['visual_context'],
            'en_dialogue': row['en_dialogue'],
        }
        for lang_code, lang_name in languages.items():
            predicted = generate_translation(row['en_dialogue'], row['visual_context'], lang_name, tokenizer, model)
            record[f'predicted_{lang_code}'] = predicted
        results.append(record)
    out_df = pd.DataFrame(results)
    suffix = "_test" if test_mode else ""
    out_file = os.path.join(output_dir, f"{movie_name}_multilingual_predicted{suffix}.csv")
    out_df.to_csv(out_file, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
    print(f"Saved {movie_name} to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate subtitles using visual context with Qwen")
    parser.add_argument("--input", required=True, help="Input CSV file (must have timeline, visual_context, en_dialogue)")
    parser.add_argument("--movie", required=True, help="Movie name (used for output naming)")
    parser.add_argument("--output_dir", default="./translations", help="Output directory")
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
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    ).eval()

    translate_movie(args.movie, args.input, args.output_dir, languages, args.test, args.max_rows)