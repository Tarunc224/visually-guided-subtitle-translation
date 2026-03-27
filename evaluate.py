# # Evaluate all files in a directory
# python evaluate.py --input_dir ./predictions --output_csv metrics.csv

# # Evaluate only certain languages
# python evaluate.py --input_dir ./predictions --output_csv metrics.csv --languages bengali telugu

# # Use a pattern to select specific files (e.g., only baseline files)
# python evaluate.py --input_dir ./predictions --output_csv metrics_baseline.csv --pattern '*baseline*.csv'

import os
import argparse
import pandas as pd
import numpy as np
from sacrebleu import corpus_bleu, corpus_chrf
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

def compute_metrics(df, pred_col, ref_col, src_col="en_dialogue", comet_model=None):
    """Compute BLEU, chrF++, COMET for a single dataframe."""
    hyps = df[pred_col].fillna("").astype(str).tolist()
    refs = [df[ref_col].fillna("").astype(str).tolist()]
    srcs = df[src_col].fillna("").astype(str).tolist()

    bleu = corpus_bleu(hyps, refs).score
    chrf = corpus_chrf(hyps, refs, word_order=2).score

    if comet_model:
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs[0])]
        scores = comet_model.predict(data, batch_size=8, gpus=1).scores
        comet = np.mean(scores)
    else:
        comet = None
    return bleu, chrf, comet

def evaluate_directory(input_dir, output_csv, lang_suffix="actual", pred_suffix="predicted",
                       src_col="en_dialogue", pattern=None, languages=None):
    """
    Scan input_dir for CSV files, compute metrics for each.
    Assumes each file has columns: en_dialogue, predicted_{lang}, actual_{lang}.
    Extracts movie name and language from filename (e.g., Titanic_bengali_predicted.csv).
    If languages provided, only process those languages.
    Saves summary CSV with columns: movie, language, BLEU, chrF++, COMET.
    """
    # Get list of CSV files
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if pattern:
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(f, pattern)]

    # Load COMET model once
    print("Loading COMET model...")
    comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    results = []
    for filename in tqdm(files, desc="Processing files"):
        # Try to extract movie and language from filename (e.g., Titanic_bengali_predicted.csv)
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        if len(parts) < 2:
            print(f"Skipping {filename}: cannot extract movie and language")
            continue
        # Assume last part is language, rest is movie
        lang = parts[-1]
        movie = '_'.join(parts[:-1])
        if languages and lang not in languages:
            continue

        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)

        # Determine prediction and reference columns
        pred_col = f"{pred_suffix}_{lang}"
        ref_col = f"{lang_suffix}_{lang}"

        if pred_col not in df.columns or ref_col not in df.columns:
            print(f"Skipping {filename}: missing columns {pred_col} or {ref_col}")
            continue

        # Compute metrics
        bleu, chrf, comet = compute_metrics(df, pred_col, ref_col, src_col, comet_model)
        results.append({
            "movie": movie,
            "language": lang,
            "BLEU": bleu,
            "chrF++": chrf,
            "COMET": comet
        })

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Results saved to {output_csv}")
    else:
        print("No files processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate subtitle translation predictions")
    parser.add_argument("--input_dir", required=True, help="Directory containing prediction CSV files")
    parser.add_argument("--output_csv", required=True, help="Output CSV file for summary metrics")
    parser.add_argument("--lang_suffix", default="actual", help="Suffix for reference column (e.g., actual_bengali)")
    parser.add_argument("--pred_suffix", default="predicted", help="Suffix for prediction column (e.g., predicted_bengali)")
    parser.add_argument("--src_col", default="en_dialogue", help="Column name for source text")
    parser.add_argument("--pattern", default=None, help="Optional glob pattern to filter files (e.g., '*_predicted.csv')")
    parser.add_argument("--languages", nargs="+", help="List of languages to process (e.g., bengali hindi telugu)")
    args = parser.parse_args()

    evaluate_directory(
        args.input_dir,
        args.output_csv,
        args.lang_suffix,
        args.pred_suffix,
        args.src_col,
        args.pattern,
        args.languages
    )