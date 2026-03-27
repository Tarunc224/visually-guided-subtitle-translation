# python selective_grounding.py \
#     --baseline Titanic_baseline.csv \
#     --visual Titanic_visual_attr.csv \
#     --output_dir selective/ \
#     --lang bengali \
#     --percentile 30 \
#     --metric cometimport os

import argparse
import numpy as np
import pandas as pd
from sacrebleu import corpus_bleu, corpus_chrf, sentence_bleu
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

def compute_segment_scores(df, hyp_col, ref_col, src_col, metric='bleu', comet_model=None):
    """
    Compute per‑segment scores for baseline.
    metric: 'bleu' or 'comet'
    If metric='comet', comet_model must be provided.
    """
    if metric == 'bleu':
        scores = []
        for _, row in df.iterrows():
            score = sentence_bleu(row[hyp_col], [row[ref_col]]).score
            scores.append(score)
        return np.array(scores)
    elif metric == 'comet':
        hyps = df[hyp_col].fillna("").astype(str).tolist()
        refs = df[ref_col].fillna("").astype(str).tolist()
        srcs = df[src_col].fillna("").astype(str).tolist()
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        scores = comet_model.predict(data, batch_size=8, gpus=1).scores
        return np.array(scores)
    else:
        raise ValueError("metric must be 'bleu' or 'comet'")

def compute_corpus_metrics(df, hyp_col, ref_col, src_col, comet_model):
    """Return (BLEU, chrF++, COMET) for given hypothesis column."""
    hyps = df[hyp_col].fillna("").astype(str).tolist()
    refs = [df[ref_col].fillna("").astype(str).tolist()]
    srcs = df[src_col].fillna("").astype(str).tolist()

    bleu = corpus_bleu(hyps, refs).score
    chrf = corpus_chrf(hyps, refs, word_order=2).score

    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs[0])]
    scores = comet_model.predict(data, batch_size=8, gpus=1).scores
    comet = np.mean(scores)
    return bleu, chrf, comet

def selective_grounding(baseline_file, visual_file, output_dir, lang, percentile=30,
                        metric='bleu', src_col='en_dialogue', ref_col_suffix='target',
                        pred_base_col='predicted', pred_vis_col='predicted'):
    """
    Perform oracle selective grounding.
    baseline_file: CSV with columns en_dialogue, predicted_{lang}, actual_{lang}
    visual_file: CSV with columns en_dialogue, predicted_{lang}
    output_dir: directory to save selective CSV and metrics
    lang: language code (e.g., 'bengali')
    percentile: percentage of worst segments to replace (e.g., 30)
    metric: ranking metric ('bleu' or 'comet')
    src_col: column name for source
    ref_col_suffix: suffix for reference column (actual_{lang})
    pred_base_col: column name for baseline predictions (default 'predicted')
    pred_vis_col: column name for visual-enhanced predictions (default 'predicted')
    """
    # Column names
    pred_base_col_lang = f"{pred_base_col}_{lang}"
    pred_vis_col_lang = f"{pred_vis_col}_{lang}"
    ref_col = f"{ref_col_suffix}_{lang}"

    # Read data
    df_base = pd.read_csv(baseline_file)
    df_vis = pd.read_csv(visual_file)

    # Merge on en_dialogue (keep rows present in both)
    df = df_base[[src_col, ref_col, pred_base_col_lang]].merge(
        df_vis[[src_col, pred_vis_col_lang]],
        on=src_col, how='inner'
    )
    if df.empty:
        raise ValueError("No common rows after merging baseline and visual files.")

    # Load COMET model if needed
    comet_model = None
    if metric == 'comet':
        print("Loading COMET model...")
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    # Compute segment scores for baseline
    print(f"Computing per‑segment {metric.upper()} scores for baseline...")
    scores = compute_segment_scores(df, pred_base_col_lang, ref_col, src_col, metric, comet_model)
    df['baseline_score'] = scores

    # Determine threshold
    threshold = np.percentile(scores, percentile)
    print(f"Threshold {metric.upper()} = {threshold:.4f} (replaces bottom {percentile}% of segments)")

    # Create selective column
    replace_mask = scores < threshold
    df['selective'] = np.where(replace_mask, df[pred_vis_col_lang], df[pred_base_col_lang])
    num_replaced = replace_mask.sum()
    total_rows = len(df)
    print(f"Replaced {num_replaced} / {total_rows} segments ({num_replaced/total_rows*100:.1f}%)")

    # Compute metrics for baseline, full visual, and selective
    if comet_model is None:
        # Load COMET model once for metrics if not already loaded
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    bleu_bl, chrf_bl, comet_bl = compute_corpus_metrics(df, pred_base_col_lang, ref_col, src_col, comet_model)
    bleu_vis, chrf_vis, comet_vis = compute_corpus_metrics(df, pred_vis_col_lang, ref_col, src_col, comet_model)
    bleu_sel, chrf_sel, comet_sel = compute_corpus_metrics(df, 'selective', ref_col, src_col, comet_model)

    print("\n=== Corpus Metrics ===")
    print(f"Baseline:      BLEU={bleu_bl:.2f}, chrF++={chrf_bl:.2f}, COMET={comet_bl:.4f}")
    print(f"Visual:        BLEU={bleu_vis:.2f}, chrF++={chrf_vis:.2f}, COMET={comet_vis:.4f}")
    print(f"Selective:     BLEU={bleu_sel:.2f}, chrF++={chrf_sel:.2f}, COMET={comet_sel:.4f}")
    print(f"Improvement (Selective - Baseline): BLEU={bleu_sel-bleu_bl:.2f}, COMET={comet_sel-comet_bl:.4f}")

    # Save selective output
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"selective_{lang}_{percentile}pct.csv")
    out_df = df[[src_col, 'selective', ref_col]].copy()
    out_df.columns = ['en_dialogue', 'selective_ground', 'actual']
    out_df.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"Selective translations saved to {out_file}")

    # Save metrics to a text file
    metrics_file = os.path.join(output_dir, f"selective_{lang}_{percentile}pct_metrics.txt")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"Selection threshold: bottom {percentile}% of baseline {metric.upper()}\n")
        f.write(f"Segments replaced: {num_replaced} / {total_rows} ({num_replaced/total_rows*100:.1f}%)\n\n")
        f.write(f"Baseline:      BLEU={bleu_bl:.2f}, chrF++={chrf_bl:.2f}, COMET={comet_bl:.4f}\n")
        f.write(f"Visual:        BLEU={bleu_vis:.2f}, chrF++={chrf_vis:.2f}, COMET={comet_vis:.4f}\n")
        f.write(f"Selective:     BLEU={bleu_sel:.2f}, chrF++={chrf_sel:.2f}, COMET={comet_sel:.4f}\n")
        f.write(f"Improvement (Selective - Baseline): BLEU={bleu_sel-bleu_bl:.2f}, COMET={comet_sel-comet_bl:.4f}\n")
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle selective grounding for subtitle translation")
    parser.add_argument("--baseline", required=True, help="Baseline translation CSV (must contain en_dialogue, predicted_{lang}, actual_{lang})")
    parser.add_argument("--visual", required=True, help="Visual-enhanced translation CSV (must contain en_dialogue, predicted_{lang})")
    parser.add_argument("--output_dir", required=True, help="Directory to save selective CSV and metrics")
    parser.add_argument("--lang", required=True, help="Language code (e.g., bengali, hindi, telugu, kannada, tamil)")
    parser.add_argument("--percentile", type=int, default=30, help="Percentage of worst segments to replace (default: 30)")
    parser.add_argument("--metric", choices=["bleu", "comet"], default="comet",
                        help="Metric to rank segments (bleu or comet). Default: comet")
    parser.add_argument("--src_col", default="en_dialogue", help="Column name for source (default: en_dialogue)")
    parser.add_argument("--ref_col_suffix", default="actual", help="Suffix for reference column (actual_{lang})")
    parser.add_argument("--pred_base_col", default="predicted", help="Column name for baseline predictions (default: predicted)")
    parser.add_argument("--pred_vis_col", default="predicted", help="Column name for visual-enhanced predictions (default: predicted)")
    args = parser.parse_args()

    selective_grounding(
        args.baseline,
        args.visual,
        args.output_dir,
        args.lang,
        args.percentile,
        args.metric,
        args.src_col,
        args.ref_col_suffix,
        args.pred_base_col,
        args.pred_vis_col
    )