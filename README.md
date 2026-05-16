# Visually-Guided Subtitle Translation for Indic Languages

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains the code and data for the EAMT 2026 paper:  
**"Towards Visually-Guided Movie Subtitle Translation for Indic Languages"**.

We translate **English movie subtitles** into five low‑resource Indian languages (Bengali, Hindi, Telugu, Kannada, Tamil) by leveraging **visual context** from video frames. Two lightweight grounding strategies are compared, and **oracle selective grounding** (replacing the worst 20–30% of text‑only segments) consistently improves COMET scores while using only a fraction of the visual processing.

---

## 🎯 Key Features

- **Visual description extraction** using **FastVLM** (frame‑by‑frame, 1 fps)
- **Two summarisation methods**:
  - *Attribute Visual Context (Attr‑VC)* – 5‑minute sliding window summarised into structured tags (setting, gender, honorifics, emotion)
  - *Inter‑Chunk Visual Summarisation (Inter‑VS)* – free‑text summary of visual gaps between dialogue turns
- **Zero‑shot translation** with **Qwen‑2.5‑7B‑Instruct** (text‑only vs. visually‑grounded)
- **Selective grounding** – replace only the worst 20‑30% of baseline segments (oracle study)
- **Evaluation** – BLEU, chrF++, COMET (wmt22‑comet‑da) on 5 full‑length movies × 5 languages

---

## 📊 Pipeline Overview

![Architecture](./images/architecture.png)

*Figure: Multimodal subtitle translation pipeline. Visual frames are described, summarised, and then passed together with the source subtitle to the translation model.*

---

## 🛠️ Prerequisites

- Python 3.10 or higher
- CUDA 12+ (recommended, with ≥16 GB VRAM for Qwen‑2.5‑7B and Llama‑3.1‑8B)
- Install dependencies:
  ```bash
  pip install -r requirements.txt

**Note**: First run downloads models (~30GB total).

## 🚀 Quick Start (Full Pipeline)
Process **Titanic** to **Bengali**:

```bash
# 1. Extract visuals (1fps, ~2hr movie takes 30-60min GPU)
python extract_visuals.py titanic.mp4 -o titanic_descriptions.log

# 2. Baseline (text-only)
python translate_baseline.py data/sample/Bengali/Titanic_en_bn_corpus.csv --movie Titanic --langs bengali --output_dir baseline/

# 3. Visual context (attribute method)
python summarise_visuals.py Titanic --lang bengali --method attr --visuals titanic_descriptions.log --subtitles data/sample/Bengali/Titanic_en_bn_corpus.csv

# 4. Visual translation
python translate_visual.py Titanic_bengali_attr_context.csv --movie Titanic --langs bengali --output_dir visual/

# 5. Selective grounding (30% worst replaced)
python selective_grounding.py baseline/Titanic_baseline_predicted.csv visual/Titanic_multilingual_predicted.csv --output_dir selective/ --lang bengali --percentile 30

# 6. Evaluate
python evaluate.py --input_dir selective/ --output_csv metrics.csv --languages bengali
```

## 🧪 Individual Scripts

### 1. `extract_visuals.py`
```bash
python extract_visuals.py movie.mp4 --output movie_descriptions.log --sample-rate 1
```

### 2. `summarise_visuals.py`
Two modes:
```bash
# Attributes (gender, setting, honorifics)
python summarise_visuals.py Movie --lang bengali --method attr --visuals movie_descriptions.log --subtitles data/sample/Bengali/Movie_en_bn_corpus.csv

# Gap summaries (inter-dialogue)
python summarise_visuals.py Movie --lang bengali --method gap --visuals movie_descriptions.log --subtitles data/sample/Bengali/Movie_en_bn_corpus.csv
```

### 3. `translate_baseline.py` / `translate_visual.py`
Multi-language:
```bash
python translate_visual.py input_context.csv --movie Movie --langs bengali telugu hindi --test  # First 10 rows
```

### 4. `selective_grounding.py`
```bash
python selective_grounding.py baseline.csv visual.csv --output_dir selective/ --lang bengali --percentile 30 --metric comet
```

### 5. `evaluate.py`
Batch:
```bash
python evaluate.py --input_dir predictions/ --output_csv metrics.csv --pattern '*predicted*.csv'
```

## 📁 Data Format
**Input Corpus** (`data/sample/Bengali/Titanic_en_bn_corpus.csv`):
```
timeline,en_dialogue,bengali_target
00:00:01,000 --> 00:00:02,000,\"I'm the king of the world!\",\"আমি বিশ্বের রাজা!\"
```

**Predictions**:
```
timeline,en_dialogue,predicted_bengali,actual_bengali
...
```


## 📚 Citation

If you use this code, dataset, or build upon our work, please cite:

```bibtex

@article{chintada2026towards,
  title={Towards Visually-Guided Movie Subtitle Translation for Indic Languages},
  author={Chintada, Tarun and Singh, Kshetrimayum Boynao and Ekbal, Asif},
  journal={arXiv preprint arXiv:2605.11993},
  year={2026}
}
```

## ⚠️ Limitations
- Requires aligned timestamps + reference translations.
- Heavy compute (GPU essential for production).
- Tested on Hollywood movies; domain-specific fine-tuning advised.

## 🤝 Contributing
1. Fork & PR improvements (e.g., new languages, models).
2. Add your metrics to the results table!

## 📄 License
MIT License - see [LICENSE](LICENSE)
