# Melody Match — Music Similarity using MERT

> **AAI-590 Applied AI Capstone · University of San Diego · Spring 2026**
> Team 3: Manikandan Perumal · Darin Verduzco · Israel Romero

---

## Documentation

Full project documentation is available at:
**[https://mperumal-usd.github.io/capstone_team_3/](https://mperumal-usd.github.io/capstone_team_3/)**

---

## Overview

**Melody Match** is a deep-learning system for classical music similarity search. Given an audio query, it retrieves the most similar pieces from a classical song corpus using semantic embeddings — not rule-based matching.

The system fine-tunes **MERT** (Music Encoder Representations from Transformers, `m-a-p/MERT-v1-95M`), a 95M-parameter audio foundation model pre-trained on large-scale music data. Fine-tuning uses LoRA adapters and a TripletMarginLoss objective. Embeddings are indexed with **FAISS** for fast nearest-neighbour retrieval.

---

## Results

| Model | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|-------|:--------:|:--------:|:--------:|:---------:|
| **MERT + LoRA + CNN (v5)** | **63.6%** | **73.6%** | **77.9%** | **83.3%** |
| Baseline MERT (no fine-tuning) | 45.5% | 55.6% | 60.7% | 67.9% |

Fine-tuning improved Recall@1 by **+18 percentage points** over the frozen baseline.

---

## Pipeline

```
590 Classical MIDI Files
        │  FluidSynth synthesis + resample 24 kHz
        ▼
    WAV chunks (7 sec, 43,663 total)
        │  MERT-v1-95M encoder
        │  + LoRA adapters (r=8, α=16)
        │  + CNN projection head → 128-dim L2-norm
        ▼
    Embeddings
        │  TripletMarginLoss fine-tuning (23,183 triplets)
        ▼
    FAISS IndexFlatL2
        │  Recall@K evaluation
        ▼
    Recall@1 = 63.6% | Recall@10 = 83.3%
```

---

## Code Organisation

```
capstone_team_3/
│
├── notebooks/                          # All experiment notebooks (Google Colab)
│   │
│   ├── ── Data ──
│   ├── DataExploration.ipynb           # EDA: note density, tempo, duration distributions
│   ├── DataExploration_v1.ipynb
│   ├── DataCleaning.ipynb              # Remove corrupt MIDIs, deduplicate
│   ├── FeatureEngineering.ipynb        # Explore piano-roll, chroma, MFCC features
│   │
│   ├── ── Primary Model (MERT + LoRA) ──
│   ├── COLAB_MERT_Finetune_v5.ipynb   # ★ FINAL MODEL — LoRA + CNN head, 10 epochs
│   ├── COLAB_MERT_Finetune_v4.ipynb   # Optimizer tuning
│   ├── COLAB_MERT_Finetune_v3.ipynb   # Added FAISS validation
│   ├── COLAB_MERT_Finetune_v2.ipynb   # Updated data loading
│   ├── COLAB_MERT_Finetune_v1.ipynb   # Initial LoRA setup
│   │
│   ├── ── Evaluation ──
│   ├── COLAB_MERT_FAISS_Recall_Eval.ipynb  # Multi-checkpoint Recall@K comparison
│   ├── mert_evaluation.ipynb           # Ad-hoc embedding quality checks
│   ├── mert_finetuned.ipynb
│   ├── Similarity_Score_EDA.ipynb      # Cosine similarity distributions
│   ├── Similarity_Score_EDA-MERT.ipynb
│   ├── COLAB_Similarity_Score_EDA_MERT.ipynb
│   ├── ResultsAnalysis.ipynb           # LSTM/GRU/Transformer prediction analysis
│   │
│   ├── ── Baseline Models ──
│   ├── LSTM_Model_v1.ipynb             # Sequence model on MIDI tokens
│   ├── GRU_Model_v1.ipynb              # GRU baseline
│   ├── Transformer_Model_v1.ipynb      # Custom transformer (trained from scratch)
│   ├── SiameseCNN_Model_v1.ipynb       # Twin-network CNN with contrastive loss
│   ├── COLAB_CNN_MEL_Similarity.ipynb  # 2D CNN on mel spectrograms
│   │
│   ├── ── Dataset & Embeddings ──
│   ├── triplet_dataset_040326.ipynb    # Anchor/positive/negative triplet generation
│   ├── Embeddings_Generation.ipynb     # Batch embedding extraction & caching
│   ├── COLAB_Piano_Roll_test.ipynb     # Piano roll visualisation
│   └── recordings/                     # Audio recordings (WAV) for testing
│
├── MidiDatasets/                       # Music corpus
│   ├── 590-Classical-music-midi/       # Primary dataset — 590 classical MIDI files
│   │   ├── albeniz/
│   │   ├── bach/
│   │   ├── beethoven/
│   │   └── ...                         # ~30 composers
│   ├── TestingSamples/                 # External test set
│   │   ├── MidiOutputs/                # Test MIDI files
│   │   ├── TestChunks/                 # Synthesised WAV chunks
│   │   └── TestingReferences.csv       # Ground-truth label mapping
│   └── OtherDatasets/                  # Supplementary sources
│
├── FrontEnd/                           # Browser-based MIDI tools (no server required)
│   ├── MidiAnalyzer.html               # MIDI player + piano roll visualiser (Tone.js)
│   └── MidiComparator.html             # Side-by-side MIDI similarity analyser (Chart.js)
│
├── website/                            # Docusaurus documentation site
│   ├── docs/                           # Page content (Markdown/MDX)
│   ├── static/                         # Images, static assets, HTML tools
│   └── docusaurus.config.js
│
├── .gitignore
└── ReadMe.md
```

---

## Model Architecture

### MERT-v1-95M

[MERT](https://arxiv.org/abs/2306.00107) is a transformer pre-trained on large-scale music audio using masked language modelling. It takes raw waveforms at 24 kHz and produces contextual 768-dim embeddings.

### LoRA Fine-tuning

Low-Rank Adaptation inserts trainable rank-8 matrices into all attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Only **442K parameters** (~0.47% of the model) are trained.

```python
LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
           target_modules=["q_proj","k_proj","v_proj","o_proj"])
```

### CNN Projection Head

Replaces mean-pooling with multi-scale temporal aggregation. Three parallel `Conv1d` branches (kernels 3, 5, 7) capture local and broader patterns, then project to a 128-dim L2-normalised embedding.

### Training

| Parameter | Value |
|-----------|-------|
| Loss | TripletMarginLoss (margin=0.3) |
| Optimizer | AdamW, lr=5e-5 |
| Scheduler | CosineAnnealingLR |
| Epochs | 10 |
| Batch size | 8 |
| Mixed precision | AMP (float16) |
| Best val loss | 0.0302 (epoch 9) |

---

## Dataset

- ** classical MIDI files** covering ~30 composers (Bach, Beethoven, Chopin, Liszt, Mozart, Schubert, …)
- Synthesised to WAV using FluidSynth + FluidR3 GM soundfont, resampled to 24 kHz
- Split into **7-second non-overlapping chunks** → 43,663 gallery chunks
- **23,183 triplets** generated for fine-tuning (80/20 train/val split, stratified by composer)

---

## Quickstart

### Run the primary model (Google Colab)

Open the final fine-tuning notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mperumal-usd/capstone_team_3/blob/main/notebooks/COLAB_MERT_Finetune_v5.ipynb)

All sections are **resume-safe** — they check Google Drive for existing checkpoints and skip completed steps automatically.

### Run the MIDI browser tools

Open `FrontEnd/MidiAnalyzer.html` or `FrontEnd/MidiComparator.html` directly in any modern browser — no server or installation required.

### Install dependencies (for local notebooks)

```bash
pip install transformers peft accelerate librosa numpy pandas scikit-learn tqdm faiss-cpu
```

---

## Why MERT over Traditional Methods

| Capability | Edit Distance | DTW | CNN (Mel) | MERT + LoRA |
|-----------|:---:|:---:|:---:|:---:|
| Key invariance | ✗ | ✗ | Partial | ✓ |
| Tempo invariance | ✗ | ✓ | Partial | ✓ |
| Semantic understanding | ✗ | ✗ | ✗ | ✓ |
| Pre-trained on music | — | — | ✗ | ✓ |
| Scalable retrieval (FAISS) | ✓ | ✗ | ✓ | ✓ |


---

## References

- Li, Y. et al. (2023). [MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training](https://arxiv.org/abs/2306.00107)
- Hu, E. et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Johnson, J. et al. (2019). [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734)
