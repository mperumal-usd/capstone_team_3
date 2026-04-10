---
id: notebooks
title: Notebooks
sidebar_label: Notebooks
---

# Notebooks

All experiments are implemented as Google Colab notebooks. Click the badge to open any notebook directly in Colab.

---

## Primary Model

### MERT Fine-Tuning v5 *(Final)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mperumal-usd/capstone_team_3/blob/main/notebooks/COLAB_MERT_Finetune_v5.ipynb)

**`notebooks/COLAB_MERT_Finetune_v5.ipynb`**

The definitive fine-tuning notebook. Trains `m-a-p/MERT-v1-95M` with LoRA adapters and a CNN projection head using TripletMarginLoss over 10 epochs. Features:

- Resume-safe training (saves every optimizer step to Drive)
- Mixed precision (AMP) training on GPU
- Intra-epoch checkpointing
- Full evaluation with cosine similarity distributions
- FAISS indexing and Recall@K comparison (fine-tuned vs baseline)
- External test set evaluation

**Sections:**

| Section | Description |
|---------|-------------|
| 0 | Setup, Drive mount, configuration |
| 1 | Triplet CSV generation (anchor/positive/negative) |
| 2 | Train/val split (stratified by composer) |
| 3 | LoRA + CNN training loop |
| 4 | Evaluation on val set |
| 5 | Extract all embeddings |
| 6 | FAISS indexing + Recall@K |
| 7 | Baseline comparison |
| 8–12 | Test set evaluation + error analysis |

---

## Evaluation

### MERT FAISS Recall Evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mperumal-usd/capstone_team_3/blob/main/notebooks/COLAB_MERT_FAISS_Recall_Eval.ipynb)

**`notebooks/COLAB_MERT_FAISS_Recall_Eval.ipynb`**

Standalone evaluation notebook. Loads multiple saved LoRA checkpoints (v3, v4, ...), builds per-checkpoint FAISS indexes from the gallery, queries with external WAV chunks, and computes a side-by-side Recall@K comparison table.

Features resume-safe embedding generation — safe to re-run after Google Colab disconnects.

---

## Alternative Models

### CNN + Mel Spectrogram Similarity

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mperumal-usd/capstone_team_3/blob/main/notebooks/COLAB_CNN_MEL_Similarity.ipynb)

**`notebooks/COLAB_CNN_MEL_Similarity.ipynb`**

A 2D CNN trained on mel spectrogram images. Explores whether visual spectrogram similarity can serve as a proxy for musical similarity without a pre-trained audio encoder.

---

### LSTM Model

**`notebooks/LSTM_Model_v1.ipynb`**

Sequence-to-label LSTM trained on MIDI token representations. Tests 210,167 (label, prediction) pairs across 38 reference compositions.

---

### GRU Model

**`notebooks/GRU_Model_v1.ipynb`**

Gated Recurrent Unit baseline — faster convergence than LSTM with comparable accuracy on the validation set.

---

### Siamese CNN

**`notebooks/SiameseCNN_Model_v1.ipynb`**

Twin-network CNN with contrastive loss. Takes two audio segments and directly predicts a similarity score.

---

### Transformer (from scratch)

**`notebooks/Transformer_Model_v1.ipynb`**

Custom encoder-only transformer trained from scratch on MIDI tokens. Demonstrates the importance of music-specific pre-training — significantly underperforms MERT.

---

## Data Preparation

### Data Exploration

**`notebooks/DataExploration.ipynb`** · **`notebooks/DataExploration_v1.ipynb`**

Initial exploratory data analysis of the MIDI dataset — note density, tempo distributions, duration histograms, composer statistics.

---

### Data Cleaning

**`notebooks/DataCleaning.ipynb`**

Identifies and removes corrupt MIDI files, deduplicates near-identical arrangements, and produces the final clean 590-file corpus.

---

### Feature Engineering

**`notebooks/FeatureEngineering.ipynb`**

Explores traditional feature extraction approaches (piano roll, REMI tokens, chroma features, MFCC) before settling on raw audio input for MERT.

---

### Triplet Dataset Creation

**`notebooks/triplet_dataset_040326.ipynb`**

Generates the anchor/positive/negative triplet CSV from the chunk directory structure.

---

### Embeddings Generation

**`notebooks/Embeddings_Generation.ipynb`**

Batch-extracts and caches embeddings from a fine-tuned MERT checkpoint.

---

## Analysis

### Results Analysis

**`notebooks/ResultsAnalysis.ipynb`**

Aggregates predictions from LSTM, GRU, and Transformer models across 210,167 test pairs. Computes accuracy, confusion matrices, and per-song error breakdowns.

---

### MERT Evaluation

**`notebooks/mert_evaluation.ipynb`** · **`notebooks/mert_finetuned.ipynb`**

Ad-hoc evaluation notebooks used during development to inspect embedding quality and retrieval behaviour.

---

### Similarity Score EDA

**`notebooks/Similarity_Score_EDA.ipynb`** · **`notebooks/Similarity_Score_EDA-MERT.ipynb`**

Exploratory analysis of cosine similarity score distributions — understanding the gap between same-song and cross-song pairs before and after fine-tuning.

---

### Piano Roll Visualization

**`notebooks/COLAB_Piano_Roll_test.ipynb`**

Interactive MIDI piano roll rendering for visual inspection of melodic content and alignment.
