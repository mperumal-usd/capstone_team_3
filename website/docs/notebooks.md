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

### Siamese CNN

**`notebooks/SiameseCNN_Model_v1.ipynb`**

Twin-network CNN with contrastive loss. Takes two audio segments and directly predicts a similarity score.

---

## Data & Analysis

### Data Exploration

**`notebooks/DataExploration.ipynb`**

Initial exploratory data analysis of the MIDI dataset — note density, tempo distributions, duration histograms, composer statistics.

---

### Feature Engineering

**`notebooks/FeatureEngineering.ipynb`**

Explores traditional feature extraction approaches (piano roll, REMI tokens, chroma features, MFCC) before settling on raw audio input for MERT.

---

### Triplet Dataset Creation

**`notebooks/triplet_dataset_040326.ipynb`**

Generates the anchor/positive/negative triplet CSV from the chunk directory structure.

---

### MERT Evaluation

**`notebooks/mert_evaluation.ipynb`** · **`notebooks/mert_finetuned.ipynb`**

Ad-hoc evaluation notebooks used during development to inspect embedding quality and retrieval behaviour.

---

### Similarity Score EDA

**`notebooks/Similarity_Score_EDA.ipynb`** · **`notebooks/COLAB_Similarity_Score_EDA_MERT.ipynb`**

Exploratory analysis of cosine similarity score distributions — understanding the gap between same-song and cross-song pairs before and after fine-tuning.

---

### Piano Roll Visualization

**`notebooks/COLAB_Piano_Roll_test.ipynb`**

Interactive MIDI piano roll rendering for visual inspection of melodic content and alignment.

---

## Archive

Older iterations moved to `notebooks/archive/`. These are preserved for reference but superseded by the notebooks above.

| Notebook | Description |
|----------|-------------|
| `archive/COLAB_MERT_Finetune_v1.ipynb` | Initial LoRA setup |
| `archive/COLAB_MERT_Finetune_v2.ipynb` | Updated data loading |
| `archive/COLAB_MERT_Finetune_v3.ipynb` | Added FAISS validation section |
| `archive/COLAB_MERT_Finetune_v4.ipynb` | Optimizer tuning |
| `archive/COLAB_MERT_FAISS_Recall_Eval.ipynb` | Multi-checkpoint Recall@K comparison |
| `archive/GRU_Model_v1.ipynb` | GRU baseline model |
| `archive/Transformer_Model_v1.ipynb` | Custom transformer (trained from scratch) |
| `archive/DataCleaning.ipynb` | Corpus cleaning and deduplication |
| `archive/DataExploration_v1.ipynb` | Earlier EDA iteration |
| `archive/Embeddings_Generation.ipynb` | Batch embedding extraction |
| `archive/ResultsAnalysis.ipynb` | LSTM/GRU/Transformer prediction analysis |
| `archive/Similarity_Score_EDA-MERT.ipynb` | Similarity EDA (earlier MERT version) |
