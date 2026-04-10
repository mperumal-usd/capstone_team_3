---
id: models
title: Models & Architecture
sidebar_label: Models & Architecture
---

# Models & Architecture

The project explored several deep learning architectures for music similarity before converging on fine-tuned MERT as the primary approach.

---

## Primary Model: MERT + LoRA + CNN Head

### Base Model — MERT-v1-95M

[MERT](https://arxiv.org/abs/2306.00107) (Music Encoder Representations from Transformers) is a 95-million-parameter transformer pre-trained on large-scale music audio data. It operates directly on raw waveforms and produces contextual embeddings that capture pitch, rhythm, harmonic structure, and temporal patterns.

| Property | Value |
|----------|-------|
| Architecture | Transformer (12 layers) |
| Hidden dimension | 768 |
| Parameters | 95M |
| Pre-training data | Large-scale music corpus |
| Input sample rate | 24,000 Hz |
| HuggingFace ID | `m-a-p/MERT-v1-95M` |

### LoRA Fine-Tuning

Instead of updating all 95M parameters, **LoRA** (Low-Rank Adaptation) inserts small trainable rank decomposition matrices into the attention projections. This trains only **~442K parameters** (~0.47% of total) while preserving the pre-trained representations.

```python
lora_config = LoraConfig(
    r            = 8,                               # rank
    lora_alpha   = 16,                              # scaling factor
    target_modules = ["q_proj", "k_proj",           # all attention
                       "v_proj", "o_proj"],         # projections
    lora_dropout = 0.05,
    bias         = "none",
)
model = get_peft_model(base_mert, lora_config)
```

| LoRA Parameter | Value |
|---------------|-------|
| Rank (r) | 8 |
| Alpha (α) | 16 |
| Dropout | 0.05 |
| Target layers | q_proj, k_proj, v_proj, o_proj |
| Trainable params | 442,368 |
| % of total | 0.47% |

### CNN Projection Head

A multi-scale CNN temporal encoder replaces simple mean-pooling. Three parallel `Conv1d` branches capture local (3-frame), mid-range (5-frame), and broader (7-frame) temporal patterns in the MERT hidden states. Their global-average-pooled outputs are concatenated and projected to a 128-dim embedding.

```
MERT last_hidden_state   (B, T, 768)
         │
         │  transpose → (B, 768, T)
         ├─── Conv1d(768→256, k=3) → GAP → (B, 256)
         ├─── Conv1d(768→256, k=5) → GAP → (B, 256)
         └─── Conv1d(768→256, k=7) → GAP → (B, 256)
                         │
                    concat → (B, 768)
                         │
                   Linear(768→384) → LayerNorm → GELU
                         │
                   Linear(384→128) → L2-normalize
                         │
                  128-dim unit-sphere embedding
```

| CNN Head Parameter | Value |
|-------------------|-------|
| Filters per branch | 256 |
| Kernel sizes | 3, 5, 7 |
| Output dimension | 128 |
| Parameters | 3,296,768 |

### Training Configuration (v5 — Final)

| Hyperparameter | Value |
|---------------|-------|
| Loss function | TripletMarginLoss |
| Margin | 0.3 |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| LR scheduler | CosineAnnealingLR |
| Epochs | 10 |
| Batch size | 8 |
| Gradient clipping | 1.0 |
| Mixed precision | AMP (float16) |
| Train triplets | 18,546 |
| Val triplets | 4,637 |
| Best val loss | 0.0302 (epoch 9) |

### Training Progression

| Epoch | Train Loss | Val Loss | Pos Sim | Neg Sim | Gap |
|-------|-----------|----------|---------|---------|-----|
| 1 | 0.0993 | 0.0950 | 0.563 | 0.078 | 0.485 |
| 3 | 0.0414 | 0.0525 | 0.512 | −0.144 | 0.656 |
| 5 | 0.0236 | 0.0399 | 0.523 | −0.114 | 0.637 |
| 7 | 0.0150 | **0.0317** | 0.519 | −0.170 | **0.689** |
| 9 | 0.0110 | **0.0302** | 0.520 | −0.130 | 0.650 |
| 10 | 0.0102 | 0.0320 | 0.521 | −0.128 | 0.649 |

*Pos Sim = mean cosine similarity of same-song pairs. Neg Sim = mean cosine similarity of cross-composer pairs. Gap = separation between them.*

---

## Baseline Models Compared

### LSTM

A sequence model built on piano-roll or MIDI token features. Trained to predict the most similar reference song given an input sequence.

- Input: MIDI token sequence
- Architecture: Stacked LSTM layers
- Output: Similarity label or embedding

### GRU

Similar to LSTM but with fewer gates — faster to train. Showed comparable accuracy to LSTM at lower compute cost.

### Transformer (custom)

A custom encoder-only transformer trained from scratch on MIDI token representations. Lower performance than the pre-trained MERT baseline due to limited training data.

### Siamese CNN

A twin-network CNN architecture that takes two MIDI/audio segments and learns to output whether they are similar. Trained with contrastive loss.

### CNN + Mel Spectrogram

A CNN trained directly on mel spectrogram images of audio chunks. Uses standard 2D convolution to extract spectro-temporal features, then cosine similarity for comparison.

---

## Why MERT Outperforms Traditional Approaches

| Capability | Edit Distance | DTW | CNN (mel) | **MERT + LoRA** |
|-----------|:---:|:---:|:---:|:---:|
| Key invariance | ✗ | ✗ | Partial | ✓ |
| Tempo invariance | ✗ | ✓ | Partial | ✓ |
| Semantic musical understanding | ✗ | ✗ | ✗ | ✓ |
| Pre-trained on music audio | — | — | ✗ | ✓ |
| Scalable (FAISS) | ✓ | ✗ | ✓ | ✓ |
| Fine-tunable with few labels | — | — | Partial | ✓ |

MERT's transformer layers, pre-trained on millions of hours of music, capture **musical semantics** — the kind of abstract similarity that a human musician would recognize — rather than surface-level note or spectrum similarity.
