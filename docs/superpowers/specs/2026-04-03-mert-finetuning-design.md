# MERT Fine-Tuning Design Spec
**Date:** 2026-04-03  
**Task:** Fine-tune MERT-v1-95M on the composer triplet dataset with Drive-backed checkpointing

---

## Overview

Fine-tune `m-a-p/MERT-v1-95M` for composer-similarity retrieval using the existing 22,458-triplet
(anchor, positive, negative) WAV dataset. The backbone is frozen; only a small projection head is
trained. Every preprocessing stage saves its output to Google Drive so a disconnected Colab session
can resume from the last completed step.

---

## Model Architecture

```
MERT-v1-95M (all weights frozen)
    → mean-pool over time axis → 768-dim embedding
    → ProjectionHead:
        Linear(768 → 256) → ReLU → Dropout(0.3)
        → Linear(256 → 128)
    → L2-normalize → 128-dim unit-sphere embedding

Loss: TripletMarginLoss(margin=0.3) with semi-hard negative mining
```

Trainable parameters: projection head only (~200K params vs. 95M frozen).

---

## Configuration

| Key | Value |
|-----|-------|
| MERT model | `m-a-p/MERT-v1-95M` |
| Sample rate | 24000 Hz |
| Embedding dim | 128 |
| Triplet margin | 0.3 |
| Batch size (training) | 256 (embeddings pre-computed) |
| Batch size (MERT embed) | 16 (GPU memory constraint) |
| Learning rate | 1e-3 |
| Scheduler | StepLR(step=5, gamma=0.5) |
| Epochs | 20 |
| Drive base | `/content/drive/MyDrive/AAI-590 Capstone/MERT_Finetune/` |
| ChunkSamples | `/content/drive/MyDrive/AAI-590 Capstone/ChunkSamples/` |

---

## Notebook Sections & Drive Checkpoints

Each section checks whether its checkpoint file already exists on Drive before running.
If the file exists, it loads and skips. If not, it runs the full computation and saves immediately.

| # | Section | Computation | Drive output |
|---|---------|-------------|--------------|
| 0 | Setup | Mount Drive, install deps, define config | — |
| 1 | Triplet CSV | Regenerate (anchor, pos, neg) DataFrame from ChunkSamples directory | `triplet_df.csv` |
| 2 | Embed Anchors | MERT-embed all anchor WAVs in batches of 16; save every 500 rows for sub-step resume | `embeddings/anchor_embeddings.npy`, `embeddings/anchor_index.json` |
| 3 | Embed Positives | Same, for positive WAVs | `embeddings/positive_embeddings.npy` |
| 4 | Embed Negatives | Same, for negative WAVs | `embeddings/negative_embeddings.npy` |
| 5 | Train/Val Split | 80/20 stratified split on anchor song identity | `splits/train_idx.npy`, `splits/val_idx.npy` |
| 6 | Training | ProjectionHead + TripletLoss loop; best model saved when val loss improves | `checkpoints/best_model.pt`, `checkpoints/last_model.pt`, `training_log.csv` |
| 7 | Evaluation | Cosine-sim positive vs. negative score distributions; compare to MERT baseline scores | `results/eval_results.csv`, `results/score_plot.png` |

### Sub-step resume for embedding (Sections 2–4)

Embedding 22,458 WAVs takes ~6 hours. The notebook saves partial progress every 500 rows
as a `.npy` partial file. On resume, it reads how many rows are already embedded and picks up
from that index. Full file is written atomically only when all rows are done.

---

## Training Loop Detail

```
For each epoch:
    For each batch of (anchor_emb, pos_emb, neg_emb):
        project(anchor_emb) → a_proj
        project(pos_emb)    → p_proj
        project(neg_emb)    → n_proj
        loss = TripletMarginLoss(a_proj, p_proj, n_proj)
        loss.backward()
        optimizer.step()

    Compute val loss on held-out 20%
    If val_loss < best_val_loss:
        save best_model.pt to Drive
    save last_model.pt to Drive (always, for crash recovery)
    append row to training_log.csv
```

**Best model criterion:** minimum validation triplet loss.

---

## Evaluation Metrics

- Mean positive cosine similarity (fine-tuned head vs. raw MERT baseline)
- Mean negative cosine similarity
- Separation gap = mean(pos_sim) − mean(neg_sim)
- Distribution plot saved to Drive

---

## Google Drive Folder Structure

```
AAI-590 Capstone/
└── MERT_Finetune/
    ├── triplet_df.csv
    ├── embeddings/
    │   ├── anchor_embeddings.npy
    │   ├── anchor_index.json
    │   ├── positive_embeddings.npy
    │   └── negative_embeddings.npy
    ├── splits/
    │   ├── train_idx.npy
    │   └── val_idx.npy
    ├── checkpoints/
    │   ├── best_model.pt
    │   └── last_model.pt
    ├── training_log.csv
    └── results/
        ├── eval_results.csv
        └── score_plot.png
```

---

## Constraints & Assumptions

- Colab GPU: T4 (16 GB VRAM). MERT-v1-95M at sr=24000, batch=16 fits comfortably.
- ChunkSamples WAV files are all accessible via the file lookup dictionary (filename → full path).
- The triplet CSV is regenerated from scratch (user preference) each run of Section 1.
- All random seeds are fixed (seed=42) for reproducibility of train/val split.
