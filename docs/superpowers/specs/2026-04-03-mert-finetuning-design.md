# MERT Fine-Tuning Design Spec (LoRA)
**Date:** 2026-04-03 (revised)
**Task:** Fine-tune MERT-v1-95M with LoRA on the composer triplet WAV dataset, Drive-backed checkpointing

---

## Overview

Fine-tune `m-a-p/MERT-v1-95M` for composer-similarity retrieval using LoRA adapters injected
into all attention projections (q, k, v, o) of each transformer layer. No projection head —
the mean-pooled, L2-normalized 768-dim MERT output is used directly for triplet loss.
Because LoRA weights change during training, embeddings cannot be pre-cached; WAV files are
loaded and forwarded on every training batch. Every preprocessing stage and an intra-epoch
step checkpoint are saved to Google Drive for disconnect-safe resumption.

---

## Model Architecture

```
MERT-v1-95M
  + LoRA adapters on q_proj, k_proj, v_proj, o_proj (all transformer layers)
    LoRA rank r=8, alpha=16, dropout=0.05
  → mean-pool last_hidden_state → 768-dim
  → L2-normalize → 768-dim unit-sphere embedding

Loss: TripletMarginLoss(margin=0.3, p=2)
```

Trainable parameters: LoRA adapters only (~3–5M out of 95M total).

---

## Configuration

| Key | Value |
|-----|-------|
| MERT model | `m-a-p/MERT-v1-95M` |
| Sample rate | 24000 Hz |
| Embedding dim | 768 (direct MERT output, no head) |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Triplet margin | 0.3 |
| Batch size | 8 (each triplet = 3 WAV forward passes) |
| Learning rate | 5e-5 |
| Scheduler | CosineAnnealingLR (T_max = total steps) |
| Epochs | 10 |
| Save every N steps | 200 |
| Drive base | `/content/drive/MyDrive/AAI-590 Capstone/MERT_Finetune/` |
| ChunkSamples | `/content/drive/MyDrive/AAI-590 Capstone/ChunkSamples/` |

---

## Notebook Sections & Drive Checkpoints

Each section checks whether its checkpoint already exists on Drive before running.

| # | Section | Computation | Drive output |
|---|---------|-------------|--------------|
| 0 | Setup | Mount Drive, install deps (`peft` + others), define config | — |
| 1 | Triplet CSV | Regenerate (anchor, pos, neg) + composer/song columns | `triplet_df.csv` |
| 2 | Train/Val Split | 80/20 stratified on composer; skips if files exist | `splits/train_idx.npy`, `splits/val_idx.npy` |
| 3 | Training | LoRA fine-tune; saves adapter every SAVE_STEPS; resumes from last checkpoint | `checkpoints/lora_last/`, `checkpoints/lora_best/`, `checkpoints/optimizer_state.pt`, `checkpoints/training_state.json`, `training_log.csv` |
| 4 | Evaluation | Load best LoRA adapter; compute pos/neg cosine sims on val set; plot | `results/eval_results.csv`, `results/score_plot.png` |

### Intra-epoch resume (Section 3)

`training_state.json` stores `{epoch, step, best_val_loss}`. On resume the training loop
skips batches up to the saved step within the current epoch, then continues normally.
The LoRA adapter and optimizer state are restored from Drive before resuming.

---

## Training Loop Detail

```
Build TripletWAVDataset (filenames only; loads WAV on __getitem__)
DataLoader(batch_size=8, shuffle=True)

For each epoch:
    For each batch of (anchor_wav, pos_wav, neg_wav):
        Forward all three through MERT+LoRA
        Mean-pool → L2-normalize → 768-dim embeddings
        loss = TripletMarginLoss(a_emb, p_emb, n_emb)
        loss.backward(); optimizer.step(); scheduler.step()
        If step % SAVE_STEPS == 0:
            Save lora_last adapter + optimizer_state.pt + training_state.json

    Compute val loss (no grad) on val set
    If val_loss < best_val_loss:
        Save lora_best adapter
    Append row to training_log.csv
```

---

## Evaluation Metrics

- Mean positive cosine similarity on val set (fine-tuned LoRA model)
- Mean negative cosine similarity on val set
- Separation gap = mean(pos_sim) − mean(neg_sim)
- 2-panel histogram: positive vs negative distributions, saved to Drive

---

## Google Drive Folder Structure

```
AAI-590 Capstone/
└── MERT_Finetune/
    ├── triplet_df.csv
    ├── splits/
    │   ├── train_idx.npy
    │   └── val_idx.npy
    ├── checkpoints/
    │   ├── lora_best/              ← best adapter weights
    │   │   ├── adapter_config.json
    │   │   └── adapter_model.safetensors
    │   ├── lora_last/              ← latest adapter for resume
    │   │   ├── adapter_config.json
    │   │   └── adapter_model.safetensors
    │   ├── optimizer_state.pt
    │   └── training_state.json     ← {epoch, step, best_val_loss}
    ├── training_log.csv
    └── results/
        ├── eval_results.csv
        └── score_plot.png
```

---

## Constraints & Assumptions

- Colab GPU: T4 (16 GB VRAM). MERT-v1-95M + LoRA at batch_size=8 fits within memory.
- ChunkSamples WAV filenames are globally unique (composer+song prefix in filename).
- Triplet CSV is regenerated from scratch each run (Section 1 always runs).
- All random seeds fixed (seed=42) for reproducibility.
- `peft` library from Hugging Face used for LoRA injection.
