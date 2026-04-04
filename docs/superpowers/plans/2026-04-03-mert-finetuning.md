# MERT Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Colab notebook that fine-tunes MERT-v1-95M on the composer triplet WAV dataset, saving every preprocessing stage and the best model to Google Drive for disconnect-safe resumption.

**Architecture:** Freeze the MERT-v1-95M backbone entirely; pre-compute and cache all 22K×3 embeddings to Drive as `.npy` files (resumable every 500 rows). Train a two-layer projection head (768→256→128) with triplet margin loss on the cached embeddings. Each notebook section checks for its Drive checkpoint before running.

**Tech Stack:** Python 3, PyTorch, Hugging Face `transformers`, `librosa`, `numpy`, `pandas`, `scikit-learn`, Google Colab + Drive

---

## File Structure

| File | Role |
|------|------|
| `notebooks/COLAB_MERT_Finetune_v1.ipynb` | Single Colab notebook — all sections |

Drive outputs (written by the notebook at runtime):
```
/content/drive/MyDrive/AAI-590 Capstone/MERT_Finetune/
├── triplet_df.csv
├── embeddings/
│   ├── anchor_embeddings.npy
│   ├── anchor_index.json          # {"completed": N, "total": M}
│   ├── positive_embeddings.npy
│   ├── positive_index.json
│   ├── negative_embeddings.npy
│   └── negative_index.json
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

## Task 1: Section 0 — Setup cell

**Files:**
- Create: `notebooks/COLAB_MERT_Finetune_v1.ipynb`

- [ ] **Step 1.1: Create the notebook file with the setup cell**

Create `notebooks/COLAB_MERT_Finetune_v1.ipynb`. The first cell (markdown) is the title; the second cell (code) installs deps and mounts Drive.

Cell 1 — Markdown:
```markdown
# MERT Fine-Tuning: Composer Similarity
## Notebook Sections
- **Section 0** — Setup (install, mount Drive, config)
- **Section 1** — Triplet CSV generation (regenerated each run)
- **Section 2** — Pre-compute anchor embeddings (resumable)
- **Section 3** — Pre-compute positive embeddings (resumable)
- **Section 4** — Pre-compute negative embeddings (resumable)
- **Section 5** — Train/val split
- **Section 6** — Train projection head
- **Section 7** — Evaluate

**Resume behaviour:** Each section checks for its Drive checkpoint. If found, it loads and skips. Run cells top-to-bottom after a disconnect to resume from where you left off.
```

Cell 2 — Code:
```python
# ── Section 0: Setup ─────────────────────────────────────────────────────────
!pip install -q torch librosa numpy pandas scikit-learn tqdm transformers

from google.colab import drive
drive.mount('/content/drive')

import os, json, random, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DRIVE_BASE  = "/content/drive/MyDrive/AAI-590 Capstone/MERT_Finetune"
CHUNK_PATH  = "/content/drive/MyDrive/AAI-590 Capstone/ChunkSamples"
MODEL_NAME  = "m-a-p/MERT-v1-95M"
SR          = 24000
EMBED_DIM   = 128        # projection head output
MERT_DIM    = 768        # MERT last_hidden_state dim
BATCH_SIZE  = 256        # training batch (embeddings already cached)
EMBED_BATCH = 16         # WAV → MERT GPU batch size
LR          = 1e-3
EPOCHS      = 20
MARGIN      = 0.3
DROPOUT     = 0.3
SEED        = 42
SAVE_EVERY  = 500        # rows between partial embedding saves

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Create Drive subdirectories
for sub in ["embeddings", "splits", "checkpoints", "results"]:
    os.makedirs(os.path.join(DRIVE_BASE, sub), exist_ok=True)
print("Drive folders ready.")
```

- [ ] **Step 1.2: Verify the cell runs**

Expected output:
```
Device: cuda
Drive folders ready.
```

---

## Task 2: Section 1 — Triplet CSV generation

- [ ] **Step 2.1: Add the triplet generation cell**

Cell 3 — Markdown:
```markdown
## Section 1 — Triplet CSV Generation
Regenerates the (anchor, positive, negative) DataFrame from ChunkSamples and saves to Drive.
Re-runs every time (no skip logic — takes ~30 seconds).
```

Cell 4 — Code:
```python
# ── Section 1: Triplet CSV ────────────────────────────────────────────────────
TRIPLET_CSV = os.path.join(DRIVE_BASE, "triplet_df.csv")

def numeric_sort(f):
    match = re.search(r'_chunk_(\d+)\.wav$', f)
    return int(match.group(1)) if match else -1

data = []
used_negatives = set()

for composer in os.listdir(CHUNK_PATH):
    composer_path = os.path.join(CHUNK_PATH, composer)
    if not os.path.isdir(composer_path):
        continue
    for song in os.listdir(composer_path):
        song_path = os.path.join(composer_path, song)
        if not os.path.isdir(song_path):
            continue
        chunks = sorted(
            [f for f in os.listdir(song_path) if f.endswith(".wav")],
            key=numeric_sort
        )
        if len(chunks) < 2:
            continue
        anchor    = chunks[0]
        positives = chunks[1:]
        other_composers = [
            c for c in os.listdir(CHUNK_PATH)
            if c != composer and os.path.isdir(os.path.join(CHUNK_PATH, c))
        ]
        if not other_composers:
            continue
        chosen_composer      = random.choice(other_composers)
        chosen_composer_path = os.path.join(CHUNK_PATH, chosen_composer)
        negative_pool = []
        for other_song in sorted(os.listdir(chosen_composer_path)):
            other_song_path = os.path.join(chosen_composer_path, other_song)
            if not os.path.isdir(other_song_path):
                continue
            for f in sorted(os.listdir(other_song_path), key=numeric_sort):
                if f.endswith(".wav") and f not in used_negatives:
                    negative_pool.append(f)
        min_len = min(len(positives), len(negative_pool))
        for i in range(min_len):
            data.append({
                "anchor":   anchor,
                "positive": positives[i],
                "negative": negative_pool[i],
                "composer": composer,         # keep for stratified split
                "song":     song,
            })
            used_negatives.add(negative_pool[i])

df = pd.DataFrame(data)
df.to_csv(TRIPLET_CSV, index=False)
print(f"Triplet CSV saved → {TRIPLET_CSV}")
print(f"Total triplets: {len(df)}")
print(df.head())
```

- [ ] **Step 2.2: Verify expected output**

Expected output (numbers may vary slightly):
```
Triplet CSV saved → /content/drive/MyDrive/AAI-590 Capstone/MERT_Finetune/triplet_df.csv
Total triplets: 22458
   anchor  positive  negative  composer  song
0  bach_847_chunk_1.wav  bach_847_chunk_2.wav  ...  bach  847
```

---

## Task 3: File lookup helper + MERT loader

These helpers are shared by Sections 2–4. They go in one cell immediately after the CSV.

- [ ] **Step 3.1: Add shared helpers cell**

Cell 5 — Markdown:
```markdown
## Shared Helpers (used by Sections 2–4)
```

Cell 6 — Code:
```python
# ── Shared helpers ────────────────────────────────────────────────────────────

# Build filename → full path lookup
file_lookup = {}
for composer in os.listdir(CHUNK_PATH):
    cp = os.path.join(CHUNK_PATH, composer)
    if not os.path.isdir(cp):
        continue
    for song in os.listdir(cp):
        sp = os.path.join(cp, song)
        if not os.path.isdir(sp):
            continue
        for f in os.listdir(sp):
            if f.endswith(".wav"):
                file_lookup[f] = os.path.join(sp, f)
print(f"File lookup built: {len(file_lookup)} files")

# Load MERT model (frozen — used only for pre-computing embeddings)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
mert_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
mert_model.eval()
mert_model.to(DEVICE)
for param in mert_model.parameters():
    param.requires_grad = False
print("MERT loaded and frozen.")

def wav_to_mert_embedding(wav_path: str) -> np.ndarray:
    """Load a WAV file and return its mean-pooled MERT embedding (768-dim)."""
    audio, _ = librosa.load(wav_path, sr=SR)
    inputs   = processor(audio, sampling_rate=SR, return_tensors="pt")
    inputs   = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = mert_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb  # shape (768,)

def embed_column(col_name: str, emb_path: str, idx_path: str):
    """
    Pre-compute MERT embeddings for a DataFrame column of filenames.
    Saves partial progress every SAVE_EVERY rows.
    Resumes from last completed row if files exist on Drive.
    """
    filenames = df[col_name].tolist()
    n         = len(filenames)

    # Resume: check how many rows are already done
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx_meta = json.load(f)
        done = idx_meta["completed"]
        print(f"[{col_name}] Resuming from row {done}/{n}")
        embeddings = np.load(emb_path)   # (done, 768)
        embeddings = list(embeddings)    # convert to list for appending
    else:
        done       = 0
        embeddings = []
        print(f"[{col_name}] Starting from scratch (0/{n})")

    for i in tqdm(range(done, n), desc=f"Embedding {col_name}", initial=done, total=n):
        fname = filenames[i]
        path  = file_lookup.get(fname)
        if path is None:
            # Missing file — use zero vector so indices stay aligned
            embeddings.append(np.zeros(MERT_DIM, dtype=np.float32))
        else:
            embeddings.append(wav_to_mert_embedding(path).astype(np.float32))

        # Partial save every SAVE_EVERY rows
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n:
            np.save(emb_path, np.array(embeddings, dtype=np.float32))
            with open(idx_path, "w") as f:
                json.dump({"completed": i + 1, "total": n}, f)

    final = np.array(embeddings, dtype=np.float32)
    np.save(emb_path, final)
    with open(idx_path, "w") as f:
        json.dump({"completed": n, "total": n}, f)
    print(f"[{col_name}] Done. Shape: {final.shape}")
    return final
```

- [ ] **Step 3.2: Verify expected output**

```
File lookup built: 43663 files
MERT loaded and frozen.
```

---

## Task 4: Section 2 — Embed anchors

- [ ] **Step 4.1: Add anchor embedding cell**

Cell 7 — Markdown:
```markdown
## Section 2 — Pre-compute Anchor Embeddings
Resumes from Drive if partially complete. Takes ~2 hours for 22K files.
```

Cell 8 — Code:
```python
# ── Section 2: Anchor embeddings ─────────────────────────────────────────────
ANCHOR_EMB = os.path.join(DRIVE_BASE, "embeddings", "anchor_embeddings.npy")
ANCHOR_IDX = os.path.join(DRIVE_BASE, "embeddings", "anchor_index.json")

anchor_embeddings = embed_column("anchor", ANCHOR_EMB, ANCHOR_IDX)
print(f"Anchor embeddings shape: {anchor_embeddings.shape}")
```

- [ ] **Step 4.2: Verify expected output**

```
[anchor] Starting from scratch (0/22458)
Embedding anchor: 100%|████████| 22458/22458
[anchor] Done. Shape: (22458, 768)
Anchor embeddings shape: (22458, 768)
```
On resume:
```
[anchor] Resuming from row 500/22458
```

---

## Task 5: Section 3 — Embed positives

- [ ] **Step 5.1: Add positive embedding cell**

Cell 9 — Markdown:
```markdown
## Section 3 — Pre-compute Positive Embeddings
```

Cell 10 — Code:
```python
# ── Section 3: Positive embeddings ───────────────────────────────────────────
POS_EMB = os.path.join(DRIVE_BASE, "embeddings", "positive_embeddings.npy")
POS_IDX = os.path.join(DRIVE_BASE, "embeddings", "positive_index.json")

positive_embeddings = embed_column("positive", POS_EMB, POS_IDX)
print(f"Positive embeddings shape: {positive_embeddings.shape}")
```

- [ ] **Step 5.2: Verify expected output**

```
[positive] Done. Shape: (22458, 768)
```

---

## Task 6: Section 4 — Embed negatives

- [ ] **Step 6.1: Add negative embedding cell**

Cell 11 — Markdown:
```markdown
## Section 4 — Pre-compute Negative Embeddings
```

Cell 12 — Code:
```python
# ── Section 4: Negative embeddings ───────────────────────────────────────────
NEG_EMB = os.path.join(DRIVE_BASE, "embeddings", "negative_embeddings.npy")
NEG_IDX = os.path.join(DRIVE_BASE, "embeddings", "negative_index.json")

negative_embeddings = embed_column("negative", NEG_EMB, NEG_IDX)
print(f"Negative embeddings shape: {negative_embeddings.shape}")
```

- [ ] **Step 6.2: Verify expected output**

```
[negative] Done. Shape: (22458, 768)
```

---

## Task 7: Section 5 — Train/val split

- [ ] **Step 7.1: Add split cell**

Cell 13 — Markdown:
```markdown
## Section 5 — Train/Val Split
80/20 stratified on composer so every composer appears in both sets.
Skips if split files already exist on Drive.
```

Cell 14 — Code:
```python
# ── Section 5: Train/val split ────────────────────────────────────────────────
TRAIN_IDX_PATH = os.path.join(DRIVE_BASE, "splits", "train_idx.npy")
VAL_IDX_PATH   = os.path.join(DRIVE_BASE, "splits",   "val_idx.npy")

if os.path.exists(TRAIN_IDX_PATH) and os.path.exists(VAL_IDX_PATH):
    train_idx = np.load(TRAIN_IDX_PATH)
    val_idx   = np.load(VAL_IDX_PATH)
    print(f"Loaded split from Drive. Train: {len(train_idx)}  Val: {len(val_idx)}")
else:
    all_idx = np.arange(len(df))
    # Stratify on composer column so each composer appears in both splits
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=0.2,
        random_state=SEED,
        stratify=df["composer"].values
    )
    np.save(TRAIN_IDX_PATH, train_idx)
    np.save(VAL_IDX_PATH,   val_idx)
    print(f"Split saved. Train: {len(train_idx)}  Val: {len(val_idx)}")

# Slice embeddings
train_a = anchor_embeddings[train_idx]
train_p = positive_embeddings[train_idx]
train_n = negative_embeddings[train_idx]

val_a = anchor_embeddings[val_idx]
val_p = positive_embeddings[val_idx]
val_n = negative_embeddings[val_idx]

print(f"train_a shape: {train_a.shape}  val_a shape: {val_a.shape}")
```

- [ ] **Step 7.2: Verify expected output**

```
Split saved. Train: 17966  Val: 4492
train_a shape: (17966, 768)  val_a shape: (4492, 768)
```

---

## Task 8: Section 6 — Projection head + training loop

- [ ] **Step 8.1: Add ProjectionHead class and TripletDataset cell**

Cell 15 — Markdown:
```markdown
## Section 6 — Train Projection Head
ProjectionHead: Linear(768→256) → ReLU → Dropout → Linear(256→128) → L2-normalize.
Trains with TripletMarginLoss on pre-computed embeddings. Saves best and last checkpoint to Drive each epoch.
```

Cell 16 — Code (model + dataset definitions):
```python
# ── Projection head ───────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


class TripletEmbeddingDataset(Dataset):
    """Dataset of pre-computed MERT embeddings (numpy arrays)."""
    def __init__(self, anchors, positives, negatives):
        self.anchors   = torch.from_numpy(anchors).float()
        self.positives = torch.from_numpy(positives).float()
        self.negatives = torch.from_numpy(negatives).float()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx], self.negatives[idx]


train_dataset = TripletEmbeddingDataset(train_a, train_p, train_n)
val_dataset   = TripletEmbeddingDataset(val_a,   val_p,   val_n)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

proj_head = ProjectionHead(
    in_dim=MERT_DIM, hidden_dim=256, out_dim=EMBED_DIM, dropout=DROPOUT
).to(DEVICE)
n_params = sum(p.numel() for p in proj_head.parameters())
print(f"ProjectionHead parameters: {n_params:,}")

criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
optimizer = torch.optim.Adam(proj_head.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
```

- [ ] **Step 8.2: Verify expected output**

```
Train batches: 70  Val batches: 18
ProjectionHead parameters: 230,016
```

- [ ] **Step 8.3: Add training loop cell**

Cell 17 — Code:
```python
# ── Training loop ─────────────────────────────────────────────────────────────
BEST_CKPT = os.path.join(DRIVE_BASE, "checkpoints", "best_model.pt")
LAST_CKPT = os.path.join(DRIVE_BASE, "checkpoints", "last_model.pt")
LOG_PATH  = os.path.join(DRIVE_BASE, "training_log.csv")

# Resume: load last checkpoint if it exists
start_epoch    = 1
best_val_loss  = float("inf")
training_log   = []

if os.path.exists(LAST_CKPT):
    ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
    proj_head.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"Resumed from epoch {ckpt['epoch']}. Continuing from epoch {start_epoch}.")
else:
    print("No checkpoint found. Training from scratch.")

if os.path.exists(LOG_PATH):
    training_log = pd.read_csv(LOG_PATH).to_dict(orient="records")

for epoch in range(start_epoch, EPOCHS + 1):
    # ── Train ──────────────────────────────────────────────────────────────
    proj_head.train()
    train_loss_total = 0.0
    for a, p, n in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
        a_proj  = proj_head(a)
        p_proj  = proj_head(p)
        n_proj  = proj_head(n)
        loss    = criterion(a_proj, p_proj, n_proj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()
    avg_train_loss = train_loss_total / len(train_loader)

    # ── Validate ──────────────────────────────────────────────────────────
    proj_head.eval()
    val_loss_total = 0.0
    val_pos_sim    = 0.0
    val_neg_sim    = 0.0
    with torch.no_grad():
        for a, p, n in val_loader:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            a_proj  = proj_head(a)
            p_proj  = proj_head(p)
            n_proj  = proj_head(n)
            loss    = criterion(a_proj, p_proj, n_proj)
            val_loss_total += loss.item()
            val_pos_sim += F.cosine_similarity(a_proj, p_proj).mean().item()
            val_neg_sim += F.cosine_similarity(a_proj, n_proj).mean().item()
    avg_val_loss = val_loss_total / len(val_loader)
    avg_pos_sim  = val_pos_sim   / len(val_loader)
    avg_neg_sim  = val_neg_sim   / len(val_loader)

    scheduler.step()

    # ── Logging ────────────────────────────────────────────────────────────
    row = {
        "epoch":      epoch,
        "train_loss": round(avg_train_loss, 6),
        "val_loss":   round(avg_val_loss,   6),
        "val_pos_sim": round(avg_pos_sim,   4),
        "val_neg_sim": round(avg_neg_sim,   4),
        "val_gap":    round(avg_pos_sim - avg_neg_sim, 4),
        "lr":         scheduler.get_last_lr()[0],
    }
    training_log.append(row)
    pd.DataFrame(training_log).to_csv(LOG_PATH, index=False)
    print(
        f"Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} "
        f"val_loss={avg_val_loss:.4f} "
        f"pos_sim={avg_pos_sim:.4f} neg_sim={avg_neg_sim:.4f} "
        f"gap={avg_pos_sim - avg_neg_sim:.4f}"
    )

    # ── Save last checkpoint (always) ──────────────────────────────────────
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    proj_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss":       best_val_loss,
    }, LAST_CKPT)

    # ── Save best checkpoint ───────────────────────────────────────────────
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            "epoch":            epoch,
            "model_state_dict": proj_head.state_dict(),
            "val_loss":         avg_val_loss,
            "val_pos_sim":      avg_pos_sim,
            "val_neg_sim":      avg_neg_sim,
            "val_gap":          avg_pos_sim - avg_neg_sim,
            "config": {
                "model_name": MODEL_NAME,
                "embed_dim":  EMBED_DIM,
                "mert_dim":   MERT_DIM,
                "margin":     MARGIN,
                "dropout":    DROPOUT,
                "lr":         LR,
                "epochs":     EPOCHS,
            },
        }, BEST_CKPT)
        print(f"  ✓ Best model saved (val_loss={avg_val_loss:.4f})")

print("\nTraining complete.")
print(f"Best model → {BEST_CKPT}")
```

- [ ] **Step 8.4: Verify training output format**

Expected per-epoch line:
```
Epoch 01 | train_loss=0.2341 val_loss=0.2198 pos_sim=0.6123 neg_sim=0.3201 gap=0.2922
  ✓ Best model saved (val_loss=0.2198)
```

---

## Task 9: Section 7 — Evaluation

- [ ] **Step 9.1: Add evaluation cell**

Cell 18 — Markdown:
```markdown
## Section 7 — Evaluation
Loads the best projection head and computes cosine similarity distributions.
Compares gap (pos_sim − neg_sim) before fine-tuning (raw MERT) vs. after (projection head).
```

Cell 19 — Code:
```python
# ── Section 7: Evaluation ─────────────────────────────────────────────────────
EVAL_CSV   = os.path.join(DRIVE_BASE, "results", "eval_results.csv")
PLOT_PATH  = os.path.join(DRIVE_BASE, "results", "score_plot.png")

# Load best model
ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
proj_head.load_state_dict(ckpt["model_state_dict"])
proj_head.eval()
print(f"Best model loaded (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

# ── Compute raw MERT cosine sims (no projection head) ────────────────────────
def batch_cosine_sim(a: np.ndarray, b: np.ndarray, batch: int = 1024) -> np.ndarray:
    """Compute cosine similarity row-wise in batches."""
    a_t  = torch.from_numpy(a).float()
    b_t  = torch.from_numpy(b).float()
    sims = []
    for i in range(0, len(a_t), batch):
        ai = F.normalize(a_t[i:i+batch], p=2, dim=1)
        bi = F.normalize(b_t[i:i+batch], p=2, dim=1)
        sims.append((ai * bi).sum(dim=1).numpy())
    return np.concatenate(sims)

raw_pos_sim = batch_cosine_sim(anchor_embeddings[val_idx], positive_embeddings[val_idx])
raw_neg_sim = batch_cosine_sim(anchor_embeddings[val_idx], negative_embeddings[val_idx])

# ── Compute fine-tuned sims (with projection head) ───────────────────────────
ft_pos_sims, ft_neg_sims = [], []
with torch.no_grad():
    for a, p, n in val_loader:
        a_proj = proj_head(a.to(DEVICE)).cpu()
        p_proj = proj_head(p.to(DEVICE)).cpu()
        n_proj = proj_head(n.to(DEVICE)).cpu()
        ft_pos_sims.append(F.cosine_similarity(a_proj, p_proj).numpy())
        ft_neg_sims.append(F.cosine_similarity(a_proj, n_proj).numpy())
ft_pos_sim = np.concatenate(ft_pos_sims)
ft_neg_sim = np.concatenate(ft_neg_sims)

# ── Summary table ─────────────────────────────────────────────────────────────
results = {
    "raw_mert_pos_mean":  round(float(raw_pos_sim.mean()), 4),
    "raw_mert_neg_mean":  round(float(raw_neg_sim.mean()), 4),
    "raw_mert_gap":       round(float(raw_pos_sim.mean() - raw_neg_sim.mean()), 4),
    "finetune_pos_mean":  round(float(ft_pos_sim.mean()), 4),
    "finetune_neg_mean":  round(float(ft_neg_sim.mean()), 4),
    "finetune_gap":       round(float(ft_pos_sim.mean() - ft_neg_sim.mean()), 4),
}
results_df = pd.DataFrame([results])
results_df.to_csv(EVAL_CSV, index=False)
print(results_df.T.to_string(header=False))

# ── Distribution plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, pos, neg, title in [
    (axes[0], raw_pos_sim, raw_neg_sim,  "Raw MERT (frozen)"),
    (axes[1], ft_pos_sim,  ft_neg_sim,   "Fine-tuned (proj head)"),
]:
    ax.hist(pos, bins=60, alpha=0.6, label="Positive", color="steelblue")
    ax.hist(neg, bins=60, alpha=0.6, label="Negative", color="tomato")
    ax.axvline(pos.mean(), color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(neg.mean(), color="tomato",    linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.show()
print(f"Plot saved → {PLOT_PATH}")
print(f"Eval results saved → {EVAL_CSV}")
```

- [ ] **Step 9.2: Verify expected output format**

```
raw_mert_pos_mean    0.7123
raw_mert_neg_mean    0.5891
raw_mert_gap         0.1232
finetune_pos_mean    0.8201
finetune_neg_mean    0.4012
finetune_gap         0.4189
Plot saved → /content/drive/MyDrive/AAI-590 Capstone/MERT_Finetune/results/score_plot.png
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Covered by |
|---|---|
| Regenerate triplet CSV from scratch | Task 2 |
| Save triplet CSV to Drive | Task 2 (`triplet_df.csv`) |
| Pre-compute MERT embeddings, resumable | Tasks 3–6 (`embed_column` + partial save every 500 rows) |
| Save anchor/pos/neg embeddings to Drive | Tasks 4–6 |
| Train/val split saved to Drive | Task 7 |
| ProjectionHead (768→256→128, L2-norm) | Task 8 |
| TripletMarginLoss (margin=0.3) | Task 8 |
| Save best model to Drive | Task 8 (BEST_CKPT) |
| Save last model for crash recovery | Task 8 (LAST_CKPT) |
| Resume training after disconnect | Task 8 (loads LAST_CKPT at start) |
| Evaluation: pos/neg cosine sim distributions | Task 9 |
| Comparison vs raw MERT baseline | Task 9 |
| Plot saved to Drive | Task 9 |
| All outputs under `MERT_Finetune/` | All tasks |

### Placeholder scan

No TBD/TODO/placeholder content found. All code blocks are complete.

### Type consistency

- `embed_column()` defined in Task 3, called in Tasks 4–6 with consistent args (`col_name`, `emb_path`, `idx_path`).
- `anchor_embeddings`, `positive_embeddings`, `negative_embeddings` are numpy `(N, 768)` arrays throughout.
- `ProjectionHead.forward(x)` takes `(batch, 768)` tensor, returns `(batch, 128)` — consistent with all usages in Tasks 8–9.
- `TripletEmbeddingDataset` returns `(anchor_tensor, positive_tensor, negative_tensor)` — consumed correctly in both train and eval loops.
- `val_loader` used in both Task 8 (training) and Task 9 (eval) — dataset slicing is consistent.
