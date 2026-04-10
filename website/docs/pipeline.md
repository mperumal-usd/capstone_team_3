---
id: pipeline
title: Data Pipeline
sidebar_label: Data Pipeline
---

# Data Pipeline

The pipeline transforms raw MIDI files into a searchable embedding index through six stages. All statistics below are taken directly from `COLAB_MERT_Finetune_v5.ipynb` outputs.

```mermaid
flowchart LR
    A(["🎼 590 MIDI Files"]) --> B["FluidSynth\nMIDI → WAV"]
    B --> C["Resample\n24 kHz mono"]
    C --> D["7-sec Chunks\n43,663 total"]
    D --> E["MERT Encoder\n768-dim"]
    E --> F["CNN Head\n128-dim L2-norm"]
    F --> G[("FAISS Index\n34,930 vectors")]
    G --> H["Recall@K\nEvaluation"]

    style A fill:#e3f2fd,stroke:#1565c0
    style G fill:#f3e5f5,stroke:#6a1b9a
    style H fill:#e8f5e9,stroke:#2e7d32
```

---

## Stage 1 — Dataset Collection

590 classical piano MIDI files spanning ~30 composers.

```mermaid
pie title Approximate Composer Distribution (590 files)
    "Bach" : 18
    "Beethoven" : 22
    "Chopin" : 25
    "Mozart" : 20
    "Schubert" : 15
    "Liszt" : 12
    "Grieg" : 10
    "Others" : 38
```

**Data cleaning steps:**
- Removed corrupt or unreadable MIDI files
- Identified and deduplicated near-identical arrangements
- Validated minimum note count and duration thresholds

---

## Stage 2 — MIDI → WAV Synthesis

MIDI files are synthesised to WAV using **FluidSynth** with the FluidR3 GM soundfont, then resampled to **24 kHz mono** to match MERT's expected input.

```mermaid
sequenceDiagram
    participant M as MIDI File
    participant F as FluidSynth
    participant L as librosa
    participant W as WAV Chunk

    M->>F: symbolic note events
    F->>F: render with FluidR3_GM.sf2
    F->>L: 44,100 Hz stereo WAV
    L->>L: resample → 24,000 Hz mono
    L->>W: float32 audio array
```

```bash
fluidsynth -ni FluidR3_GM.sf2 input.mid -F output.wav -r 44100
```

---

## Stage 3 — Chunking

Each WAV is split into **fixed 7-second non-overlapping chunks** — the retrieval unit for FAISS.

```mermaid
xychart-beta
    title "Dataset Scale after Chunking"
    x-axis ["Gallery Chunks", "Train Triplets", "Val Triplets", "Test Chunks"]
    y-axis "Count (thousands)" 0 --> 50
    bar [43.663, 18.546, 4.637, 4.116]
```

**Actual notebook output:**
```
File lookup built: 43663 files
Total triplets:    23183
Train: 18546  Val: 4637   (80/20 stratified by composer)
Test chunks: 4116
```

| Metric | Value |
|--------|-------|
| Total WAV chunks (gallery) | **43,663** |
| Total triplets generated | **23,183** |
| Train split | 18,546 (80%) |
| Val split | 4,637 (20%) |
| External test chunks | 4,116 |
| Chunk duration | 7 seconds |
| Sample rate | 24,000 Hz |

---

## Stage 4 — Model Setup

**Trainable parameters from notebook output:**

```
trainable params: 442,368 || all params: 94,814,080 || trainable%: 0.4666
CNN projection head: 3,296,768 params  (filters=256, kernels=(3, 5, 7), embed=128)
Train batches: 2319  Val batches: 580
Total training steps: 23190
```

```mermaid
graph TD
    A["WAV chunk · 168,000 samples"] --> B["AutoProcessor\nfeature extraction"]
    B --> C["MERT-v1-95M\n12 transformer layers\n768-dim hidden states\n94.8M total params"]
    C --> D["LoRA adapters\nq/k/v/o projections\nr=8 · α=16\n442K trainable params (0.47%)"]
    D --> E["CNN Projection Head\nConv1d × 3 (k=3,5,7)\n3.3M params\n→ 128-dim L2-norm"]

    style A fill:#e3f2fd,stroke:#1565c0
    style C fill:#f3e5f5,stroke:#6a1b9a
    style D fill:#fff8e1,stroke:#f57f17
    style E fill:#e8f5e9,stroke:#2e7d32
```

---

## Stage 5 — Triplet Training

Each training example is a `(anchor, positive, negative)` triple:

```mermaid
graph LR
    subgraph same_song["Same Song (e.g. bach_847)"]
        A["Anchor\nbash_847_chunk_1.wav"]
        P["Positive\nbach_847_chunk_3.wav"]
    end
    subgraph diff_composer["Different Composer (e.g. Liszt)"]
        N["Negative\nislam_chunk_1.wav"]
    end
    A -->|"↓ pull together"| P
    A -->|"↑ push apart"| N

    style same_song fill:#e8f5e9,stroke:#2e7d32
    style diff_composer fill:#ffebee,stroke:#c62828
```

**Actual triplet sample from notebook:**

```
anchor               positive              negative              composer  song
bach_847_chunk_1.wav bach_847_chunk_2.wav  islamei_chunk_1.wav   bach      bach_847
bach_847_chunk_1.wav bach_847_chunk_3.wav  islamei_chunk_2.wav   bach      bach_847
```

Loss: `TripletMarginLoss(margin=0.3)`

---

## Stage 6 — Embedding Extraction & FAISS Index

After training, all 43,663 chunks are embedded with the best checkpoint (epoch 9):

```
Extracting all embeddings: 100%|██████████| 5458/5458 [35:09<00:00, 2.59s/it]
Saved 43663 embeddings → all_embeddings.pkl
FAISS index built. Total vectors indexed: 34930
```

```mermaid
graph LR
    Q["Query chunk\n(7-sec WAV)"] --> QE["MERT + LoRA\n+ CNN Head\n128-dim embedding"]
    QE --> FS["FAISS IndexFlatL2\nsearch top-K"]
    DB[("Gallery Index\n34,930 × 128-dim\nL2-normalised")] --> FS
    FS --> R["Top-K results\nwith similarity scores\ncosine_sim = 1 − L2²/2"]

    style Q fill:#e3f2fd,stroke:#1565c0
    style DB fill:#f3e5f5,stroke:#6a1b9a
    style R fill:#e8f5e9,stroke:#2e7d32
```

**Sample FAISS search output:**
```
Query: Aragon (Fantasia) Op.47 part 6_chunk_1.wav
  Rank 1: alb_se6_chunk_41.wav   sim=0.9710  ✓ correct
  Rank 2: alb_se6_chunk_6.wav    sim=0.9618  ✓ correct
  Rank 3: alb_se6_chunk_39.wav   sim=0.9608  ✓ correct
  Rank 4: alb_se6_chunk_40.wav   sim=0.9581  ✓ correct
```
