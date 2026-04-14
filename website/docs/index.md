---
id: index
slug: /
title: Project Overview
sidebar_label: Project Overview
---

<div style={{borderRadius: '16px', overflow: 'hidden', marginBottom: '2rem', position: 'relative'}}>
  <img src="/capstone_team_3/img/capstone-hero.jpg" alt="AAI-590 Capstone" style={{width: '100%', height: '280px', objectFit: 'cover', objectPosition: 'center 30%', display: 'block'}} />
  <div style={{position: 'absolute', inset: 0, background: 'linear-gradient(to bottom, rgba(0,0,0,0.15) 0%, rgba(0,0,0,0.65) 100%)', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', padding: '2rem'}}>
    <h1 style={{color: '#fff', margin: 0, fontSize: '2rem', fontWeight: 800}}>Melody Match — Music Similarity with MERT</h1>
    <p style={{color: 'rgba(255,255,255,0.85)', margin: '0.5rem 0 0', fontSize: '1rem'}}>
      AAI-590 Applied AI Capstone · University of San Diego · Spring 2026 · Team 3
    </p>
    <p style={{color: 'rgba(255,255,255,0.7)', margin: '0.25rem 0 0', fontSize: '0.9rem'}}>
      Manikandan Perumal · Darin Verduzco · Israel Romero
    </p>
  </div>
</div>

---

## What Is This Project?

**Melody Match** is a deep-learning system that identifies whether two pieces of classical music are similar — even across different tempos, keys, and recordings. Given an audio query, the system retrieves the most similar pieces from a 295-song classical catalogue using semantic embeddings, not rule-based matching.

The core idea is to fine-tune **MERT** (Music Encoder Representations from Transformers), a 95M-parameter audio foundation model, to produce embeddings that cluster music by composer and melody, then index those embeddings with **FAISS** for fast nearest-neighbour retrieval.

---

## Presentation Video

<div style={{position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', borderRadius: '12px', marginBottom: '2rem', boxShadow: '0 4px 24px rgba(0,0,0,0.15)'}}>
  <iframe
    src="https://www.youtube.com/embed/D2opYEuRxb4"
    title="Melody Match Demo"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 'none'}}
  />
</div>

---

## Motivation

Traditional music similarity methods fail to capture **musical meaning**:

```mermaid
graph LR
    subgraph Traditional["❌ Traditional Approaches"]
        ED["Edit Distance<br/>No key/tempo invariance<br/>No semantic understanding"]
        DTW["Dynamic Time Warping<br/>Some rhythm handling<br/>No semantic understanding"]
    end

    subgraph DL["✅ Deep Learning (MERT + LoRA)"]
        MERT["MERT Embeddings<br/>Key-invariant ✓<br/>Tempo-invariant ✓<br/>Semantic understanding ✓"]
    end

    Music["🎵 Musical Data"] --> ED
    Music --> DTW
    Music --> MERT
    MERT --> Best["Best Performance<br/>Captures Musical Meaning"]

    style Traditional fill:#ffdddd,stroke:#cc4444
    style DL fill:#ddffdd,stroke:#44cc44
    style Best fill:#aaffaa,stroke:#22aa22
```

---

## System Overview

```mermaid
flowchart TB
    Start(["🎼 295 Classical MIDI Files"])

    subgraph Preprocessing["Preprocessing"]
        direction LR
        P1["FluidSynth<br/>MIDI → WAV"] --> P2["Resample<br/>24 kHz mono"] --> P3["7-sec chunks<br/>43,663 total"]
    end

    subgraph Encoder["MERT Encoder  (95M params)"]
        direction LR
        E1["Token / waveform<br/>feature extraction"] --> E2["12 Transformer layers<br/>768-dim hidden"] --> E3["CNN Projection Head<br/>→ 128-dim embedding"]
    end

    subgraph Training["Fine-tuning (LoRA)"]
        direction LR
        T1["Triplet dataset<br/>23,183 pairs"] --> T2["TripletMarginLoss<br/>margin = 0.3"] --> T3["10 epochs<br/>AdamW + AMP"]
    end

    subgraph Retrieval["Retrieval"]
        direction LR
        R1["FAISS IndexFlatL2<br/>34,930 gallery vectors"] --> R2["Top-K search<br/>milliseconds"]
    end

    Start --> Preprocessing --> Encoder --> Training
    Encoder --> Retrieval

    style Preprocessing fill:#e3f2fd,stroke:#1565c0
    style Encoder fill:#f3e5f5,stroke:#6a1b9a
    style Training fill:#e8f5e9,stroke:#2e7d32
    style Retrieval fill:#fff8e1,stroke:#f57f17
```

---

## Key Results

| Model | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|-------|:--------:|:--------:|:--------:|:---------:|
| **MERT + LoRA + CNN (v5)** | **63.6%** | **73.6%** | **77.9%** | **83.3%** |
| Baseline MERT (no fine-tuning) | 45.5% | 55.6% | 60.7% | 67.9% |
| Improvement | +18.1 pp | +18.1 pp | +17.2 pp | +15.4 pp |

```mermaid
xychart-beta
    title "Recall@K — Fine-tuned vs Baseline"
    x-axis ["R@1", "R@3", "R@5", "R@10"]
    y-axis "Recall (%)" 0 --> 100
    bar [63.6, 73.6, 77.9, 83.3]
    bar [45.5, 55.6, 60.7, 67.9]
```

---

## Embedding Separation After Fine-tuning

The model produces well-separated embedding clusters:

```mermaid
graph LR
    subgraph After["After Fine-tuning (Epoch 9)"]
        P["Same-song pairs<br/>cosine sim ≈ +0.52"]
        N["Cross-composer pairs<br/>cosine sim ≈ −0.13"]
    end
    Gap["Gap = 0.65<br/>✓ Large, discriminative margin"]
    P --> Gap
    N --> Gap
    style P fill:#bbdefb,stroke:#1976d2
    style N fill:#ffcdd2,stroke:#c62828
    style Gap fill:#c8e6c9,stroke:#2e7d32
```

---

## Training Progression (v5)

```mermaid
xychart-beta
    title "Validation Loss over 10 Epochs"
    x-axis ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]
    y-axis "Val Loss" 0 --> 0.12
    line [0.095, 0.063, 0.053, 0.045, 0.040, 0.037, 0.032, 0.040, 0.030, 0.032]
```

---

## Applications

```mermaid
mindmap
  root((Melody Match))
    Copyright Protection
      Plagiarism Detection
      Unauthorized Copying
      Derivative Works
    Music Discovery
      Recommendation Systems
      Similar Song Search
      Query by Humming
    Research
      Motif Discovery
      Composer Attribution
      Influence Analysis
    Education
      Tracing Musical Lineages
      Style Analysis
```

---

## Repository Structure

```
capstone_team_3/
├── notebooks/
│   ├── COLAB_MERT_Finetune_v5.ipynb    ← Final model (primary)
│   ├── COLAB_CNN_MEL_Similarity.ipynb
│   ├── LSTM_Model_v1.ipynb
│   ├── SiameseCNN_Model_v1.ipynb
│   ├── DataExploration.ipynb
│   ├── FeatureEngineering.ipynb
│   ├── triplet_dataset_040326.ipynb
│   ├── mert_evaluation.ipynb
│   ├── Similarity_Score_EDA.ipynb
│   ├── COLAB_Piano_Roll_test.ipynb
│   └── archive/                        ← Older iterations (v1–v4, GRU, Transformer, etc.)
├── MidiDatasets/
│   ├── 590-Classical-music-midi/       ← 295 MIDI files
│   └── TestingSamples/
├── FrontEnd/
│   ├── MidiAnalyzer.html
│   └── MidiComparator.html
└── website/                            ← This documentation
```

---

## Project Timeline

```mermaid
gantt
    title Capstone Project Timeline — Spring 2026
    dateFormat YYYY-MM-DD
    section Research
        Literature Review           :done, 2026-03-06, 4d
        Methodology Definition      :done, 2026-03-10, 1d
    section Data Engineering
        Corpus curation + cleaning  :done, 2026-03-11, 3d
        MIDI → WAV synthesis        :done, 2026-03-14, 2d
        Audio chunking pipeline     :done, 2026-03-16, 2d
    section Modelling
        Baseline models (LSTM/GRU)  :done, 2026-03-18, 5d
        MERT fine-tuning v1–v4      :done, 2026-03-23, 7d
        CNN projection head (v5)    :done, 2026-03-30, 4d
    section Evaluation
        FAISS indexing + Recall@K   :done, 2026-04-03, 4d
        Test set + error analysis   :done, 2026-04-07, 3d
    section Deliverables
        Capstone Report             :active, 2026-04-10, 5d
        GitHub + Documentation      :active, 2026-04-10, 5d
        Final Presentation          :active, 2026-04-12, 3d
```
