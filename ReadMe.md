# Melody Match using MusicBERT

This guide explains how to compute **melody similarity using MusicBERT**, a transformer-based model for symbolic music understanding. The workflow converts MIDI files into embeddings and compares them using cosine similarity.

---

## Table of Contents

```mermaid
mindmap
  root((MusicBERT<br/>Guide))
    Overview
      What MusicBERT Captures
      Pipeline Architecture
    Setup
      Dependencies
      Installation
    Implementation
      MIDI to Tokens
      Model Loading
      Embeddings
      Similarity Computation
    Advanced Features
      Melody Extraction
      Sliding Window
      Batch Processing
      FAISS Integration
    Use Cases
```

---

# Overview

MusicBERT converts symbolic music (MIDI) into **semantic embeddings** that capture:

* Pitch relationships
* Rhythm patterns
* Harmonic context
* Musical motifs and structure

Similarity between melodies is computed using **vector similarity** instead of rule-based matching.

---

# Complete Pipeline Architecture

```mermaid
flowchart TB
    Start([MIDI Files]) --> Extract[Melody Extraction]
    Extract --> Tokenize[REMI Tokenization]
    Tokenize --> Embed[MusicBERT Embeddings]
    Embed --> Storage{Storage Strategy}

    Storage -->|Direct Compare| Cosine[Cosine Similarity]
    Storage -->|Batch Processing| VectorDB[(Vector Database<br/>FAISS)]

    Cosine --> Results1[Similarity Score]
    VectorDB --> Search[Similarity Search]
    Search --> Results2[Top K Matches]

    Results1 --> Apps[Applications]
    Results2 --> Apps

    Apps --> Use1[Copyright Detection]
    Apps --> Use2[Music Recommendation]
    Apps --> Use3[Motif Discovery]
    Apps --> Use4[Dataset Clustering]

    style Start fill:#e1f5ff
    style Extract fill:#fff4e1
    style Tokenize fill:#ffe1f5
    style Embed fill:#e1ffe1
    style VectorDB fill:#f5e1ff
    style Apps fill:#ffe1e1
```

---

# Step 1 — Install Dependencies

```mermaid
graph LR
    A[Install Base<br/>Dependencies] --> B[Clone MusicBERT<br/>Repository]
    B --> C[Install MusicBERT<br/>Package]
    C --> D[Download Pretrained<br/>Weights]
    D --> E[Ready to Use]

    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style E fill:#c8e6c9
```

```bash
pip install music21 pretty_midi torch numpy scikit-learn
```

Clone and install MusicBERT:

```bash
git clone https://github.com/microsoft/muzic.git
cd muzic/musicbert
pip install -e .
```

Download pretrained weights:

```bash
bash scripts/download_pretrained.sh
```

---

# Step 2 — Convert MIDI → REMI Tokens

```mermaid
sequenceDiagram
    participant User
    participant MIDI as MIDI File
    participant Parser as MIDI Parser
    participant REMI as REMI Converter
    participant Tokens as Token Sequence

    User->>MIDI: Load melody1.mid
    MIDI->>Parser: Raw MIDI data
    Parser->>REMI: Parse notes, timing, velocity
    REMI->>Tokens: Convert to REMI tokens
    Tokens-->>User: Return token sequence
```

MusicBERT expects **REMI token representation**, not raw MIDI.

```python
import pretty_midi
from musicbert.preprocess import midi_to_remi

def midi_to_tokens(midi_path):
    tokens = midi_to_remi(midi_path)
    return tokens
```

---

# Step 3 — Load MusicBERT Model

```mermaid
stateDiagram-v2
    [*] --> LoadCheckpoint: Load pretrained weights
    LoadCheckpoint --> InitModel: Initialize MusicBERT
    InitModel --> SetEvalMode: model.eval()
    SetEvalMode --> Ready: Model ready for inference
    Ready --> [*]
```

```python
import torch
from musicbert.model.musicbert import MusicBERTModel

model = MusicBERTModel.from_pretrained(
    "musicbert_base",
    checkpoint_file="checkpoint_last_musicbert_base.pt"
)

model.eval()
```

---

# Step 4 — Generate Embeddings

```mermaid
graph TD
    A[Token Sequence] --> B[Convert to Tensor]
    B --> C[Add Batch Dimension]
    C --> D[Forward Pass<br/>through MusicBERT]
    D --> E[Extract CLS Token<br/>from last_hidden_state]
    E --> F[Convert to NumPy]
    F --> G[Embedding Vector<br/>768 dimensions]

    style A fill:#e1f5fe
    style D fill:#fff9c4
    style E fill:#f8bbd0
    style G fill:#c8e6c9
```

Use the CLS token embedding as the **global melody representation**.

```python
def get_embedding(tokens):
    input_ids = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids)

    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze().numpy()
```

---

# Step 5 — Compute Melody Similarity

```mermaid
flowchart LR
    M1[Melody 1<br/>MIDI] --> T1[Tokens 1]
    M2[Melody 2<br/>MIDI] --> T2[Tokens 2]

    T1 --> E1[Embedding 1<br/>768-dim vector]
    T2 --> E2[Embedding 2<br/>768-dim vector]

    E1 --> CS[Cosine Similarity<br/>Computation]
    E2 --> CS

    CS --> Score[Similarity Score<br/>0.0 to 1.0]

    style M1 fill:#bbdefb
    style M2 fill:#bbdefb
    style E1 fill:#c5e1a5
    style E2 fill:#c5e1a5
    style Score fill:#ffccbc
```

Use cosine similarity between embeddings.

```python
from sklearn.metrics.pairwise import cosine_similarity

tokens1 = midi_to_tokens("melody1.mid")
tokens2 = midi_to_tokens("melody2.mid")

emb1 = get_embedding(tokens1)
emb2 = get_embedding(tokens2)

similarity = cosine_similarity([emb1], [emb2])[0][0]
print("Melody similarity:", similarity)
```

### Similarity Interpretation

```mermaid
graph LR
    A[Similarity Score] --> B{Score Range}
    B -->|0.9 - 1.0| C[Nearly Identical<br/>Melodies]
    B -->|0.7 - 0.9| D[Strong<br/>Similarity]
    B -->|0.4 - 0.7| E[Moderate<br/>Similarity]
    B -->|< 0.4| F[Different<br/>Melodies]

    style C fill:#4caf50
    style D fill:#8bc34a
    style E fill:#ffc107
    style F fill:#f44336
```

| Score     | Meaning                   |
| --------- | ------------------------- |
| 0.9 – 1.0 | Nearly identical melodies |
| 0.7 – 0.9 | Strong similarity         |
| 0.4 – 0.7 | Moderate similarity       |
| < 0.4     | Different melodies        |

---

# Step 6 — Extract Melody Track (Recommended)

```mermaid
flowchart TD
    A[Multi-track MIDI File] --> B[Load with pretty_midi]
    B --> C[Analyze all instruments]
    C --> D{Find track with<br/>most notes}
    D --> E[Extract melody track]
    E --> F[Create new MIDI file<br/>with melody only]
    F --> G[Save as _melody.mid]

    style A fill:#e1bee7
    style D fill:#fff59d
    style F fill:#80deea
    style G fill:#a5d6a7
```

Most MIDI files contain multiple tracks. Extract the **dominant melody track** first.

```python
def extract_melody(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    melody = max(midi.instruments, key=lambda inst: len(inst.notes))

    new_midi = pretty_midi.PrettyMIDI()
    new_midi.instruments.append(melody)

    out_path = midi_path.replace(".mid", "_melody.mid")
    new_midi.write(out_path)
    return out_path
```

---

# Step 7 — Sliding Window Similarity (Motif Detection)

```mermaid
sequenceDiagram
    participant Full as Full Melody
    participant Window as Sliding Window
    participant Target as Target Melody
    participant Scores as Score Array

    loop For each window position
        Full->>Window: Extract chunk (128 tokens)
        Window->>Window: Get embedding
        Target->>Target: Get embedding
        Window->>Scores: Compute similarity
    end

    Scores->>Scores: Find maximum score
    Scores-->>Full: Return best match score
```

Useful for detecting **partial similarity or plagiarism**.

```python
def sliding_similarity(tokens1, tokens2, window=128):
    scores = []

    for i in range(0, len(tokens1) - window, window):
        chunk = tokens1[i:i+window]
        emb1 = get_embedding(chunk)
        emb2 = get_embedding(tokens2)
        score = cosine_similarity([emb1], [emb2])[0][0]
        scores.append(score)

    return max(scores)
```

---

# Step 8 — Full End-to-End Example

```mermaid
graph TB
    Start([song1.mid & song2.mid]) --> Step1[Extract Melody Tracks]
    Step1 --> Step2[Convert to REMI Tokens]
    Step2 --> Step3[Generate Embeddings]
    Step3 --> Step4[Compute Cosine Similarity]
    Step4 --> Result[Final Similarity Score]

    style Start fill:#e1f5fe
    style Step1 fill:#f3e5f5
    style Step2 fill:#e8f5e9
    style Step3 fill:#fff9c4
    style Step4 fill:#ffccbc
    style Result fill:#c8e6c9
```

```python
melody1 = extract_melody("song1.mid")
melody2 = extract_melody("song2.mid")

tokens1 = midi_to_tokens(melody1)
tokens2 = midi_to_tokens(melody2)

emb1 = get_embedding(tokens1)
emb2 = get_embedding(tokens2)

similarity = cosine_similarity([emb1], [emb2])[0][0]
print("Final similarity score:", similarity)
```

---

# Why MusicBERT Works Better Than Traditional Methods

```mermaid
graph TB
    subgraph Traditional Methods
    ED[Edit Distance<br/>❌ No key invariance<br/>❌ No rhythm handling<br/>❌ No semantic understanding]
    DTW[Dynamic Time Warping<br/>❌ No key invariance<br/>✅ Rhythm handling<br/>❌ No semantic understanding]
    end

    subgraph Deep Learning
    MB[MusicBERT<br/>✅ Key invariance<br/>✅ Rhythm handling<br/>✅ Semantic understanding]
    end

    Music[Musical Data] --> ED
    Music --> DTW
    Music --> MB

    MB --> Best[Best Performance<br/>Captures Musical Meaning]

    style ED fill:#ffcdd2
    style DTW fill:#fff9c4
    style MB fill:#c8e6c9
    style Best fill:#a5d6a7
```

| Method        | Key Invariance | Rhythm Handling | Semantic Understanding |
| ------------- | -------------- | --------------- | ---------------------- |
| Edit Distance | ❌              | ❌               | ❌                      |
| DTW           | ❌              | ✅               | ❌                      |
| MusicBERT     | ✅              | ✅               | ✅                      |

MusicBERT captures **musical meaning**, not just note sequences.

---

# Optional: Batch Similarity for Many MIDI Files

```mermaid
flowchart LR
    Files[MIDI Files Collection] --> Loop{For each file}
    Loop --> Extract[Extract Melody]
    Extract --> Tokenize[Tokenize]
    Tokenize --> Embed[Get Embedding]
    Embed --> Store[Store in Dictionary]
    Store --> Loop
    Loop --> DB[(Embedding Database)]

    style Files fill:#e1f5fe
    style Loop fill:#fff9c4
    style DB fill:#c8e6c9
```

```python
def build_embedding_database(midi_files):
    database = {}

    for path in midi_files:
        melody = extract_melody(path)
        tokens = midi_to_tokens(melody)
        embedding = get_embedding(tokens)
        database[path] = embedding

    return database
```

---

# Optional: Build Melody Search Engine with FAISS

```mermaid
graph TD
    A[Embedding Database] --> B[Convert to NumPy Array<br/>float32]
    B --> C[Create FAISS Index<br/>IndexFlatL2]
    C --> D[Add all embeddings<br/>to index]
    D --> E[Index Ready]

    Q[Query MIDI] --> QT[Tokenize & Embed]
    QT --> QS[Search Index<br/>k nearest neighbors]

    E --> QS
    QS --> R[Return Top K<br/>Similar Melodies]

    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style E fill:#c8e6c9
    style Q fill:#fff9c4
    style R fill:#ffccbc
```

```bash
pip install faiss-cpu
```

```python
import faiss
import numpy as np

embeddings = np.array(list(database.values())).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

query = get_embedding(midi_to_tokens("query.mid")).astype("float32")
D, I = index.search(np.array([query]), k=5)
print("Most similar melodies:", I)
```

---

# Recommended System Architecture

```mermaid
graph TB
    Input[MIDI Files] --> Prep[Preprocessing Pipeline]

    subgraph Prep[Preprocessing Pipeline]
        P1[Melody Extraction] --> P2[REMI Tokenization]
    end

    P2 --> Model[MusicBERT Model]

    subgraph Model[MusicBERT Inference]
        M1[Token Embedding] --> M2[Transformer Layers]
        M2 --> M3[CLS Token Extraction]
    end

    M3 --> Storage[Storage Layer]

    subgraph Storage[Storage Layer]
        S1[(Vector Database<br/>FAISS)]
        S2[(Metadata Store<br/>File paths, titles)]
    end

    Storage --> Apps[Applications]

    subgraph Apps[Applications]
        A1[Similarity Search]
        A2[Plagiarism Detection]
        A3[Music Recommendation]
        A4[Clustering & Analysis]
    end

    style Input fill:#e1f5fe
    style Prep fill:#f3e5f5
    style Model fill:#e8f5e9
    style Storage fill:#fff9c4
    style Apps fill:#ffccbc
```

---

# Use Cases

```mermaid
mindmap
  root((MusicBERT<br/>Applications))
    Similarity Analysis
      Melody Comparison
      Cover Song Detection
      Musical Fingerprinting
    Copyright Protection
      Plagiarism Detection
      Copyright Infringement
      Derivative Work Analysis
    Music Discovery
      Recommendation Systems
      Similar Song Search
      Query-by-humming
    Research & Analysis
      Motif Discovery
      Dataset Clustering
      Musical Pattern Analysis
      Genre Classification
```

* **Melody similarity scoring** — Compare two melodies for similarity
* **Copyright / plagiarism detection** — Detect unauthorized copying
* **Music recommendation** — Find similar songs for recommendations
* **Motif discovery** — Identify recurring musical patterns
* **Dataset clustering** — Group similar melodies together
* **Query-by-humming** — Match hummed melodies (symbolic form)

---

# Implementation Timeline

```mermaid
gantt
    title Project Implementation Timeline
    dateFormat YYYY-MM-DD
    section Setup
    Install dependencies           :setup1, 2026-03-06, 1d
    Clone & install MusicBERT     :setup2, after setup1, 1d
    Download pretrained weights    :setup3, after setup2, 1d
    section Core Development
    Implement MIDI preprocessing   :dev1, after setup3, 2d
    Implement embedding generation :dev2, after dev1, 2d
    Implement similarity compute   :dev3, after dev2, 1d
    section Advanced Features
    Batch processing system        :adv1, after dev3, 2d
    FAISS integration             :adv2, after adv1, 2d
    Sliding window detection       :adv3, after adv2, 2d
    section Testing & Deployment
    Testing & validation           :test, after adv3, 3d
    Optimization & deployment      :deploy, after test, 2d
```

---

# Performance Considerations

```mermaid
graph LR
    subgraph Input Optimization
    I1[Melody Extraction<br/>Reduces noise]
    I2[Token Length Limit<br/>Faster inference]
    end

    subgraph Model Optimization
    M1[Batch Processing<br/>Process multiple at once]
    M2[GPU Acceleration<br/>CUDA support]
    M3[Model Quantization<br/>Reduced memory]
    end

    subgraph Storage Optimization
    S1[FAISS Indexing<br/>Fast similarity search]
    S2[Embedding Caching<br/>Avoid recomputation]
    end

    Input --> I1
    Input --> I2
    I1 --> M1
    I2 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> S1
    S1 --> S2
    S2 --> Output[Optimized System]

    style I1 fill:#e1f5fe
    style M2 fill:#f3e5f5
    style S1 fill:#e8f5e9
    style Output fill:#c8e6c9
```

---

# Next Steps

```mermaid
flowchart TD
    Current[Current Implementation] --> Branch{Enhancement Path}

    Branch -->|Audio Integration| A1[MIDI to WAV Conversion]
    Branch -->|API Development| A2[REST API for Comparison]
    Branch -->|Scalability| A3[Batch Processing System]
    Branch -->|Customization| A4[Fine-tune for Genre]

    A1 --> A1B[Audio Playback Feature]
    A2 --> A2B[Web Interface]
    A3 --> A3B[Process Thousands of Songs]
    A4 --> A4B[Domain-Specific Models]

    A1B --> Production[Production System]
    A2B --> Production
    A3B --> Production
    A4B --> Production

    style Current fill:#e1f5fe
    style Branch fill:#fff9c4
    style Production fill:#c8e6c9
```

You can extend this system by:

* **Converting MIDI → WAV** for audio playback
* **Building a web API** for melody comparison
* **Running batch similarity** across thousands of songs
* **Fine-tuning MusicBERT** for genre-specific similarity

---

# Complete Code Example

```python
# Complete working example combining all steps

import pretty_midi
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from musicbert.preprocess import midi_to_remi
from musicbert.model.musicbert import MusicBERTModel

class MelodyMatcher:
    def __init__(self, model_path="musicbert_base", checkpoint="checkpoint_last_musicbert_base.pt"):
        """Initialize MusicBERT model"""
        self.model = MusicBERTModel.from_pretrained(model_path, checkpoint_file=checkpoint)
        self.model.eval()
        self.cache = {}

    def extract_melody(self, midi_path):
        """Extract dominant melody track from MIDI"""
        midi = pretty_midi.PrettyMIDI(midi_path)
        melody = max(midi.instruments, key=lambda inst: len(inst.notes))
        new_midi = pretty_midi.PrettyMIDI()
        new_midi.instruments.append(melody)
        out_path = midi_path.replace(".mid", "_melody.mid")
        new_midi.write(out_path)
        return out_path

    def get_embedding(self, midi_path):
        """Get MusicBERT embedding for a MIDI file"""
        if midi_path in self.cache:
            return self.cache[midi_path]

        tokens = midi_to_remi(midi_path)
        input_ids = torch.tensor(tokens).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids)

        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        self.cache[midi_path] = embedding
        return embedding

    def compute_similarity(self, midi1, midi2):
        """Compute similarity between two MIDI files"""
        emb1 = self.get_embedding(midi1)
        emb2 = self.get_embedding(midi2)
        return cosine_similarity([emb1], [emb2])[0][0]

    def find_similar(self, query_midi, database_midis, top_k=5):
        """Find top-k similar melodies from database"""
        query_emb = self.get_embedding(query_midi)
        similarities = []

        for midi_path in database_midis:
            emb = self.get_embedding(midi_path)
            sim = cosine_similarity([query_emb], [emb])[0][0]
            similarities.append((midi_path, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Usage example
if __name__ == "__main__":
    matcher = MelodyMatcher()

    # Extract melodies
    melody1 = matcher.extract_melody("song1.mid")
    melody2 = matcher.extract_melody("song2.mid")

    # Compute similarity
    similarity = matcher.compute_similarity(melody1, melody2)
    print(f"Similarity: {similarity:.4f}")

    # Find similar songs in database
    database = ["song1.mid", "song2.mid", "song3.mid"]
    results = matcher.find_similar("query.mid", database, top_k=3)

    print("\nTop matches:")
    for path, score in results:
        print(f"{path}: {score:.4f}")
```

---

# End of Guide

**Created with enhanced Mermaid visualizations for better understanding of the MusicBERT melody matching workflow.**
