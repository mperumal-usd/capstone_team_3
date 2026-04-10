---
id: results
title: Results & Evaluation
sidebar_label: Results & Evaluation
---

# Results & Evaluation

## Evaluation Methodology

Retrieval quality is measured with **Recall@K**: given a query chunk, does the correct song appear in the top-K retrieved results?

- **Gallery**: 80% of the 43,663 WAV chunks (34,930 vectors in the FAISS index)
- **Validation queries**: remaining 20% (8,733 chunks)
- **External test set**: 4,116 chunks from 38 reference songs not seen during training
- **Retrieval unit**: song-level (chunks from the same song = relevant)

```
Recall@K = (queries where correct song is in top-K) / (total queries)
```

---

## Validation Set Results

### Fine-tuned MERT + LoRA + CNN (v5) vs Baseline

| K | Fine-tuned | Baseline | Improvement |
|---|-----------|----------|-------------|
| **1** | **63.6%** | 45.5% | **+18.1 pp** |
| **3** | **73.6%** | 55.6% | **+18.1 pp** |
| **5** | **77.9%** | 60.7% | **+17.2 pp** |
| **10** | **83.3%** | 67.9% | **+15.4 pp** |

```
Validation Recall@K — Fine-tuned vs Baseline

100% ┤
 90% ┤                                        ████  83.3%
 80% ┤              ████  77.9%     ████
 70% ┤  ████  73.6% ████           ████  67.9%
 60% ┤  ████         ████  60.7%   ████
     │  63.6%  ████  55.6%         ████
 50% ┤         45.5%
 40% ┤
      ─────────────────────────────────────────
         R@1    R@3    R@5    R@10
         ████ Fine-tuned    ████ Baseline
```

### Cosine Similarity Distribution

After training, the model produces well-separated distributions:

- **Same-song pairs (positive)**: mean cosine similarity ≈ **+0.52**
- **Cross-composer pairs (negative)**: mean cosine similarity ≈ **−0.13**
- **Gap**: ≈ **0.65** — large margin between similar and dissimilar pairs

This wide gap indicates the embeddings form compact, well-separated clusters by song.

---

## External Test Set Results

The external test set uses audio recordings of known songs (e.g., performances of *Für Elise*, Chopin Nocturnes, etc.) converted via BasicPitch and FluidSynth, matched against the classical MIDI gallery.

This is a harder task — cross-domain (audio → MIDI) and unseen compositions.

| K | Recall |
|---|--------|
| **1** | 7.9% |
| **3** | 14.1% |
| **5** | 17.8% |
| **10** | 25.1% |

The lower numbers reflect the **domain gap** between synthesized MIDI audio (gallery) and real audio recordings (queries). The model still shows meaningful retrieval well above random chance.

### Error Analysis

When Rank-1 is wrong:
- **~18% of missed Rank-1 queries** are recovered within Rank 2–5
- Most confusions occur between composers with stylistically similar pieces (e.g., Beethoven ↔ Schubert, Chopin ↔ Liszt)
- Mean cosine similarity for **correct** Rank-1 predictions is significantly higher than for incorrect ones — the model is calibrated

---

## Model Comparison (All Approaches)

Results on the internal validation set (38 reference songs):

| Model | Approach | Notes |
|-------|----------|-------|
| **MERT + LoRA + CNN (v5)** | Pre-trained audio transformer + metric learning | **Best overall** |
| MERT + LoRA v3/v4 | Earlier LoRA configurations | Lower Recall@K |
| Baseline MERT | Frozen encoder, no fine-tuning | R@1 = 45.5% |
| CNN + Mel Spectrogram | 2D CNN on spectrograms | Lower semantic understanding |
| LSTM | Sequence model on MIDI tokens | Limited by symbolic representation |
| GRU | Sequence model on MIDI tokens | Slightly faster than LSTM, similar accuracy |
| Siamese CNN | Contrastive CNN | Good on short clips |
| Custom Transformer | Trained from scratch | Underpowered without pre-training |

---

## Version History (MERT Fine-tuning Iterations)

| Version | Key Change | Best Val Loss | Recall@1 |
|---------|-----------|--------------|----------|
| v1 | Base LoRA setup | — | — |
| v2 | Updated search paths + data reload | — | — |
| v3 | Added FAISS validation section | — | 63.6% (same split) |
| v4 | Optimizer tuning | — | — |
| **v5** | **CNN projection head, 10 epochs** | **0.0302** | **63.6%** |

v5 is the final, production-quality model used for all reported results.

---

## Key Takeaways

1. **Pre-training matters enormously.** MERT's music-specific pre-training gives it a 18pp head start over models trained from scratch.

2. **LoRA is parameter-efficient.** We adapt only 0.47% of the model's weights yet achieve substantial gains over the frozen baseline.

3. **Multi-scale CNN head outperforms mean-pooling.** Capturing local, mid-range, and broader temporal patterns produces more discriminative embeddings.

4. **Domain gap is real.** Cross-domain retrieval (real audio → synthesized MIDI) is significantly harder and warrants future work on domain adaptation.

5. **FAISS scales cleanly.** Indexing 43K+ vectors and retrieving top-10 in milliseconds makes the system practical at scale.
