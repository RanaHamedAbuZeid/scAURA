<div align="center">

# scAURA
### Single-Cell ATAC-seq Unified Regulatory Analysis

*Cell-type annotation and out-of-distribution detection for scATAC-seq data*  
*using chromatin accessibility–derived transcription factor motif signatures*

<br>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)


<br>

| Metric | Value |
|--------|-------|
| 🎯 Ensemble Accuracy | **87.75%** |
| ⚖️ Balanced Accuracy | **86.02%** |
| 📐 Macro F1 | **86.22%** |
| 🔍 Best OOD AUROC | **0.8787** (KNN) |
| 🧬 Cell Types | **13 PBMC lineages** |
| ⏱️ Training Time | **23.1 min** (5 models, GPU) |

</div>

---

## Overview

**scAURA** is a deep learning framework for **supervised immune cell-type annotation** and **out-of-distribution (OOD) detection** in single-cell ATAC-seq data.

Rather than operating on the raw, dataset-specific peak matrix, scAURA uses **ChromVAR transcription factor motif deviation scores** as input — a compact, biologically interpretable, and cross-dataset-compatible feature representation derived from the JASPAR 2024 CORE database (879 vertebrate motifs). This design enables the model to classify known immune populations while simultaneously flagging novel or aberrant cell types that were never seen during training.

**Key capabilities:**

- ✅ Multi-class immune cell classification across 13 PBMC lineages
- ✅ Out-of-distribution detection via 6 complementary uncertainty methods
- ✅ Ensemble of 5 independent WideNet models for robust, calibrated predictions
- ✅ Full reproducibility — fixed seeds, saved checkpoints, and split indices
- ✅ End-to-end pipeline from raw `.h5ad` to trained model and OOD scores

---

## Architecture

scAURA is built on a **5-model deep ensemble** of WideNet feed-forward networks. Each model is trained independently from a different random seed, and their probability outputs are averaged at inference time.

```
Input: 879-dim ChromVAR TF motif deviation vector
       │
       ▼
  ┌─────────────────────────────┐
  │  Linear(879 → 1024)         │
  │  BatchNorm  ·  ReLU         │
  │  Dropout(p=0.4)             │
  ├─────────────────────────────┤
  │  Linear(1024 → 512)         │
  │  BatchNorm  ·  ReLU         │
  │  Dropout(p=0.4)             │
  ├─────────────────────────────┤
  │  Linear(512 → 256)          │  ← penultimate (feature) layer
  │  BatchNorm  ·  ReLU         │
  │  Dropout(p=0.4)             │
  ├─────────────────────────────┤
  │  Linear(256 → 13)           │  ← classification head
  └─────────────────────────────┘
       │
       ▼
  Softmax → cell-type probabilities
```

| Component | Detail |
|-----------|--------|
| Parameters per model | 1,564,173 |
| Total parameters (ensemble) | 7,820,865 |
| Ensemble size | 5 models (seeds 42 – 46) |
| Loss function | Label-smoothing cross-entropy (ε = 0.1) + class weighting |
| Optimizer | AdamW (lr = 1e-3, weight decay = 5e-5) |
| Scheduler | OneCycleLR (per-batch, cosine annealing) |
| Regularization | Dropout (p = 0.4) · MixUp (α = 0.2) · Gradient clipping (norm = 1.0) |
| Early stopping | Patience = 40 epochs · Max 150 epochs |

---

## Results

### Classification Performance — Test Set (n = 3,894 cells)

| Metric | Score |
|--------|-------|
| **Accuracy** | **87.75%** |
| **Balanced Accuracy** | **86.02%** |
| **Macro F1** | **86.22%** |
| **Weighted F1** | **87.60%** |
| **Cohen's κ** | **0.8606** |

### Per-class Performance

| Cell Type | N | Precision | Recall | F1 |
|-----------|---|-----------|--------|----|
| Monocyte | 779 | 0.9961 | 0.9769 | **0.9864** |
| Naive B | 326 | 0.9493 | 0.9755 | **0.9622** |
| pDC | 11 | 1.0000 | 0.9091 | **0.9524** |
| NK | 285 | 0.9375 | 0.9474 | **0.9424** |
| Treg | 291 | 0.9161 | 0.9003 | **0.9081** |
| Memory B | 143 | 0.9333 | 0.8811 | **0.9065** |
| cDC | 27 | 0.8125 | 0.9630 | **0.8814** |
| Effector Memory CD8 T | 273 | 0.8169 | 0.8498 | **0.8330** |
| Memory CD4 T | 669 | 0.8281 | 0.8640 | **0.8456** |
| MAIT | 88 | 0.8372 | 0.8182 | **0.8276** |
| Naive CD4 T | 503 | 0.8157 | 0.8449 | **0.8301** |
| Naive CD8 T | 335 | 0.7522 | 0.7701 | **0.7611** |
| Central Memory CD8 T | 164 | 0.7054 | 0.4817 | **0.5725** |

### OOD Detection — Plasma Cells vs 13 ID Types

Six uncertainty estimators are evaluated simultaneously. Penultimate-layer features are used for geometry-based scoring (Mahalanobis, KNN).

| Method | AUROC | AUPR-OUT | FPR@95 TPR |
|--------|-------|----------|-----------|
| **KNN Distance** | **0.8787** | **0.4853** | 0.4964 |
| Mahalanobis | 0.8643 | 0.3438 | 0.4556 |
| MSP Ensemble | 0.8590 | 0.3829 | **0.3901** |
| Energy Score | 0.8314 | 0.3597 | 0.6328 |
| Ensemble Variance | 0.7589 | 0.1665 | 0.4605 |
| MC Dropout Variance | 0.7174 | 0.1440 | 0.4854 |

### Ensemble vs Individual Models

| Model | Seed | Best Epoch | Val Acc | Test Acc |
|-------|------|-----------|---------|----------|
| M1 | 42 | 126 | 88.59% | 86.95% |
| M2 | 43 | 85 | 88.75% | 86.80% |
| M3 | 44 | 62 | 88.90% | 86.80% |
| M4 | 45 | 42 | 88.54% | 87.06% |
| M5 | 46 | 61 | 88.24% | 87.31% |
| **Ensemble** | — | — | — | **87.75% (+0.44%)** |

---

## Dataset

scAURA was developed and evaluated on the **healthy adult human PBMC scATAC-seq atlas** ([GSE184462](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184462)).

| Split | Cells | Description |
|-------|-------|-------------|
| Train | 31,147 | Stratified 80% of 13 ID types |
| Validation | 3,893 | Stratified 10% of 13 ID types |
| Test (ID) | 3,894 | Stratified 10% of 13 ID types |
| Test (OOD) | 415 | Plasma cells — held out entirely |
| **Total** | **39,349** | |

Each cell is represented by an **879-dimensional vector** of ChromVAR bias-corrected TF motif deviation scores, derived from the [JASPAR 2024 CORE vertebrates](https://jaspar.elixir.no) non-redundant database.

---

## Repository Structure

```
scAURA/
│
├── preprocessing/
│   ├── part1.py             # Steps 1–7: peaks, genome download, MOODS scanning
│   ├── part2.py             # ChromVAR bias-corrected deviation computation
│   ├── Part3.py             # Final feature matrix (X_dnn.npy — 879 motifs)
│   └── Part4.py             # Train / val / test / OOD data splitting
│
├── model/
│   └── Model7_ensemble_.py  # 5-WideNet ensemble: training + OOD evaluation
│
├── README.md
└── requirements.txt
```

---

## Installation
### Environment Specifications

| Package | Installed Version | Minimum Required |
| :--- | :--- | :--- |
| **numpy** | 1.26.4 | 1.21.0 |
| **pandas** | 2.3.3 | 1.3.0 |
| **scipy** | 1.15.3 | 1.7.0 |
| **torch** | 2.12.0.dev20260407 | - |
| **scikit-learn** | 1.7.2 | 1.0.0 |
| **scanpy** | 1.11.5 | 1.9.0 |
| **anndata** | 0.11.4 | 0.9.0 |
| **pyfaidx** | 0.9.0.3 | 0.7.0 |
| **requests** | 2.32.5 | 2.26.0 |
| **tqdm** | 4.67.1 | 4.60.0 |
| **matplotlib** | 3.10.8 | 3.4.0 |
| **seaborn** | 0.13.2 | 0.11.0 |
| **MOODS-python** | Installed | 1.9.4 |

## Usage

### Step 1 — Preprocessing

Run the four preprocessing scripts in order. Each step checkpoints its output — interrupted runs resume automatically from where they stopped.

```bash
# 1. Peak extraction, genome + JASPAR download, motif scanning
python preprocessing/part1.py

# 2. ChromVAR bias-corrected deviations  (GPU recommended)
python preprocessing/part2.py

# 3. Assemble the 879-motif feature matrix
python preprocessing/Part3.py

# 4. Create train / val / test / OOD splits
python preprocessing/Part4.py
```

> ⚠️ Before running, update the base directory paths at the top of each script to match your environment.

For HPC clusters, motif scanning in Step 1 can be parallelized across chunks:

```bash
# Run chunks in parallel (adjust --n-chunks as needed)
python preprocessing/part1.py --chunk 0 --chunk-size 50000
python preprocessing/part1.py --chunk 1 --chunk-size 50000
python preprocessing/part1.py --chunk 2 --chunk-size 50000
python preprocessing/part1.py --chunk 3 --chunk-size 50000

# Then merge and build the motif-peak matrix
python preprocessing/part1.py --merge --n-chunks 4
```

---

### Step 2 — Train the Ensemble

```bash
python model/Model7_ensemble_.py
```

**Optional arguments** (all defaults match the published configuration):

```bash
python model/Model7_ensemble_.py \
    --epochs       150         \
    --batch        256         \
    --lr           1e-3        \
    --dropout      0.4         \
    --label-smooth 0.1         \
    --mixup-alpha  0.2         \
    --mc-passes    30          \
    --patience     40          \
    --seeds  42 43 44 45 46
```

The script trains all 5 models sequentially, then runs full ensemble inference and evaluates all 6 OOD detection methods automatically. Checkpoints, metrics, and figures are saved to `stage_e_model7_fixed/`.

---

## Output Files

After a complete run, the following files are produced under `stage_e_model7_fixed/`:

```
stage_e_model7_fixed/
├── model_1_widenet_seed42.pt          # model checkpoint + fitted scaler
├── model_2_widenet_seed43.pt
├── model_3_widenet_seed44.pt
├── model_4_widenet_seed45.pt
├── model_5_widenet_seed46.pt
│
├── ensemble_predictions.npz           # mean probabilities + stacked per-model probs
├── mahalanobis_stats.npz              # class means + precision matrix (for scoring)
├── ood_scores_full.csv                # per-cell OOD scores — all 6 methods
│
├── ensemble_classification_metrics.json
├── ood_results.json
├── per_class_metrics.csv
├── confusion_matrix.csv
│
├── confusion_matrix.png
├── ood_roc_curves.png
└── ood_score_distributions.png
```

---

## Reproducibility

All stochastic components use fixed seeds:

| Component | Seed |
|-----------|------|
| Model training (5 runs) | 42, 43, 44, 45, 46 |
| Background peak sampling (ChromVAR) | 2017 (matching the original R paper) |
| Data splitting | 42 |
| NumPy / PyTorch global seeds | 42 |

cuDNN deterministic mode is enabled during training (`deterministic = True`, `benchmark = False`).

---
---

<div align="center">

Made with ❤️ for the single-cell genomics community

</div>
