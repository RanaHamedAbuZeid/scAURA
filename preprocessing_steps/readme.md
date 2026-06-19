## Pipeline Execution Order ( Preprocessing Step )


# scAURA — ChromVAR Preprocessing Pipeline

This repository contains the preprocessing pipeline for **scAURA**, a single-cell chromatin accessibility model for PBMC immune cell classification and out-of-distribution (OOD) detection.  
The pipeline converts a raw scATAC-seq peak matrix into a TF motif deviation matrix ready for model training.

---

## Input Data

| Item | Details |
|------|---------|
| **File** | `PBMC_atlas.h5ad` |
| **Format** | AnnData (`.h5ad`) — single-cell ATAC-seq fragment counts |
| **Size** | 39,441 cells × ~196,000 chromatin accessibility peaks |
| **Cell types** | 14 annotated PBMC types stored in `obs['celltypes']` |
| **OOD class** | Plasma cell (415 cells) — held out for OOD evaluation |

> Important Note:
>  Before running the pipeline, please open part1.py (and subsequent scripts) and update the hardcoded paths in the PATHS section to match your local machine's directory where the dataset is stored.

---

## External Resources (auto-downloaded by the pipeline)

| Resource | Source | Size |
|----------|--------|------|
| Human genome reference (hg38) | [UCSC Genome Browser](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz) | ~3 GB |
| JASPAR 2024 CORE Vertebrates motifs | [JASPAR](https://jaspar.elixir.no) | ~10 MB |

Both files are downloaded automatically when running `part1.py` for the first time.  
If the download fails, the script will print the manual download URL.

---

## Requirements

```bash
pip install numpy pandas scipy scanpy scikit-learn tqdm pyfaidx requests torch
pip install MOODS-python   # for motif scanning in part1.py
```

> A **GPU** is strongly recommended for `part2.py` (background peak sampling and chromVAR deviation computation). The pipeline will fall back to CPU automatically if no GPU is available.

---

## Before Running — Update Paths

Open each script and update the base directory to match your system:

```python
# In part1.py, part2.py, Part3.py, Part4.py — change this line:
"/home/rhamed/results/pbmc_atlas/"  →  your own base path

# In part1.py — update the input file path:
adata = sc.read_h5ad("/home/rhamed/results/pbmc_atlas/PBMC_atlas.h5ad")
```

---

## Pipeline Overview

```
part1.py  →  part2.py  →  Part3.py  →  Part4.py
```

Each script saves intermediate checkpoints automatically.  
If a run is interrupted, re-running the same script will resume from where it stopped.

---

## Step-by-Step Instructions

### Step 1 — Prepare peaks, download references, scan motifs

```bash
python part1.py
```

This runs steps 1–7 in sequence:
- Loads `PBMC_atlas.h5ad`
- Extracts peak coordinates and DNA sequences (uses hg38)
- Downloads JASPAR 2024 motif PWMs
- Scans all peaks for transcription factor binding motifs (MOODS)
- Builds a motif × peak binary matrix

**On large HPC clusters** — motif scanning can be split into parallel chunks:

```bash
# Run 4 chunks in parallel (adjust --n-chunks to match your setup)
python part1.py --chunk 0 --chunk-size 50000
python part1.py --chunk 1 --chunk-size 50000
python part1.py --chunk 2 --chunk-size 50000
python part1.py --chunk 3 --chunk-size 50000

# Then merge and finish
python part1.py --merge --n-chunks 4
```

---

### Step 2 — Compute chromVAR TF deviation scores

```bash
python part2.py
```

Computes bias-corrected chromVAR deviation scores and z-scores for all 879 motifs across all cells.

**Optional parameters** (defaults are paper-recommended values):

```bash
python part2.py --min-depth 1500   # minimum fragment depth per cell
python part2.py --n-iter 50        # background sampling iterations
python part2.py --n-bins 50        # Mahalanobis bias bins
```

**Main outputs** (saved to `chromvar_final_gh38/`):

| File | Description |
|------|-------------|
| `chromvar_deviations.npy` | Bias-corrected deviation scores (cells × 879 motifs) |
| `chromvar_zscores.npy` | Z-scores (cells × 879 motifs) |
| `cell_ids_kept.npy` | Cell IDs that passed quality filtering |
| `variability.csv` | Per-motif variability statistics |

---

### Step 3 — Build the final feature matrix

```bash
python Part3.py
```

Assembles the final input matrix for the model using all 879 motifs (no filtering applied).

**Outputs** (saved to `chromvar_final_gh38/`):

| File | Description |
|------|-------------|
| `X_dnn.npy` | Feature matrix — 39,349 cells × 879 motifs (primary input) |
| `X_dnn_z.npy` | Same matrix using z-scores (alternative) |
| `X_dnn.csv` | Same as above with cell ID and motif name headers |
| `X_dnn_motifs.csv` | Motif metadata including immune TF annotations |

---

### Step 4 — Split data for training

```bash
python Part4.py
```

Splits cells into train / validation / test sets with Plasma cells held out as the OOD class.

**Optional parameters:**

```bash
python Part4.py --ood "Plasma cell"   # OOD class (default)
python Part4.py --train 0.80 --val 0.10 --test 0.10   # split ratios (default)
python Part4.py --seed 42             # random seed (default)
```

**Outputs** (saved to `chromvar_final_gh38/splits_fixed/`):

| File | Description |
|------|-------------|
| `X_train.npy` / `labels_train.npy` | Training set (31,147 cells) |
| `X_val.npy` / `labels_val.npy` | Validation set (3,893 cells) |
| `X_test_id.npy` / `labels_test_id.npy` | In-distribution test set (3,894 cells) |
| `X_test_ood.npy` / `labels_test_ood.npy` | OOD test set — Plasma cells (415 cells) |
| `label_encoder.npy` | Integer → cell type name mapping |
| `split_summary.txt` | Human-readable split report |

> OOD cells receive label `−1`. ID classes are encoded as integers `0–12`.

---
---

## Quick Reference

| Script | What it does | Approx. runtime |
|--------|-------------|-----------------|
| `part1.py` | Genome alignment & motif scanning | 6–24 hours (HPC) |
| `part2.py` | chromVAR deviation computation | 2–4 hours (GPU) |
| `Part3.py` | Build feature matrix | < 5 minutes |
| `Part4.py` | Train/val/test split | < 1 minute |

---

## Notes

- **Checkpointing**: Every step saves its result to disk. Restarting a script will automatically skip completed steps.
- **Memory**: `part2.py` requires approximately 6–8 GB of RAM. The pipeline is designed to stay within this budget by avoiding dense matrix operations.
- **Reproducibility**: All random operations use fixed seeds (`seed=42` for splitting, `seed=2017` for background sampling, matching the original chromVAR R implementation).
