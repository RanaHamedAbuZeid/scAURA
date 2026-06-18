## Pipeline Execution Order

To reproduce the results, execute the scripts in the following order:

1. **`Part1.py` (Current File) — ChromVAR Features Extraction**
2. *[File 2 Name]* — *[Brief Description]*
3. *[File 3 Name]* — *[Brief Description]*
4. *[File 4 Name]* — *[Brief Description]*

--- 
##  1. ChromVAR Feature Extraction Pipeline

### Overview

First File summary: 

This script is a **highly optimized, low-RAM implementation of ChromVAR** designed to run within a **6 GB RAM budget** for large-scale single-cell datasets (e.g., 39K cells × 196K peaks). It bypasses memory-heavy dense matrix conversions by using strict sparse computations (`scipy.sparse`) and column-slicing throughout the entire pipeline.

### Key Steps & Workflow

The script automatically executes the following 10 steps, utilizing a custom checkpoint manager to allow safe resuming if interrupted:

* **Data Loading & Peak Prep (Steps 1-2):** Loads the cell atlas (`.h5ad`) and extracts genomic peak coordinates into a standardized BED format.
* **Reference Acquisition (Steps 3-5):** Downloads the `hg38` reference genome and `JASPAR 2024 CORE` vertebrate motifs, then extracts fasta sequences for all peak regions.
* **Motif Scanning (Step 6):** Uses `MOODS` to scan peak sequences for transcription factor binding sites. It includes built-in progress tracking and failure handling.
* **Sparse Mapping (Step 7):** Builds a highly memory-efficient sparse Motif-Peak binary matrix.
* **Low-RAM ChromVAR (Step 8):** Computes background-peak selection, bias-corrected deviations, and Z-scores using strictly sparse math without expanding to a dense grid.
* **Feature Filtering (Step 9):** Applies a two-stage filter: drops highly correlated redundant motifs (threshold > 0.9) and removes low-variability motifs (Z-score SD < 1.5).
* **Artifact Export (Step 10):** Generates and saves the final inputs for your Deep Learning model (`X_dnn.npy` / `X_dnn.csv`), a full execution summary, and metadata.

### How to Run

Depending on your compute resources, you can run this script in **Standard Mode** (all-in-one) or **Chunked Mode** (parallel motif scanning).

#### Option A: Standard Mode (Recommended for uninterrupted single-shot execution)

```bash
python Part1.py

```

#### Option B: Chunked Mode (To break the heavy motif-scanning step into parts)

1. Run individual chunk processing (e.g., split into 4 chunks from 0 to 3):

```bash
python Part1.py --chunk 0 --chunk-size 50000
python Part1.py --chunk 1 --chunk-size 50000
python Part1.py --chunk 2 --chunk-size 50000
python Part1.py --chunk 3 --chunk-size 50000

```

2. Merge the chunks and complete the remaining ChromVAR steps automatically:

```bash
python Part1.py --merge --n-chunks 4

```

### Main Outputs Generated

* `X_dnn.npy` & `X_dnn.csv`: Final filtered Z-score matrix ready for Neural Network training.
* `chromvar_deviations.npy` / `chromvar_zscores.npy`: Raw computed matrices.
* `motif_variability.csv`: Detailed report of skipped vs. kept motifs.
* `CHROMVAR_SUMMARY.txt`: Detailed log summary of the run execution.

---
--- 
جاهز لتنظيم الملف الثاني أول ما تشاركيني تفاصيله أو ترفعيه!
