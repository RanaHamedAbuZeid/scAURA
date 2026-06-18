## Pipeline Execution Order

To reproduce the results, execute the scripts in the following order:

1. **`Part1.py`  ChromVAR Features Extraction**
2. **`Part2.py`  ChromVAR GPU Pipeline**
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


## 📄 2. Part-2

### Overview

`Part2.py`

This script serves as a complete, paper-accurate implementation of the chromVAR pipeline (Schep et al., Nat Methods 2017), directly translated from the original R source code. It is heavily optimized via a hybrid architecture using **CPU-sparse data slicing and GPU-dense tensor arithmetic** to ensure high performance on large datasets without hitting RAM/VRAM bottlenecks.

### Key Steps & Workflow

The script handles data validation, extensive biological filtering, background matching, and execution across 9 discrete stages (Stages 0 to 8):

* **Checkpoint Ingestion & Validation (Stage 0):** Automatically loads intermediate outputs (`step1_data`, `step4_sequences`, `step5_jaspar`, `step7_matrix`) from the previous script and ensures all peak and motif array dimensions match perfectly. It also computes or loads peak GC content percentages.


* **Strict Sample & Feature Filtering (Stages 1-2):** Filters out low-quality cells based on a configurable minimum fragment depth threshold (default $\ge 1500$). It simultaneously drops empty peaks with no fragments or those with invalid GC content (failures where GC is exactly $0$ or $1$).


* **Expectation Modeling (Stage 3):** Calculates the fraction of total expected reads mapped per peak, ensuring the cumulative baseline expectation sums to $1.0$.


* **Mahalanobis Bias Binning (Stage 4):** Transforms peak GC metrics and $\log_{10}(\text{depth} + 1)$ values into a decorrelated Mahalanobis space, then projects them onto an even $50 \times 50$ spatial grid.


* **GPU-Accelerated Background Sampling (Stage 5):** Leverages PyTorch multinomial sampling on the GPU to generate background peak indices across multiple iterations (default $50$ iterations), perfectly mimicking the exact random seed structure of the R implementation.


* **Hybrid Deviation Arithmetic (Stage 6):** Combines efficient SciPy CSC column slicing on the CPU with dense vector operations on the GPU. It computes observed versus background deviations per cell and marks unreliable low-expectation inputs as `NaN` values.


* **TF Variability & Redundancy Filtering (Stage 7):** Gauges transcription factor (TF) variability across cells, sorts the highest-variance TFs, performs a greedy correlation analysis to drop redundant motifs (correlation threshold $> 0.9$), and screens for high-variability motifs (variance threshold $> 1.5$).


* **Consolidated Data Export (Stage 8):** Writes clean, standalone output matrices, background references, filtered sparse counts, full logs, and ready-to-train deep learning tensors to a dedicated final directory.



### How to Run

```bash
python Part2.py

```

#### Customized Threshold Run (e.g., higher cell depth and custom iteration counts)

```bash
python Part2.py --min-depth 2000 --n-iter 50 --variability-cutoff 1.5 --corr-cutoff 0.9

```

### Main Outputs Generated

All outputs are organized and stored inside a dedicated workspace (`/chromvar_final_gh38/`):

* `X_dnn.npy` & `X_dnn.csv`: Final curated input tensor containing cells $\times$ highly variable, non-redundant motif Z-scores tailored for Deep Neural Network training.


* `filtered_counts.npz` & `motif_matrix_filtered.npz`: Cleaned sparse SciPy arrays representing cell counts and motif mappings after rigorous peak-filtering.


* `chromvar_deviations.npy` & `chromvar_zscores.npy`: Complete computed baseline matrices before downstream model-specific filtering.


* `background_peaks.npy`: Main matrix containing sampled matching background peak groups.


* `variability.csv`: Tabular breakdown summarizing individual motif IDs, names, metric variance scores, and inclusion status.


* `pipeline_summary.txt`: A clean, human-readable execution summary recording dataset dimensions, parameter settings, filtering progress, and top-ranking variable elements.
