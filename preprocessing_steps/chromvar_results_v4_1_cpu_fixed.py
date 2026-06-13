#!/usr/bin/env python3
"""
Complete ChromVAR Implementation - LOW RAM VERSION
=====================================================
Designed for 6 GB available RAM with 39K cells × 196K peaks.

Key constraint:
  Dense float64 matrix = 39441 × 196618 × 8 bytes ≈ 58 GB  → impossible
  Sparse nnz only      = 343M  × 8 bytes           ≈  2.7 GB → fine

Step 8 never calls .toarray() on the full matrix.
All cell_total / peak_total / E computations use sparse operations.
Background peak selection uses sparse column slicing only.
"""

import os
import sys
import pickle
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================

CHECKPOINT_DIR = Path("/home/rhamed/results/pbmc_atlas/checkpoints_v4_1_copy")
RESULTS_DIR    = Path("/home/rhamed/results/pbmc_atlas/chromvar_results_v4_1_copy")
LOG_DIR        = Path("/home/rhamed/results/pbmc_atlas/logs")
GENOME_DIR     = Path("/home/rhamed/results/pbmc_atlas/genome_data_38")

for d in [CHECKPOINT_DIR, RESULTS_DIR, GENOME_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'chromvar_complete_38.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)

    def save(self, step_name, data):
        path = self.checkpoint_dir / f"{step_name}.pkl"
        logger.info(f"💾 Saving: {step_name}")
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, step_name):
        path = self.checkpoint_dir / f"{step_name}.pkl"
        if path.exists():
            logger.info(f"📂 Loading: {step_name}")
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def exists(self, step_name):
        return (self.checkpoint_dir / f"{step_name}.pkl").exists()


checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def step1_load_data():
    step_name = "step1_data"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    logger.info("=" * 80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 80)

    adata = sc.read_h5ad("/home/rhamed/results/pbmc_atlas/PBMC_atlas.h5ad")
    logger.info(f"✅ Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")

    checkpoint_mgr.save(step_name, adata)
    return adata

# ============================================================================
# STEP 2: EXTRACT & PREPARE PEAKS
# ============================================================================

def step2_prepare_peaks(adata):
    step_name = "step2_peaks"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    logger.info("=" * 80)
    logger.info("STEP 2: PREPARING PEAKS")
    logger.info("=" * 80)

    peaks_list = []
    for peak in tqdm(adata.var_names, desc="Parsing peaks"):
        try:
            if ':' in peak and '-' in peak:
                chrom, coords = peak.split(':')
                start, end = coords.split('-')
            elif '_' in peak:
                parts = peak.split('_')
                chrom, start, end = parts[0], parts[1], parts[2]
            else:
                continue
            peaks_list.append({
                'chr':   chrom,
                'start': int(start),
                'end':   int(end),
                'name':  peak
            })
        except Exception:
            continue

    peaks_df = pd.DataFrame(peaks_list)
    logger.info(f"✅ Extracted {len(peaks_df):,} peaks")

    bed_file = RESULTS_DIR / "peaks.bed"
    peaks_df[['chr', 'start', 'end', 'name']].to_csv(
        bed_file, sep='\t', header=False, index=False
    )
    logger.info(f"📄 BED file: {bed_file}")

    checkpoint_mgr.save(step_name, peaks_df)
    return peaks_df

# ============================================================================
# STEP 3: DOWNLOAD GENOME REFERENCE
# ============================================================================

def step3_download_genome():
    step_name = "step3_genome"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    logger.info("=" * 80)
    logger.info("STEP 3: DOWNLOADING GENOME REFERENCE (hg38)")
    logger.info("=" * 80)

    # 1. تعديل اسم الملف المستهدف ليكون hg38
    genome_file = GENOME_DIR / "hg38.fa"

    if genome_file.exists():
        logger.info(f"✅ Genome already exists: {genome_file}")
        result = {'genome_file': str(genome_file), 'build': 'hg38'}
        checkpoint_mgr.save(step_name, result)
        return result

    # 2. تعديل الرسائل ورابط التحميل والملف المضغوط لـ hg38 الرسمي
    logger.info("Downloading hg38 genome (~3 GB) ...")
    genome_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    gz_file    = GENOME_DIR / "hg38.fa.gz"

    import urllib.request
    try:
        def reporthook(count, block_size, total_size):
            if count % 100 == 0:
                pct = int(count * block_size * 100 / total_size)
                logger.info(f"   Downloaded: {pct}%")

        urllib.request.urlretrieve(genome_url, gz_file, reporthook)
        logger.info("✅ Download complete")

        import gzip, shutil
        logger.info("Decompressing hg38 ...")
        with gzip.open(gz_file, 'rb') as fi, open(genome_file, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        gz_file.unlink()
        logger.info(f"✅ Genome ready: {genome_file}")

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.info(f"Download manually:\n  {genome_url}\n  → {GENOME_DIR}/\n  gunzip hg38.fa.gz")
        sys.exit(1)

    # 3. تعديل القاموس المرتجع لتوثيق الإصدار الجديد hg38
    result = {'genome_file': str(genome_file), 'build': 'hg38'}
    checkpoint_mgr.save(step_name, result)
    return result
# ============================================================================
# STEP 4: EXTRACT PEAK SEQUENCES FROM GENOME
# ============================================================================

def step4_extract_sequences(peaks_df, genome_info):
    step_name = "step4_sequences"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    logger.info("=" * 80)
    logger.info("STEP 4: EXTRACTING PEAK SEQUENCES")
    logger.info("=" * 80)

    try:
        from pyfaidx import Fasta
    except ImportError:
        logger.error("❌ pyfaidx not installed!  pip install pyfaidx")
        sys.exit(1)

    genome = Fasta(genome_info['genome_file'])
    sequences, failed = [], 0

    for _, row in tqdm(peaks_df.iterrows(), total=len(peaks_df), desc="Extracting"):
        try:
            seq = genome[row['chr']][row['start']:row['end']].seq.upper()
            sequences.append(seq)
        except Exception:
            sequences.append('')
            failed += 1

    logger.info(f"✅ Sequences: {len(sequences)-failed:,} ok  {failed} failed")

    peaks_df = peaks_df.copy()
    peaks_df['sequence']   = sequences
    peaks_df['seq_length'] = [len(s) for s in sequences]

    fasta_file = RESULTS_DIR / "peaks.fasta"
    with open(fasta_file, 'w') as f:
        for _, row in peaks_df.iterrows():
            if row['sequence']:
                f.write(f">{row['name']}\n{row['sequence']}\n")

    result = {
        'peaks_with_seq': peaks_df,
        'fasta_file':     str(fasta_file),
        'n_sequences':    len(peaks_df) - failed
    }
    checkpoint_mgr.save(step_name, result)
    return result

# ============================================================================
# STEP 5: DOWNLOAD JASPAR MOTIFS
# ============================================================================

def step5_download_jaspar_pwms():
    step_name = "step5_jaspar"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    logger.info("=" * 80)
    logger.info("STEP 5: DOWNLOADING JASPAR MOTIFS")
    logger.info("=" * 80)

    import requests

    jaspar_url = (
        "https://jaspar.elixir.no/download/data/2024/CORE/"
        "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
    )
    meme_file = GENOME_DIR / "jaspar_motifs.meme"

    if not meme_file.exists():
        logger.info("Downloading JASPAR 2024 CORE vertebrates ...")
        try:
            r = requests.get(jaspar_url)
            r.raise_for_status()
            meme_file.write_text(r.text)
            logger.info(f"✅ Downloaded: {meme_file}")
        except Exception as e:
            logger.error(f"❌ {e}")
            sys.exit(1)
    else:
        logger.info(f"✅ Already exists: {meme_file}")

    motif_ids, motif_names = [], []
    with open(meme_file) as f:
        for line in f:
            if line.startswith('MOTIF'):
                parts = line.strip().split()
                motif_ids.append(parts[1])
                motif_names.append(parts[2] if len(parts) > 2 else parts[1])

    logger.info(f"✅ Found {len(motif_ids)} motifs")

    motifs_df = pd.DataFrame({'motif_id': motif_ids, 'motif_name': motif_names})
    motifs_df.to_csv(RESULTS_DIR / "jaspar_motifs_list.csv", index=False)

    result = {
        'meme_file': str(meme_file),
        'n_motifs':  len(motif_ids),
        'motifs_df': motifs_df
    }
    checkpoint_mgr.save(step_name, result)
    return result

# ============================================================================
# STEP 6: SCAN SEQUENCES FOR MOTIFS USING MOODS
# ============================================================================

def step6_scan_motifs_moods(sequences_result, jaspar_result,
                             chunk_idx=0, chunk_size=50000):
    step_name = f"step6_chunk_{chunk_idx}"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    try:
        import MOODS.scan
        import MOODS.tools
    except ImportError:
        logger.error("❌ MOODS not installed!")
        sys.exit(1)

    peaks_df  = sequences_result['peaks_with_seq']
    meme_file = jaspar_result['meme_file']

    start = chunk_idx * chunk_size
    end   = min(start + chunk_size, len(peaks_df))
    peaks_chunk = peaks_df.iloc[start:end]
    logger.info(f"Chunk {chunk_idx}: peaks {start:,} → {end:,} ({len(peaks_chunk):,})")

    matrices, motif_ids = parse_meme_for_moods(meme_file)
    bg         = compute_background_frequencies(peaks_df['sequence'].tolist())
    thresholds = [MOODS.tools.threshold_from_p(m, bg, 5e-5) for m in matrices]

    matches_file = RESULTS_DIR / f"moods_matches_chunk{chunk_idx}.csv"
    buffer, total, failed = [], 0, 0

    with open(matches_file, 'w') as f:
        f.write("motif_id,sequence_name,n_matches,max_score\n")
        for _, row in tqdm(peaks_chunk.iterrows(), total=len(peaks_chunk),
                           desc=f"Chunk {chunk_idx}"):
            seq = row['sequence']
            if not seq or len(seq) < 10:
                failed += 1
                continue
            results = MOODS.scan.scan_dna(seq, matrices, bg, thresholds, 7)
            for mi, motif_matches in enumerate(results):
                if motif_matches:
                    buffer.append(
                        f"{motif_ids[mi]},{row['name']},"
                        f"{len(motif_matches)},"
                        f"{max(m.score for m in motif_matches)}\n"
                    )
                    total += 1
            if len(buffer) >= 5000:
                f.writelines(buffer)
                buffer = []
        if buffer:
            f.writelines(buffer)

    logger.info(f"✅ Chunk {chunk_idx}: {total:,} matches  {failed} failed")
    result = {'matches_file': str(matches_file), 'n_matches': total, 'chunk_idx': chunk_idx}
    checkpoint_mgr.save(step_name, result)
    return result


def step6_merge_chunks(n_chunks=4):
    step_name = "step6_fimo"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    logger.info("Merging chunks ...")
    all_files = [RESULTS_DIR / f"moods_matches_chunk{i}.csv" for i in range(n_chunks)]
    for f in all_files:
        if not f.exists():
            logger.error(f"Missing: {f}")
            sys.exit(1)

    fimo_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    logger.info(f"Total matches: {len(fimo_df):,}")
    fimo_df.to_csv(RESULTS_DIR / "moods_matches_all.csv", index=False)

    result = {'fimo_results': fimo_df, 'n_matches': len(fimo_df),
              'method': 'MOODS', 'pvalue_used': 5e-5}
    checkpoint_mgr.save(step_name, result)
    return result


def step6_run_full(sequences_result, jaspar_result):
    """Run MOODS on all peaks in one shot (no chunking needed)."""
    step_name = "step6_fimo"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    try:
        import MOODS.scan
        import MOODS.tools
    except ImportError:
        logger.error("❌ MOODS not installed!")
        sys.exit(1)

    peaks_df  = sequences_result['peaks_with_seq']
    meme_file = jaspar_result['meme_file']

    logger.info(f"Scanning {len(peaks_df):,} peaks (no chunking) ...")

    matrices, motif_ids = parse_meme_for_moods(meme_file)
    bg         = compute_background_frequencies(peaks_df['sequence'].tolist())
    thresholds = [MOODS.tools.threshold_from_p(m, bg, 5e-5) for m in matrices]

    matches_file  = RESULTS_DIR / "moods_matches_all.csv"
    progress_file = RESULTS_DIR / "moods_progress.txt"

    # Resume support
    start_idx, total = 0, 0
    if progress_file.exists():
        lines     = progress_file.read_text().strip().split('\n')
        start_idx = int(lines[0])
        total     = int(lines[1])
        logger.info(f"▶️  Resuming from peak {start_idx:,}  matches so far: {total:,}")

    file_mode = 'a' if start_idx > 0 else 'w'
    buffer, failed = [], 0
    FLUSH = 5000

    with open(matches_file, file_mode) as f:
        if file_mode == 'w':
            f.write("motif_id,sequence_name,n_matches,max_score\n")

        for idx, row in tqdm(
            peaks_df.iloc[start_idx:].iterrows(),
            total=len(peaks_df) - start_idx,
            desc="Scanning"
        ):
            seq = row['sequence']
            if not seq or len(seq) < 10:
                failed += 1
                continue

            results = MOODS.scan.scan_dna(seq, matrices, bg, thresholds, 7)
            for mi, motif_matches in enumerate(results):
                if motif_matches:
                    buffer.append(
                        f"{motif_ids[mi]},{row['name']},"
                        f"{len(motif_matches)},"
                        f"{max(m.score for m in motif_matches)}\n"
                    )
                    total += 1

            current = idx + 1
            if (current - start_idx) % FLUSH == 0:
                f.writelines(buffer)
                buffer = []
                progress_file.write_text(f"{current}\n{total}\n")
                logger.info(f"  Peak {current:,}/{len(peaks_df):,}  matches: {total:,}")

        if buffer:
            f.writelines(buffer)

    if progress_file.exists():
        progress_file.unlink()

    logger.info(f"✅ Scanning done  total matches: {total:,}  failed: {failed}")

    fimo_df = pd.read_csv(matches_file)
    result  = {'fimo_results': fimo_df, 'n_matches': len(fimo_df),
               'method': 'MOODS', 'pvalue_used': 5e-5}
    checkpoint_mgr.save(step_name, result)
    return result

# ============================================================================
# HELPERS
# ============================================================================

def parse_meme_for_moods(meme_file):
    matrices, motif_ids = [], []
    current_id, current_rows, in_matrix = None, [], False

    with open(meme_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('MOTIF'):
                if current_id and current_rows:
                    mat = _rows_to_moods(current_rows)
                    if mat:
                        matrices.append(mat)
                        motif_ids.append(current_id)
                parts = line.split()
                current_id, current_rows, in_matrix = parts[1], [], False
            elif line.startswith('letter-probability matrix'):
                in_matrix = True
            elif in_matrix and line and line[0].isdigit():
                vals = [float(x) for x in line.split()]
                if len(vals) == 4:
                    current_rows.append(vals)
            elif in_matrix and line.startswith('URL'):
                in_matrix = False

    if current_id and current_rows:
        mat = _rows_to_moods(current_rows)
        if mat:
            matrices.append(mat)
            motif_ids.append(current_id)

    return matrices, motif_ids


def _rows_to_moods(rows):
    if not rows:
        return None
    try:
        return [[r[i] for r in rows] for i in range(4)]
    except Exception:
        return None


def compute_background_frequencies(sequences):
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    total  = 0
    for seq in sequences:
        if not seq:
            continue
        for base in seq.upper():
            if base in counts:
                counts[base] += 1
                total += 1
    if total == 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [counts[b] / total for b in ('A', 'C', 'G', 'T')]


def compute_gc(seq):
    if not seq:
        return 0.5
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq)

# ============================================================================
# STEP 7: CREATE MOTIF-PEAK MATRIX  (vectorized)
# ============================================================================

def step7_create_motif_matrix(peaks_df, jaspar_result, fimo_result):
    step_name = "step7_matrix"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    logger.info("=" * 80)
    logger.info("STEP 7: CREATING MOTIF-PEAK MATRIX")
    logger.info("=" * 80)

    fimo_df   = pd.read_csv(RESULTS_DIR / "moods_matches_all.csv")
    motifs_df = jaspar_result['motifs_df']

    peak_to_idx  = {name: idx for idx, name in enumerate(peaks_df['name'])}
    motif_to_idx = {mid:  idx for idx, mid  in enumerate(motifs_df['motif_id'])}

    n_peaks  = len(peaks_df)
    n_motifs = len(motifs_df)
    logger.info(f"Matrix: {n_motifs} motifs × {n_peaks} peaks")

    fimo_df['peak_idx']  = fimo_df['sequence_name'].map(peak_to_idx)
    fimo_df['motif_idx'] = fimo_df['motif_id'].map(motif_to_idx)
    fimo_df = fimo_df.dropna(subset=['peak_idx', 'motif_idx'])

    rows = fimo_df['motif_idx'].astype(int).values
    cols = fimo_df['peak_idx'].astype(int).values
    data = np.ones(len(rows), dtype=np.float32)

    motif_matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_motifs, n_peaks),
        dtype=np.float32
    )

    motif_freq = np.array(motif_matrix.sum(axis=1)).flatten()
    logger.info(f"✅ nnz: {motif_matrix.nnz:,}  avg peaks/motif: {motif_freq.mean():.1f}")

    result = {
        'motif_matrix': motif_matrix,
        'motif_freq':   motif_freq,
        'peak_to_idx':  peak_to_idx,
        'motif_to_idx': motif_to_idx
    }
    checkpoint_mgr.save(step_name, result)
    return result

# ============================================================================
# STEP 8: CHROMVAR DEVIATIONS — FULLY SPARSE, LOW RAM
# ============================================================================
#
# Memory budget (6 GB available):
#   counts sparse  ~2.7 GB  (343M nnz × float64)
#   E vectors      computed on-the-fly per motif  → O(n_cells) only
#   bg_devs        (n_cells × n_background) = 39441 × 50 × 8 ≈ 15 MB per motif
#   output arrays  (n_cells × n_motifs) × float32 ≈ manageable
#
# We NEVER call counts.toarray() or build a full dense E matrix.
# ============================================================================

def step8_compute_chromvar(adata, peaks_df, motif_matrix_result,
                            n_background=50):
    step_name = "step8_chromvar"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    logger.info("=" * 80)
    logger.info("STEP 8: CHROMVAR DEVIATIONS (fully sparse, low-RAM)")
    logger.info("=" * 80)

    # ── Keep counts as sparse CSC for fast column slicing ────────────────
    if not sparse.issparse(adata.X):
        counts_sp = adata.X.tocsc().astype(np.float64)
    else:
        counts_sp = adata.X.tocsc().astype(np.float64)

    motif_matrix = motif_matrix_result['motif_matrix']   # (n_motifs, n_peaks) CSR
    n_cells, n_peaks = counts_sp.shape
    n_motifs         = motif_matrix.shape[0]

    logger.info(f"Cells: {n_cells:,}  Peaks: {n_peaks:,}  Motifs: {n_motifs:,}")
    logger.info(f"counts_sp  RAM ≈ {counts_sp.data.nbytes / 1e9:.2f} GB")

    # ── Sparse totals (never dense) ───────────────────────────────────────
    # cell_totals : (n_cells,)   — sum over all peaks per cell
    # peak_totals : (n_peaks,)   — sum over all cells per peak
    cell_totals  = np.asarray(counts_sp.sum(axis=1)).flatten()   # (n_cells,)
    peak_totals  = np.asarray(counts_sp.sum(axis=0)).flatten()   # (n_peaks,)
    total_counts = cell_totals.sum()

    logger.info("Computing GC content per peak ...")
    peak_gc = np.load(str(RESULTS_DIR / "peak_gc.npy"))
    peak_mean_acc = peak_totals / n_cells   # mean accessibility ≈ peak_total / n_cells

    # ── Output arrays (float32 to save RAM) ──────────────────────────────
    bias_corrected_devs = np.zeros((n_cells, n_motifs), dtype=np.float32)
    z_scores            = np.zeros((n_cells, n_motifs), dtype=np.float32)

    rng = np.random.default_rng(42)

    logger.info(f"Processing {n_motifs:,} motifs  (n_background={n_background}) ...")

    for k in tqdm(range(n_motifs), desc="Motifs"):

        # ── Motif peak indices ────────────────────────────────────────────
        motif_row  = motif_matrix[k, :]                            # (1, n_peaks) sparse
        motif_cols = motif_row.indices                             # peak indices with motif
        n_mp       = len(motif_cols)

        if n_mp < 5:
            continue

        # ── Observed counts for motif peaks: sparse column slice ─────────
        # counts_sp[:, motif_cols]  shape (n_cells, n_mp)  — stays sparse
        obs_k = np.asarray(
            counts_sp[:, motif_cols].sum(axis=1)
        ).flatten()                                                # (n_cells,)

        # ── Expected counts (paper formula) ──────────────────────────────
        # E[i,j] = cell_total[i] * peak_total[j] / total_counts
        # sum over motif peaks:
        #   exp_k[i] = cell_total[i] * sum(peak_totals[motif_cols]) / total_counts
        sum_peak_totals_k = peak_totals[motif_cols].sum()
        exp_k   = cell_totals * (sum_peak_totals_k / total_counts)   # (n_cells,)
        denom_k = exp_k.sum() if exp_k.sum() > 0 else 1.0

        raw_dev_k = (obs_k - exp_k) / denom_k                       # (n_cells,)

        # ── Background sampling ───────────────────────────────────────────
        motif_mask = np.zeros(n_peaks, dtype=bool)
        motif_mask[motif_cols] = True

        candidate_idx = np.where(~motif_mask)[0]
        motif_gc      = peak_gc[motif_cols]
        motif_acc     = np.log1p(peak_mean_acc[motif_cols])
        cand_gc       = peak_gc[candidate_idx]
        cand_acc      = np.log1p(peak_mean_acc[candidate_idx])

        # Weight matrix (n_mp, n_candidates) — same formula as paper
        gc_diff  = (cand_gc[None, :]  - motif_gc[:, None])  ** 2
        acc_diff = (cand_acc[None, :] - motif_acc[:, None]) ** 2
        W        = np.exp(-(gc_diff + acc_diff) / (2 * 0.01))
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W /= row_sums                                                # row-normalise

        bg_devs = np.zeros((n_cells, n_background), dtype=np.float64)

        for b in range(n_background):
            # Choose one background peak per motif peak
            chosen_local  = np.array([
                rng.choice(len(candidate_idx), p=W[i])
                for i in range(n_mp)
            ])
            bg_cols = candidate_idx[chosen_local]                   # (n_mp,) global indices

            # Observed background (sparse slice, same formula)
            obs_bg = np.asarray(
                counts_sp[:, bg_cols].sum(axis=1)
            ).flatten()

            # Expected background
            sum_peak_totals_bg = peak_totals[bg_cols].sum()
            exp_bg = cell_totals * (sum_peak_totals_bg / total_counts)

            bg_devs[:, b] = (obs_bg - exp_bg) / denom_k

        # ── Bias correction ───────────────────────────────────────────────
        bc_dev = raw_dev_k - bg_devs.mean(axis=1)
        bg_std = bg_devs.std(axis=1)
        bg_std[bg_std == 0] = 1.0

        bias_corrected_devs[:, k] = bc_dev.astype(np.float32)
        z_scores[:, k]            = (bc_dev / bg_std).astype(np.float32)

    logger.info("✅ Deviations computed!")

    result = {
        'deviations': bias_corrected_devs,
        'z_scores':   z_scores,
        'method':     'ChromVAR_bias_corrected_sparse',
        'shape':      bias_corrected_devs.shape
    }
    checkpoint_mgr.save(step_name, result)
    return result

# ============================================================================
# STEP 9: FILTER VARIABLE MOTIFS
# ============================================================================

def _batched_correlation_filter(z_scores, threshold=0.9):
    """Greedy correlation filter — same result as np.corrcoef approach."""
    Z     = z_scores - z_scores.mean(axis=0)
    norms = np.linalg.norm(Z, axis=0)
    norms[norms == 0] = 1.0
    Z_norm = (Z / norms).astype(np.float32)

    n    = Z_norm.shape[1]
    keep = np.ones(n, dtype=bool)

    for i in tqdm(range(n), desc="Correlation filter"):
        if not keep[i]:
            continue
        col_i       = Z_norm[:, i]
        corr        = Z_norm.T @ col_i                 # (n,) dot = corr for unit-normed
        remove      = np.zeros(n, dtype=bool)
        remove[i+1:] = np.abs(corr[i+1:]) > threshold
        keep[remove & keep] = False

    return keep


def step9_filter_variable_motifs(chromvar_result, jaspar_result):
    step_name = "step9_variable_motifs"
    if checkpoint_mgr.exists(step_name):
        return checkpoint_mgr.load(step_name)

    logger.info("=" * 80)
    logger.info("STEP 9: FILTERING MOTIFS")
    logger.info("=" * 80)

    z_scores       = chromvar_result['z_scores']
    motifs_df      = jaspar_result['motifs_df']
    n_motifs_total = z_scores.shape[1]

    logger.info("Filter 1: correlation > 0.9 ...")
    keep = _batched_correlation_filter(z_scores, threshold=0.9)
    logger.info(f"  kept: {keep.sum()}  removed: {(~keep).sum()}")

    z_scores_f    = z_scores[:, keep]
    names_f       = motifs_df['motif_name'].values[keep]
    ids_f         = motifs_df['motif_id'].values[keep]

    logger.info("Filter 2: variability > 1.5 ...")
    variability   = z_scores_f.std(axis=0)
    variable_mask = variability > 1.5
    logger.info(f"  kept: {variable_mask.sum()}  removed: {(~variable_mask).sum()}")

    X_dnn             = z_scores_f[:, variable_mask]
    motif_names_final = names_f[variable_mask]
    motif_ids_final   = ids_f[variable_mask]

    logger.info(f"✅ DNN input shape: {X_dnn.shape}")

    kept_indices = np.where(keep)[0]
    final_mask   = np.zeros(n_motifs_total, dtype=bool)
    final_mask[kept_indices[variable_mask]] = True

    variability_all = z_scores.std(axis=0)

    result = {
        'X_dnn':                X_dnn,
        'variability':          variability_all,
        'variable_mask':        final_mask,
        'motif_names_variable': motif_names_final,
        'motif_ids_variable':   motif_ids_final,
        'threshold_used':       1.5,
        'corr_threshold_used':  0.9
    }
    checkpoint_mgr.save(step_name, result)
    return result

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================

def step10_save_results(adata, chromvar_result, jaspar_result,
                         fimo_result, variable_motifs_result):
    step_name = "step10_saved"
    if checkpoint_mgr.exists(step_name):
        return

    logger.info("=" * 80)
    logger.info("STEP 10: SAVING RESULTS")
    logger.info("=" * 80)

    X_dnn       = variable_motifs_result['X_dnn']
    motif_names = variable_motifs_result['motif_names_variable']
    variability = variable_motifs_result['variability']
    motifs_df   = jaspar_result['motifs_df']

    np.save(RESULTS_DIR / "chromvar_deviations.npy", chromvar_result['deviations'])
    np.save(RESULTS_DIR / "chromvar_zscores.npy",    chromvar_result['z_scores'])
    np.save(RESULTS_DIR / "X_dnn.npy",               X_dnn)

    pd.DataFrame(X_dnn, index=adata.obs_names, columns=motif_names
                 ).to_csv(RESULTS_DIR / "X_dnn.csv")

    pd.DataFrame({
        'motif_name':   motifs_df['motif_name'].values,
        'motif_id':     motifs_df['motif_id'].values,
        'variability':  variability,
        'kept_for_dnn': variable_motifs_result['variable_mask']
    }).to_csv(RESULTS_DIR / "motif_variability.csv", index=False)

    adata.obs.to_csv(RESULTS_DIR / "cell_metadata.csv")

    with open(RESULTS_DIR / "CHROMVAR_SUMMARY.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CHROMVAR RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date                 : {datetime.now()}\n\n")
        f.write(f"Cells                : {adata.n_obs:,}\n")
        f.write(f"Peaks                : {adata.n_vars:,}\n")
        f.write(f"Total motifs scanned : {len(variability)}\n")
        f.write(f"Motif matches        : {fimo_result['n_matches']:,}\n")
        f.write(f"Scanning method      : {fimo_result['method']}\n\n")
        f.write(f"Variability threshold: {variable_motifs_result['threshold_used']}\n")
        f.write(f"Variable motifs kept : {variable_motifs_result['variable_mask'].sum()}\n\n")
        f.write(f"DNN input shape      : {X_dnn.shape}\n\n")
        f.write("Output files:\n")
        f.write("  X_dnn.npy / X_dnn.csv\n")
        f.write("  chromvar_deviations.npy\n")
        f.write("  chromvar_zscores.npy\n")
        f.write("  motif_variability.csv\n")
        f.write("  cell_metadata.csv\n")

    logger.info(f"✅ Saved to: {RESULTS_DIR}/")
    checkpoint_mgr.save(step_name, True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk',      type=int,  default=None,
                        help='Run single MOODS chunk (0-based)')
    parser.add_argument('--chunk-size', type=int,  default=50000)
    parser.add_argument('--merge',      action='store_true',
                        help='Merge chunks + run steps 7-10')
    parser.add_argument('--n-chunks',   type=int,  default=4)
    args = parser.parse_args()

    # Steps 1-5 always load from checkpoint if available
    adata            = step1_load_data()
    peaks_df         = step2_prepare_peaks(adata)
    genome_info      = step3_download_genome()
    sequences_result = step4_extract_sequences(peaks_df, genome_info)
    jaspar_result    = step5_download_jaspar_pwms()

    # ── Chunk mode ────────────────────────────────────────────────────────
    if args.chunk is not None:
        step6_scan_motifs_moods(
            sequences_result, jaspar_result,
            chunk_idx=args.chunk,
            chunk_size=args.chunk_size
        )
        logger.info(f"✅ Chunk {args.chunk} done")
        return 0

    # ── Merge mode ────────────────────────────────────────────────────────
    if args.merge:
        fimo_result = step6_merge_chunks(n_chunks=args.n_chunks)

    else:
        # ── Default: run everything in one shot (24-hour job) ─────────────
        fimo_result = step6_run_full(sequences_result, jaspar_result)

    motif_matrix    = step7_create_motif_matrix(peaks_df, jaspar_result, fimo_result)
    sequences_result = checkpoint_mgr.load("step4_sequences")
    peaks_df_with_seq = sequences_result["peaks_with_seq"]
    chromvar_result = step8_compute_chromvar(adata, peaks_df_with_seq, motif_matrix,
                                             n_background=50)
    variable_motifs = step9_filter_variable_motifs(chromvar_result, jaspar_result)
    step10_save_results(adata, chromvar_result, jaspar_result,
                        fimo_result, variable_motifs)
    return 0


if __name__ == "__main__":
    sys.exit(main())