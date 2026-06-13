#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
CHROMVAR PIPELINE — COMPLETE & PAPER-ACCURATE
═══════════════════════════════════════════════════════════════════════════

Reference implementation of chromVAR (Schep et al., Nat Methods 2017)
translated from the R source code at:
  https://github.com/GreenleafLab/chromVAR/blob/master/R/compute_deviations.R
  https://github.com/GreenleafLab/chromVAR/blob/master/R/background_peaks.R

Inputs (from existing checkpoints — outputs of Steps 1-7):
  • step1_data.pkl       — AnnData: (n_cells, n_peaks) fragment counts
  • step4_sequences.pkl  — peaks_df with DNA sequences
  • step5_jaspar.pkl     — motifs_df (879 JASPAR motifs)
  • step7_matrix.pkl     — motif_matrix: (n_motifs, n_peaks) binary

Outputs (for downstream DNN model):
  • filtered_counts.npz           — final sparse counts matrix used
  • cell_ids_kept.npy             — cell IDs after filtering
  • peak_ids_kept.npy             — peak names after filtering
  • motif_matrix_filtered.npz     — motif × peak (after peak filtering)
  • peak_gc_filtered.npy          — GC content per kept peak
  • peak_depth_filtered.npy       — total fragments per kept peak
  • cell_depth.npy                — total fragments per kept cell
  • expectation.npy               — peak_total / total_counts (chromVAR expectation)
  • bias_bins.npz                 — Mahalanobis bins + centers
  • background_peaks.npy          — (n_peaks, n_iter) bg peak indices
  • chromvar_deviations.npy       — (n_cells, n_motifs_tested) float32
  • chromvar_zscores.npy          — (n_cells, n_motifs_tested) float32
  • motif_ids_tested.npy          — motif IDs corresponding to columns
  • variability.csv               — motif_id, name, variability, kept_for_dnn
  • X_dnn.npy                     — (n_cells, n_variable_motifs) — DNN input
  • X_dnn.csv                     — same, with cell and motif IDs
  • pipeline_summary.txt          — full human-readable log

CLI:
    python chromvar_pipeline.py                          # run full pipeline
    python chromvar_pipeline.py --stage 4                # resume from a stage
    python chromvar_pipeline.py --n-iter 50              # bg iterations (paper default)
    python chromvar_pipeline.py --min-depth 1500         # min fragments per cell
    python chromvar_pipeline.py --variability-cutoff 1.5 # for DNN input

SLURM submission:
    sbatch run_chromvar_pipeline.slurm
"""

# ───────────────────────────────────────────────────────────────────────────
# CUDA_VISIBLE_DEVICES fix (MUST come before torch import)
# ───────────────────────────────────────────────────────────────────────────
import os
_cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if _cvd is None or _cvd.strip() == '' or _cvd.strip().lower() == 'nodevfiles':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

# ───────────────────────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("/home/rhamed/results/pbmc_atlas/checkpoints_v4_1_copy")
RESULTS_DIR    = Path("/home/rhamed/results/pbmc_atlas/chromvar_results_v4_1_copy")
LOG_DIR        = Path("/home/rhamed/results/pbmc_atlas/logs")
OUT_DIR        = Path("/home/rhamed/results/pbmc_atlas/chromvar_final_gh38")   # NEW clean dir
OUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# ───────────────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'chromvar_pipeline_step2_gh38.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def banner(title):
    log.info("")
    log.info("═" * 72)
    log.info(f"  {title}")
    log.info("═" * 72)


def save_npz_sparse(path, M):
    sparse.save_npz(str(path), M.tocsr())


def load_ckpt(name):
    with open(CHECKPOINT_DIR / f"{name}.pkl", 'rb') as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 0 — LOAD INPUTS & VALIDATE
# ═══════════════════════════════════════════════════════════════════════════

def stage0_load_inputs():
    banner("STAGE 0 — LOAD INPUTS FROM CHECKPOINTS")

    # Check all required checkpoints exist
    required = ["step1_data", "step4_sequences", "step5_jaspar", "step7_matrix"]
    for name in required:
        p = CHECKPOINT_DIR / f"{name}.pkl"
        if not p.exists():
            log.error(f"Missing checkpoint: {p}")
            sys.exit(1)
        log.info(f"  ✓ found {p.name}")

    # Load
    log.info("Loading ...")
    adata = load_ckpt("step1_data")
    step4 = load_ckpt("step4_sequences")
    step5 = load_ckpt("step5_jaspar")
    step7 = load_ckpt("step7_matrix")

    counts = adata.X
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    counts = counts.astype(np.float32).tocsr()

    motif_matrix = step7['motif_matrix'].astype(np.float32).tocsr()   # (n_motifs, n_peaks)
    peaks_df = step4['peaks_with_seq']
    motifs_df = step5['motifs_df']

    n_cells, n_peaks = counts.shape
    n_motifs = motif_matrix.shape[0]

    # Sanity
    assert motif_matrix.shape[1] == n_peaks, "motif_matrix peak count mismatch"
    assert len(peaks_df) == n_peaks, "peaks_df length mismatch"
    assert len(motifs_df) == n_motifs, "motifs_df length mismatch"

    log.info(f"  cells:    {n_cells:,}")
    log.info(f"  peaks:    {n_peaks:,}")
    log.info(f"  motifs:   {n_motifs:,}")
    log.info(f"  nnz:      {counts.nnz:,}")
    log.info(f"  density:  {counts.nnz/(n_cells*n_peaks):.4f}")

    # Compute GC if not already present
    peak_gc_path = RESULTS_DIR / "peak_gc.npy"
    if peak_gc_path.exists():
        peak_gc = np.load(str(peak_gc_path)).astype(np.float32)
        log.info(f"  ✓ peak_gc loaded ({peak_gc_path})")
    else:
        log.info("  computing GC content from sequences ...")
        seqs = peaks_df['sequence'].tolist()
        peak_gc = np.zeros(n_peaks, dtype=np.float32)
        for i, s in enumerate(seqs):
            if s:
                s = s.upper()
                L = len(s)
                peak_gc[i] = (s.count('G') + s.count('C')) / L if L > 0 else 0.5
            else:
                peak_gc[i] = 0.5
        np.save(str(peak_gc_path), peak_gc)

    assert len(peak_gc) == n_peaks, "peak_gc length mismatch"
    log.info(f"  GC content: mean={peak_gc.mean():.3f}  range=[{peak_gc.min():.3f}, {peak_gc.max():.3f}]")

    return {
        'counts':        counts,
        'motif_matrix':  motif_matrix,
        'peaks_df':      peaks_df,
        'motifs_df':     motifs_df,
        'peak_gc':       peak_gc,
        'cell_ids':      np.array(adata.obs_names),
        'peak_ids':      np.array(adata.var_names),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — FILTER CELLS (filterSamples equivalent)
# ═══════════════════════════════════════════════════════════════════════════
# chromVAR filterSamples: min_depth (default 500 or 10% median), min_in_peaks
# We only have fragments in peaks, so we apply min_depth directly to that.

def stage1_filter_cells(data, min_depth=1500):
    banner(f"STAGE 1 — FILTER CELLS (min_depth = {min_depth})")

    counts = data['counts']
    cell_depth_all = np.asarray(counts.sum(axis=1)).flatten()

    log.info(f"  cell depth stats:")
    log.info(f"    min={cell_depth_all.min():.0f}  "
             f"median={np.median(cell_depth_all):.0f}  "
             f"max={cell_depth_all.max():.0f}")

    keep_cells = cell_depth_all >= min_depth
    n_kept = int(keep_cells.sum())
    n_drop = int((~keep_cells).sum())
    log.info(f"  cells kept: {n_kept:,} / {len(keep_cells):,}  "
             f"(dropped {n_drop:,} with depth < {min_depth})")

    if n_kept < 100:
        log.error(f"  Too few cells pass filter — try lower --min-depth")
        sys.exit(1)

    data['counts']   = counts[keep_cells, :]
    data['cell_ids'] = data['cell_ids'][keep_cells]
    data['cell_depth'] = cell_depth_all[keep_cells].astype(np.float64)
    return data


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — FILTER PEAKS (filterPeaks equivalent)
# ═══════════════════════════════════════════════════════════════════════════
# chromVAR requires "all peaks must have at least one fragment in one sample"
# We also drop peaks with pathological GC (sequence extraction failed).

def stage2_filter_peaks(data, min_fragments=1):
    banner(f"STAGE 2 — FILTER PEAKS (min_fragments ≥ {min_fragments}, valid GC)")

    counts = data['counts']
    peak_depth_all = np.asarray(counts.sum(axis=0)).flatten()
    peak_gc_all = data['peak_gc']

    # Filter 1: peaks with enough fragments
    keep_coverage = peak_depth_all >= min_fragments
    log.info(f"  peaks with ≥{min_fragments} fragments: "
             f"{keep_coverage.sum():,} / {len(keep_coverage):,}")

    # Filter 2: peaks with valid GC (not 0 or 1 — those are usually failures)
    keep_gc = (peak_gc_all > 0.0) & (peak_gc_all < 1.0)
    log.info(f"  peaks with 0 < GC < 1:             "
             f"{keep_gc.sum():,} / {len(keep_gc):,}")

    keep_peaks = keep_coverage & keep_gc
    n_kept = int(keep_peaks.sum())
    log.info(f"  peaks kept (both filters):        {n_kept:,} / {len(keep_peaks):,}")

    # Apply
    data['counts']       = data['counts'][:, keep_peaks]
    data['motif_matrix'] = data['motif_matrix'][:, keep_peaks]
    data['peak_ids']     = data['peak_ids'][keep_peaks]
    data['peak_gc']      = peak_gc_all[keep_peaks].astype(np.float64)
    data['peak_depth']   = peak_depth_all[keep_peaks].astype(np.float64)
    data['peaks_df']     = data['peaks_df'].iloc[keep_peaks].reset_index(drop=True)

    log.info(f"  final counts shape: {data['counts'].shape}")
    log.info(f"  final motif_matrix shape: {data['motif_matrix'].shape}")
    return data


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — COMPUTE EXPECTATION (computeExpectations equivalent)
# ═══════════════════════════════════════════════════════════════════════════
# R code (compute_expectations_core):
#   expectation <- getFragmentsPerPeak(object) / getTotalFragments(object)
# → expectation[j] = peak_total[j] / total_counts  (a probability over peaks)

def stage3_compute_expectation(data):
    banner("STAGE 3 — COMPUTE EXPECTATION")

    peak_depth = data['peak_depth']
    total = peak_depth.sum()
    expectation = peak_depth / total   # fraction of reads expected per peak

    assert abs(expectation.sum() - 1.0) < 1e-6, "expectation should sum to 1"
    log.info(f"  total fragments:       {total:,.0f}")
    log.info(f"  expectation sum:       {expectation.sum():.6f} (should be 1)")
    log.info(f"  expectation range:     [{expectation.min():.2e}, {expectation.max():.2e}]")
    log.info(f"  expectation median:    {np.median(expectation):.2e}")

    data['expectation'] = expectation
    data['total_counts'] = total
    return data


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — BUILD MAHALANOBIS BIAS BINS
# ═══════════════════════════════════════════════════════════════════════════
# R getBackgroundPeaks (background_peaks.R):
#   - bias_df = (GC, log10(depth+1))    — R uses log10
#   - Mahalanobis transform (decorrelate)
#   - bin into bs × bs grid (bs=50 default)
#   - w=0.1 default (Gaussian kernel sigma for similarity)

def stage4_build_bias_bins(data, n_bins=50, w=0.1):
    banner(f"STAGE 4 — BUILD BIAS BINS ({n_bins}×{n_bins}, w={w})")

    peak_gc = data['peak_gc']
    peak_depth = data['peak_depth']

    # chromVAR R uses log10(depth+1); some forks use log. We use log10 to match paper.
    bias_x = peak_gc
    bias_y = np.log10(peak_depth + 1.0)
    bias = np.stack([bias_x, bias_y], axis=1)   # (n_peaks, 2)

    # Mahalanobis: bias_M = (bias − μ) Σ^{-1/2}
    mu = bias.mean(axis=0)
    cov = np.cov(bias.T)
    cov += 1e-8 * np.eye(2)                     # regularize
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    bias_M = (bias - mu) @ inv_sqrt

    # Even grid over the transformed range
    edges_x = np.linspace(bias_M[:, 0].min() - 1e-6,
                          bias_M[:, 0].max() + 1e-6, n_bins + 1)
    edges_y = np.linspace(bias_M[:, 1].min() - 1e-6,
                          bias_M[:, 1].max() + 1e-6, n_bins + 1)

    bx = np.clip(np.searchsorted(edges_x, bias_M[:, 0]) - 1, 0, n_bins - 1)
    by = np.clip(np.searchsorted(edges_y, bias_M[:, 1]) - 1, 0, n_bins - 1)
    bin_of_peak = (bx * n_bins + by).astype(np.int64)   # flat bin index per peak

    # Bin centers (in Mahalanobis space)
    cx = 0.5 * (edges_x[:-1] + edges_x[1:])
    cy = 0.5 * (edges_y[:-1] + edges_y[1:])
    XX, YY = np.meshgrid(cx, cy, indexing='ij')
    bin_centers = np.stack([XX.flatten(), YY.flatten()], axis=1)

    B = n_bins * n_bins
    bin_counts = np.bincount(bin_of_peak, minlength=B)

    log.info(f"  transformed bias cov (should be ~identity):")
    M_cov = np.cov(bias_M.T)
    log.info(f"    [[{M_cov[0,0]:.3f}, {M_cov[0,1]:.3f}],")
    log.info(f"     [{M_cov[1,0]:.3f}, {M_cov[1,1]:.3f}]]")

    log.info(f"  bin counts: min={bin_counts.min()}  median={int(np.median(bin_counts))}  max={bin_counts.max()}")
    log.info(f"  empty bins: {(bin_counts == 0).sum()} / {B}")

    data['bin_of_peak']   = bin_of_peak
    data['bin_centers']   = bin_centers
    data['bin_counts']    = bin_counts
    data['n_bins']        = n_bins
    data['mahalanobis_w'] = w
    return data


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5 — SAMPLE BACKGROUND PEAKS (getBackgroundPeaks equivalent)
# ═══════════════════════════════════════════════════════════════════════════
# R does this on CPU. For our 187K peaks × 50 iterations, doing it once
# upfront and saving is cleanest (matches R exactly — one background_peaks
# matrix reused across all motifs).
#
# Output: background_peaks[p, b] = index of a background peak for peak p
#   in iteration b. Shape (n_peaks, n_iter).

def stage5_sample_background_peaks(data, n_iter=50, device='cuda'):
    banner(f"STAGE 5 — SAMPLE BACKGROUND PEAKS (n_iter={n_iter})")

    n_peaks = data['counts'].shape[1]
    bin_of_peak = data['bin_of_peak']
    bin_centers = data['bin_centers']
    bin_counts = data['bin_counts']
    n_bins = data['n_bins']
    w = data['mahalanobis_w']
    B = n_bins * n_bins

    # log p(bin_i | bin_j) ∝ −d²/(2w²) + log(ρ_i)
    d2 = ((bin_centers[:, None, :] - bin_centers[None, :, :]) ** 2).sum(axis=2)
    log_kernel = -d2 / (2.0 * w * w)
    with np.errstate(divide='ignore'):
        log_rho = np.log(bin_counts.astype(np.float64))
    log_rho[np.isinf(log_rho)] = -1e30
    log_transition = log_kernel + log_rho[None, :]   # (B, B): row j, col i

    # Build peaks_by_bin index (sorted by bin)
    order = np.argsort(bin_of_peak, kind='stable')
    peaks_by_bin_np = order.astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(bin_counts)]).astype(np.int64)

    # Move to GPU for fast sampling
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    log.info(f"  sampling on {dev}")

    bin_of_peak_t    = torch.from_numpy(bin_of_peak).long().to(dev)
    log_transition_t = torch.from_numpy(log_transition).float().to(dev)
    peaks_by_bin_t   = torch.from_numpy(peaks_by_bin_np).to(dev)
    offsets_t        = torch.from_numpy(offsets).to(dev)

    gen = torch.Generator(device=dev)
    gen.manual_seed(2017)    # chromVAR paper uses set.seed(2017) — matches

    # For each peak, its bin → probs over all bins → multinomial sample for each iteration
    # Do it in chunks of peaks to save VRAM
    chunk = 20000
    bg_peaks = np.empty((n_peaks, n_iter), dtype=np.int64)

    t0 = time.time()
    for start in range(0, n_peaks, chunk):
        end = min(start + chunk, n_peaks)
        peak_bins = bin_of_peak_t[start:end]                    # (c,)
        logits = log_transition_t[peak_bins]                    # (c, B)
        probs  = torch.softmax(logits, dim=1)                   # (c, B)

        # Multinomial: each row independently → n_iter samples
        sampled_bins = torch.multinomial(probs, n_iter, replacement=True,
                                          generator=gen)        # (c, n_iter)

        # Uniformly pick a peak within each sampled bin
        starts_of_bin = offsets_t[sampled_bins]                 # (c, n_iter)
        counts_in_bin = (offsets_t[sampled_bins + 1] - starts_of_bin).clamp(min=1)
        rand_u = torch.rand(sampled_bins.shape, device=dev, generator=gen)
        pick = torch.minimum(
            (rand_u * counts_in_bin.float()).long(),
            counts_in_bin - 1
        )
        chosen_peaks = peaks_by_bin_t[starts_of_bin + pick]     # (c, n_iter)
        bg_peaks[start:end, :] = chosen_peaks.cpu().numpy()

        if (start // chunk) % 5 == 0:
            log.info(f"    chunk {start:,}-{end:,} / {n_peaks:,}")

    log.info(f"  done in {time.time()-t0:.1f}s")

    # Sanity: each bg peak must exist (0 ≤ idx < n_peaks)
    assert bg_peaks.min() >= 0 and bg_peaks.max() < n_peaks, "bad bg index"

    data['background_peaks'] = bg_peaks   # (n_peaks, n_iter)
    data['n_iter'] = n_iter
    return data


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6 — COMPUTE DEVIATIONS (compute_deviations_single equivalent)
# ═══════════════════════════════════════════════════════════════════════════
# EXACT R FORMULA (from compute_deviations_single, lines ~440-470):
#
#   tf_vec        : (1, n_peaks) sparse, 1 where peak ∈ motif set
#   observed      : (n_cells,)  = tf_vec @ counts  (sum across motif peaks)
#   expected      : (n_cells,)  = (tf_vec @ expectation) * fragments_per_sample
#                                  i.e. (sum_over_peaks_in_set of expectation[p])
#                                       × cell_depth[i]
#   observed_deviation = (observed - expected) / expected       ← PER-CELL !!
#
#   sample_mat    : (n_iter, n_peaks) sparse, 1 for each bg set
#   sampled           : (n_iter, n_cells) = sample_mat @ counts
#   sampled_expected  : (n_iter, n_cells) = (sample_mat @ expectation) * cell_depth[:, None]
#   sampled_deviation = (sampled - sampled_expected) / sampled_expected    ← PER-CELL !!
#
#   normdev = observed_deviation - colMeans(sampled_deviation)
#   z       = normdev / colStds(sampled_deviation)
#
# Also: threshold filter — if expected[i] < 1, z[i] and dev[i] := NaN

def stage6_compute_deviations(data, min_peaks=5, device='cuda'):
    banner("STAGE 6 — COMPUTE DEVIATIONS (paper-exact, hybrid CPU-sparse + GPU-dense)")

    # Strategy:
    #   - Keep counts as scipy.sparse CSC on CPU (fast column slicing)
    #   - For each motif: slice columns (motif peaks + bg peaks) → dense on CPU
    #   - Ship the small dense block (n_cells, m + m*n_iter) to GPU for arithmetic
    #   - This avoids torch.sparse.mm on GPU (unreliable on Blackwell/RTX5070)
    #
    # Memory per motif: m columns + m*n_iter bg columns, each n_cells floats
    #   worst case: m=20000, n_iter=50 → 20000*50*39349*4 bytes = 157 GB ✗
    #   so we batch bg iterations if m is large.

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    log.info(f"  arithmetic on: {dev}")
    if dev.type == 'cuda':
        log.info(f"    {torch.cuda.get_device_name(0)}  "
                 f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    counts       = data['counts']                  # (n_cells, n_peaks) sparse
    motif_matrix = data['motif_matrix']            # (n_motifs, n_peaks) sparse
    expectation  = data['expectation']             # (n_peaks,) float64, sums to 1
    cell_depth   = data['cell_depth']              # (n_cells,) float64
    bg_peaks     = data['background_peaks']        # (n_peaks, n_iter) int64
    motifs_df    = data['motifs_df']

    n_cells, n_peaks = counts.shape
    n_motifs = motif_matrix.shape[0]
    n_iter   = bg_peaks.shape[1]

    # Convert counts to CSC for fast column slicing (one-time cost ~10s)
    log.info(f"  converting counts to CSC (one-time, for fast column ops) ...")
    t0 = time.time()
    counts_csc = counts.tocsc().astype(np.float32)
    log.info(f"    done in {time.time()-t0:.1f}s  nnz={counts_csc.nnz:,}")

    # Upload small dense constants to GPU
    expectation_t = torch.from_numpy(expectation.astype(np.float32)).to(dev)   # (n_peaks,)
    cell_depth_t  = torch.from_numpy(cell_depth.astype(np.float32)).to(dev)    # (n_cells,)

    # Verify a basic op works on GPU
    log.info("  testing GPU arithmetic ...")
    _test = (cell_depth_t * 2.0).sum().item()
    _expected = cell_depth.sum() * 2.0
    assert abs(_test - _expected) / max(_expected, 1) < 1e-4, \
        f"GPU arithmetic broken: {_test} vs {_expected}"
    log.info(f"    ✓ basic GPU ops work")

    # ── Output arrays ──
    deviations = np.full((n_cells, n_motifs), np.nan, dtype=np.float32)
    z_scores   = np.full((n_cells, n_motifs), np.nan, dtype=np.float32)
    skipped = []
    threshold = 1.0

    log.info(f"  processing {n_motifs} motifs ...")
    t0 = time.time()
    motif_matrix_csr = motif_matrix.tocsr()

    # Helper: column-sum a set of columns from counts_csc → returns numpy (n_cells,)
    # scipy gives fast column slicing + sum; this is the key speed-up.
    def sum_columns(col_indices):
        """
        Efficiently compute sum across the given peak columns of counts_csc.
        Returns (n_cells,) float32 numpy array.
        """
        # counts_csc[:, col_indices] — CSC slicing is fast
        # .sum(axis=1) returns a (n_cells, 1) matrix; flatten + cast
        return np.asarray(
            counts_csc[:, col_indices].sum(axis=1)
        ).flatten().astype(np.float32)

    for k in range(n_motifs):
        row = motif_matrix_csr[k]
        peak_set_np = row.indices
        m = len(peak_set_np)

        if m < min_peaks:
            skipped.append((k, m, 'too_few_peaks'))
            continue

        # ── Observed: sum over motif peaks of counts per cell ──
        observed_np = sum_columns(peak_set_np)                    # (n_cells,)
        observed = torch.from_numpy(observed_np).to(dev)

        # ── Expected: sum(expectation[peak_set]) * cell_depth ──
        sum_exp_k = expectation_t[
            torch.from_numpy(peak_set_np.astype(np.int64)).to(dev)
        ].sum()
        expected = sum_exp_k * cell_depth_t                       # (n_cells,)

        # observed_deviation = (obs - exp) / exp (per cell)
        obs_dev = (observed - expected) / expected.clamp(min=1e-10)

        # ── Background: for each of n_iter iterations, compute sampled deviation ──
        # We need: sampled[i, b] = sum over q∈peak_set of counts[i, bg_peaks[q, b]]
        # This is equivalent to summing over bg_peaks[peak_set, b]
        # → for each b, a subset of bg peak indices of size m (possibly with dupes)

        bg_for_motif = bg_peaks[peak_set_np, :]                   # (m, n_iter) int64
        # Unique peaks that need slicing
        # For efficiency, process all n_iter at once:
        # Build a sparse matrix S of shape (n_iter, n_peaks) where
        #   S[b, p] = count of p in bg_for_motif[:, b]
        # Then sampled = S @ counts_csc.T → (n_iter, n_cells)
        # But that's another sparse matmul — slow on CPU for large m×n_iter.
        #
        # Faster: for each b, sum_columns(bg_for_motif[:, b])
        # That's n_iter scipy slicing ops. With m ~ 1000-2000 and n_iter=50,
        # this is 50 small slicing ops — fast (<1s per motif).
        sampled_np = np.empty((n_cells, n_iter), dtype=np.float32)
        for b in range(n_iter):
            sampled_np[:, b] = sum_columns(bg_for_motif[:, b])
        sampled = torch.from_numpy(sampled_np).to(dev)            # (n_cells, n_iter)

        # sampled_expected[i, b] = cell_depth[i] * sum(expectation[bg_for_motif[:, b]])
        sum_exp_bg_np = expectation[bg_for_motif].sum(axis=0).astype(np.float32)  # (n_iter,)
        sum_exp_bg = torch.from_numpy(sum_exp_bg_np).to(dev)
        sampled_expected = cell_depth_t.unsqueeze(1) * sum_exp_bg.unsqueeze(0)
        # (n_cells, n_iter)

        samp_dev = (sampled - sampled_expected) / sampled_expected.clamp(min=1e-10)

        # ── Bias correction & Z ──
        mean_sd = samp_dev.mean(dim=1)
        std_sd  = samp_dev.std(dim=1)

        normdev = obs_dev - mean_sd
        z = normdev / std_sd.clamp(min=1e-10)

        # Threshold: expected < 1 ⇒ NaN
        fail = expected < threshold
        normdev_np = normdev.cpu().numpy()
        z_np = z.cpu().numpy()
        fail_np = fail.cpu().numpy()
        normdev_np[fail_np] = np.nan
        z_np[fail_np] = np.nan

        deviations[:, k] = normdev_np.astype(np.float32)
        z_scores[:, k]   = z_np.astype(np.float32)

        # Progress every 25 motifs (this is slower than pure-GPU; more frequent reports)
        if (k + 1) % 25 == 0 or k == n_motifs - 1:
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed
            eta = (n_motifs - k - 1) / rate
            log.info(f"    {k+1:4d}/{n_motifs} motifs  "
                     f"elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m  "
                     f"({rate:.2f} motifs/s)")

    log.info(f"  ✓ done in {(time.time()-t0)/60:.1f} min")
    log.info(f"  skipped: {len(skipped)} motifs")

    data['deviations']      = deviations
    data['z_scores']        = z_scores
    data['motif_ids_all']   = motifs_df['motif_id'].values
    data['motif_names_all'] = motifs_df['motif_name'].values
    data['skipped_motifs']  = skipped
    return data


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 7 — VARIABILITY & MOTIF FILTERING
# ═══════════════════════════════════════════════════════════════════════════
# R computeVariability: variability = sqrt( Var(Z) / expected_var_under_null )
# We use the simpler std(Z) which is the common practice for DNN input.

def stage7_variability(data, variability_cutoff=1.5, corr_cutoff=0.9):
    banner("STAGE 7 — VARIABILITY & FILTERING")

    Z = data['z_scores']    # (n_cells, n_motifs) may contain NaN
    n_motifs = Z.shape[1]

    # variability per motif: std over cells, ignoring NaN
    with np.errstate(invalid='ignore'):
        variability = np.nanstd(Z, axis=0)
    variability = np.nan_to_num(variability, nan=0.0)

    log.info(f"  variability distribution:")
    log.info(f"    min={variability.min():.3f}  "
             f"median={np.median(variability):.3f}  "
             f"max={variability.max():.3f}")
    log.info(f"    motifs with var > 1.0:  {(variability > 1.0).sum()}")
    log.info(f"    motifs with var > 1.5:  {(variability > 1.5).sum()}")
    log.info(f"    motifs with var > 2.0:  {(variability > 2.0).sum()}")

    # Print top 20 TFs for biological sanity
    top = np.argsort(variability)[::-1][:20]
    log.info(f"\n  TOP 20 MOST VARIABLE MOTIFS:")
    log.info(f"    {'rank':>4}  {'motif_id':15s}  {'name':20s}  {'var':>8s}")
    for rank, i in enumerate(top, 1):
        log.info(f"    {rank:>4}  {data['motif_ids_all'][i]:15s}  "
                 f"{data['motif_names_all'][i]:20s}  {variability[i]:>8.3f}")

    # Filter 1: correlation redundancy
    Z_clean = np.nan_to_num(Z, nan=0.0)
    Z_c = Z_clean - Z_clean.mean(axis=0)
    norms = np.linalg.norm(Z_c, axis=0)
    norms[norms == 0] = 1.0
    Z_n = (Z_c / norms).astype(np.float32)

    keep_corr = np.ones(n_motifs, dtype=bool)
    log.info(f"\n  correlation filter (threshold = {corr_cutoff}) ...")
    for i in range(n_motifs):
        if not keep_corr[i]:
            continue
        # Correlate motif i with all subsequent motifs
        c = Z_n.T[i+1:] @ Z_n[:, i]        # (n_motifs - i - 1,)
        highly_corr = np.abs(c) > corr_cutoff
        keep_corr[i+1:][highly_corr] = False

    n_after_corr = int(keep_corr.sum())
    log.info(f"    kept after corr filter: {n_after_corr} / {n_motifs}")

    # Filter 2: variability
    keep_var = variability > variability_cutoff
    keep_final = keep_corr & keep_var
    n_variable = int(keep_final.sum())
    log.info(f"    kept after var > {variability_cutoff}: {n_variable}")

    data['variability'] = variability
    data['keep_final']  = keep_final
    data['X_dnn']       = Z_clean[:, keep_final].astype(np.float32)
    data['variable_motif_ids']   = data['motif_ids_all'][keep_final]
    data['variable_motif_names'] = data['motif_names_all'][keep_final]

    log.info(f"  X_dnn shape: {data['X_dnn'].shape}")
    return data


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 8 — SAVE ALL OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

def stage8_save(data, params):
    banner("STAGE 8 — SAVE OUTPUTS")

    # ── Filtered data ──
    save_npz_sparse(OUT_DIR / "filtered_counts.npz", data['counts'])
    save_npz_sparse(OUT_DIR / "motif_matrix_filtered.npz", data['motif_matrix'])
    np.save(OUT_DIR / "cell_ids_kept.npy",      data['cell_ids'])
    np.save(OUT_DIR / "peak_ids_kept.npy",      data['peak_ids'])
    np.save(OUT_DIR / "peak_gc_filtered.npy",   data['peak_gc'])
    np.save(OUT_DIR / "peak_depth_filtered.npy",data['peak_depth'])
    np.save(OUT_DIR / "cell_depth.npy",         data['cell_depth'])
    np.save(OUT_DIR / "expectation.npy",        data['expectation'])

    # ── Bias & background ──
    np.savez(OUT_DIR / "bias_bins.npz",
             bin_of_peak=data['bin_of_peak'],
             bin_centers=data['bin_centers'],
             bin_counts=data['bin_counts'],
             n_bins=data['n_bins'],
             w=data['mahalanobis_w'])
    np.save(OUT_DIR / "background_peaks.npy",   data['background_peaks'])

    # ── chromVAR outputs ──
    np.save(OUT_DIR / "chromvar_deviations.npy", data['deviations'])
    np.save(OUT_DIR / "chromvar_zscores.npy",    data['z_scores'])
    np.save(OUT_DIR / "motif_ids_tested.npy",    data['motif_ids_all'])

    # ── Variability & DNN input ──
    pd.DataFrame({
        'motif_id':     data['motif_ids_all'],
        'motif_name':   data['motif_names_all'],
        'variability':  data['variability'],
        'kept_for_dnn': data['keep_final'],
    }).to_csv(OUT_DIR / "variability.csv", index=False)

    np.save(OUT_DIR / "X_dnn.npy", data['X_dnn'])
    pd.DataFrame(
        data['X_dnn'],
        index=data['cell_ids'],
        columns=data['variable_motif_names'],
    ).to_csv(OUT_DIR / "X_dnn.csv")

    # ── Summary ──
    with open(OUT_DIR / "pipeline_summary.txt", 'w') as f:
        f.write("=" * 72 + "\n")
        f.write("CHROMVAR PIPELINE SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Date: {datetime.now()}\n\n")

        f.write(f"Parameters:\n")
        for k, v in params.items():
            f.write(f"  {k:25s} = {v}\n")
        f.write("\n")

        f.write(f"Pipeline stages:\n")
        f.write(f"  Stage 0  Load               : OK\n")
        f.write(f"  Stage 1  Filter cells       : {len(data['cell_ids']):,} kept\n")
        f.write(f"  Stage 2  Filter peaks       : {len(data['peak_ids']):,} kept\n")
        f.write(f"  Stage 3  Expectation        : sum = {data['expectation'].sum():.6f}\n")
        f.write(f"  Stage 4  Bias bins          : {data['n_bins']}x{data['n_bins']}\n")
        f.write(f"  Stage 5  Background peaks   : {data['background_peaks'].shape[1]} iter\n")
        f.write(f"  Stage 6  Deviations         : {data['deviations'].shape}\n")
        f.write(f"  Stage 7  Variability filter : {data['X_dnn'].shape[1]} variable motifs\n")
        f.write(f"  Stage 8  Save               : {OUT_DIR}\n\n")

        f.write(f"DNN INPUT (X_dnn): {data['X_dnn'].shape}\n")
        f.write(f"   cells × variable_motifs\n\n")

        var = data['variability']
        f.write(f"Deviation stats:\n")
        f.write(f"   min    : {np.nanmin(data['deviations']):.4f}\n")
        f.write(f"   max    : {np.nanmax(data['deviations']):.4f}\n")
        f.write(f"   median : {np.nanmedian(np.abs(data['deviations'])):.4f}\n\n")

        f.write(f"Z-score stats:\n")
        f.write(f"   min    : {np.nanmin(data['z_scores']):.3f}\n")
        f.write(f"   max    : {np.nanmax(data['z_scores']):.3f}\n")
        f.write(f"   median : {np.nanmedian(np.abs(data['z_scores'])):.3f}\n\n")

        f.write(f"Top 20 variable motifs:\n")
        top = np.argsort(var)[::-1][:20]
        for rank, i in enumerate(top, 1):
            f.write(f"  {rank:>3}. {data['motif_ids_all'][i]:15s} "
                    f"{data['motif_names_all'][i]:20s} var={var[i]:6.3f}\n")

    log.info(f"  ✓ all outputs in: {OUT_DIR}")
    log.info(f"  Summary: {OUT_DIR / 'pipeline_summary.txt'}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Complete chromVAR pipeline")
    parser.add_argument('--min-depth', type=int, default=1500,
                        help='min fragments per cell (default 1500)')
    parser.add_argument('--min-peak-fragments', type=int, default=1,
                        help='min fragments per peak (default 1)')
    parser.add_argument('--n-bins', type=int, default=50,
                        help='bias bins per axis (default 50, paper)')
    parser.add_argument('--w', type=float, default=0.1,
                        help='Mahalanobis kernel width (default 0.1, chromVAR R default)')
    parser.add_argument('--n-iter', type=int, default=50,
                        help='background iterations (default 50, paper)')
    parser.add_argument('--min-peaks', type=int, default=5,
                        help='skip motifs with fewer peaks')
    parser.add_argument('--variability-cutoff', type=float, default=1.5)
    parser.add_argument('--corr-cutoff', type=float, default=0.9)
    args = parser.parse_args()

    params = vars(args)

    log.info("╔" + "═" * 70 + "╗")
    log.info("║" + "  CHROMVAR PIPELINE — PAPER-ACCURATE GPU IMPLEMENTATION  ".center(70) + "║")
    log.info("╚" + "═" * 70 + "╝")
    log.info("")
    log.info("Parameters:")
    for k, v in params.items():
        log.info(f"  {k:25s} = {v}")

    t_start = time.time()

    data = stage0_load_inputs()
    data = stage1_filter_cells(data, min_depth=args.min_depth)
    data = stage2_filter_peaks(data, min_fragments=args.min_peak_fragments)
    data = stage3_compute_expectation(data)
    data = stage4_build_bias_bins(data, n_bins=args.n_bins, w=args.w)
    data = stage5_sample_background_peaks(data, n_iter=args.n_iter)
    data = stage6_compute_deviations(data, min_peaks=args.min_peaks)
    data = stage7_variability(data,
                               variability_cutoff=args.variability_cutoff,
                               corr_cutoff=args.corr_cutoff)
    stage8_save(data, params)

    total_time = time.time() - t_start
    log.info("")
    log.info(f"✅ Pipeline completed in {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()