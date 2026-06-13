#!/usr/bin/env python3
"""
Final X_dnn builder for OOD detection model.

Strategy: USE ALL MOTIFS, NO FILTERING
───────────────────────────────────────
Rationale:
  1. For OOD detection, features distinguishing rare/novel cells cannot
     be known a priori from in-distribution variability alone.
  2. The deep learning model with proper regularization (dropout, L2)
     handles 879 features without issues.
  3. Post-hoc feature importance (SHAP, integrated gradients) gives a
     more accurate picture of what the model uses than pre-filtering.
  4. No arbitrary choices to defend in peer review — every JASPAR
     vertebrate motif is included.

Outputs:
  X_dnn.npy          (n_cells, 879)   float32  — BIAS-CORRECTED DEVIATIONS
  X_dnn_z.npy        (n_cells, 879)   float32  — Z-SCORES (alternative)
  X_dnn.csv          with cell IDs × motif names
  X_dnn_motifs.csv   ordered list of motifs (by JASPAR order)
  variability.csv    variability stats (for reference, not for filtering)

Usage:
    python build_x_dnn_all.py
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
OUT_DIR        = Path("/home/rhamed/results/pbmc_atlas/chromvar_final_gh38")
CHECKPOINT_DIR = Path("/home/rhamed/results/pbmc_atlas/checkpoints_v4_1_copy")


def main():
    print("═" * 72)
    print("BUILD X_dnn FROM ALL 879 MOTIFS (no filtering)")
    print("═" * 72)

    # ── Load ──
    print("\nLoading ...")
    D = np.load(OUT_DIR / "chromvar_deviations.npy")
    Z = np.load(OUT_DIR / "chromvar_zscores.npy")
    cell_ids = np.load(OUT_DIR / "cell_ids_kept.npy", allow_pickle=True)
    motif_ids = np.load(OUT_DIR / "motif_ids_tested.npy", allow_pickle=True)

    with open(CHECKPOINT_DIR / "step5_jaspar.pkl", 'rb') as f:
        step5 = pickle.load(f)
    motif_names = step5['motifs_df']['motif_name'].values

    n_cells, n_motifs = D.shape
    print(f"  cells:  {n_cells:,}")
    print(f"  motifs: {n_motifs}")

    # ── Clean NaN (from threshold filter in chromVAR) ──
    # chromVAR produces NaN where expected < 1 (rare cells × small motifs).
    # For DNN input, replace with 0 (neutral signal).
    nan_D = np.isnan(D).sum()
    nan_Z = np.isnan(Z).sum()
    pct_D = nan_D / D.size * 100
    pct_Z = nan_Z / Z.size * 100
    print(f"\nNaN handling:")
    print(f"  deviations: {nan_D:,} NaN  ({pct_D:.2f}%) → replaced with 0")
    print(f"  z_scores:   {nan_Z:,} NaN  ({pct_Z:.2f}%) → replaced with 0")

    D_clean = np.nan_to_num(D, nan=0.0).astype(np.float32)
    Z_clean = np.nan_to_num(Z, nan=0.0).astype(np.float32)

    # ── Compute variability (for reference only, NOT used for filtering) ──
    with np.errstate(invalid='ignore'):
        variability = np.nanstd(Z, axis=0)
    variability = np.nan_to_num(variability, nan=0.0)

    # ── Immune TF sanity ──
    known_immune = {
        'SPI1','SPIB','SPIC','CEBPA','CEBPB','CEBPD','CEBPE',
        'IRF1','IRF2','IRF3','IRF4','IRF5','IRF7','IRF8',
        'GATA3','GATA2','GATA1','TBX21','EOMES',
        'ETS1','ETV6','ELF1','ELF4','FLI1',
        'BCL11A','BCL11B','BCL6','KLF2','KLF4','KLF6',
        'RUNX1','RUNX2','RUNX3','BATF','BATF3',
        'JUN','JUNB','JUND','FOS','FOSB','FOSL1','FOSL2',
        'NFKB1','RELA','RELB',
        'STAT1','STAT3','STAT4','STAT5A','STAT5B',
        'FOXP3','FOXO1','PAX5','EBF1',
        'TCF7','LEF1','TCF3','TCF4',
        'ID2','ID3','BACH1','BACH2',
        'NFATC1','NFATC2','NFATC3',
        'IKZF1','IKZF2','IKZF3',
    }

    def canonical(name):
        return str(name).upper().replace('(', '').replace(')', '').strip()

    present_immune = set()
    for i in range(n_motifs):
        n = canonical(motif_names[i])
        if n in known_immune:
            present_immune.add(n)
        for part in n.split('::'):
            if part.strip() in known_immune:
                present_immune.add(part.strip())

    print(f"\nImmune TFs present in the 879 motif set: {len(present_immune)}")
    print(f"  {sorted(present_immune)}")

    # ── Distribution checks ──
    print(f"\nX_dnn (deviations) distribution:")
    print(f"  shape:  {D_clean.shape}")
    print(f"  dtype:  {D_clean.dtype}")
    print(f"  range:  [{D_clean.min():.4f}, {D_clean.max():.4f}]")
    print(f"  mean:   {D_clean.mean():.6f}")
    print(f"  std:    {D_clean.std():.4f}")
    print(f"  |D|.p50: {np.percentile(np.abs(D_clean), 50):.4f}")
    print(f"  |D|.p95: {np.percentile(np.abs(D_clean), 95):.4f}")
    print(f"  |D|.p99: {np.percentile(np.abs(D_clean), 99):.4f}")

    print(f"\nX_dnn_z (z-scores) distribution:")
    print(f"  shape:  {Z_clean.shape}")
    print(f"  range:  [{Z_clean.min():.3f}, {Z_clean.max():.3f}]")
    print(f"  mean:   {Z_clean.mean():.4f}")
    print(f"  std:    {Z_clean.std():.4f}")

    # ── Normalization recommendation ──
    print(f"\nDNN input prep suggestions:")
    d_scale = D_clean.std()
    z_scale = Z_clean.std()
    print(f"  For deviations (X_dnn):")
    print(f"    - Already approximately centered (mean ≈ 0)")
    print(f"    - StandardScaler recommended: (X - mean) / std")
    print(f"    - Or scale by global std: X / {d_scale:.4f}")
    print(f"  For z-scores (X_dnn_z):")
    print(f"    - Already standardized per motif (std ≈ 1)")
    print(f"    - Use directly, or clip extreme values (±10)")

    # ── Save ──
    print(f"\nSaving to {OUT_DIR} ...")
    np.save(OUT_DIR / "X_dnn.npy", D_clean)
    np.save(OUT_DIR / "X_dnn_z.npy", Z_clean)

    pd.DataFrame(D_clean, index=cell_ids, columns=motif_names).to_csv(
        OUT_DIR / "X_dnn.csv")

    pd.DataFrame({
        'column_index': np.arange(n_motifs),
        'motif_id':     motif_ids,
        'motif_name':   motif_names,
        'variability':  variability,
        'is_immune_tf': [canonical(n) in known_immune or
                         any(p.strip() in known_immune for p in canonical(n).split('::'))
                         for n in motif_names],
    }).to_csv(OUT_DIR / "X_dnn_motifs.csv", index=False)

    pd.DataFrame({
        'motif_id':     motif_ids,
        'motif_name':   motif_names,
        'variability':  variability,
        'rank':         np.argsort(np.argsort(variability)[::-1]) + 1,
        'kept_for_dnn': np.ones(n_motifs, dtype=bool),   # all kept
    }).sort_values('rank').to_csv(OUT_DIR / "variability.csv", index=False)

    # Summary
    with open(OUT_DIR / "x_dnn_summary.txt", 'w') as f:
        f.write("X_dnn FINAL SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Strategy: ALL 879 motifs, no filtering\n")
        f.write(f"Rationale: OOD detection needs full feature set\n\n")
        f.write(f"Shape: {D_clean.shape} (cells × motifs)\n")
        f.write(f"Primary file: X_dnn.npy (bias-corrected deviations)\n")
        f.write(f"Alternative:  X_dnn_z.npy (z-scores)\n\n")
        f.write(f"Deviations: min={D_clean.min():.4f}  max={D_clean.max():.4f}\n")
        f.write(f"            mean={D_clean.mean():.6f}  std={D_clean.std():.4f}\n\n")
        f.write(f"Z-scores:   min={Z_clean.min():.3f}  max={Z_clean.max():.3f}\n")
        f.write(f"            mean={Z_clean.mean():.4f}  std={Z_clean.std():.4f}\n\n")
        f.write(f"Immune TFs in feature set: {len(present_immune)}\n")
        f.write(f"  {sorted(present_immune)}\n\n")
        f.write(f"Cell types in dataset (from obs['celltypes']):\n")
        f.write(f"  Monocyte (7786), Memory CD4 T (6700), Naive CD4 T (5061),\n")
        f.write(f"  Naive CD8 T (3365), Naive B (3261), Treg (2913), NK (2851),\n")
        f.write(f"  Effector memory CD8 T (2742), Central memory CD8 T (1641),\n")
        f.write(f"  Memory B (1430), MAIT (888), Plasma cell (415), cDC (277),\n")
        f.write(f"  pDC (111)\n")

    print(f"  ✓ X_dnn.npy              — {D_clean.shape}  (DEVIATIONS, primary)")
    print(f"  ✓ X_dnn_z.npy            — {Z_clean.shape}  (Z-SCORES, alternative)")
    print(f"  ✓ X_dnn.csv              — with cell_id × motif_name headers")
    print(f"  ✓ X_dnn_motifs.csv       — motif metadata (incl. is_immune_tf flag)")
    print(f"  ✓ variability.csv        — all 879 motifs ranked by variability")
    print(f"  ✓ x_dnn_summary.txt      — human-readable summary")

    print("\n" + "═" * 72)
    print("DONE. No filtering applied. All 879 motifs available to the DNN.")
    print("Next: train your OOD detection model with X_dnn.npy as input.")
    print("═" * 72)


if __name__ == "__main__":
    main()
