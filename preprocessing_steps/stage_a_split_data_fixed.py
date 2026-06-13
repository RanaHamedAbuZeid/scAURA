#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
STAGE A — DATA SPLITTING FOR LEAVE-ONE-OUT OOD DETECTION
═══════════════════════════════════════════════════════════════════════════

Configuration
─────────────
  OOD target class : "Plasma cell" (415 cells)
  ID classes       : 13 remaining cell types (38,934 cells)
  Split ratios     : 80 / 10 / 10  (train / val / test)
  Random seed      : 42
  Stratification   : by celltype within ID data

Inputs
──────
  X_dnn.npy                           — (39349, 879) bias-corrected deviations
  X_dnn_z.npy                         — (39349, 879) z-scores (also split for later comparison)
  cell_ids_kept.npy                   — (39349,) cell identifiers
  step1_data.pkl                      — AnnData with obs['celltypes'] labels

Outputs (in OUT_DIR / 'splits/')
───────
  splits/X_train.npy      labels_train.npy       cell_ids_train.npy
  splits/X_val.npy        labels_val.npy         cell_ids_val.npy
  splits/X_test_id.npy    labels_test_id.npy     cell_ids_test_id.npy
  splits/X_test_ood.npy   labels_test_ood.npy    cell_ids_test_ood.npy
  splits/X_train_z.npy  ...  (same 4 splits for z-scores)
  splits/label_encoder.npy            — mapping: integer → cell type name
  splits/split_summary.txt            — human-readable report
  splits/split_indices.npz            — row indices into original X_dnn

Usage
─────
  python stage_a_split_data.py
  python stage_a_split_data.py --ood "pDC"       # different OOD class
  python stage_a_split_data.py --train 0.70 --val 0.15 --test 0.15
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path("/home/rhamed/results/pbmc_atlas/chromvar_final_gh38")
CHECKPOINT_DIR = Path("/home/rhamed/results/pbmc_atlas/checkpoints_v4_1_copy")
OUT_DIR        = BASE_DIR / "splits_fixed"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ood', type=str, default='Plasma cell',
                        help='OOD target cell type name (must match obs[celltypes])')
    parser.add_argument('--train', type=float, default=0.80)
    parser.add_argument('--val', type=float, default=0.10)
    parser.add_argument('--test', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Validate ratios
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        print(f"❌ Ratios sum to {total}, must be 1.0")
        return 1

    print("═" * 72)
    print("STAGE A — DATA SPLITTING FOR OOD DETECTION")
    print("═" * 72)
    print(f"OOD target:   {args.ood}")
    print(f"Split ratios: train={args.train}  val={args.val}  test={args.test}")
    print(f"Random seed:  {args.seed}")
    print()

    # ── Load features ─────────────────────────────────────────────────
    print("Loading features ...")
    X      = np.load(BASE_DIR / "X_dnn.npy")           # deviations (primary)
    X_z    = np.load(BASE_DIR / "X_dnn_z.npy")         # z-scores (alternative)
    cells  = np.load(BASE_DIR / "cell_ids_kept.npy", allow_pickle=True)
    assert X.shape == X_z.shape, "shape mismatch between X_dnn and X_dnn_z"
    assert X.shape[0] == len(cells), "cell count mismatch"
    print(f"  X_dnn shape:     {X.shape}  dtype={X.dtype}")
    print(f"  X_dnn_z shape:   {X_z.shape}  dtype={X_z.dtype}")
    print(f"  cell_ids:        {len(cells):,}")

    # ── Load labels from AnnData obs ──────────────────────────────────
    print("\nLoading labels from AnnData ...")
    with open(CHECKPOINT_DIR / "step1_data.pkl", 'rb') as f:
        adata = pickle.load(f)

    # Build lookup: cell_id → celltype
    obs_names = np.array(adata.obs_names)
    obs_celltypes = np.array(adata.obs['celltypes'].astype(str))
    cell_to_type = dict(zip(obs_names, obs_celltypes))

    # Map our kept cells to their celltypes
    labels_str = np.array([cell_to_type[c] for c in cells])
    assert len(labels_str) == len(cells)

    print(f"  labels retrieved: {len(labels_str):,}")
    unique_types, counts = np.unique(labels_str, return_counts=True)
    print(f"\n  Cell type distribution:")
    for t, c in sorted(zip(unique_types, counts), key=lambda x: -x[1]):
        marker = "  ← OOD" if t == args.ood else ""
        print(f"    {t:32s} {c:>6,}{marker}")

    # Validate OOD class exists
    if args.ood not in unique_types:
        print(f"\n❌ OOD class '{args.ood}' not found. Available:")
        for t in unique_types:
            print(f"     - {t}")
        return 1

    # ── Split: OOD vs ID ──────────────────────────────────────────────
    ood_mask = (labels_str == args.ood)
    id_mask  = ~ood_mask

    n_ood = int(ood_mask.sum())
    n_id  = int(id_mask.sum())

    print(f"\nInitial partition:")
    print(f"  OOD cells ({args.ood}): {n_ood:,}")
    print(f"  ID cells (13 types):     {n_id:,}")

    # ── Stratified split on ID cells only ─────────────────────────────
    print(f"\nStratified split on ID cells ({n_id:,}) ...")
    id_indices = np.where(id_mask)[0]
    id_labels  = labels_str[id_mask]

    # First split: train vs (val + test)
    # train_test_split is stratified by default when given stratify=
    train_idx, valtest_idx, _, valtest_labels = train_test_split(
        id_indices, id_labels,
        train_size=args.train,
        random_state=args.seed,
        stratify=id_labels,
    )

    # Second split: val vs test (from the valtest portion)
    # Preserve relative ratio of val vs test
    val_fraction_of_valtest = args.val / (args.val + args.test)
    val_idx, test_idx = train_test_split(
        valtest_idx,
        train_size=val_fraction_of_valtest,
        random_state=args.seed,
        stratify=valtest_labels,
    )

    ood_idx = np.where(ood_mask)[0]

    print(f"  train:    {len(train_idx):,}  ({len(train_idx)/n_id*100:.1f}% of ID)")
    print(f"  val:      {len(val_idx):,}  ({len(val_idx)/n_id*100:.1f}% of ID)")
    print(f"  id_test:  {len(test_idx):,}  ({len(test_idx)/n_id*100:.1f}% of ID)")
    print(f"  ood_test: {len(ood_idx):,}  (100% of {args.ood})")

    # ── Sanity: no overlap ────────────────────────────────────────────
    all_idx = np.concatenate([train_idx, val_idx, test_idx, ood_idx])
    assert len(all_idx) == len(set(all_idx)), "OVERLAP DETECTED between splits!"
    assert len(all_idx) == len(X), f"missing cells: {len(all_idx)} != {len(X)}"
    print(f"\n  ✓ no overlap between splits")
    print(f"  ✓ all {len(X):,} cells accounted for")

    # ── Label encoding (ID classes only — for classifier) ─────────────
    # The classifier will only see the 13 ID types. OOD class gets a
    # special label (-1) used only for OOD evaluation.
    id_type_names = sorted(list(set(unique_types) - {args.ood}))
    type_to_int = {t: i for i, t in enumerate(id_type_names)}

    def encode(labels_arr, for_ood=False):
        """Encode string labels to integers. OOD class → -1."""
        out = np.full(len(labels_arr), -1, dtype=np.int32)
        for i, lab in enumerate(labels_arr):
            if lab in type_to_int:
                out[i] = type_to_int[lab]
        return out

    y_train   = encode(labels_str[train_idx])
    y_val     = encode(labels_str[val_idx])
    y_test_id = encode(labels_str[test_idx])
    y_test_ood = encode(labels_str[ood_idx])  # all -1

    # Sanity: no -1 in ID splits, all -1 in OOD split
    assert (y_train >= 0).all(),   "ID train contains OOD labels!"
    assert (y_val >= 0).all(),     "ID val contains OOD labels!"
    assert (y_test_id >= 0).all(), "ID test contains OOD labels!"
    assert (y_test_ood == -1).all(), "OOD test contains non-OOD labels!"

    print(f"\nLabel encoding ({len(id_type_names)} ID classes):")
    for t, i in type_to_int.items():
        print(f"  {i:>2}: {t}")

    # ── Verify per-class distribution across splits ───────────────────
    print(f"\nPer-class distribution (ID splits):")
    print(f"  {'class':32s}{'train':>8s}{'val':>7s}{'test':>7s}{'total':>7s}")
    print(f"  " + "─" * 62)
    for t, i in type_to_int.items():
        n_tr = int((y_train == i).sum())
        n_vl = int((y_val == i).sum())
        n_te = int((y_test_id == i).sum())
        n_tot = n_tr + n_vl + n_te
        print(f"  {t:32s}{n_tr:>8,}{n_vl:>7,}{n_te:>7,}{n_tot:>7,}")
    print(f"  " + "─" * 62)
    print(f"  {'TOTAL':32s}{len(y_train):>8,}{len(y_val):>7,}{len(y_test_id):>7,}{n_id:>7,}")

    # Warn if any class has < 10 cells in val or test (statistical concern)
    print()
    for t, i in type_to_int.items():
        n_vl = int((y_val == i).sum())
        n_te = int((y_test_id == i).sum())
        if n_vl < 10:
            print(f"  ⚠ {t}: only {n_vl} cells in val — limited power")
        if n_te < 10:
            print(f"  ⚠ {t}: only {n_te} cells in test — limited power")

    # ── Save ──────────────────────────────────────────────────────────
    print(f"\nSaving to {OUT_DIR} ...")

    # Deviations (primary)
    np.save(OUT_DIR / "X_train.npy",    X[train_idx])
    np.save(OUT_DIR / "X_val.npy",      X[val_idx])
    np.save(OUT_DIR / "X_test_id.npy",  X[test_idx])
    np.save(OUT_DIR / "X_test_ood.npy", X[ood_idx])

    # Z-scores (alternative)
    np.save(OUT_DIR / "X_train_z.npy",    X_z[train_idx])
    np.save(OUT_DIR / "X_val_z.npy",      X_z[val_idx])
    np.save(OUT_DIR / "X_test_id_z.npy",  X_z[test_idx])
    np.save(OUT_DIR / "X_test_ood_z.npy", X_z[ood_idx])

    # Labels
    np.save(OUT_DIR / "labels_train.npy",    y_train)
    np.save(OUT_DIR / "labels_val.npy",      y_val)
    np.save(OUT_DIR / "labels_test_id.npy",  y_test_id)
    np.save(OUT_DIR / "labels_test_ood.npy", y_test_ood)

    # Cell IDs (for traceability)
    np.save(OUT_DIR / "cell_ids_train.npy",    cells[train_idx])
    np.save(OUT_DIR / "cell_ids_val.npy",      cells[val_idx])
    np.save(OUT_DIR / "cell_ids_test_id.npy",  cells[test_idx])
    np.save(OUT_DIR / "cell_ids_test_ood.npy", cells[ood_idx])

    # Label encoder (integer → name mapping)
    label_encoder = np.array(id_type_names)
    np.save(OUT_DIR / "label_encoder.npy", label_encoder)

    # Original indices (for reproducibility / joining back)
    np.savez(OUT_DIR / "split_indices.npz",
             train=train_idx, val=val_idx, test_id=test_idx, test_ood=ood_idx,
             ood_class_name=np.array([args.ood]),
             seed=np.array([args.seed]))

    # Human-readable summary
    with open(OUT_DIR / "split_summary.txt", 'w') as f:
        f.write("DATA SPLIT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date:          {datetime.now()}\n")
        f.write(f"OOD class:     {args.ood}\n")
        f.write(f"Seed:          {args.seed}\n")
        f.write(f"Split ratios:  {args.train}/{args.val}/{args.test}\n\n")
        f.write(f"Feature matrix: X_dnn.npy  shape={X.shape}  (deviations)\n")
        f.write(f"Also split:     X_dnn_z.npy              (z-scores)\n\n")
        f.write(f"ID classes ({len(id_type_names)}):\n")
        for t, i in type_to_int.items():
            f.write(f"  {i:>2}: {t}\n")
        f.write(f"\nOOD class:\n  {args.ood}  ({len(ood_idx)} cells, label=-1)\n\n")
        f.write(f"Split sizes:\n")
        f.write(f"  train:    {len(y_train):,}\n")
        f.write(f"  val:      {len(y_val):,}\n")
        f.write(f"  id_test:  {len(y_test_id):,}\n")
        f.write(f"  ood_test: {len(y_test_ood):,}\n")

    print(f"  ✓ X_train.npy          — {X[train_idx].shape}")
    print(f"  ✓ X_val.npy            — {X[val_idx].shape}")
    print(f"  ✓ X_test_id.npy        — {X[test_idx].shape}")
    print(f"  ✓ X_test_ood.npy       — {X[ood_idx].shape}")
    print(f"  ✓ labels_*.npy         — integer-encoded (OOD = -1)")
    print(f"  ✓ cell_ids_*.npy       — cell identifiers per split")
    print(f"  ✓ label_encoder.npy    — integer → celltype mapping")
    print(f"  ✓ split_indices.npz    — original indices (reproducibility)")
    print(f"  ✓ X_*_z.npy            — same splits, z-score version")
    print(f"  ✓ split_summary.txt    — human-readable report")

    print()
    print("═" * 72)
    print("✅ STAGE A COMPLETE — splits ready for Stage B (classifier training)")
    print("═" * 72)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
