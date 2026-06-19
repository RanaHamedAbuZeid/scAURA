#!/usr/bin/env python3


import os
_cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if _cvd is None or _cvd.strip() == '' or _cvd.strip().lower() == 'nodevfiles':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors



# ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/home/rhamed/results/pbmc_atlas/chromvar_final_gh38/")
SPLITS_DIR = BASE_DIR / "splits_fixed"
OUT_DIR    = BASE_DIR / "stage_e_model7_fixed"
LOG_DIR    = Path("/home/rhamed/results/pbmc_atlas/logs")
OUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'stage_e_model7_fixed.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def banner(title, char="═"):
    log.info("")
    log.info(char * 72)
    log.info(f"  {title}")
    log.info(char * 72)


class WideNet(nn.Module):
    hidden = [1024, 512, 256]
    dropout_p = 0.4

    def __init__(self, input_dim, n_classes, dropout=None):
        super().__init__()
        dp = self.dropout_p if dropout is None else dropout
        dims = [input_dim] + list(self.hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dp),
            ])
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(self.hidden[-1], n_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        logits = self.head(feats)
        return (logits, feats) if return_features else logits


# ═══════════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════════

class LabelSmoothingCE(nn.Module):
    """CE with label smoothing and class weights, supports MixUp soft targets."""
    def __init__(self, n_classes, class_weights=None, epsilon=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.epsilon = epsilon
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            w = w / w.mean()
            self.register_buffer('weights', w)
        else:
            self.weights = None

    def _smooth_one_hot(self, targets):
        oh = F.one_hot(targets, self.n_classes).float()
        return oh * (1 - self.epsilon) + self.epsilon / self.n_classes

    def forward(self, logits, targets, targets_b=None, lam=1.0):
        log_probs = F.log_softmax(logits, dim=1)
        smooth_a = self._smooth_one_hot(targets)
        if targets_b is not None:
            smooth_b = self._smooth_one_hot(targets_b)
            smooth = lam * smooth_a + (1 - lam) * smooth_b
        else:
            smooth = smooth_a
        nll = -(smooth * log_probs).sum(dim=1)
        if self.weights is not None:
            nll = nll * self.weights[targets]
        return nll.mean()

def mixup_batch(x, y, alpha=0.2, small_class_ids=None):

    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    lam = max(lam, 1.0 - lam)  # keep original label dominant

    idx = torch.randperm(x.size(0), device=x.device)

    # Protect small classes: if a sample is in a small class, don't mix
    if small_class_ids is not None and len(small_class_ids) > 0:
        mask_small = torch.zeros_like(y, dtype=torch.bool)
        for c in small_class_ids:
            mask_small |= (y == c)
        # For samples in small classes, use identity mixing (no change)
        idx_final = torch.where(mask_small, torch.arange(x.size(0), device=x.device), idx)
    else:
        idx_final = idx

    x_mixed = lam * x + (1 - lam) * x[idx_final]
    return x_mixed, y, y[idx_final], lam

def compute_class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    return len(y) / (n_classes * counts.clip(min=1))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def identify_small_classes(y, threshold=300):
    """Return class IDs with fewer samples than threshold (protect in MixUp)."""
    counts = np.bincount(y)
    return [c for c, n in enumerate(counts) if n < threshold]

def train_epoch(model, loader, criterion, optimizer, scheduler, device,
                mixup_alpha, small_classes, grad_clip=1.0):
    model.train()
    total_loss, total_n = 0.0, 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()

        if mixup_alpha > 0 and np.random.rand() < 0.5:
            x_m, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha, small_classes)
            logits = model(x_m)
            loss = criterion(logits, y_a, targets_b=y_b, lam=lam)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()  # OneCycleLR per batch (correct this time)

        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        with torch.no_grad():
            correct += (logits.argmax(1) == y).sum().item()
    return total_loss / total_n, correct / total_n


@torch.no_grad()
def eval_labels(model, loader, criterion, device, return_features=False):
    model.eval()
    total_loss, total_n = 0.0, 0
    preds, labels_all, probs_all, feats_all = [], [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if return_features:
            logits, feats = model(x, return_features=True)
            feats_all.append(feats.cpu().numpy())
        else:
            logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        labels_all.append(y.cpu().numpy())
        probs_all.append(F.softmax(logits, dim=1).cpu().numpy())

    result = {
        'loss': total_loss / total_n,
        'preds': np.concatenate(preds),
        'labels': np.concatenate(labels_all),
        'probs': np.concatenate(probs_all, axis=0),
        'acc': accuracy_score(np.concatenate(labels_all), np.concatenate(preds)),
        'bal_acc': balanced_accuracy_score(
            np.concatenate(labels_all), np.concatenate(preds)),
    }
    if return_features:
        result['features'] = np.concatenate(feats_all, axis=0)
    return result

@torch.no_grad()
def safe_inference(model, X_scaled, device, batch_size=256):

    model.eval()
    ds = TensorDataset(torch.from_numpy(X_scaled.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        pin_memory=(device.type == 'cuda'))
    probs_all, logits_all, feats_all = [], [], []
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        logits, feats = model(x, return_features=True)
        probs_all.append(F.softmax(logits, dim=1).cpu().numpy())
        logits_all.append(logits.cpu().numpy())
        feats_all.append(feats.cpu().numpy())
    return (np.concatenate(probs_all, axis=0),
            np.concatenate(logits_all, axis=0),
            np.concatenate(feats_all, axis=0))

@torch.no_grad()
def safe_mc_dropout(model, X_scaled, device, n_passes, batch_size=256):
    """MC Dropout — no loss."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    ds = TensorDataset(torch.from_numpy(X_scaled.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        pin_memory=(device.type == 'cuda'))
    all_passes = []
    for _ in range(n_passes):
        probs_list = []
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            probs_list.append(F.softmax(model(x), dim=1).cpu().numpy())
        all_passes.append(np.concatenate(probs_list, axis=0))
    model.eval()
    return np.stack(all_passes, axis=0)


def fit_mahalanobis(F_train, y_train, n_classes, ridge=1e-4):
    D = F_train.shape[1]
    class_means = np.zeros((n_classes, D))
    centered = np.zeros_like(F_train)
    for c in range(n_classes):
        mask = (y_train == c)
        if mask.sum() == 0:
            continue
        class_means[c] = F_train[mask].mean(axis=0)
        centered[mask] = F_train[mask] - class_means[c]
    cov = (centered.T @ centered) / len(F_train) + np.eye(D) * ridge
    return class_means, np.linalg.inv(cov)


def mahalanobis_scores(F_query, class_means, precision):
    N = F_query.shape[0]
    C = class_means.shape[0]
    d = np.zeros((N, C))
    for c in range(C):
        diff = F_query - class_means[c]
        d[:, c] = np.sum((diff @ precision) * diff, axis=1)
    return d.min(axis=1)


def energy_score(logits, T=1.0):
    return -T * np.log(np.exp(logits / T).sum(axis=1) + 1e-10)


def knn_scores(F_train, F_query, k=10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(F_train)
    d, _ = nbrs.kneighbors(F_query)
    return d.mean(axis=1)


def fpr_at_tpr(y_true, scores, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= tpr_target)[0]
    return float(fpr[idx[0]]) if len(idx) > 0 else 1.0

def train_one_widenet(seed, X_train, y_train, X_val, y_val,
                      n_classes, class_weights, device, args):
    """Train one WideNet with the full improvement stack."""
    banner(f"TRAINING WIDENET — seed {seed}", char="─")
    set_seed(seed)

    # Each model gets its OWN scaler (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_s),
                      torch.from_numpy(y_train.astype(np.int64))),
        batch_size=args.batch, shuffle=True,
        pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_s),
                      torch.from_numpy(y_val.astype(np.int64))),
        batch_size=args.batch, shuffle=False,
        pin_memory=(device.type == 'cuda'))

    model = WideNet(input_dim=X_train.shape[1], n_classes=n_classes,
                    dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  parameters: {n_params:,}  dropout: {args.dropout}")

    criterion = LabelSmoothingCE(
        n_classes=n_classes, class_weights=class_weights,
        epsilon=args.label_smooth).to(device)
    eval_crit = LabelSmoothingCE(
        n_classes=n_classes, class_weights=None, epsilon=0.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    # FIXED: OneCycleLR stepped per-BATCH now (inside train_epoch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.1, anneal_strategy='cos',
        div_factor=25, final_div_factor=1000)

    small_classes = identify_small_classes(y_train, threshold=300)
    log.info(f"  small classes protected from MixUp: {small_classes}")

    best_val_acc = 0.0
    best_epoch = 0
    patience_ctr = 0
    best_state = None

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            mixup_alpha=args.mixup_alpha, small_classes=small_classes)
        val = eval_labels(model, val_loader, eval_crit, device)

        improved = val['acc'] > best_val_acc
        if improved:
            best_val_acc = val['acc']
            best_epoch = epoch
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 20 == 0 or improved:
            marker = "  ← best" if improved else ""
            lr_now = optimizer.param_groups[0]['lr']
            log.info(f"  epoch {epoch:>3}  lr={lr_now:.1e}  "
                     f"train_acc={train_acc:.4f}  val_acc={val['acc']:.4f}  "
                     f"bal={val['bal_acc']:.4f}{marker}")

        if patience_ctr >= args.patience:
            log.info(f"  early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t_start
    log.info(f"  best val_acc={best_val_acc:.4f} at epoch {best_epoch}  "
             f"time={elapsed/60:.1f} min")

    model.load_state_dict(best_state)
    return model, scaler, best_val_acc, best_epoch


def plot_roc_curves(y_ood, scores_dict, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    colors = {
        'ensemble_var':   '#7F77DD',
        'mahalanobis':    '#E24B4A',
        'energy_mean':    '#185FA5',
        'knn_concat':     '#1D9E75',
        'msp_ensemble':   '#BA7517',
        'mc_variance':    '#888780',
    }
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_ood, scores)
        auroc = roc_auc_score(y_ood, scores)
        plt.plot(fpr, tpr, linewidth=2.0,
                 color=colors.get(name, None),
                 label=f"{name:18s} AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('OOD Detection — 5-WideNet Ensemble (Plasma cell vs 10 ID types)',
              fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_distributions(id_scores_dict, ood_scores_dict, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = list(id_scores_dict.keys())
    n = len(methods)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, method in zip(axes, methods):
        id_s, ood_s = id_scores_dict[method], ood_scores_dict[method]
        ax.hist(id_s,  bins=50, alpha=0.6, color='#378ADD',
                label=f'ID (n={len(id_s)})', density=True)
        ax.hist(ood_s, bins=50, alpha=0.6, color='#E24B4A',
                label=f'OOD (n={len(ood_s)})', density=True)
        auroc = roc_auc_score(
            np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))]),
            np.concatenate([id_s, ood_s]))
        ax.set_title(f'{method}\nAUROC = {auroc:.4f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('OOD score')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_confusion(y_true, y_pred, class_names, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, data, t, fmt in zip(
        axes, [cm_norm, cm], ["Normalized (row %)", "Raw counts"], [".2f", "d"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt,
                    xticklabels=class_names, yticklabels=class_names,
                    cmap="Blues", linewidths=0.5, ax=ax)
        ax.set_title(t, fontsize=13, fontweight='bold')
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    plt.suptitle("Stage E — 5-WideNet Ensemble (Test Set)",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--label-smooth', type=float, default=0.1)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--mc-passes', type=int, default=30)
    parser.add_argument('--knn-k', type=int, default=10)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    banner("STAGE E — ENSEMBLE OF 5 IMPROVED WIDENETS")
    log.info(f"  splits:       {SPLITS_DIR}")
    log.info(f"  output:       {OUT_DIR}")
    log.info(f"  seeds:        {args.seeds}")
    log.info(f"  epochs:       {args.epochs}")
    log.info(f"  peak lr:      {args.lr}")
    log.info(f"  mixup alpha:  {args.mixup_alpha}")
    log.info(f"  label smooth: {args.label_smooth}")
    log.info(f"  MC passes:    {args.mc_passes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log.info(f"  device:       {torch.cuda.get_device_name(0)}")

    # ── Load data ─────────────────────────────────────────────────────
    banner("LOADING DATA")
    X_train    = np.load(SPLITS_DIR / "X_train.npy")
    X_val      = np.load(SPLITS_DIR / "X_val.npy")
    X_test_id  = np.load(SPLITS_DIR / "X_test_id.npy")
    X_test_ood = np.load(SPLITS_DIR / "X_test_ood.npy")
    y_train    = np.load(SPLITS_DIR / "labels_train.npy")
    y_val      = np.load(SPLITS_DIR / "labels_val.npy")
    y_test_id  = np.load(SPLITS_DIR / "labels_test_id.npy")
    y_test_ood = np.load(SPLITS_DIR / "labels_test_ood.npy")
    class_names = np.load(SPLITS_DIR / "label_encoder.npy", allow_pickle=True)
    idx_npz = np.load(SPLITS_DIR / "split_indices.npz", allow_pickle=True)

    n_classes = len(class_names)
    input_dim = X_train.shape[1]

    log.info(f"  train:    {X_train.shape}")
    log.info(f"  val:      {X_val.shape}")
    log.info(f"  test_id:  {X_test_id.shape}")
    log.info(f"  test_ood: {X_test_ood.shape}")
    log.info(f"  n_classes: {n_classes}  input_dim: {input_dim}")

    # Data leakage check
    all_idx = np.concatenate([idx_npz['train'], idx_npz['val'],
                              idx_npz['test_id'], idx_npz['test_ood']])
    assert len(all_idx) == len(np.unique(all_idx)), "❌ DATA LEAKAGE"
    assert (y_train >= 0).all() and (y_test_ood == -1).all()
    log.info(f"  ✓ no data leakage, labels correct")

    class_weights = compute_class_weights(y_train, n_classes)

    # ── Train 5 models ────────────────────────────────────────────────
    banner("TRAINING 5 WIDENETS")
    models, scalers, val_accs, best_epochs = [], [], [], []

    total_start = time.time()
    for model_idx, seed in enumerate(args.seeds, start=1):
        log.info(f"\n[model {model_idx}/{len(args.seeds)}]")
        model, scaler, val_acc, best_ep = train_one_widenet(
            seed, X_train, y_train, X_val, y_val,
            n_classes, class_weights, device, args)
        models.append(model)
        scalers.append(scaler)
        val_accs.append(val_acc)
        best_epochs.append(best_ep)

        ckpt_path = OUT_DIR / f"model_{model_idx}_widenet_seed{seed}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_mean':  scaler.mean_,
            'scaler_scale': scaler.scale_,
            'best_val_acc': val_acc,
            'best_epoch':   best_ep,
            'seed':         seed,
            'class_names':  class_names,
            'args':         vars(args),
        }, ckpt_path)
        log.info(f"  ✓ saved: {ckpt_path.name}")

    total_time = time.time() - total_start
    log.info(f"\n  total training time: {total_time/60:.1f} min")

    # ── Ensemble inference ────────────────────────────────────────────
    banner("ENSEMBLE INFERENCE (test_id + test_ood)")
    # Per-model outputs
    all_probs_id, all_logits_id, all_feats_id = [], [], []
    all_probs_ood, all_logits_ood, all_feats_ood = [], [], []
    all_mc_id, all_mc_ood = [], []

    for i, (model, scaler) in enumerate(zip(models, scalers), start=1):
        # Apply THIS model's scaler to test data
        X_id_s  = scaler.transform(X_test_id).astype(np.float32)
        X_ood_s = scaler.transform(X_test_ood).astype(np.float32)

        p_id, l_id, f_id = safe_inference(model, X_id_s, device, args.batch)
        p_ood, l_ood, f_ood = safe_inference(model, X_ood_s, device, args.batch)

        all_probs_id.append(p_id);    all_probs_ood.append(p_ood)
        all_logits_id.append(l_id);   all_logits_ood.append(l_ood)
        all_feats_id.append(f_id);    all_feats_ood.append(f_ood)

        mc_id  = safe_mc_dropout(model, X_id_s,  device, args.mc_passes, args.batch)
        mc_ood = safe_mc_dropout(model, X_ood_s, device, args.mc_passes, args.batch)
        all_mc_id.append(mc_id);      all_mc_ood.append(mc_ood)

        acc = accuracy_score(y_test_id, p_id.argmax(axis=1))
        log.info(f"  M{i} (seed {args.seeds[i-1]}): val={val_accs[i-1]:.4f}  test={acc:.4f}")

    # Stack across models
    probs_id_stack   = np.stack(all_probs_id, axis=0)    
    probs_ood_stack  = np.stack(all_probs_ood, axis=0)   
    logits_id_stack  = np.stack(all_logits_id, axis=0)
    logits_ood_stack = np.stack(all_logits_ood, axis=0)

    # Ensemble averaging
    mean_probs_id  = probs_id_stack.mean(axis=0)
    mean_probs_ood = probs_ood_stack.mean(axis=0)
    y_pred_id = mean_probs_id.argmax(axis=1)

    # Classification metrics
    banner("ENSEMBLE CLASSIFICATION METRICS")
    ens_acc    = accuracy_score(y_test_id, y_pred_id)
    ens_bal    = balanced_accuracy_score(y_test_id, y_pred_id)
    ens_kappa  = cohen_kappa_score(y_test_id, y_pred_id)
    ens_macro  = f1_score(y_test_id, y_pred_id, average='macro')
    ens_wtdF1  = f1_score(y_test_id, y_pred_id, average='weighted')
    log.info(f"  ensemble accuracy:     {ens_acc:.4f}  ({ens_acc*100:.2f}%)")
    log.info(f"  ensemble balanced acc: {ens_bal:.4f}")
    log.info(f"  Cohen's kappa:         {ens_kappa:.4f}")
    log.info(f"  macro F1:              {ens_macro:.4f}")
    log.info(f"  weighted F1:           {ens_wtdF1:.4f}")

    log.info(f"\n  Individual vs ensemble test accuracy:")
    ind_test_accs = [accuracy_score(y_test_id, p.argmax(axis=1))
                      for p in all_probs_id]
    for i, acc in enumerate(ind_test_accs, start=1):
        log.info(f"    M{i} (seed {args.seeds[i-1]}): {acc:.4f}")
    log.info(f"    ensemble:           {ens_acc:.4f}  "
             f"(gain: {(ens_acc - max(ind_test_accs))*100:+.2f}%)")

    log.info(f"\n  Classification report:\n"
             f"{classification_report(y_test_id, y_pred_id, target_names=class_names, digits=4, zero_division=0)}")

    pc_rows = []
    for i, name in enumerate(class_names):
        m = (y_test_id == i)
        n = int(m.sum())
        if n == 0: continue
        pc_rows.append({
            'class_id': i, 'class_name': str(name), 'N': n,
            'accuracy': round(accuracy_score(y_test_id[m], y_pred_id[m]), 4),
            'precision': round(precision_score(y_test_id, y_pred_id, labels=[i],
                               average='micro', zero_division=0), 4),
            'recall': round(recall_score(y_test_id, y_pred_id, labels=[i],
                            average='micro', zero_division=0), 4),
            'f1': round(f1_score(y_test_id, y_pred_id, labels=[i],
                        average='micro', zero_division=0), 4),
        })
    pc_df = pd.DataFrame(pc_rows)
    pc_df.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)
    log.info(f"\n  Per-class:\n{pc_df.to_string(index=False)}")

    save_confusion(y_test_id, y_pred_id, class_names,
                   OUT_DIR / "confusion_matrix.png")
    pd.DataFrame(confusion_matrix(y_test_id, y_pred_id,
                                   labels=np.arange(n_classes)),
                 index=class_names, columns=class_names).to_csv(
        OUT_DIR / "confusion_matrix.csv")

    banner("OOD DETECTION — multiple methods")

    # Method 1: Ensemble variance (across models' probs)
    ens_var_id  = probs_id_stack.var(axis=0).sum(axis=1)  
    ens_var_ood = probs_ood_stack.var(axis=0).sum(axis=1)

    # Method 2: Energy averaged across models
    energy_per_model_id  = np.stack([energy_score(l) for l in all_logits_id])
    energy_per_model_ood = np.stack([energy_score(l) for l in all_logits_ood])
    energy_id  = energy_per_model_id.mean(axis=0)
    energy_ood = energy_per_model_ood.mean(axis=0)

    # Method 3: Mahalanobis on CONCATENATED features
    # (best single model's features for fitting, concat of all for scoring)
    best_model_idx = int(np.argmax(val_accs))
    F_tr_best, F_id_best, F_ood_best = [], [], []

    # Need train features from best model for Mahalanobis/KNN
    X_train_s_best = scalers[best_model_idx].transform(X_train).astype(np.float32)
    _, _, F_tr_best = safe_inference(models[best_model_idx],
                                      X_train_s_best, device, args.batch)
    F_id_best  = all_feats_id[best_model_idx]
    F_ood_best = all_feats_ood[best_model_idx]

    class_means, precision = fit_mahalanobis(F_tr_best, y_train, n_classes)
    maha_id  = mahalanobis_scores(F_id_best,  class_means, precision)
    maha_ood = mahalanobis_scores(F_ood_best, class_means, precision)

    # Method 4: KNN on best model's features
    knn_id  = knn_scores(F_tr_best, F_id_best,  k=args.knn_k)
    knn_ood = knn_scores(F_tr_best, F_ood_best, k=args.knn_k)

    # Method 5: MSP on ensemble mean
    msp_id  = 1.0 - mean_probs_id.max(axis=1)
    msp_ood = 1.0 - mean_probs_ood.max(axis=1)

    # Method 6: MC variance averaged across models
    mc_var_per_model_id  = np.stack([mc.var(axis=0).sum(axis=1) for mc in all_mc_id])
    mc_var_per_model_ood = np.stack([mc.var(axis=0).sum(axis=1) for mc in all_mc_ood])
    mc_variance_id  = mc_var_per_model_id.mean(axis=0)
    mc_variance_ood = mc_var_per_model_ood.mean(axis=0)

    # Compile results
    y_ood = np.concatenate([
        np.zeros(len(X_test_id),  dtype=np.int32),
        np.ones(len(X_test_ood), dtype=np.int32),
    ])

    methods = {
        'ensemble_var': (ens_var_id,    ens_var_ood),
        'energy_mean':  (energy_id,     energy_ood),
        'mahalanobis':  (maha_id,       maha_ood),
        'knn_concat':   (knn_id,        knn_ood),
        'msp_ensemble': (msp_id,        msp_ood),
        'mc_variance':  (mc_variance_id, mc_variance_ood),
    }

    log.info(f"  {'method':18s}{'AUROC':>10s}{'AUPR-OUT':>12s}{'AUPR-IN':>10s}{'FPR@95':>10s}")
    log.info("  " + "─" * 60)

    ood_results = {}
    all_scores = {}
    id_scores_dict, ood_scores_dict = {}, {}

    for name, (id_s, ood_s) in methods.items():
        all_s = np.concatenate([id_s, ood_s])
        auroc = roc_auc_score(y_ood, all_s)
        aupr_out = average_precision_score(y_ood, all_s)
        aupr_in = average_precision_score(1 - y_ood, -all_s)
        fpr95 = fpr_at_tpr(y_ood, all_s, 0.95)
        ood_results[name] = {
            'auroc': float(auroc), 'aupr_out': float(aupr_out),
            'aupr_in': float(aupr_in), 'fpr_at_95_tpr': float(fpr95),
            'id_mean': float(id_s.mean()), 'id_std': float(id_s.std()),
            'ood_mean': float(ood_s.mean()), 'ood_std': float(ood_s.std()),
        }
        log.info(f"  {name:18s}{auroc:>10.4f}{aupr_out:>12.4f}"
                 f"{aupr_in:>10.4f}{fpr95:>10.4f}")
        all_scores[name] = all_s
        id_scores_dict[name] = id_s
        ood_scores_dict[name] = ood_s

    # Save per-cell scores
    pd.DataFrame({
        'is_ood': y_ood,
        'pred_class': np.concatenate([y_pred_id, mean_probs_ood.argmax(axis=1)]),
        **all_scores,
    }).to_csv(OUT_DIR / "ood_scores_full.csv", index=False)

    # Save Mahalanobis stats
    np.savez(OUT_DIR / "mahalanobis_stats.npz",
             class_means=class_means, precision=precision,
             class_names=class_names, best_model_idx=best_model_idx)

    # ── Plots ─────────────────────────────────────────────────────────
    plot_roc_curves(y_ood, all_scores, OUT_DIR / "ood_roc_curves.png")
    plot_distributions(id_scores_dict, ood_scores_dict,
                       OUT_DIR / "ood_score_distributions.png")
    log.info(f"  ✓ plots saved")

    # ── Save ensemble predictions ─────────────────────────────────────
    np.savez(OUT_DIR / "ensemble_predictions.npz",
             y_test_id=y_test_id, y_pred_id=y_pred_id,
             mean_probs_id=mean_probs_id, mean_probs_ood=mean_probs_ood,
             probs_id_stack=probs_id_stack, probs_ood_stack=probs_ood_stack)

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        'date': str(datetime.now()),
        'n_models': len(models),
        'seeds': args.seeds,
        'individual_val_accs': [float(a) for a in val_accs],
        'individual_test_accs': [float(a) for a in ind_test_accs],
        'individual_best_epochs': [int(e) for e in best_epochs],
        'best_model_idx': int(best_model_idx),
        'classification': {
            'ensemble_test_accuracy': float(ens_acc),
            'ensemble_balanced_accuracy': float(ens_bal),
            'ensemble_cohens_kappa': float(ens_kappa),
            'ensemble_macro_f1': float(ens_macro),
            'ensemble_weighted_f1': float(ens_wtdF1),
            'best_single_test_accuracy': float(max(ind_test_accs)),
            'ensemble_gain_over_best_single': float(ens_acc - max(ind_test_accs)),
        },
        'ood_detection': ood_results,
        'total_training_time_min': round(total_time / 60, 1),
        'args': vars(args),
    }
    with open(OUT_DIR / "ensemble_classification_metrics.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    with open(OUT_DIR / "ood_results.json", 'w') as f:
        json.dump({'ood_detection': ood_results,
                   'ensemble_accuracy': float(ens_acc)}, f, indent=2, default=str)

    # ── Verdict ───────────────────────────────────────────────────────
    banner("VERDICT")
    best_method, best_res = max(ood_results.items(), key=lambda x: x[1]['auroc'])
    log.info(f"  ensemble accuracy:   {ens_acc*100:.2f}% "
             f"(vs 84.03% single, Δ={((ens_acc-0.8403)*100):+.2f}%)")
    log.info(f"  best OOD method:     {best_method}")
    log.info(f"  best OOD AUROC:      {best_res['auroc']:.4f} "
             f"(vs 0.8297 single, Δ={((best_res['auroc']-0.8297)):+.4f})")
    log.info("")
    if best_res['auroc'] >= 0.90:
        log.info("  ✅ STRONG OOD detection achieved (≥0.90)")
        log.info("     → Stage F (Outlier Exposure) may not be needed.")
    elif best_res['auroc'] >= 0.87:
        log.info("  🟡 GOOD improvement; Stage F (OE) can push to 0.90+")
    elif best_res['auroc'] > 0.84:
        log.info("  ⚠  Modest improvement; Stage F (OE) is essential")
    else:
        log.info("  ❌ Ensemble didn't help — need architectural changes")

    log.info("")
    log.info("═" * 72)
    log.info("STAGE E COMPLETE")
    log.info("═" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
