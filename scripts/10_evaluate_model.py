"""
10 EVALUATE MODEL
======================================================================
Model Evaluation on Held-Out Test Set
======================================================================

Loads the best model checkpoint and evaluates on the 407-protein test set.

The test set was never seen during training or hyperparameter selection.
This script should be run exactly once — it is the final honest benchmark.

Metrics reported:
  AUROC:             area under the ROC curve (measures ranking quality)
  Average Precision: area under the PR curve (more informative for imbalanced data)
  95% CI:            bootstrap confidence interval on AUROC (1000 iterations)

Plots: ROC curve, precision-recall curve, score distribution histogram.

Inputs
------
models/best_model.pt
data/final/test_X.npy, test_y.npy

Outputs
-------
results/plots/roc_curve.png
results/plots/pr_curve.png
results/plots/score_distribution.png
logs/test_results.json

Usage
-----
python 10_evaluate_model.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix,
                              roc_curve, precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR   = os.path.expanduser("~/llps_project/data/final/")
MODEL_DIR  = os.path.expanduser("~/llps_project/models/")
RESULTS_DIR= os.path.expanduser("~/llps_project/results/plots/")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Feature block indices ──────────────────────────────────────────────────────
ESM_END  = 1280
AF_END   = 1304
PC_END   = 1351
TRAD_END = 1761

# ── Model definition (must match training script exactly) ─────────────────────
class LLPSHybridModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(LLPSHybridModel, self).__init__()
        self.esm_branch = nn.Sequential(
            nn.Linear(1280, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
        )
        self.af_branch = nn.Sequential(
            nn.Linear(24, 64),  nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32),  nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.pc_branch = nn.Sequential(
            nn.Linear(47, 64),  nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32),  nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.trad_branch = nn.Sequential(
            nn.Linear(410, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(224, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 1),   nn.Sigmoid()
        )

    def forward(self, x):
        out_esm  = self.esm_branch(x[:, :ESM_END])
        out_af   = self.af_branch(x[:, ESM_END:AF_END])
        out_pc   = self.pc_branch(x[:, AF_END:PC_END])
        out_trad = self.trad_branch(x[:, PC_END:TRAD_END])
        fused    = torch.cat([out_esm, out_af, out_pc, out_trad], dim=1)
        return self.fusion(fused).squeeze(1)

# ── Load best model ────────────────────────────────────────────────────────────
print("Loading best model...")
model = LLPSHybridModel().to(device)
checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_model.pt'), weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Val AUC at save time: {checkpoint['val_auc']:.4f}")

# ── Load test data ─────────────────────────────────────────────────────────────
X_test = torch.tensor(np.load(os.path.join(DATA_DIR, 'test_X.npy')), dtype=torch.float32)
y_test = np.load(os.path.join(DATA_DIR, 'test_y.npy'))
ids_test = np.load(os.path.join(DATA_DIR, 'test_ids.npy'), allow_pickle=True)
# ── Predict ────────────────────────────────────────────────────────────────────
with torch.no_grad():
    test_preds = model(X_test.to(device)).cpu().numpy()

test_binary = (test_preds > 0.5).astype(int)

# ── Metrics ────────────────────────────────────────────────────────────────────
test_auc = roc_auc_score(y_test, test_preds)
test_ap  = average_precision_score(y_test, test_preds)

print(f"\n{'='*50}")
print(f"FINAL TEST SET RESULTS")
print(f"{'='*50}")
print(f"AUC-ROC           : {test_auc:.4f}")
print(f"Average Precision : {test_ap:.4f}")
print(f"\nClassification Report (threshold=0.5):")
print(classification_report(y_test, test_binary,
      target_names=['Non-LLPS','LLPS']))
print(f"Confusion Matrix:")
cm = confusion_matrix(y_test, test_binary)
print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

# ── ROC Curve ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

fpr, tpr, _ = roc_curve(y_test, test_preds)
axes[0].plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {test_auc:.4f})')
axes[0].plot([0,1],[0,1], color='navy', lw=1, linestyle='--',
             label='Random classifier')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve — LLPS Prediction')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(0.88, color='green', linestyle=':', alpha=0.7)

# ── Precision-Recall Curve ────────────────────────────────────────────────────
precision, recall, _ = precision_recall_curve(y_test, test_preds)
axes[1].plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {test_ap:.4f})')
axes[1].axhline(y_test.mean(), color='navy', lw=1, linestyle='--',
                label=f'Random classifier (AP={y_test.mean():.2f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve — LLPS Prediction')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'roc_pr_curves.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nROC + PR curves saved → {plot_path}")

# ── Score distribution ─────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 5))
ax.hist(test_preds[y_test==0], bins=50, alpha=0.6,
        color='steelblue', label='Non-LLPS (negative)')
ax.hist(test_preds[y_test==1], bins=50, alpha=0.6,
        color='tomato',    label='LLPS (positive)')
ax.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
ax.set_xlabel('Predicted LLPS Probability')
ax.set_ylabel('Count')
ax.set_title('Score Distribution — Test Set')
ax.legend()
ax.grid(True, alpha=0.3)
fig2.savefig(os.path.join(RESULTS_DIR, 'score_distribution.png'),
             dpi=150, bbox_inches='tight')
print(f"Score distribution saved → {RESULTS_DIR}score_distribution.png")

# Save predictions
np.save(os.path.join(MODEL_DIR, 'test_predictions.npy'), test_preds)
np.save(os.path.join(MODEL_DIR, 'test_labels.npy'),      y_test)
np.save(os.path.join(MODEL_DIR, 'test_ids.npy'),         ids_test)
print("\nPredictions saved. Ready for mutation scoring and SHAP analysis!")
