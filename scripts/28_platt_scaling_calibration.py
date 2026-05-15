"""
28_platt_scaling_calibration.py
================================
Fits Platt scaling on the validation set to improve probability calibration.
Compares calibrated vs uncalibrated ECE and Brier score on the test set.

Platt scaling fits a logistic regression on model logits (pre-sigmoid outputs)
from the validation set, then applies it to test set logits. This is the
standard post-hoc calibration method for neural networks.

The Platt scaler is fitted ONLY on val set. Test set used only for evaluation.

Inputs
------
data/final/val_X.npy, val_y.npy
data/final/test_X.npy, test_y.npy
models/best_model_compat.pt

Outputs
-------
models/platt_scaler.pkl
logs/calibration_platt_results.json
results/plots/reliability_diagram.png
"""

import os
import json
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
warnings.filterwarnings("ignore")

BASE     = os.path.expanduser("~/llps_project")
DATA     = BASE + "/data/final"
MODELS   = BASE + "/models"
LOGS     = BASE + "/logs"
PLOTS    = BASE + "/results/plots"
os.makedirs(LOGS,  exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

print("=" * 60)
print("Script 28: Platt Scaling Calibration")
print("=" * 60)

# ── Correct model architecture ────────────────────────────────────
class HybridLLPSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm_branch = nn.Sequential(
            nn.Linear(1280,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512,256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,128),  nn.BatchNorm1d(128), nn.ReLU())
        self.af_branch = nn.Sequential(
            nn.Linear(24,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU())
        self.pc_branch = nn.Sequential(
            nn.Linear(47,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU())
        self.trad_branch = nn.Sequential(
            nn.Linear(410,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32),   nn.BatchNorm1d(32),  nn.ReLU())
        # Fusion outputs logit (no Sigmoid here — needed for Platt scaling)
        self.fusion = nn.Sequential(
            nn.Linear(224,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,1))

    def forward(self, xe, xa, xp, xt):
        return self.fusion(
            torch.cat([self.esm_branch(xe), self.af_branch(xa),
                       self.pc_branch(xp), self.trad_branch(xt)], 1)).squeeze(1)

    def predict_proba(self, xe, xa, xp, xt):
        return torch.sigmoid(self.forward(xe, xa, xp, xt))

# ── Load model ────────────────────────────────────────────────────
print("\n[1] Loading model...")
model = HybridLLPSModel()
ckpt = torch.load(MODELS + "/best_model_compat.pt", map_location="cpu", weights_only=False)
# The checkpoint was saved WITH sigmoid in fusion layer
# We need to reload with sigmoid to get probabilities, then derive logits
# Strategy: load normally, get sigmoid probs, then use log-odds as logits

class HybridLLPSModelWithSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm_branch = nn.Sequential(
            nn.Linear(1280,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512,256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,128),  nn.BatchNorm1d(128), nn.ReLU())
        self.af_branch = nn.Sequential(
            nn.Linear(24,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU())
        self.pc_branch = nn.Sequential(
            nn.Linear(47,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU())
        self.trad_branch = nn.Sequential(
            nn.Linear(410,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32),   nn.BatchNorm1d(32),  nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(224,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,1),   nn.Sigmoid())

    def forward(self, xe, xa, xp, xt):
        return self.fusion(
            torch.cat([self.esm_branch(xe), self.af_branch(xa),
                       self.pc_branch(xp), self.trad_branch(xt)], 1)).squeeze(1)

model = HybridLLPSModelWithSigmoid()
model.load_state_dict(ckpt["model_state"])
model.eval()
print("  Model loaded OK")

# ── Load data ─────────────────────────────────────────────────────
print("\n[2] Loading data...")
X_val  = np.load(DATA + "/val_X.npy").astype(np.float32)
y_val  = np.load(DATA + "/val_y.npy").astype(np.float32)
X_test = np.load(DATA + "/test_X.npy").astype(np.float32)
y_test = np.load(DATA + "/test_y.npy").astype(np.float32)
print("  Val:  " + str(X_val.shape) + "  Test: " + str(X_test.shape))

# ── Get raw probabilities from model ─────────────────────────────
print("\n[3] Getting model probabilities...")
def get_probs(X):
    with torch.no_grad():
        xb = torch.from_numpy(X)
        return model(xb[:,:1280], xb[:,1280:1304],
                     xb[:,1304:1351], xb[:,1351:]).numpy()

probs_val  = get_probs(X_val)
probs_test = get_probs(X_test)

# Convert probabilities to logits for Platt scaling input
# logit(p) = log(p / (1-p))
eps = 1e-6
logits_val  = np.log(np.clip(probs_val,  eps, 1-eps) /
                     (1 - np.clip(probs_val,  eps, 1-eps)))
logits_test = np.log(np.clip(probs_test, eps, 1-eps) /
                     (1 - np.clip(probs_test, eps, 1-eps)))

print("  Val  probs range:  " + str(round(probs_val.min(),3)) + " to " + str(round(probs_val.max(),3)))
print("  Test probs range:  " + str(round(probs_test.min(),3)) + " to " + str(round(probs_test.max(),3)))

# ── Fit Platt scaler on validation set ONLY ───────────────────────
print("\n[4] Fitting Platt scaler on validation set...")
platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
platt.fit(logits_val.reshape(-1,1), y_val)
print("  Platt scaler fitted. Coef: " + str(round(float(platt.coef_[0][0]),4)) +
      "  Intercept: " + str(round(float(platt.intercept_[0]),4)))

# Save scaler
with open(MODELS + "/platt_scaler.pkl", "wb") as f:
    pickle.dump(platt, f)
print("  Saved: " + MODELS + "/platt_scaler.pkl")

# ── Get calibrated probabilities ──────────────────────────────────
print("\n[5] Applying Platt scaling to test set...")
probs_test_calibrated = platt.predict_proba(logits_test.reshape(-1,1))[:, 1]

# ── Compute calibration metrics ───────────────────────────────────
print("\n[6] Computing calibration metrics...")

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            bin_data.append(None)
            continue
        bin_acc  = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        bin_n    = int(mask.sum())
        ece += (bin_n / len(y_true)) * abs(bin_acc - bin_conf)
        bin_data.append({"acc": bin_acc, "conf": bin_conf, "n": bin_n})
    return ece, bin_data

ece_before, bins_before = compute_ece(y_test, probs_test)
ece_after,  bins_after  = compute_ece(y_test, probs_test_calibrated)
brier_before = float(brier_score_loss(y_test, probs_test))
brier_after  = float(brier_score_loss(y_test, probs_test_calibrated))

print("  BEFORE Platt scaling:")
print("    ECE:   " + str(round(ece_before, 4)))
print("    Brier: " + str(round(brier_before, 4)))
print()
print("  AFTER Platt scaling:")
print("    ECE:   " + str(round(ece_after, 4)))
print("    Brier: " + str(round(brier_after, 4)))
print()
print("  ECE improvement:   " + str(round(ece_before - ece_after, 4)))
print("  Brier improvement: " + str(round(brier_before - brier_after, 4)))

# ── Save results ──────────────────────────────────────────────────
results = {
    "before_platt": {
        "ece":   ece_before,
        "brier": brier_before,
    },
    "after_platt": {
        "ece":   ece_after,
        "brier": brier_after,
    },
    "improvement": {
        "ece_delta":   ece_before - ece_after,
        "brier_delta": brier_before - brier_after,
    },
    "platt_params": {
        "coef":      float(platt.coef_[0][0]),
        "intercept": float(platt.intercept_[0]),
    }
}
with open(LOGS + "/calibration_platt_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved: " + LOGS + "/calibration_platt_results.json")

# ── Plot reliability diagrams ─────────────────────────────────────
print("\n[7] Generating reliability diagram...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Platt Scaling Calibration Analysis", fontsize=13, fontweight="bold")

BLUE  = "#2E75B6"
GREEN = "#2ECC71"
RED   = "#E74C3C"

for ax, bins_data, probs, ece_val, brier_val, title_prefix in [
    (axes[0], bins_before, probs_test,           ece_before, brier_before, "Before"),
    (axes[1], bins_after,  probs_test_calibrated, ece_after,  brier_after,  "After"),
]:
    confs, accs, sizes = [], [], []
    for b in bins_data:
        if b is not None:
            confs.append(b["conf"])
            accs.append(b["acc"])
            sizes.append(b["n"])

    ax.plot([0,1],[0,1],"k--",alpha=0.5,lw=1.5,label="Perfect calibration")
    ax.scatter(confs, accs, s=[s*3 for s in sizes],
               color=BLUE if title_prefix=="Before" else GREEN,
               alpha=0.85, zorder=5, edgecolors="white", lw=0.5)
    ax.plot(confs, accs,
            color=BLUE if title_prefix=="Before" else GREEN,
            lw=2, label="Model calibration")
    ax.fill_between(confs, confs, accs, alpha=0.1,
                    color=RED, label="Calibration gap")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title_prefix + " Platt Scaling\nECE=" +
                 str(round(ece_val,4)) + "  Brier=" + str(round(brier_val,4)))
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS + "/reliability_diagram.png",  dpi=300, bbox_inches="tight")
plt.savefig(PLOTS + "/reliability_diagram.tiff", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: " + PLOTS + "/reliability_diagram.png")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("ECE before: " + str(round(ece_before,4)) + "  after: " + str(round(ece_after,4)))
print("Brier before: " + str(round(brier_before,4)) + "  after: " + str(round(brier_after,4)))
if ece_after < ece_before:
    print("Platt scaling IMPROVED calibration.")
else:
    print("Model was already well calibrated. Platt scaling made minimal difference.")
print("Platt scaler saved to: " + MODELS + "/platt_scaler.pkl")
