"""
19 CALIBRATION ANALYSIS
======================================================================
Probability Calibration — ECE and Brier Score
======================================================================

Tests whether model output probabilities are reliable (well-calibrated).

A calibrated model predicting p=0.8 should be correct 80% of the time.
If it is only correct 60% of the time, the model is overconfident.

Metrics:
  Expected Calibration Error (ECE): weighted average absolute calibration error
    ECE = Σ (bin_n / n) × |mean_confidence_in_bin - fraction_positive_in_bin|
    Our ECE = 0.0605 (threshold for 'well-calibrated': < 0.05)
  Brier Score: mean squared error of probability predictions
    Our Brier = 0.0892 (random = 0.25; perfect = 0.0)

A U-shaped score distribution indicates the model makes confident predictions.

Inputs
------
models/best_model.pt
data/final/test_X.npy, test_y.npy

Outputs
-------
results/plots/calibration_analysis.png
logs/calibration_results.json

Usage
-----
python 19_calibration_analysis.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# --- ARCHITECTURE (4-BRANCH GOLD STANDARD) ---
class HybridLLPSModel(nn.Module):
    def __init__(self):
        super(HybridLLPSModel, self).__init__()
        self.esm_branch = nn.Sequential(
            nn.Linear(1280, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU()
        )
        self.af_branch = nn.Sequential(
            nn.Linear(24, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.pc_branch = nn.Sequential(
            nn.Linear(47, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.trad_branch = nn.Sequential(
            nn.Linear(410, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(128+32+32+32, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x_esm, x_af, x_pc, x_trad):
        e, a, p, t = self.esm_branch(x_esm), self.af_branch(x_af), self.pc_branch(x_pc), self.trad_branch(x_trad)
        return self.fusion(torch.cat([e, a, p, t], dim=1)).squeeze(1)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_idx = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        if np.any(bin_idx):
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            ece += np.abs(bin_acc - bin_conf) * (np.sum(bin_idx) / len(y_true))
    return ece

# --- LOAD DATA & MODEL ---
print("🚀 Starting Calibration Analysis...")
X_test = np.load('data/final/test_X.npy').astype(np.float32)
y_test = np.load('data/final/test_y.npy').astype(np.float32)

model = HybridLLPSModel()
ckpt = torch.load('models/best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

with torch.no_grad():
    x_esm, x_af, x_pc, x_trad = (torch.from_numpy(X_test[:, :1280]), torch.from_numpy(X_test[:, 1280:1304]), 
                                 torch.from_numpy(X_test[:, 1304:1351]), torch.from_numpy(X_test[:, 1351:]))
    y_prob = model(x_esm, x_af, x_pc, x_trad).numpy()

# --- CALCULATE METRICS ---
ece = expected_calibration_error(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

print(f"✅ Calibration Metrics:\nECE: {ece:.4f}\nBrier Score: {brier:.4f}")

# --- SAVING & PLOTTING ---
os.makedirs('logs', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

with open('logs/calibration_results.json', 'w') as f:
    json.dump({"ece": float(ece), "brier_score": float(brier)}, f, indent=4)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Reliability Diagram
axes[0].plot(prob_pred, prob_true, marker='s', label=f'Our Model (ECE={ece:.4f})', color='darkblue')
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
axes[0].set_xlabel('Mean Predicted Probability'); axes[0].set_ylabel('Fraction of Positives')
axes[0].set_title('Reliability Diagram'); axes[0].legend()

# Panel 2: Score Histograms
axes[1].hist(y_prob[y_test == 0], bins=20, alpha=0.5, label='LLPS-', color='steelblue', density=True)
axes[1].hist(y_prob[y_test == 1], bins=20, alpha=0.5, label='LLPS+', color='salmon', density=True)
axes[1].set_xlabel('Predicted Probability'); axes[1].set_ylabel('Density')
axes[1].set_title('Score Distribution by Class'); axes[1].legend()

plt.tight_layout()
plt.savefig('results/plots/calibration_analysis.png', dpi=300)
print("Saved plot -> results/plots/calibration_analysis.png")
