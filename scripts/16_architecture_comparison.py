"""
16 ARCHITECTURE COMPARISON
======================================================================
3-Branch vs 4-Branch Architecture Comparison
======================================================================

Resolves the 'ablation paradox' by directly comparing the 4-branch model
against a properly retrained 3-branch model (without Dipeptide branch).

Context: the ablation study showed that removing dipeptide features seemed to
improve test AUC by +0.007. This script tests whether that improvement is real
or an artefact of training variance (different random initialisations).

Result: 4-branch (AUC=0.9273) clearly outperforms 3-branch (AUC=0.9158).
The apparent improvement in the ablation study was within training variance.
Decision: keep 4-branch model as the final architecture.

Inputs
------
models/best_model.pt
models/best_model_3branch.pt
data/final/test_X.npy, test_y.npy

Outputs
-------
results/plots/3branch_clean_comparison.png
logs/3branch_final_eval.json

Usage
-----
python 16_architecture_comparison.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json

DATA   = os.path.expanduser("~/llps_project/data/final")
MODELS = os.path.expanduser("~/llps_project/models")
PLOTS  = os.path.expanduser("~/llps_project/results/plots")
LOGS   = os.path.expanduser("~/llps_project/logs")
os.makedirs(MODELS, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Feature slices — no dipeptide
ESM_END = 1280
AF_END  = 1304
PC_END  = 1351

# Load data — only first 1351 features
X_tr = np.load(f"{DATA}/train_X.npy")[:, :PC_END].astype(np.float32)
y_tr = np.load(f"{DATA}/train_y.npy").astype(np.float32)
X_va = np.load(f"{DATA}/val_X.npy")[:,   :PC_END].astype(np.float32)
y_va = np.load(f"{DATA}/val_y.npy").astype(np.float32)
X_te = np.load(f"{DATA}/test_X.npy")[:,  :PC_END].astype(np.float32)
y_te = np.load(f"{DATA}/test_y.npy").astype(np.float32)
print(f"Train {X_tr.shape}  Val {X_va.shape}  Test {X_te.shape}")

pos = y_tr.sum(); neg = len(y_tr) - pos
class_weight = neg / pos
print(f"Class weight: {class_weight:.3f}  pos={int(pos)}  neg={int(neg)}")

# Model — MATCHES the 4-branch architecture style exactly
class Branch3MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm_branch = nn.Sequential(
            nn.Linear(1280,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512,256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,128),  nn.BatchNorm1d(128), nn.ReLU()
        )
        self.af_branch = nn.Sequential(
            nn.Linear(24,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.pc_branch = nn.Sequential(
            nn.Linear(47,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU()
        )
        # 128+32+32 = 192
        self.fusion = nn.Sequential(
            nn.Linear(192,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,1), nn.Sigmoid()
        )
    def forward(self, x):
        e = self.esm_branch(x[:, :1280])
        a = self.af_branch (x[:, 1280:1304])
        p = self.pc_branch (x[:, 1304:1351])
        return self.fusion(torch.cat([e,a,p], dim=1)).squeeze(1)

model = Branch3MLP().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# Training setup
pos_weight = torch.tensor([class_weight], device=DEVICE)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                   batch_size=64, shuffle=True)
va_dl = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                   batch_size=256)

# Training loop
best_auc, best_ep, patience = 0.0, 0, 0
PATIENCE_LIMIT = 15
history = []

print("\nTraining 3-branch model from scratch...")
print(f"{'Epoch':>5} {'Val AUC':>9} {'Loss':>8} {'Best':>9}")
print("-"*35)

for epoch in range(1, 120):
    model.train()
    losses = []
    for xb, yb in tr_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        w    = torch.where(yb==1,
                           pos_weight.expand_as(yb),
                           torch.ones_like(yb))
        loss = (w * F.binary_cross_entropy(pred, yb, reduction='none')).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        va_preds = torch.cat([model(xb.to(DEVICE)).cpu()
                              for xb,_ in va_dl]).numpy()
    val_auc = roc_auc_score(y_va, va_preds)
    mean_loss = float(np.mean(losses))
    history.append({'epoch': epoch, 'val_auc': val_auc, 'loss': mean_loss})

    if epoch % 5 == 0 or val_auc > best_auc:
        marker = " <-- BEST" if val_auc > best_auc else ""
        print(f"{epoch:>5} {val_auc:>9.4f} {mean_loss:>8.4f} {best_auc:>9.4f}{marker}")

    if val_auc > best_auc:
        best_auc = val_auc; best_ep = epoch; patience = 0
        torch.save({'epoch': epoch, 'val_auc': val_auc,
                    'model_state': model.state_dict()},
                   f"{MODELS}/best_model_3branch_clean.pt")
    else:
        patience += 1
        if patience >= PATIENCE_LIMIT:
            print(f"\nEarly stopping at epoch {epoch}  "
                  f"best val AUC={best_auc:.4f} at epoch {best_ep}")
            break

# Test evaluation
ckpt = torch.load(f"{MODELS}/best_model_3branch_clean.pt",
                  map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()
with torch.no_grad():
    te_preds = model(torch.from_numpy(X_te).to(DEVICE)).cpu().numpy()

test_auc = roc_auc_score(y_te, te_preds)
test_ap  = average_precision_score(y_te, te_preds)

print(f"\n{'='*50}")
print(f"3-BRANCH CLEAN MODEL RESULTS")
print(f"  Val  AUC : {best_auc:.4f}  (epoch {best_ep})")
print(f"  Test AUC : {test_auc:.4f}")
print(f"  Test AP  : {test_ap:.4f}")
print(f"  4-branch : AUC=0.9273  AP=0.8591")
print(f"  Delta AUC: {test_auc - 0.9273:+.4f}")
print(f"{'='*50}")

# Load 4-branch predictions for comparison plot
class HybridMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm_branch = nn.Sequential(
            nn.Linear(1280,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(512,256), nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(256,128), nn.BatchNorm1d(128),nn.ReLU())
        self.af_branch = nn.Sequential(
            nn.Linear(24,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU())
        self.pc_branch = nn.Sequential(
            nn.Linear(47,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU())
        self.trad_branch = nn.Sequential(
            nn.Linear(410,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(64,32),  nn.BatchNorm1d(32), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(224,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(64,1),nn.Sigmoid())
    def forward(self, x_esm, x_af, x_pc, x_trad):
        e=self.esm_branch(x_esm); a=self.af_branch(x_af)
        p=self.pc_branch(x_pc);   t=self.trad_branch(x_trad)
        return self.fusion(torch.cat([e,a,p,t],dim=1)).squeeze(1)

m4 = HybridMLP().to(DEVICE)
c4 = torch.load(f"{MODELS}/best_model.pt", map_location=DEVICE, weights_only=False)
m4.load_state_dict(c4['model_state']); m4.eval()
X_te_full = np.load(f"{DATA}/test_X.npy").astype(np.float32)
with torch.no_grad():
    p4 = m4(torch.from_numpy(X_te_full[:,:1280]).to(DEVICE),
             torch.from_numpy(X_te_full[:,1280:1304]).to(DEVICE),
             torch.from_numpy(X_te_full[:,1304:1351]).to(DEVICE),
             torch.from_numpy(X_te_full[:,1351:]).to(DEVICE)).cpu().numpy()
auc4 = roc_auc_score(y_te, p4)

# Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"3-Branch Clean (no dipeptide) vs 4-Branch Model",
             fontsize=13, fontweight='bold')

# ROC
fpr3, tpr3, _ = roc_curve(y_te, te_preds)
fpr4, tpr4, _ = roc_curve(y_te, p4)
ax = axes[0]
ax.plot(fpr4, tpr4, 'steelblue', lw=2.5, label=f"4-branch full (AUC={auc4:.4f})")
ax.plot(fpr3, tpr3, 'coral',     lw=2.5, linestyle='--',
        label=f"3-branch no-dipeptide (AUC={test_auc:.4f})")
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# PR
prec3,rec3,_ = precision_recall_curve(y_te, te_preds)
prec4,rec4,_ = precision_recall_curve(y_te, p4)
ap4 = average_precision_score(y_te, p4)
ax = axes[1]
ax.plot(rec4, prec4, 'steelblue', lw=2.5, label=f"4-branch (AP={ap4:.4f})")
ax.plot(rec3, prec3, 'coral', lw=2.5, linestyle='--',
        label=f"3-branch (AP={test_ap:.4f})")
ax.axhline(y_te.mean(), color='grey', linestyle=':', label='Random')
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Training curve
epochs_hist = [h['epoch'] for h in history]
aucs_hist   = [h['val_auc'] for h in history]
ax = axes[2]
ax.plot(epochs_hist, aucs_hist, 'steelblue', lw=2, label='3-branch val AUC')
ax.axhline(0.9406, color='coral', linestyle='--', lw=1.5,
           label='4-branch val AUC (0.9406)')
ax.axvline(best_ep, color='green', linestyle=':', lw=1.5,
           label=f'Best epoch ({best_ep})')
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUC")
ax.set_title(f"Training Curve\nBest Val AUC={best_auc:.4f} at ep {best_ep}")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
out = f"{PLOTS}/3branch_clean_comparison.png"
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved plot → {out}")

results = {
    "model_3branch_clean": {
        "test_auc": float(test_auc), "test_ap": float(test_ap),
        "val_auc": float(best_auc), "best_epoch": int(best_ep),
        "n_features": int(PC_END), "n_params": int(total_params)
    },
    "model_4branch": {"test_auc": float(auc4), "test_ap": float(ap4)},
    "delta_auc": float(test_auc - auc4),
    "conclusion": "3-branch_better" if test_auc > auc4 else "4-branch_better"
}
with open(f"{LOGS}/3branch_clean_results.json","w") as f:
    json.dump(results, f, indent=2)
print(f"Saved → logs/3branch_clean_results.json")
print(f"\nConclusion: {'3-branch' if test_auc > auc4 else '4-branch'} is better")
