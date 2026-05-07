"""
13 ABLATION STUDY
======================================================================
Ablation Study — Quantifying Each Feature Branch's Contribution
======================================================================

Measures each feature branch's contribution by retraining with one removed.

Four additional models trained from scratch:
  No_ESM2             (481 features):  largest drop, ΔAUC = -0.060  [HIGH IMPACT]
  No_Physicochemical  (1714 features): moderate drop, ΔAUC = -0.021 [MODERATE]
  No_AlphaFold        (1737 features): small drop, ΔAUC = -0.009    [LOW]
  No_Dipeptide        (1351 features): no drop, ΔAUC = +0.007       [NONE]

Each ablated model uses identical training settings to the full model.
Results confirm that ESM-2 is the dominant branch — removing it causes
the largest performance decrease by a wide margin.

Inputs
------
data/final/train_X.npy, val_X.npy, test_X.npy (and y files)

Outputs
-------
results/plots/ablation_study.png
results/ablation/ablation_results.csv
logs/ablation_study.json

Usage
-----
python 13_ablation_study.py
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
import os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR    = os.path.expanduser("~/llps_project/data/final/")
RESULTS_DIR = os.path.expanduser("~/llps_project/results/ablation/")
PLOT_DIR    = os.path.expanduser("~/llps_project/results/plots/")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading data...")
X_train = np.load(os.path.join(DATA_DIR, "train_X.npy"))
y_train = np.load(os.path.join(DATA_DIR, "train_y.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "val_X.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "val_y.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "test_X.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "test_y.npy"))
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

ESM_START, ESM_END   = 0,    1280
AF_START,  AF_END    = 1280, 1304
PC_START,  PC_END    = 1304, 1351
TRAD_START,TRAD_END  = 1351, 1761

ablation_configs = [
    {
        "name"       : "No_ESM2",
        "description": "AlphaFold + Physicochemical + Dipeptide only",
        "keep_ranges": [(AF_START, AF_END), (PC_START, PC_END), (TRAD_START, TRAD_END)],
        "n_features" : 481,
        "esm_dim"    : 0, "af_dim": 24, "pc_dim": 47, "trad_dim": 410,
    },
    {
        "name"       : "No_AlphaFold",
        "description": "ESM-2 + Physicochemical + Dipeptide only",
        "keep_ranges": [(ESM_START, ESM_END), (PC_START, PC_END), (TRAD_START, TRAD_END)],
        "n_features" : 1737,
        "esm_dim"    : 1280, "af_dim": 0, "pc_dim": 47, "trad_dim": 410,
    },
    {
        "name"       : "No_Physicochemical",
        "description": "ESM-2 + AlphaFold + Dipeptide only",
        "keep_ranges": [(ESM_START, ESM_END), (AF_START, AF_END), (TRAD_START, TRAD_END)],
        "n_features" : 1714,
        "esm_dim"    : 1280, "af_dim": 24, "pc_dim": 0, "trad_dim": 410,
    },
    {
        "name"       : "No_Dipeptide",
        "description": "ESM-2 + AlphaFold + Physicochemical only",
        "keep_ranges": [(ESM_START, ESM_END), (AF_START, AF_END), (PC_START, PC_END)],
        "n_features" : 1351,
        "esm_dim"    : 1280, "af_dim": 24, "pc_dim": 47, "trad_dim": 0,
    },
]

def get_ablated_features(X, keep_ranges):
    parts = [X[:, start:end] for start, end in keep_ranges]
    return np.concatenate(parts, axis=1)

class AblationModel(nn.Module):
    def __init__(self, esm_dim, af_dim, pc_dim, trad_dim, dr=0.3):
        super().__init__()
        self.esm_dim  = esm_dim
        self.af_dim   = af_dim
        self.pc_dim   = pc_dim
        self.trad_dim = trad_dim
        fusion_in = 0
        if esm_dim > 0:
            self.esm_branch = nn.Sequential(
                nn.Linear(esm_dim,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(512,256),     nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(256,128),     nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dr),
            )
            fusion_in += 128
        if af_dim > 0:
            self.af_branch = nn.Sequential(
                nn.Linear(af_dim,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(64,32),     nn.BatchNorm1d(32), nn.ReLU(),
            )
            fusion_in += 32
        if pc_dim > 0:
            self.pc_branch = nn.Sequential(
                nn.Linear(pc_dim,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(64,32),     nn.BatchNorm1d(32), nn.ReLU(),
            )
            fusion_in += 32
        if trad_dim > 0:
            self.trad_branch = nn.Sequential(
                nn.Linear(trad_dim,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(128,64),       nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dr),
                nn.Linear(64,32),        nn.BatchNorm1d(32),  nn.ReLU(),
            )
            fusion_in += 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(64,1), nn.Sigmoid()
        )
    def forward(self, x):
        parts = []
        idx = 0
        if self.esm_dim > 0:
            parts.append(self.esm_branch(x[:, idx:idx+self.esm_dim]))
            idx += self.esm_dim
        if self.af_dim > 0:
            parts.append(self.af_branch(x[:, idx:idx+self.af_dim]))
            idx += self.af_dim
        if self.pc_dim > 0:
            parts.append(self.pc_branch(x[:, idx:idx+self.pc_dim]))
            idx += self.pc_dim
        if self.trad_dim > 0:
            parts.append(self.trad_branch(x[:, idx:idx+self.trad_dim]))
        out = torch.cat(parts, dim=1)
        return self.fusion(out).squeeze(1)

def train_model(config, X_tr, y_tr, X_v, y_v, X_te, y_te):
    print(f"\nTraining: {config['name']} ({config['n_features']} features)")
    X_tr_abl = get_ablated_features(X_tr, config["keep_ranges"])
    X_v_abl  = get_ablated_features(X_v,  config["keep_ranges"])
    X_te_abl = get_ablated_features(X_te, config["keep_ranges"])

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    train_ds     = TensorDataset(
        torch.tensor(X_tr_abl, dtype=torch.float32),
        torch.tensor(y_tr,     dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = AblationModel(
        esm_dim=config["esm_dim"], af_dim=config["af_dim"],
        pc_dim=config["pc_dim"],   trad_dim=config["trad_dim"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=False)
    criterion = nn.BCELoss()

    best_val_auc   = 0.0
    best_state     = None
    patience_count = 0
    patience       = 10

    for epoch in range(50):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds   = model(X_batch)
            weights = torch.where(y_batch==1, pos_weight, torch.ones_like(y_batch))
            loss    = (criterion(preds, y_batch) * weights).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(
                torch.tensor(X_v_abl, dtype=torch.float32).to(device)
            ).cpu().numpy()
        val_auc = roc_auc_score(y_v, val_preds)
        scheduler.step(1 - val_auc)

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: val_AUC={val_auc:.4f} (best={best_val_auc:.4f})")

        if patience_count >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(
            torch.tensor(X_te_abl, dtype=torch.float32).to(device)
        ).cpu().numpy()

    test_auc = roc_auc_score(y_te, test_preds)
    test_ap  = average_precision_score(y_te, test_preds)
    print(f"  RESULT: Val AUC={best_val_auc:.4f} | Test AUC={test_auc:.4f} | AP={test_ap:.4f}")
    return best_val_auc, test_auc, test_ap

results = [{
    "model"      : "Full Model",
    "description": "ESM-2 + AlphaFold + Physicochemical + Dipeptide",
    "n_features" : 1761,
    "val_auc"    : 0.9406,
    "test_auc"   : 0.9273,
    "test_ap"    : 0.8591,
    "auc_drop"   : 0.0
}]

for config in ablation_configs:
    val_auc, test_auc, test_ap = train_model(
        config, X_train, y_train, X_val, y_val, X_test, y_test)
    results.append({
        "model"      : config["name"],
        "description": config["description"],
        "n_features" : config["n_features"],
        "val_auc"    : val_auc,
        "test_auc"   : test_auc,
        "test_ap"    : test_ap,
        "auc_drop"   : 0.9273 - test_auc
    })

df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)

print("\n" + "="*60)
print("ABLATION STUDY RESULTS")
print("="*60)
print(f"{'Model':<22} {'Features':>10} {'Val AUC':>9} {'Test AUC':>10} {'AUC Drop':>10}")
print("-"*65)
for _, row in df.iterrows():
    print(f"{row['model']:<22} {int(row['n_features']):>10} "
          f"{row['val_auc']:>9.4f} {row['test_auc']:>10.4f} "
          f"{row['auc_drop']:>10.4f}")

print("\n=== INTERPRETATION ===")
for _, row in df[df["model"] != "Full Model"].iterrows():
    drop = row["auc_drop"]
    if drop > 0.05:
        impact = "HIGH IMPACT"
    elif drop > 0.02:
        impact = "MODERATE IMPACT"
    elif drop > 0.0:
        impact = "LOW IMPACT"
    else:
        impact = "NO IMPACT"
    print(f"Removing {row['model'].replace('No_',''):<22}: AUC drop={drop:+.4f} -> {impact}")

fig, ax = plt.subplots(figsize=(10, 6))
models  = df["model"].tolist()
aucs    = df["test_auc"].tolist()
colors  = ["#2196F3" if m == "Full Model" else "#FF7043" for m in models]
bars    = ax.bar(models, aucs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
ax.axhline(0.9273, color="blue", linestyle="--", linewidth=1.5, label="Full model AUC=0.9273")
ax.set_ylim(0.7, 1.0)
ax.set_ylabel("Test AUC-ROC", fontsize=12)
ax.set_title("Ablation Study: Contribution of Each Feature Block\nBlue=Full model, Red=Ablated", fontsize=12)
ax.tick_params(axis="x", rotation=15)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, auc + 0.003,
            f"{auc:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = os.path.join(PLOT_DIR, "ablation_study.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {out}")
print("="*60)
