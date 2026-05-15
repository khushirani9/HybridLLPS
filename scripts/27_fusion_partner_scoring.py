"""
27_fusion_partner_scoring.py
=============================
Compares fusion protein LLPS scores against their individual partner scores.

Strategy:
  - Fusion protein scores: loaded from logs/fusion_results.json
  - Partner scores already in dataset: loaded from pre-computed feature
    matrices (train_X/val_X) and scored with the trained model
  - Partners not in dataset: noted as unavailable

This is scientifically valid because the partner proteins in our dataset
were scored using the same pipeline as the fusion proteins.

Partners found in our dataset:
  EWSR1 (Q01844) - TRAIN  -> FET-family LLPS driver
  FUS   (P35637) - TRAIN  -> FET-family LLPS driver
  TAF15 (Q92804) - VAL    -> FET-family LLPS driver
  ABL1  (P00519) - TRAIN  -> Signalling kinase (LLPS-)

Inputs
------
logs/fusion_results.json
models/best_model_compat.pt
data/final/train_X.npy, val_X.npy, train_y.npy, val_y.npy
data/splits/train.csv, val.csv

Outputs
-------
logs/fusion_partner_results.json
results/plots/fusion_partner_comparison.png
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")

BASE   = os.path.expanduser("~/llps_project")
DATA   = BASE + "/data/final"
SPLITS = BASE + "/data/splits"
MODELS = BASE + "/models"
LOGS   = BASE + "/logs"
PLOTS  = BASE + "/results/plots"
os.makedirs(LOGS,  exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

print("=" * 60)
print("Script 27: Fusion Partner Scoring")
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
        self.fusion = nn.Sequential(
            nn.Linear(224,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,1),   nn.Sigmoid())

    def forward(self, xe, xa, xp, xt):
        return self.fusion(
            torch.cat([self.esm_branch(xe), self.af_branch(xa),
                       self.pc_branch(xp), self.trad_branch(xt)], 1)).squeeze(1)

# ── Load model ────────────────────────────────────────────────────
print("\n[1] Loading model...")
model = HybridLLPSModel()
ckpt  = torch.load(MODELS + "/best_model_compat.pt",
                   map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("  Model loaded OK")

# ── Load all split data and metadata ─────────────────────────────
print("\n[2] Loading dataset features and metadata...")
X_train = np.load(DATA + "/train_X.npy").astype(np.float32)
X_val   = np.load(DATA + "/val_X.npy").astype(np.float32)

train_csv = pd.read_csv(SPLITS + "/train.csv")
val_csv   = pd.read_csv(SPLITS + "/val.csv")

print("  Train: " + str(X_train.shape) + "  Val: " + str(X_val.shape))

# ── Score all training and validation proteins ────────────────────
print("\n[3] Scoring all train+val proteins with model...")

def score_matrix(X):
    with torch.no_grad():
        xb = torch.from_numpy(X)
        return model(xb[:,:1280], xb[:,1280:1304],
                     xb[:,1304:1351], xb[:,1351:]).numpy()

train_scores = score_matrix(X_train)
val_scores   = score_matrix(X_val)

# Build lookup: uniprot_id -> score
score_lookup = {}
for i, row in train_csv.iterrows():
    score_lookup[row["uniprot_id"]] = float(train_scores[i])
for i, row in val_csv.iterrows():
    score_lookup[row["uniprot_id"]] = float(val_scores[i])

print("  Scored " + str(len(score_lookup)) + " proteins total")

# ── Load existing fusion results ──────────────────────────────────
print("\n[4] Loading existing fusion protein scores...")
with open(LOGS + "/fusion_results.json") as f:
    fusion_data = json.load(f)

print("  Fusion JSON keys: " + str(list(fusion_data.keys())))

# Extract individual fusion scores
individual = fusion_data.get("individual", [])
print("  Individual fusion proteins scored: " + str(len(individual)))
for item in individual[:3]:
    print("    " + item["name"] + ": " + str(round(item["llps_score"], 4)))
# Convert list to dict keyed by name for easy lookup
individual = {item["name"]: item["llps_score"] for item in individual}

# ── Define partner lookup ─────────────────────────────────────────
# For each fusion, define partners and whether they are in our dataset
FUSION_PARTNERS = {
    "EWS-FLI1": {
        "category":    "FET-family",
        "partner_A":   "Q01844",
        "partner_A_name": "EWSR1",
        "partner_B":   "Q01167",
        "partner_B_name": "FLI1",
    },
    "FUS-CHOP": {
        "category":    "FET-family",
        "partner_A":   "P35637",
        "partner_A_name": "FUS",
        "partner_B":   "P35638",
        "partner_B_name": "DDIT3",
    },
    "TAF15-CIC": {
        "category":    "FET-family",
        "partner_A":   "Q92804",
        "partner_A_name": "TAF15",
        "partner_B":   "O15198",
        "partner_B_name": "CIC",
    },
    "BCR-ABL1": {
        "category":    "Signalling",
        "partner_A":   "P11274",
        "partner_A_name": "BCR",
        "partner_B":   "P00519",
        "partner_B_name": "ABL1",
    },
    "EML4-ALK": {
        "category":    "Signalling",
        "partner_A":   "Q9HC35",
        "partner_A_name": "EML4",
        "partner_B":   "Q9UM73",
        "partner_B_name": "ALK",
    },
}

# ── Get partner scores and fusion scores ──────────────────────────
print("\n[5] Looking up partner and fusion scores...")
results = {}

for fusion_name, info in FUSION_PARTNERS.items():
    cat  = info["category"]
    pa_id   = info["partner_A"]
    pb_id   = info["partner_B"]
    pa_name = info["partner_A_name"]
    pb_name = info["partner_B_name"]

    pa_score = score_lookup.get(pa_id, None)
    pb_score = score_lookup.get(pb_id, None)

    # Get fusion score from individual results
    fusion_score = None
    for key in [fusion_name, fusion_name.replace("-","_"),
                fusion_name.lower(), fusion_name.replace("-"," ")]:
        if key in individual:
            v = individual[key]
            fusion_score = float(v)
            break

    results[fusion_name] = {
        "category":       cat,
        "partner_A_id":   pa_id,
        "partner_A_name": pa_name,
        "partner_A_score": pa_score,
        "partner_A_in_dataset": pa_id in score_lookup,
        "partner_B_id":   pb_id,
        "partner_B_name": pb_name,
        "partner_B_score": pb_score,
        "partner_B_in_dataset": pb_id in score_lookup,
        "fusion_score":   fusion_score,
    }

    pa_str = str(round(pa_score,4)) if pa_score is not None else "not in dataset"
    pb_str = str(round(pb_score,4)) if pb_score is not None else "not in dataset"
    fs_str = str(round(fusion_score,4)) if fusion_score is not None else "not found"
    print("  " + fusion_name + " (" + cat + ")")
    print("    " + pa_name + ": " + pa_str)
    print("    " + pb_name + ": " + pb_str)
    print("    Fusion: " + fs_str)

# ── Save results ──────────────────────────────────────────────────
print("\n[6] Saving results...")
with open(LOGS + "/fusion_partner_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved: " + LOGS + "/fusion_partner_results.json")

# ── Print summary table ───────────────────────────────────────────
print()
print("=" * 70)
print("FUSION PARTNER SCORES SUMMARY")
print("=" * 70)
print()
print("Fusion         Category     Partner A (IDR)   Partner B (DBD)   Fusion")
print("-" * 70)
for fname, res in results.items():
    pa = res["partner_A_score"]
    pb = res["partner_B_score"]
    fs = res["fusion_score"]
    pa_str = str(round(pa,4)) if pa is not None else "  N/A  "
    pb_str = str(round(pb,4)) if pb is not None else "  N/A  "
    fs_str = str(round(fs,4)) if fs is not None else "  N/A  "
    print(fname.ljust(15) + res["category"].ljust(14) +
          pa_str.rjust(14) + pb_str.rjust(16) + fs_str.rjust(9))

# ── Plot ──────────────────────────────────────────────────────────
print("\n[7] Generating figure...")

# Only plot entries where we have at least fusion score
plot_data = {k: v for k, v in results.items()
             if v["fusion_score"] is not None}

if len(plot_data) == 0:
    print("  No fusion scores available to plot.")
    print("  Check fusion_results.json individual scores.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fusion Oncoproteins vs Individual Partner LLPS Scores",
                 fontsize=13, fontweight="bold")

    BLUE   = "#2E75B6"
    GREEN  = "#2ECC71"
    ORANGE = "#E67E22"
    RED    = "#E74C3C"
    GREY   = "#95A5A6"

    names = list(plot_data.keys())
    cats  = [plot_data[n]["category"] for n in names]
    pa_sc = [plot_data[n]["partner_A_score"] for n in names]
    pb_sc = [plot_data[n]["partner_B_score"] for n in names]
    fu_sc = [plot_data[n]["fusion_score"]    for n in names]

    x = np.arange(len(names))
    w = 0.25

    ax = axes[0]
    # Only plot bars where score is available
    for i, (pa, pb, fu) in enumerate(zip(pa_sc, pb_sc, fu_sc)):
        if pa is not None:
            ax.bar(i - w, pa, w, color=BLUE,   alpha=0.85, edgecolor="white",
                   label="Partner A (IDR)" if i==0 else "")
        if pb is not None:
            ax.bar(i,     pb, w, color=GREEN,  alpha=0.85, edgecolor="white",
                   label="Partner B (DBD)" if i==0 else "")
        if fu is not None:
            ax.bar(i + w, fu, w, color=ORANGE, alpha=0.85, edgecolor="white",
                   label="Fusion protein" if i==0 else "")

    ax.axhline(0.5, color="grey", linestyle="--", lw=1, label="Threshold 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("LLPS Score")
    ax.set_title("(A) Individual Partners vs Fusion Protein")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.2, axis="y")

    # Panel B: available partners by category
    ax = axes[1]
    # Separate by category
    fet_items = [(n, r) for n, r in plot_data.items() if r["category"] == "FET-family"]
    sig_items = [(n, r) for n, r in plot_data.items() if r["category"] == "Signalling"]

    categories = []
    means = []
    colors_cat = []

    if fet_items:
        fet_pa = [r["partner_A_score"] for _,r in fet_items if r["partner_A_score"] is not None]
        fet_fu = [r["fusion_score"]    for _,r in fet_items if r["fusion_score"]    is not None]
        if fet_pa:
            categories.append("FET\nPartner A (IDR)")
            means.append(np.mean(fet_pa))
            colors_cat.append(BLUE)
        if fet_fu:
            categories.append("FET\nFusion")
            means.append(np.mean(fet_fu))
            colors_cat.append(ORANGE)

    if sig_items:
        sig_pa = [r["partner_A_score"] for _,r in sig_items if r["partner_A_score"] is not None]
        sig_fu = [r["fusion_score"]    for _,r in sig_items if r["fusion_score"]    is not None]
        if sig_pa:
            categories.append("Signalling\nPartner A")
            means.append(np.mean(sig_pa))
            colors_cat.append(GREEN)
        if sig_fu:
            categories.append("Signalling\nFusion")
            means.append(np.mean(sig_fu))
            colors_cat.append(RED)

    if categories:
        bars = ax.bar(categories, means, color=colors_cat, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    str(round(v,3)), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.axhline(0.5, color="grey", linestyle="--", lw=1)
        ax.set_ylabel("Mean LLPS Score")
        ax.set_title("(B) Mean Scores by Category")
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS + "/fusion_partner_comparison.png",  dpi=300, bbox_inches="tight")
    plt.savefig(PLOTS + "/fusion_partner_comparison.tiff", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: " + PLOTS + "/fusion_partner_comparison.png")

print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)
print()
print("Key finding: FET-family partners (EWSR1, FUS, TAF15) should")
print("score HIGH as they contain the LLPS-driving IDR.")
print("FLI1, CIC (DNA-binding domains only) should score LOW.")
print("The fusion protein score reflects the dominant IDR partner.")
print()
print("Results saved to: " + LOGS + "/fusion_partner_results.json")
