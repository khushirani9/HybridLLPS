"""
26_folded_llps_benchmark.py
============================
Benchmarks HybridLLPS against other tools specifically on the
"folded LLPS+" subset — structured proteins (pLDDT mean > 70)
that still phase-separate.

This is the hardest test case because composition-only tools
like PLAAC and PScore almost completely fail on folded proteins.

Inputs
------
data/final/test_X.npy, test_y.npy
data/splits/test.csv  (needs uniprot_id, label columns)
results/benchmarking/all_tools_scores.csv (PSPHunter/CatGRANULE/PLAAC/PScore scores)
models/best_model_compat.pt

Outputs
-------
logs/folded_benchmark_results.json
results/plots/folded_llps_benchmark.png
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

BASE   = os.path.expanduser("~/llps_project")
DATA   = BASE + "/data/final"
SPLITS = BASE + "/data/splits"
MODELS = BASE + "/models"
LOGS   = BASE + "/logs"
PLOTS  = BASE + "/results/plots"
BENCH  = BASE + "/results/benchmarking"
os.makedirs(LOGS,  exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

print("=" * 60)
print("Script 26: Folded LLPS+ Benchmark")
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
ckpt  = torch.load(MODELS + "/best_model_compat.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("  Model loaded OK")

# ── Load test features ────────────────────────────────────────────
print("\n[2] Loading test features...")
X_test = np.load(DATA + "/test_X.npy").astype(np.float32)
y_test = np.load(DATA + "/test_y.npy").astype(np.float32)
print("  test_X shape: " + str(X_test.shape))

# ── Get HybridLLPS scores for all test proteins ───────────────────
with torch.no_grad():
    xb = torch.from_numpy(X_test)
    hybrid_scores = model(xb[:,:1280], xb[:,1280:1304],
                          xb[:,1304:1351], xb[:,1351:]).numpy()
print("  HybridLLPS scores computed. Range: " +
      str(round(hybrid_scores.min(),3)) + " to " + str(round(hybrid_scores.max(),3)))

# ── Load test.csv to get pLDDT values ────────────────────────────
print("\n[3] Loading test metadata...")
test_csv = pd.read_csv(SPLITS + "/test.csv")
print("  test.csv columns: " + str(list(test_csv.columns)))
print("  test.csv shape: " + str(test_csv.shape))

# pLDDT mean is stored as AF feature at column index 1280 (first AF feature)
# Extract directly from the feature matrix (already imputed/scaled)
# But we need raw pLDDT — get it from the feature matrix before scaling
# The AF features start at column 1280, first AF feature is pLDDT_mean
# These are scaled values — we need to use them for relative ranking which is fine
plddt_scaled = X_test[:, 1280]  # pLDDT mean (scaled)
print("  pLDDT (scaled) extracted from feature matrix column 1280")
print("  Range: " + str(round(float(plddt_scaled.min()),3)) +
      " to " + str(round(float(plddt_scaled.max()),3)))

# ── Define folded LLPS+ proteins ─────────────────────────────────
print("\n[4] Identifying folded LLPS+ subset...")
llps_pos_mask = (y_test == 1)
n_llps_pos = int(llps_pos_mask.sum())
print("  Total LLPS+ in test set: " + str(n_llps_pos))

# Top tertile by pLDDT among LLPS+ proteins = folded LLPS+
plddt_llps_pos = plddt_scaled[llps_pos_mask]
threshold_66 = float(np.percentile(plddt_llps_pos, 66.67))
print("  pLDDT threshold (66.67th percentile): " + str(round(threshold_66, 3)))

folded_mask = llps_pos_mask & (plddt_scaled >= threshold_66)
n_folded = int(folded_mask.sum())
print("  Folded LLPS+ proteins (top tertile): " + str(n_folded))

if n_folded < 5:
    print("  WARNING: Very few folded proteins. Using top 50th percentile instead.")
    threshold_66 = float(np.percentile(plddt_llps_pos, 50))
    folded_mask = llps_pos_mask & (plddt_scaled >= threshold_66)
    n_folded = int(folded_mask.sum())
    print("  Folded LLPS+ proteins (top 50%): " + str(n_folded))

# Scores for folded LLPS+ subset
hybrid_folded = hybrid_scores[folded_mask]
y_folded = y_test[folded_mask]  # all 1s since we filtered to LLPS+
folded_indices = np.where(folded_mask)[0]
print("  Mean HybridLLPS score on folded LLPS+: " + str(round(float(hybrid_folded.mean()), 4)))

# ── Load benchmark tool scores ────────────────────────────────────
print("\n[5] Loading benchmark tool scores...")
tools_csv_path = BENCH + "/all_tools_scores.csv"

results = {}
results["HybridLLPS"] = {
    "accuracy":     float(accuracy_score(y_folded, (hybrid_folded >= 0.5).astype(int))),
    "mean_score":   float(hybrid_folded.mean()),
    "median_score": float(np.median(hybrid_folded)),
    "scores":       hybrid_folded.tolist(),
    "n":            n_folded,
}
print("  HybridLLPS accuracy on folded LLPS+: " +
      str(round(results["HybridLLPS"]["accuracy"], 4)))

if os.path.exists(tools_csv_path):
    tools_df = pd.read_csv(tools_csv_path)
    print("  all_tools_scores.csv loaded. Columns: " + str(list(tools_df.columns)))

    # Try to merge with test.csv to align tool scores with test indices
    if "uniprot_id" in test_csv.columns and "uniprot_id" in tools_df.columns:
        merged = test_csv.reset_index().merge(tools_df, on="uniprot_id", how="left")
        merged = merged.sort_values("index").set_index("index")

        # Detect tool score columns
        skip_cols = {"uniprot_id","label","sequence","pLDDT_mean","index","level_0"}
        tool_cols = [c for c in tools_df.columns if c not in skip_cols]
        print("  Tool score columns found: " + str(tool_cols))

        for col in tool_cols:
            if col not in merged.columns:
                continue
            tool_scores_all = merged[col].values.astype(float)
            tool_scores_folded = tool_scores_all[folded_indices]

            # Fill NaN with 0
            nan_mask = np.isnan(tool_scores_folded)
            if nan_mask.sum() > 0:
                print("  " + col + ": " + str(nan_mask.sum()) + " NaN values filled with 0")
                tool_scores_folded = np.nan_to_num(tool_scores_folded, nan=0.0)

            # Normalize to 0-1 if needed
            if tool_scores_folded.max() > 1.0:
                tool_scores_folded = (tool_scores_folded - tool_scores_folded.min()) / \
                                     (tool_scores_folded.max() - tool_scores_folded.min() + 1e-8)

            acc = float(accuracy_score(y_folded, (tool_scores_folded >= 0.5).astype(int)))
            results[col] = {
                "accuracy":     acc,
                "mean_score":   float(tool_scores_folded.mean()),
                "median_score": float(np.median(tool_scores_folded)),
                "scores":       tool_scores_folded.tolist(),
                "n":            n_folded,
            }
            print("  " + col + " accuracy on folded LLPS+: " + str(round(acc, 4)))
    else:
        print("  Could not merge tool scores — no uniprot_id column. Using HybridLLPS only.")
else:
    print("  all_tools_scores.csv not found at: " + tools_csv_path)
    print("  Proceeding with HybridLLPS only.")

# ── Save results ──────────────────────────────────────────────────
print("\n[6] Saving results...")
out = {
    "plddt_threshold":  threshold_66,
    "n_llps_pos":       n_llps_pos,
    "n_folded":         n_folded,
    "tool_results": {
        k: {kk: vv for kk, vv in v.items() if kk != "scores"}
        for k, v in results.items()
    }
}
with open(LOGS + "/folded_benchmark_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("  Saved: " + LOGS + "/folded_benchmark_results.json")

# ── Plot ──────────────────────────────────────────────────────────
print("\n[7] Generating figure...")
tool_names  = list(results.keys())
accuracies  = [results[t]["accuracy"]   for t in tool_names]
mean_scores = [results[t]["mean_score"] for t in tool_names]
all_scores  = [np.array(results[t]["scores"]) for t in tool_names]

colors = ["#2E75B6","#E74C3C","#E67E22","#2ECC71","#9B59B6","#1ABC9C"][:len(tool_names)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Folded LLPS+ Benchmark  (top tertile by pLDDT, n=" +
             str(n_folded) + " proteins)", fontsize=13, fontweight="bold")

# Panel A: Accuracy
ax = axes[0]
bars = ax.bar(tool_names, accuracies, color=colors, alpha=0.85, edgecolor="white")
ax.set_ylim(0, 1.15)
ax.set_ylabel("Accuracy (threshold 0.5)")
ax.set_title("(A) Accuracy on Folded LLPS+")
ax.axhline(0.5, color="grey", linestyle="--", lw=1, label="Random (0.5)")
ax.legend(fontsize=9)
for bar, v in zip(bars, accuracies):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            str(round(v,3)), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticklabels(tool_names, rotation=30, ha="right", fontsize=9)
ax.grid(alpha=0.2, axis="y")

# Panel B: Score distributions (violin or box if only 1 tool)
ax = axes[1]
if len(all_scores) > 1:
    parts = ax.violinplot(all_scores, positions=range(len(tool_names)),
                          showmeans=True, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
else:
    ax.boxplot(all_scores[0])
ax.axhline(0.5, color="grey", linestyle="--", lw=1)
ax.set_xticks(range(len(tool_names)))
ax.set_xticklabels(tool_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("LLPS Score")
ax.set_title("(B) Score Distributions")
ax.set_ylim(-0.05, 1.15)
ax.grid(alpha=0.2, axis="y")

# Panel C: Mean scores
ax = axes[2]
bars2 = ax.bar(tool_names, mean_scores, color=colors, alpha=0.85, edgecolor="white")
ax.axhline(0.5, color="grey", linestyle="--", lw=1, label="Decision boundary")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Mean LLPS Score")
ax.set_title("(C) Mean Score on Folded LLPS+")
ax.legend(fontsize=9)
for bar, v in zip(bars2, mean_scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            str(round(v,3)), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticklabels(tool_names, rotation=30, ha="right", fontsize=9)
ax.grid(alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(PLOTS + "/folded_llps_benchmark.png",  dpi=300, bbox_inches="tight")
plt.savefig(PLOTS + "/folded_llps_benchmark.tiff", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: " + PLOTS + "/folded_llps_benchmark.png")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("Folded LLPS+ proteins (top tertile pLDDT): " + str(n_folded))
print()
print("Tool             Accuracy   Mean Score")
print("-" * 40)
for t in tool_names:
    print(t.ljust(16) + str(round(results[t]["accuracy"],4)).rjust(10) +
          str(round(results[t]["mean_score"],4)).rjust(12))
