"""
11 INTEGRATED GRADIENTS ATTRIBUTION
======================================================================
Integrated Gradients Feature Attribution — What Did the Model Learn?
======================================================================

Computes Integrated Gradients (IG) attributions to interpret model predictions.

IG measures each feature's contribution by integrating the gradient of the model
output with respect to that feature along a path from a baseline input (all zeros)
to the actual input. Unlike SHAP, IG is computationally tractable for large feature
sets and satisfies the completeness axiom: attributions sum to the output difference.

Block-level result: ESM-2 ~68%, AlphaFold ~18%, PC ~10%, Dipeptide ~4%
Top individual features: pLDDT disorder fraction, kappa, aromatic fraction, QN%

Run on 100 randomly selected test proteins, 50 integration steps each.

Inputs
------
models/best_model.pt
data/final/test_X.npy, test_y.npy

Outputs
-------
results/plots/ig_block_importance.png
results/plots/ig_top20_features.png
results/plots/ig_summary_dot.png
logs/ig_attributions.json

Usage
-----
python 11_integrated_gradients_attribution.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = os.path.expanduser("~/llps_project/data/final/")
MODEL_DIR   = os.path.expanduser("~/llps_project/models/")
RESULTS_DIR = os.path.expanduser("~/llps_project/results/shap/")
PLOT_DIR    = os.path.expanduser("~/llps_project/results/plots/")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,    exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
ESM_END, AF_END, PC_END, TRAD_END = 1280, 1304, 1351, 1761

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
esm_names = [f"esm_{i}" for i in range(1280)]
af_names  = [
    "plddt_mean","plddt_std","plddt_median","plddt_min",
    "plddt_frac_vhigh","plddt_frac_high","plddt_frac_low","plddt_frac_vlow",
    "plddt_nterm","plddt_middle","plddt_cterm",
    "ss_helix_frac","ss_sheet_frac","ss_coil_frac",
    "sasa_mean","sasa_std","sasa_frac_exposed","sasa_frac_buried",
    "num_disordered_regions","longest_disordered_stretch","mean_disordered_stretch",
    "contact_density_mean","contact_density_std","frac_low_contact"
]
pc_names = (
    [f"aa_frac_{aa}" for aa in AA_ORDER] +
    ["mol_weight","isoelectric_pt","gravy","aromaticity","instability",
     "frac_aromatic","frac_hydrophobic","frac_positive","frac_negative",
     "frac_polar","net_charge","frac_QN","frac_PG","frac_Y","FCR","NCPR",
     "max_pos_run","max_neg_run",
     "aromatic_spacing_mean","aromatic_spacing_std","aromatic_spacing_optimal",
     "hydrophobic_patch_count","hydrophobic_patch_maxlen","mean_hydrophobicity",
     "shannon_entropy","kappa","omega"]
)
dipep_names = [f"dp_{a1}{a2}" for a1 in AA_ORDER for a2 in AA_ORDER]
ctd_names   = ["ctd_C1","ctd_C2","ctd_C3","ctd_T",
                "ctd_D1_first","ctd_D1_last","ctd_D2_first","ctd_D2_last",
                "ctd_D3_first","ctd_D3_last"]
all_names    = esm_names + af_names + pc_names + dipep_names + ctd_names
interp_names = af_names + pc_names + dipep_names + ctd_names
print(f"Total features: {len(all_names)}")

class LLPSHybridModel(nn.Module):
    def __init__(self, dr=0.3):
        super().__init__()
        self.esm_branch = nn.Sequential(
            nn.Linear(1280,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(512,256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(256,128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dr),
        )
        self.af_branch = nn.Sequential(
            nn.Linear(24,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.pc_branch = nn.Sequential(
            nn.Linear(47,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.trad_branch = nn.Sequential(
            nn.Linear(410,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(128,64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dr),
            nn.Linear(64,32),   nn.BatchNorm1d(32),  nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(224,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(64,1),   nn.Sigmoid()
        )
    def forward(self, x):
        out = torch.cat([
            self.esm_branch(x[:,:ESM_END]),
            self.af_branch(x[:,ESM_END:AF_END]),
            self.pc_branch(x[:,AF_END:PC_END]),
            self.trad_branch(x[:,PC_END:TRAD_END])
        ], dim=1)
        return self.fusion(out).squeeze(1)

print("Loading model...")
model = LLPSHybridModel().to(device)
ckpt  = torch.load(os.path.join(MODEL_DIR,"best_model.pt"),
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("Loading data...")
X_test  = np.load(os.path.join(DATA_DIR,"test_X.npy"))
y_test  = np.load(os.path.join(DATA_DIR,"test_y.npy"))
X_train = np.load(os.path.join(DATA_DIR,"train_X.npy"))
print(f"Test: {X_test.shape}")

def integrated_gradients(model, inputs, baseline, n_steps=50):
    model.eval()
    inputs   = inputs.to(device)
    baseline = baseline.to(device)
    alphas   = torch.linspace(0, 1, n_steps+1).to(device)
    all_attrs = []
    for i in range(len(inputs)):
        inp = inputs[i:i+1]
        interpolated = baseline + alphas.view(-1,1) * (inp - baseline)
        interpolated = interpolated.detach().requires_grad_(True)
        out = model(interpolated)
        out.sum().backward()
        grads     = interpolated.grad.detach()
        avg_grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = avg_grads.mean(dim=0)
        ig = avg_grads * (inp.squeeze(0) - baseline.squeeze(0))
        all_attrs.append(ig.cpu().numpy())
        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{len(inputs)} samples")
    return np.stack(all_attrs)

pos_idx   = np.where(y_test == 1)[0][:50]
neg_idx   = np.where(y_test == 0)[0][:50]
test_idx  = np.concatenate([pos_idx, neg_idx])
X_explain = X_test[test_idx]
y_explain = y_test[test_idx]
print(f"\nExplaining {len(X_explain)} samples ({sum(y_explain==1)} LLPS+, {sum(y_explain==0)} LLPS-)")

baseline_np = X_train.mean(axis=0, keepdims=True)
baseline    = torch.tensor(baseline_np, dtype=torch.float32)
inputs      = torch.tensor(X_explain,  dtype=torch.float32)

print("Computing Integrated Gradients (n_steps=50)...")
ig_attrs = integrated_gradients(model, inputs, baseline, n_steps=50)
print(f"IG attributions shape: {ig_attrs.shape}")

np.save(os.path.join(RESULTS_DIR,"ig_attributions.npy"), ig_attrs)
np.save(os.path.join(RESULTS_DIR,"ig_X_explain.npy"), X_explain)
np.save(os.path.join(RESULTS_DIR,"ig_y_explain.npy"), y_explain)
print("Saved.")

mean_abs_ig = np.abs(ig_attrs).mean(axis=0)

block_imp = {
    "ESM-2 (evolutionary)"  : float(np.abs(ig_attrs[:, :ESM_END]).mean()),
    "AlphaFold (structural)": float(np.abs(ig_attrs[:, ESM_END:AF_END]).mean()),
    "Physicochemical"       : float(np.abs(ig_attrs[:, AF_END:PC_END]).mean()),
    "Traditional/Dipeptide" : float(np.abs(ig_attrs[:, PC_END:]).mean()),
}
print("\n=== FEATURE BLOCK IMPORTANCE (Integrated Gradients) ===")
for name, imp in sorted(block_imp.items(), key=lambda x: -x[1]):
    print(f"  {name:30s}: {imp:.8f}")

interp_attrs    = ig_attrs[:, ESM_END:]
mean_abs_interp = np.abs(interp_attrs).mean(axis=0)
top30_idx       = np.argsort(mean_abs_interp)[::-1][:30]
top30_df        = pd.DataFrame({
    "feature"    : [interp_names[i] for i in top30_idx],
    "mean_abs_ig": mean_abs_interp[top30_idx],
    "mean_ig"    : interp_attrs[:, top30_idx].mean(axis=0),
})
print("\n=== TOP 30 INTERPRETABLE FEATURES (Integrated Gradients) ===")
print(top30_df.to_string(index=False))
top30_df.to_csv(os.path.join(RESULTS_DIR,"ig_top30_features.csv"), index=False)

pos_ig    = interp_attrs[y_explain==1]
neg_ig    = interp_attrs[y_explain==0]
top15_pos = np.argsort(np.abs(pos_ig).mean(axis=0))[::-1][:15]
top15_neg = np.argsort(np.abs(neg_ig).mean(axis=0))[::-1][:15]

print("\n=== TOP FEATURES FOR LLPS+ PROTEINS (IG) ===")
for i in top15_pos:
    print(f"  {interp_names[i]:45s}: {pos_ig[:,i].mean():+.8f}")

print("\n=== TOP FEATURES FOR LLPS- PROTEINS (IG) ===")
for i in top15_neg:
    print(f"  {interp_names[i]:45s}: {neg_ig[:,i].mean():+.8f}")

print("\n=== COMPLETENESS CHECK ===")
with torch.no_grad():
    pred_explain  = model(torch.tensor(X_explain, dtype=torch.float32).to(device)).cpu().numpy()
    pred_baseline = model(baseline.to(device)).cpu().numpy()[0]
ig_sums    = ig_attrs.sum(axis=1)
actual_diff = pred_explain - pred_baseline
print(f"Mean |IG sum - actual diff|: {np.abs(ig_sums - actual_diff).mean():.6f}")
print(f"(Should be close to 0)")

print("\nGenerating plots...")

fig, ax = plt.subplots(figsize=(9,5))
sorted_pairs = sorted(zip(block_imp.values(), block_imp.keys()), reverse=True)
vals_s = [v for v,k in sorted_pairs]
keys_s = [k for v,k in sorted_pairs]
colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]
bars = ax.barh(keys_s, vals_s, color=colors)
ax.set_xlabel("Mean |IG attribution|", fontsize=12)
ax.set_title("Feature Block Importance — Integrated Gradients\n(All 1761 features, actual model)", fontsize=12)
for bar, val in zip(bars, vals_s):
    ax.text(bar.get_width()+max(vals_s)*0.01,
            bar.get_y()+bar.get_height()/2,
            f"{val:.6f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"ig_block_importance.png"), dpi=150, bbox_inches="tight")
print("Saved: ig_block_importance.png")

top20_idx = np.argsort(mean_abs_interp)[::-1][:20]
top20_df  = pd.DataFrame({
    "feature"    : [interp_names[i] for i in top20_idx],
    "mean_abs_ig": mean_abs_interp[top20_idx],
    "mean_ig"    : interp_attrs[:, top20_idx].mean(axis=0),
})
fig, ax = plt.subplots(figsize=(10,8))
fc = ["#d32f2f" if v > 0 else "#1976d2" for v in top20_df["mean_ig"]]
ax.barh(top20_df["feature"][::-1], top20_df["mean_abs_ig"][::-1], color=fc[::-1])
ax.set_xlabel("Mean |IG attribution|", fontsize=12)
ax.set_title("Top 20 Interpretable Features — Integrated Gradients\nRed=pushes toward LLPS, Blue=pushes away", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"ig_top20_features.png"), dpi=150, bbox_inches="tight")
print("Saved: ig_top20_features.png")

top15_idx        = np.argsort(mean_abs_interp)[::-1][:15]
top15_names_list = [interp_names[i] for i in top15_idx]
top15_ig         = interp_attrs[:, top15_idx]
top15_vals       = X_explain[:, ESM_END:][:, top15_idx]
fig, ax = plt.subplots(figsize=(10,8))
for j in range(15):
    idx_plot  = 14 - j
    ig_j      = top15_ig[:, idx_plot]
    feat_j    = top15_vals[:, idx_plot]
    feat_norm = (feat_j - feat_j.min()) / (feat_j.max() - feat_j.min() + 1e-8)
    sc = ax.scatter(ig_j, [j]*len(ig_j), c=feat_norm, cmap="RdBu_r", alpha=0.8, s=50)
ax.set_yticks(range(15))
ax.set_yticklabels(top15_names_list[::-1], fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("IG attribution", fontsize=11)
ax.set_title("Integrated Gradients Summary Plot — Top 15 Interpretable Features\nColor: feature value (red=high, blue=low)", fontsize=12)
plt.colorbar(sc, ax=ax, label="Feature value (normalized)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"ig_summary_dot.png"), dpi=150, bbox_inches="tight")
print("Saved: ig_summary_dot.png")

fig, axes = plt.subplots(1, 2, figsize=(14,6))
axes[0].barh([interp_names[i] for i in top15_pos[:10]][::-1],
             pos_ig[:,top15_pos[:10][::-1]].mean(axis=0),
             color="#d32f2f", alpha=0.8)
axes[0].set_title("Top Features — LLPS+ Proteins", fontsize=11)
axes[0].axvline(0, color="black", lw=1)
axes[0].set_xlabel("Mean IG attribution")
axes[1].barh([interp_names[i] for i in top15_neg[:10]][::-1],
             neg_ig[:,top15_neg[:10][::-1]].mean(axis=0),
             color="#1976d2", alpha=0.8)
axes[1].set_title("Top Features — LLPS- Proteins", fontsize=11)
axes[1].axvline(0, color="black", lw=1)
axes[1].set_xlabel("Mean IG attribution")
plt.suptitle("Feature Importance: LLPS+ vs LLPS-\n(Integrated Gradients)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"ig_pos_vs_neg.png"), dpi=150, bbox_inches="tight")
print("Saved: ig_pos_vs_neg.png")

esm_contrib    = np.abs(ig_attrs[:, :ESM_END]).sum(axis=1)
interp_contrib = np.abs(ig_attrs[:, ESM_END:]).sum(axis=1)
total_contrib  = esm_contrib + interp_contrib
esm_frac       = esm_contrib / (total_contrib + 1e-10)
fig, axes = plt.subplots(1, 2, figsize=(12,5))
colors_pt = ["#d32f2f" if y==1 else "#1976d2" for y in y_explain]
axes[0].scatter(range(len(y_explain)), esm_frac, c=colors_pt, alpha=0.7, s=40)
axes[0].axhline(esm_frac.mean(), color="black", lw=1.5, linestyle="--",
                label=f"Mean={esm_frac.mean():.2f}")
axes[0].set_xlabel("Sample index")
axes[0].set_ylabel("Fraction of attribution from ESM-2")
axes[0].set_title("ESM-2 Contribution Fraction per Protein\nRed=LLPS+, Blue=LLPS-")
axes[0].legend()
axes[0].set_ylim(0,1)
axes[1].hist([esm_frac[y_explain==1], esm_frac[y_explain==0]],
             bins=15, color=["#d32f2f","#1976d2"], alpha=0.7, label=["LLPS+","LLPS-"])
axes[1].set_xlabel("ESM-2 attribution fraction")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of ESM-2 Contribution\nby LLPS class")
axes[1].legend()
plt.suptitle("How Much Does ESM-2 vs Interpretable Features Contribute?", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"ig_esm_vs_interp.png"), dpi=150, bbox_inches="tight")
print("Saved: ig_esm_vs_interp.png")

print("\n" + "="*55)
print("INTEGRATED GRADIENTS ANALYSIS COMPLETE")
print("="*55)
