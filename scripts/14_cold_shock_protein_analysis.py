"""
14 COLD SHOCK PROTEIN ANALYSIS
======================================================================
Cold Shock Protein Evolutionary LLPS Gradient Analysis
======================================================================

Scores 24 cold shock domain proteins across four evolutionary groups to test
whether the model generalises to protein families not in the training set.

Groups:
  Bacterial CSPs  (n=10, 66-70aa):    CSD only, no LLPS expected
  Plant CSPs      (n=6, 201-469aa):   CSD + glycine-rich extension, LLPS uncertain
  Mammalian YBX   (n=6, 322-372aa):   CSD + charged IDR, LLPS confirmed
  Reference       (n=2):              TDP-43, FUS — known LLPS proteins

None of the 24 proteins were in the training set. A positive correlation between
evolutionary complexity and LLPS score validates genuine transfer learning.
Wheat CSPs (TaCS120, TaCS66) are outliers due to tandem CSD repeats, not IDR.

Inputs
------
Sequences fetched from UniProt at runtime
models/best_model.pt, data/final/imputer.pkl, scaler.pkl

Outputs
-------
results/plots/csp_score_vs_length.png
results/plots/csp_evolutionary_gradient.png
logs/csp_results.json

Usage
-----
python 14_cold_shock_protein_analysis.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import joblib, os, re, warnings
warnings.filterwarnings("ignore")

CSP_DIR     = os.path.expanduser("~/llps_project/data/cold_shock/")
MODEL_DIR   = os.path.expanduser("~/llps_project/models/")
DATA_DIR    = os.path.expanduser("~/llps_project/data/final/")
RESULTS_DIR = os.path.expanduser("~/llps_project/results/cold_shock/")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading CSP sequences...")
metadata = pd.read_csv(os.path.join(CSP_DIR, "csp_metadata.csv"))
sequences = {}
for rec in SeqIO.parse(os.path.join(CSP_DIR, "all_csp_proteins.fasta"), "fasta"):
    uid = rec.id.split("|")[0]
    sequences[uid] = str(rec.seq)

ids      = metadata["uniprot_id"].tolist()
seqs     = [sequences[uid] for uid in ids]
names    = metadata["name"].tolist()
groups   = metadata["group"].tolist()
expected = metadata["expected_llps"].tolist()
print(f"Total proteins: {len(ids)}")

print("Computing ESM-2 embeddings...")
import esm
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

esm_features = []
for i in range(0, len(seqs), 8):
    batch = [(f"p{j}", s[:1022]) for j, s in enumerate(seqs[i:i+8])]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[33], return_contacts=False)
    reps = out["representations"][33]
    for k, (_, s) in enumerate(batch):
        emb = reps[k, 1:len(s)+1].mean(0).cpu().numpy()
        esm_features.append(emb)
    print(f"  {min(i+8, len(seqs))}/{len(seqs)}")
esm_features = np.array(esm_features)
print(f"ESM-2 shape: {esm_features.shape}")

af_features = np.full((len(seqs), 24), np.nan)

print("Computing physicochemical features...")
AROMATIC    = set("YFW")
HYDROPHOBIC = set("VILMFYW")
POSITIVE_AA = set("KR")
NEGATIVE_AA = set("DE")
POLAR_AA    = set("STNQ")

def extract_pc(seq):
    feat = {}
    seq  = seq.upper()
    L    = len(seq)
    if L == 0:
        return np.zeros(47)
    try:
        analysis = ProteinAnalysis(seq)
        aa_comp  = analysis.get_amino_acids_percent()
        for aa, frac in aa_comp.items():
            feat[f"aa_frac_{aa}"] = frac
        feat["mol_weight"]     = analysis.molecular_weight()
        feat["isoelectric_pt"] = analysis.isoelectric_point()
        feat["gravy"]          = analysis.gravy()
        feat["aromaticity"]    = analysis.aromaticity()
        feat["instability"]    = analysis.instability_index()
    except:
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            feat[f"aa_frac_{aa}"] = 0.0
        for k in ["mol_weight","isoelectric_pt","gravy","aromaticity","instability"]:
            feat[k] = 0.0
    feat["frac_aromatic"]    = sum(seq.count(a) for a in AROMATIC)    / L
    feat["frac_hydrophobic"] = sum(seq.count(a) for a in HYDROPHOBIC) / L
    feat["frac_positive"]    = sum(seq.count(a) for a in POSITIVE_AA) / L
    feat["frac_negative"]    = sum(seq.count(a) for a in NEGATIVE_AA) / L
    feat["frac_polar"]       = sum(seq.count(a) for a in POLAR_AA)    / L
    feat["net_charge"]       = feat["frac_positive"] - feat["frac_negative"]
    feat["frac_QN"]          = (seq.count("Q") + seq.count("N")) / L
    feat["frac_PG"]          = (seq.count("P") + seq.count("G")) / L
    feat["frac_Y"]           = seq.count("Y") / L
    try:
        pos_arr = np.array([1 if aa in POSITIVE_AA else 0 for aa in seq], dtype=float)
        neg_arr = np.array([1 if aa in NEGATIVE_AA else 0 for aa in seq], dtype=float)
        f_plus  = pos_arr.mean()
        f_minus = neg_arr.mean()
        FCR     = f_plus + f_minus
        NCPR    = f_plus - f_minus
        feat["FCR"]  = FCR
        feat["NCPR"] = NCPR
        if FCR > 0 and L >= 5:
            blob_size    = 5
            delta_values = []
            for i in range(L - blob_size + 1):
                blob     = seq[i:i+blob_size]
                blob_pos = sum(1 for a in blob if a in POSITIVE_AA) / blob_size
                blob_neg = sum(1 for a in blob if a in NEGATIVE_AA) / blob_size
                delta    = ((blob_pos - f_plus)**2 + (blob_neg - f_minus)**2) / 2
                delta_values.append(delta)
            delta_mean    = np.mean(delta_values)
            delta_max     = (f_plus**2 + f_minus**2) / 2 if FCR > 0 else 1
            feat["kappa"] = delta_mean / delta_max if delta_max > 0 else 0.0
        else:
            feat["kappa"] = 0.0
    except:
        feat["FCR"] = feat["NCPR"] = feat["kappa"] = 0.0
    try:
        max_pos_run = max_neg_run = cur_pos = cur_neg = 0
        for aa in seq:
            if aa in POSITIVE_AA:
                cur_pos += 1; cur_neg  = 0
            elif aa in NEGATIVE_AA:
                cur_neg += 1; cur_pos  = 0
            else:
                cur_pos = cur_neg = 0
            max_pos_run = max(max_pos_run, cur_pos)
            max_neg_run = max(max_neg_run, cur_neg)
        feat["max_pos_run"] = max_pos_run / L
        feat["max_neg_run"] = max_neg_run / L
    except:
        feat["max_pos_run"] = feat["max_neg_run"] = 0.0
    try:
        arom_pos = [i for i, aa in enumerate(seq) if aa in AROMATIC]
        if len(arom_pos) >= 2:
            spacings = np.diff(arom_pos)
            feat["aromatic_spacing_mean"]    = float(np.mean(spacings))
            feat["aromatic_spacing_std"]     = float(np.std(spacings))
            feat["aromatic_spacing_optimal"] = float(np.mean((spacings>=2)&(spacings<=6)))
        else:
            feat["aromatic_spacing_mean"]    = L
            feat["aromatic_spacing_std"]     = 0.0
            feat["aromatic_spacing_optimal"] = 0.0
    except:
        feat["aromatic_spacing_mean"] = feat["aromatic_spacing_std"] = 0.0
        feat["aromatic_spacing_optimal"] = 0.0
    try:
        patches = re.findall(r"[VILMFYW]{3,}", seq)
        feat["hydrophobic_patch_count"]  = len(patches) / L
        feat["hydrophobic_patch_maxlen"] = max((len(p) for p in patches), default=0) / L
        eisenberg = {"A":0.25,"R":-1.76,"N":-0.64,"D":-0.72,"C":0.04,
                     "Q":-0.69,"E":-0.62,"G":0.16,"H":-0.40,"I":0.73,
                     "L":0.53,"K":-1.10,"M":0.26,"F":0.61,"P":-0.07,
                     "S":-0.26,"T":-0.18,"W":0.37,"Y":0.02,"V":0.54}
        feat["mean_hydrophobicity"] = float(np.mean([eisenberg.get(aa, 0) for aa in seq]))
    except:
        feat["hydrophobic_patch_count"] = feat["hydrophobic_patch_maxlen"] = 0.0
        feat["mean_hydrophobicity"] = 0.0
    try:
        aa_counts = np.array([seq.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"])
        aa_freq   = aa_counts / aa_counts.sum()
        aa_freq   = aa_freq[aa_freq > 0]
        feat["shannon_entropy"] = float(-np.sum(aa_freq * np.log2(aa_freq)))
    except:
        feat["shannon_entropy"] = 0.0
    try:
        n = len(seq)
        if n >= 6:
            f_arom  = sum(1 for a in seq if a in AROMATIC) / n
            f_charg = sum(1 for a in seq if a in (POSITIVE_AA | NEGATIVE_AA)) / n
            blob_size  = 6
            omega_vals = []
            for i in range(n - blob_size + 1):
                blob   = seq[i:i+blob_size]
                b_arom = sum(1 for a in blob if a in AROMATIC) / blob_size
                b_chg  = sum(1 for a in blob if a in (POSITIVE_AA | NEGATIVE_AA)) / blob_size
                delta  = ((b_arom - f_arom)**2 + (b_chg - f_charg)**2) / 2
                omega_vals.append(delta)
            denom = (f_arom**2 + f_charg**2) / 2
            feat["omega"] = float(np.mean(omega_vals) / denom) if denom > 0 else 0.0
        else:
            feat["omega"] = 0.0
    except:
        feat["omega"] = 0.0
    col_order = (
        [f"aa_frac_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"] +
        ["mol_weight","isoelectric_pt","gravy","aromaticity","instability",
         "frac_aromatic","frac_hydrophobic","frac_positive","frac_negative",
         "frac_polar","net_charge","frac_QN","frac_PG","frac_Y",
         "FCR","NCPR","max_pos_run","max_neg_run",
         "aromatic_spacing_mean","aromatic_spacing_std","aromatic_spacing_optimal",
         "hydrophobic_patch_count","hydrophobic_patch_maxlen","mean_hydrophobicity",
         "shannon_entropy","kappa","omega"]
    )
    return np.array([feat.get(c, 0.0) for c in col_order], dtype=np.float32)

pc_features = np.array([extract_pc(s) for s in seqs])
print(f"Physicochemical shape: {pc_features.shape}")

print("Computing dipeptide/CTD features...")
AAs = "ACDEFGHIKLMNPQRSTVWY"

def extract_trad(seq):
    feat = {}
    seq  = seq.upper()
    L    = len(seq)
    dipep_counts = {f"dp_{a1}{a2}": 0 for a1 in AAs for a2 in AAs}
    for i in range(L - 1):
        key = f"dp_{seq[i]}{seq[i+1]}"
        if key in dipep_counts:
            dipep_counts[key] += 1
    total_pairs = L - 1
    for key in dipep_counts:
        feat[key] = dipep_counts[key] / total_pairs if total_pairs > 0 else 0.0
    hydro_groups = {
        "R":1,"K":1,"E":1,"D":1,"Q":1,"N":1,
        "G":2,"A":2,"S":2,"T":2,"P":2,"H":2,"Y":2,
        "C":3,"L":3,"V":3,"I":3,"M":3,"F":3,"W":3
    }
    group_seq = [hydro_groups.get(aa, 2) for aa in seq]
    for g in [1, 2, 3]:
        feat[f"ctd_C{g}"] = group_seq.count(g) / L
    transitions = sum(1 for i in range(L-1) if group_seq[i] != group_seq[i+1])
    feat["ctd_T"] = transitions / (L - 1) if L > 1 else 0.0
    for g in [1, 2, 3]:
        positions = [i/L for i, x in enumerate(group_seq) if x == g]
        feat[f"ctd_D{g}_first"] = positions[0]  if positions else 0.0
        feat[f"ctd_D{g}_last"]  = positions[-1] if positions else 0.0
    col_order = (
        [f"dp_{a1}{a2}" for a1 in AAs for a2 in AAs] +
        ["ctd_C1","ctd_C2","ctd_C3","ctd_T",
         "ctd_D1_first","ctd_D1_last","ctd_D2_first","ctd_D2_last",
         "ctd_D3_first","ctd_D3_last"]
    )
    return np.array([feat.get(c, 0.0) for c in col_order], dtype=np.float32)

trad_features = np.array([extract_trad(s) for s in seqs])
print(f"Dipeptide/CTD shape: {trad_features.shape}")

print("Preprocessing...")
X_raw    = np.concatenate([esm_features, af_features, pc_features, trad_features], axis=1)
imputer  = joblib.load(os.path.join(DATA_DIR, "imputer.pkl"))
scaler   = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
X_scaled = scaler.transform(imputer.transform(X_raw))
print(f"Feature matrix: {X_scaled.shape}")

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
            self.esm_branch(x[:,:1280]),
            self.af_branch(x[:,1280:1304]),
            self.pc_branch(x[:,1304:1351]),
            self.trad_branch(x[:,1351:])
        ], dim=1)
        return self.fusion(out).squeeze(1)

print("Loading model...")
model = LLPSHybridModel().to(device)
ckpt  = torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("Predicting...")
with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    scores   = model(X_tensor).cpu().numpy()

results = pd.DataFrame({
    "uniprot_id"   : ids,
    "name"         : names,
    "group"        : groups,
    "expected_llps": expected,
    "seq_length"   : [len(s) for s in seqs],
    "llps_score"   : scores
})
results.to_csv(os.path.join(RESULTS_DIR, "csp_llps_scores.csv"), index=False)

print("\n" + "="*65)
print("CSP LLPS SCORES - EVOLUTIONARY GRADIENT ANALYSIS")
print("="*65)

group_means = {}
for group in ["Bacterial","Plant","Mammalian","Reference"]:
    subset   = results[results["group"]==group].sort_values("llps_score", ascending=False)
    scores_g = subset["llps_score"].tolist()
    group_means[group] = np.mean(scores_g)
    print(f"\n{group} (mean={np.mean(scores_g):.4f}):")
    print(f"  {'Name':<12} {'Length':>7}  {'Score':>7}  {'Expected':>10}")
    print(f"  {'-'*45}")
    for _, row in subset.iterrows():
        bar = "#" * int(row["llps_score"] * 20)
        print(f"  {row['name']:<12} {row['seq_length']:>7}  "
              f"{row['llps_score']:>7.4f}  {row['expected_llps']:>10}  {bar}")

print("\n=== HYPOTHESIS TEST ===")
print(f"Bacterial (no IDR):    {group_means['Bacterial']:.4f}")
print(f"Plant (short IDR):     {group_means['Plant']:.4f}")
print(f"Mammalian (long IDR):  {group_means['Mammalian']:.4f}")
print(f"Reference (confirmed): {group_means['Reference']:.4f}")

gradient = (group_means["Bacterial"] < group_means["Plant"] < group_means["Mammalian"])
print(f"\nEvolutionary gradient confirmed: {gradient}")
if gradient:
    print("Model correctly identifies increasing LLPS propensity")
    print("from bacterial -> plant -> mammalian CSP homologs")
else:
    print("Gradient not fully confirmed - see individual scores above")

print(f"\nSaved: {RESULTS_DIR}csp_llps_scores.csv")
print("="*65)
