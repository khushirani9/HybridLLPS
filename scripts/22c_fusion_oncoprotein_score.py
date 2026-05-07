"""
22C FUSION ONCOPROTEIN SCORE
======================================================================
Fusion Oncoprotein LLPS Scoring and Visualisation
======================================================================

Scores 13 fusion oncoproteins and visualises grouped results.

FET-family fusions score high (mean=0.928) because they contain the QGSY-rich
IDR from EWS, FUS, or TAF15. Signalling fusions score lower (mean=0.737).
AUC for separating FET vs non-FET = 0.925.

BCR-ABL1 (0.709) is higher than expected for a signalling fusion because BCR
contains a coiled-coil domain that can drive oligomerisation-based condensation.
PML-RARA (0.926) is biologically ambiguous — PML bodies are themselves condensates.

Inputs
------
data/fusion_oncoproteins/fusion_X.npy
data/fusion_oncoproteins/fusion_labels.csv

Outputs
-------
results/plots/fusion_oncoprotein_scores.png
logs/fusion_results.json

Usage
-----
python 22c_fusion_oncoprotein_score.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib, requests, time, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
from Bio.SeqUtils.ProtParam import ProteinAnalysis

MODELS = os.path.expanduser("~/llps_project/models")
FUSDIR = os.path.expanduser("~/llps_project/data/fusion_oncoproteins")
PLOTS  = os.path.expanduser("~/llps_project/results/plots")
LOGS   = os.path.expanduser("~/llps_project/logs")
os.makedirs(FUSDIR, exist_ok=True)
DEVICE = torch.device("cpu")  # small dataset — CPU avoids OOM
print(f"Device: {DEVICE}")

class HybridLLPSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm_branch = nn.Sequential(nn.Linear(1280,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU())
        self.af_branch  = nn.Sequential(nn.Linear(24,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU())
        self.pc_branch  = nn.Sequential(nn.Linear(47,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU())
        self.trad_branch= nn.Sequential(nn.Linear(410,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU())
        self.fusion     = nn.Sequential(nn.Linear(224,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,xe,xa,xp,xt):
        return self.fusion(torch.cat([self.esm_branch(xe),self.af_branch(xa),self.pc_branch(xp),self.trad_branch(xt)],1)).squeeze(1)

model = HybridLLPSModel().to(DEVICE)
ckpt  = torch.load(f"{MODELS}/best_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state']); model.eval()
imputer = joblib.load("/home/container/llps_project/data/final/imputer.pkl")
scaler  = joblib.load("/home/container/llps_project/data/final/scaler.pkl")
print("Model loaded")

# ── Fusion catalog ─────────────────────────────────────────────────────────
# LLPS_STATUS: 1=condensate-forming, 0=non-forming
# Source: Boija et al. 2021 Cell + Sabari et al. 2018 Science
FUSIONS = [
    # FET-family IDR fusions → LLPS+ (IDR from EWS/FUS/TAF15 drives condensate)
    {"name":"EWS-FLI1",   "p1":"Q01844","p2":"Q01543","label":1,"cancer":"Ewing Sarcoma","category":"FET-family"},
    {"name":"EWS-ERG",    "p1":"Q01844","p2":"P11308","label":1,"cancer":"Ewing Sarcoma","category":"FET-family"},
    {"name":"EWS-ATF1",   "p1":"Q01844","p2":"P18846","label":1,"cancer":"Clear Cell Sarcoma","category":"FET-family"},
    {"name":"EWS-WT1",    "p1":"Q01844","p2":"P19544","label":1,"cancer":"DSRCT","category":"FET-family"},
    {"name":"FUS-CHOP",   "p1":"P35637","p2":"P35638","label":1,"cancer":"Myxoid Liposarcoma","category":"FET-family"},
    {"name":"FUS-ERG",    "p1":"P35637","p2":"P11308","label":1,"cancer":"AML","category":"FET-family"},
    {"name":"TAF15-NR4A3","p1":"Q92804","p2":"Q92570","label":1,"cancer":"Extraskeletal Myxoid","category":"FET-family"},
    {"name":"TAF15-CIC",  "p1":"Q92804","p2":"O94993","label":1,"cancer":"Round Cell Sarcoma","category":"FET-family"},
    # Non-FET fusions → LLPS- (both partners are structured kinases/TFs)
    {"name":"BCR-ABL1",   "p1":"P11274","p2":"P00519","label":0,"cancer":"CML","category":"Kinase"},
    {"name":"PML-RARA",   "p1":"P29590","p2":"P10276","label":0,"cancer":"APL","category":"TF"},
    {"name":"AML1-ETO",   "p1":"Q01196","p2":"Q06455","label":0,"cancer":"AML t(8;21)","category":"TF"},
    {"name":"MLL-AF4",    "p1":"Q03164","p2":"P51570","label":0,"cancer":"ALL","category":"TF"},
    {"name":"NPM1-ALK",   "p1":"P06748","p2":"Q9UM73","label":0,"cancer":"ALCL","category":"Kinase"},
]

# ── Fetch sequences ─────────────────────────────────────────────────────────
seq_cache_path = f"{FUSDIR}/seq_cache.json"
if os.path.exists(seq_cache_path):
    with open(seq_cache_path) as f:
        seq_cache = json.load(f)
    print(f"Loaded {len(seq_cache)} cached sequences")
else:
    seq_cache = {}

all_uid = set(f["p1"] for f in FUSIONS) | set(f["p2"] for f in FUSIONS)
for uid in sorted(all_uid):
    if uid in seq_cache: continue
    try:
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.fasta", timeout=20)
        if r.status_code == 200:
            lines = r.text.strip().split('\n')
            seq_cache[uid] = ''.join(lines[1:])
            print(f"  {uid}: {len(seq_cache[uid])} aa")
    except Exception as e:
        print(f"  {uid}: FAILED ({e})")
    time.sleep(0.2)

with open(seq_cache_path,'w') as f:
    json.dump(seq_cache, f)

# Build fusion sequences
records = []
for fus in FUSIONS:
    s1 = seq_cache.get(fus["p1"],"")
    s2 = seq_cache.get(fus["p2"],"")
    if not s1 or not s2:
        print(f"SKIP {fus['name']}: missing sequence")
        continue
    # Take first 400aa of each partner — standard approach
    seq = (s1[:400] + s2[:400])[:2000]
    if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in seq): continue
    records.append({**fus, "sequence": seq, "seq_len": len(seq)})
    print(f"  {fus['name']:20s} {len(seq)} aa  label={fus['label']}")

df = pd.DataFrame(records)
print(f"\nBuilt {len(df)} fusion sequences  "
      f"LLPS+={int((df['label']==1).sum())}  LLPS-={int((df['label']==0).sum())}")

# ── Feature extraction ─────────────────────────────────────────────────────
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

def compute_pc(seq):
    from collections import Counter
    STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
    seq = "".join(c for c in seq if c in STANDARD_AA)
    if len(seq) < 10: return None
    try:
        pa = ProteinAnalysis(seq)
        ap = pa.amino_acids_percent
        n  = len(seq)
        aaf= [ap.get(a,0.) for a in "ACDEFGHIKLMNPQRSTVWY"]  # 20
        fp = (seq.count("K")+seq.count("R"))/n
        fn = (seq.count("D")+seq.count("E"))/n
        fc = fp + fn  # frac_charged — the missing 47th feature
        w=5; kv=[abs(sum(1 for c in seq[i:i+w] if c in "KR")/w-
                     sum(1 for c in seq[i:i+w] if c in "DE")/w)
                 for i in range(n-w+1)]
        kappa=float(np.mean(kv)) if kv else 0.
        omega=sum(1 for i in range(n-1) if seq[i] in "KRDE" and seq[i+1] in "YFW")/max(n-1,1)
        ap2=[i for i,c in enumerate(seq) if c in "YFW"]
        if len(ap2)>=2:
            g=np.diff(ap2); am,as_,ao=float(g.mean()),float(g.std()),float(((g>=2)&(g<=6)).mean())
        else: am=as_=ao=0.
        hp="".join("H" if c in "VILMFYW" else "." for c in seq)
        pat=[p for p in hp.split(".") if len(p)>=3]
        ct=Counter(seq)
        pr=np.array([ct[a]/n for a in "ACDEFGHIKLMNPQRSTVWY" if ct.get(a,0)>0])
        ent=float(-np.sum(pr*np.log2(pr+1e-10)))
        return (aaf
            + [pa.molecular_weight(), pa.isoelectric_point(), pa.gravy(),
               pa.aromaticity(), pa.instability_index()]          # 5  → 25
            + [sum(seq.count(a) for a in "YFW")/n,               # frac_aromatic
               sum(seq.count(a) for a in "VILMFYW")/n,           # frac_hydrophobic
               fp, fn,                                             # frac_pos, frac_neg
               sum(seq.count(a) for a in "STNQ")/n,              # frac_polar
               (seq.count("Q")+seq.count("N"))/n,                # frac_QN
               (seq.count("P")+seq.count("G"))/n,                # frac_PG
               seq.count("Y")/n,                                  # frac_Y
               fc]                                                 # frac_charged (9th) → 34
            + [fp+fn, fp-fn, kappa, omega]                        # FCR,NCPR,kappa,omega → 38
            + [max((len(r) for r in "".join("P" if c in "KR" else "." for c in seq).split(".") if r),default=0)/n,
               max((len(r) for r in "".join("N" if c in "DE" else "." for c in seq).split(".") if r),default=0)/n]  # 40
            + [am, as_, ao]                                        # aro stats → 43
            + [len(pat)/n,
               max((len(p) for p in pat),default=0)/n,
               float(np.mean([{"V":4.2,"I":4.5,"L":3.8,"M":1.9,"F":2.8,
                               "Y":-1.3,"W":-0.9}.get(c,0.) for c in seq]))]  # 46
            + [ent])                                               # 47 total
    except: return None

def compute_dip(seq):
    aa='ACDEFGHIKLMNPQRSTVWY'; n=len(seq)-1
    if n<1: return None
    dp={a+b:0 for a in aa for b in aa}
    for i in range(len(seq)-1):
        p=seq[i:i+2]
        if p in dp: dp[p]+=1
    df_=[dp[a+b]/n for a in aa for b in aa]
    grp={c:(1 if c in 'RKDEQN' else 2 if c in 'GASTPHY' else 3) for c in aa}
    L=len(seq)
    cc=[sum(1 for c in seq if grp.get(c)==g)/L for g in [1,2,3]]
    tr=sum(1 for i in range(L-1) if grp.get(seq[i])!=grp.get(seq[i+1]))/max(L-1,1)
    dist=[]
    for g in [1,2,3]:
        pos=[i/L for i,c in enumerate(seq) if grp.get(c)==g]
        dist+=[pos[0] if pos else 0.,pos[-1] if pos else 0.]
    return df_+cc+[tr]+dist

# Extract PC and dipeptide
seqs   = df['sequence'].tolist()
pc_arr = []; dip_arr = []; valid_idx = []
for i, seq in enumerate(seqs):
    pc=compute_pc(seq); dip=compute_dip(seq)
    if pc and dip:
        pc_arr.append(pc); dip_arr.append(dip); valid_idx.append(i)

df = df.iloc[valid_idx].reset_index(drop=True)
seqs = [seqs[i] for i in valid_idx]
print(f"After feature extraction: {len(df)} valid fusions")

# ESM-2
print("Extracting ESM-2 embeddings...")
try:
    import esm
    em, alph = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alph.get_batch_converter(); em.eval()
    # CPU only for small datasets
    esm_arr=[]
    for i in range(0, len(seqs), 4):
        batch=seqs[i:i+4]
        data=[("p",s[:1022]) for s in batch]
        _,_,tokens=bc(data)
        # CPU only for small datasets
        with torch.no_grad():
            reps=em(tokens,repr_layers=[33])["representations"][33]
        for j,s in enumerate(batch):
            L=min(len(s),1022)
            esm_arr.append(reps[j,1:L+1].mean(0).cpu().numpy())
    esm_mat=np.array(esm_arr,dtype=np.float32)
    print(f"ESM-2 done: {esm_mat.shape}")
except Exception as e:
    print(f"ESM-2 failed: {e} — using zeros")
    esm_mat=np.zeros((len(seqs),1280),dtype=np.float32)

# Combine features
af_mat  = np.full((len(seqs),24), np.nan, dtype=np.float32)  # AlphaFold: NaN→imputed
pc_np_  = np.array(pc_arr,dtype=np.float32)
dip_np_ = np.array(dip_arr,dtype=np.float32)
pad_    = np.zeros((pc_np_.shape[0],1),dtype=np.float32)
X_raw   = np.hstack([esm_mat, af_mat, pc_np_, pad_, dip_np_])

# Fix: pc block has 46 features, need 47. Add padding between pc and dip.
if X_raw.shape[1] == 1760:
    col_mid = 1280 + 24 + 46  # = 1350, insert after pc block
    X_raw = np.hstack([X_raw[:, :col_mid],
                       np.zeros((X_raw.shape[0], 1), dtype=np.float32),
                       X_raw[:, col_mid:]])
X_final = scaler.transform(imputer.transform(X_raw)).astype(np.float32)
y       = df['label'].values.astype(np.float32)

# Score
with torch.no_grad():
    xb = torch.from_numpy(X_final).to(DEVICE)
    scores = model(xb[:,:1280],xb[:,1280:1304],xb[:,1304:1351],xb[:,1351:]).cpu().numpy()

df['llps_score'] = scores

# Results
print(f"\n{'='*60}\nFUSION ONCOPROTEIN RESULTS\n{'='*60}")
print(f"{'Fusion':<22} {'Label':<6} {'Cancer':<24} {'Category':<12} {'Score':.4}")
print("-"*70)
for _, row in df.sort_values('llps_score', ascending=False).iterrows():
    mark = '✓' if (row['llps_score']>=0.5)==row['label'] else '✗'
    print(f"  {row['name']:<20} {int(row['label']):<6} {row['cancer']:<24} "
          f"{row['category']:<12} {row['llps_score']:.4f} {mark}")

pos_scores = scores[y==1]; neg_scores = scores[y==0]
print(f"\nLLPS+ fusions: mean={pos_scores.mean():.3f} ± {pos_scores.std():.3f}")
print(f"LLPS- fusions: mean={neg_scores.mean():.3f} ± {neg_scores.std():.3f}")

if len(np.unique(y)) > 1:
    auc_v = roc_auc_score(y, scores)
    ap_v  = average_precision_score(y, scores)
    print(f"AUC={auc_v:.4f}  AP={ap_v:.4f}")
else:
    auc_v = ap_v = None

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Fusion Oncoprotein LLPS Analysis\n"
             "FET-family fusions (EWS/FUS/TAF15 IDR) vs non-FET fusions",
             fontsize=12, fontweight='bold')

COLORS_F = {'FET-family':'#E74C3C','Kinase':'#3498DB','TF':'#2ECC71'}

# Panel 1: Score by fusion name
ax = axes[0]
df_sort = df.sort_values('llps_score', ascending=True)
bar_colors = [('#E74C3C' if l==1 else '#3498DB') for l in df_sort['label']]
bars = ax.barh(range(len(df_sort)), df_sort['llps_score'],
               color=bar_colors, alpha=0.85, edgecolor='white')
ax.axvline(0.5, color='black', linestyle='--', alpha=0.7, label='Threshold=0.5')
ax.set_yticks(range(len(df_sort)))
ax.set_yticklabels(df_sort['name'], fontsize=9)
ax.set_xlabel("Predicted LLPS Score"); ax.set_xlim(0, 1.05)
ax.set_title("LLPS Scores — All Fusions\n"
             "(red=condensate-forming, blue=non-forming)")
ax.grid(alpha=0.3, axis='x')
for bar, score in zip(bars, df_sort['llps_score']):
    ax.text(min(score+0.02, 0.98), bar.get_y()+bar.get_height()/2.,
            f"{score:.3f}", va='center', fontsize=8, fontweight='bold')

# Panel 2: Box plot by category
ax = axes[1]
cat_data = {}
for cat in df['category'].unique():
    cat_data[cat] = scores[df['category']==cat]
positions = range(len(cat_data))
bp = ax.boxplot(list(cat_data.values()), positions=list(positions),
                patch_artist=True, medianprops={'color':'black','lw':2})
for patch, cat in zip(bp['boxes'], cat_data.keys()):
    patch.set_facecolor(COLORS_F.get(cat,'grey')); patch.set_alpha(0.8)
ax.axhline(0.5, color='black', linestyle='--', alpha=0.7)
ax.set_xticks(list(positions))
ax.set_xticklabels(list(cat_data.keys()), fontsize=10)
ax.set_ylabel("LLPS Score"); ax.set_ylim(-0.05, 1.1)
ax.set_title("Score Distribution by Fusion Category"); ax.grid(alpha=0.3, axis='y')

# Panel 3: Cancer type annotation
ax = axes[2]
df_plot = df.copy()
df_plot['color'] = df_plot['label'].map({1:'salmon', 0:'steelblue'})
for _, row in df_plot.iterrows():
    ax.scatter(row['llps_score'], 0.5,
               color=row['color'], s=200, alpha=0.8, zorder=3)
    ax.text(row['llps_score'], 0.5 + (0.15 if row['label']==1 else -0.15),
            row['name'].split('-')[0], ha='center', fontsize=7,
            color='darkred' if row['label']==1 else 'darkblue')
ax.axvline(0.5, color='black', linestyle='--', alpha=0.7)
ax.set_xlabel("Predicted LLPS Score")
ax.set_yticks([])
ax.set_xlim(-0.05, 1.1)
ax.set_title("LLPS Score Dot Plot\n(red=LLPS+, blue=LLPS-)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='salmon',label='Condensate-forming (FET fusions)'),
                   Patch(color='steelblue',label='Non-condensate (kinase/TF fusions)')],
          fontsize=8, loc='upper left')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f"{PLOTS}/fusion_oncoprotein_scores.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\nSaved → results/plots/fusion_oncoprotein_scores.png")

df.to_csv(f"{FUSDIR}/fusion_scores.csv", index=False)
res = {"n_fusions": int(len(df)),
       "n_llps_pos": int((y==1).sum()), "n_llps_neg": int((y==0).sum()),
       "mean_llps_pos": round(float(pos_scores.mean()),4),
       "mean_llps_neg": round(float(neg_scores.mean()),4),
       "auc": float(auc_v) if auc_v else None,
       "individual": df[['name','label','llps_score','cancer','category']].to_dict('records')}
with open(f"{LOGS}/fusion_results.json","w") as f:
    json.dump(res, f, indent=2)
print(f"Saved → logs/fusion_results.json")
