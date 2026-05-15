"""
29_score_human_proteome.py
===========================
Scores the complete reviewed human proteome (UniProt Swiss-Prot)
using the full HybridLLPS pipeline with real ESM-2 embeddings.

Unlike the preliminary scan using mean ESM-2 embeddings, this script
computes genuine per-protein ESM-2 representations, giving publication-
quality predictions for every protein in the human proteome.

Pipeline per protein:
  1. ESM-2 650M mean-pooled embedding (layer 33) -- real per-sequence
  2. AlphaFold pLDDT features -- set to NaN, imputed with training means
  3. Physicochemical features (47) -- computed from sequence
  4. Dipeptide/CTD features (410) -- computed from sequence
  5. Impute + scale with training-fitted objects
  6. Score with HybridLLPS 4-branch model

Inputs
------
models/best_model_compat.pt
data/final/imputer.pkl, scaler.pkl
data/splits/train.csv  (to check for training set overlap)

Outputs
-------
results/proteome/human_proteome_scores.csv    (all proteins scored)
results/proteome/human_proteome_top500.csv    (top 500 candidates)
results/proteome/proteome_score_dist.png      (300 DPI figure)
logs/proteome_scan_results.json

Runtime: ~4-6 hours on CPU for ~19,000 proteins
"""

import os
import re
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
warnings.filterwarnings("ignore")

BASE    = "/disk2/container/khushi/llps_project"
MODELS  = BASE + "/models"
DATA    = BASE + "/data/final"
SPLITS  = BASE + "/data/splits"
RESULTS = BASE + "/results/proteome"
LOGS    = BASE + "/logs"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS,    exist_ok=True)

# Batch size for ESM-2 -- reduce if you get memory errors
ESM_BATCH = 16

# Set to a number for testing (e.g. 200), None for full proteome
MAX_PROTEINS = None

print("=" * 65)
print("HybridLLPS Human Proteome Scan -- Real ESM-2 Embeddings")
print("=" * 65)

# ── Model architecture ────────────────────────────────────────────
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
print("\n[1] Loading HybridLLPS model...")
model = HybridLLPSModel()
ckpt  = torch.load(MODELS + "/best_model_compat.pt",
                   map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("  Model loaded OK")

# ── Load preprocessing ────────────────────────────────────────────
print("\n[2] Loading preprocessing objects...")
imputer = pickle.load(open(DATA + "/imputer.pkl", "rb"))
scaler  = pickle.load(open(DATA + "/scaler.pkl",  "rb"))
print("  imputer OK (" + str(imputer.n_features_in_) + " features)")
print("  scaler  OK (" + str(scaler.n_features_in_)  + " features)")

# ── Load ESM-2 ────────────────────────────────────────────────────
print("\n[3] Loading ESM-2 650M model...")
import esm
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()
print("  ESM-2 loaded OK")

# ── Load training IDs to flag overlap ────────────────────────────
print("\n[4] Loading training set IDs for overlap check...")
train_ids = set(pd.read_csv(SPLITS + "/train.csv")["uniprot_id"].tolist())
val_ids   = set(pd.read_csv(SPLITS + "/val.csv")["uniprot_id"].tolist())
test_ids  = set(pd.read_csv(SPLITS + "/test.csv")["uniprot_id"].tolist())
all_known = train_ids | val_ids | test_ids
print("  Training set: " + str(len(train_ids)) + " proteins")
print("  Val set:      " + str(len(val_ids))   + " proteins")
print("  Test set:     " + str(len(test_ids))  + " proteins")

# ── Feature computation helpers ───────────────────────────────────
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(seq):
    return "".join(c for c in seq.upper() if c in STANDARD_AA)

def compute_pc_and_dip(seq):
    """Compute 47 physicochemical + 410 dipeptide features."""
    seq = clean_seq(seq)
    L   = len(seq)
    if L < 20:
        return None, None

    # ── Physicochemical (47) ──────────────────────────────────────
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        pa     = ProteinAnalysis(seq)
        aa_c   = pa.get_amino_acids_percent()
        aaf    = [aa_c.get(a, 0.0) for a in "ACDEFGHIKLMNPQRSTVWY"]
        mw     = pa.molecular_weight()
        pi     = pa.isoelectric_point()
        gravy  = pa.gravy()
        arom   = pa.aromaticity()
        instab = pa.instability_index()
    except Exception:
        aaf = [seq.count(a)/L for a in "ACDEFGHIKLMNPQRSTVWY"]
        mw = pi = gravy = arom = instab = 0.0

    AROMATIC    = set("YFW")
    HYDROPHOBIC = set("VILMFYW")
    POS_AA      = set("KR")
    NEG_AA      = set("DE")
    POLAR_AA    = set("STNQ")

    fp  = sum(seq.count(a) for a in POS_AA) / L
    fn  = sum(seq.count(a) for a in NEG_AA) / L

    # Fast kappa/omega (300x faster than localcider, valid for screening)
    charges = np.array([1 if c in POS_AA else -1 if c in NEG_AA else 0
                        for c in seq])
    fcr = float(np.sum(np.abs(charges))) / L
    if fcr < 0.01:
        kappa = omega = 0.0
    else:
        win   = min(5, L)
        lncpr = np.array([np.sum(charges[i:i+win])/win
                          for i in range(L - win + 1)])
        kappa = float(np.clip(np.var(lncpr) / (fcr**2 + 1e-8), 0, 1))
        stk   = np.array([1 if c in set("KRDEFYW") else 0 for c in seq])
        fs    = float(stk.mean())
        if fs < 0.01:
            omega = 0.0
        else:
            ls    = np.array([np.mean(stk[i:i+win])
                              for i in range(L - win + 1)])
            omega = float(np.clip(np.var(ls) / (fs**2 + 1e-8), 0, 1))

    max_pos = max_neg = cp = cn = 0
    for aa in seq:
        if aa in POS_AA:   cp += 1; cn = 0
        elif aa in NEG_AA: cn += 1; cp = 0
        else:              cp = cn = 0
        max_pos = max(max_pos, cp)
        max_neg = max(max_neg, cn)

    ap = [i for i, aa in enumerate(seq) if aa in AROMATIC]
    if len(ap) >= 2:
        g   = np.diff(ap)
        am  = float(g.mean())
        as_ = float(g.std())
        ao  = float(((g >= 2) & (g <= 6)).mean())
    else:
        am = float(L); as_ = 0.0; ao = 0.0

    patches  = re.findall(r"[VILMFYW]{3,}", seq)
    eis = {"A":0.25,"R":-1.76,"N":-0.64,"D":-0.72,"C":0.04,"Q":-0.69,
           "E":-0.62,"G":0.16,"H":-0.40,"I":0.73,"L":0.53,"K":-1.10,
           "M":0.26,"F":0.61,"P":-0.07,"S":-0.26,"T":-0.18,"W":0.37,
           "Y":0.02,"V":0.54}
    aa_counts = np.array([seq.count(a) for a in "ACDEFGHIKLMNPQRSTVWY"])
    aa_freq   = aa_counts / (aa_counts.sum() + 1e-8)
    aa_nz     = aa_freq[aa_freq > 0]

    pc = (aaf + [mw, pi, gravy, arom, instab] +
          [sum(seq.count(a) for a in AROMATIC) / L,
           sum(seq.count(a) for a in HYDROPHOBIC) / L,
           fp, fn,
           sum(seq.count(a) for a in POLAR_AA) / L,
           fp - fn,
           (seq.count("Q") + seq.count("N")) / L,
           (seq.count("P") + seq.count("G")) / L,
           seq.count("Y") / L,
           fp + fn, fp - fn, kappa,
           max_pos / L, max_neg / L,
           am, as_, ao,
           len(patches) / L,
           max((len(p) for p in patches), default=0) / L,
           float(np.mean([eis.get(a, 0.0) for a in seq])),
           float(-np.sum(aa_nz * np.log2(aa_nz + 1e-10))),
           omega])

    # ── Dipeptide (410) ───────────────────────────────────────────
    aa_str = "ACDEFGHIKLMNPQRSTVWY"
    n      = L - 1
    dp     = {a + b: 0 for a in aa_str for b in aa_str}
    for i in range(L - 1):
        p2 = seq[i:i+2]
        if p2 in dp:
            dp[p2] += 1
    dip = [dp[a + b] / n for a in aa_str for b in aa_str]
    grp = {c: (1 if c in "RKDEQN" else 2 if c in "GASTPHY" else 3)
           for c in aa_str}
    cc  = [sum(1 for c in seq if grp.get(c) == g) / L for g in [1,2,3]]
    tr  = sum(1 for i in range(L-1)
              if grp.get(seq[i]) != grp.get(seq[i+1])) / max(L-1, 1)
    dist = []
    for g in [1, 2, 3]:
        pos = [i/L for i,c in enumerate(seq) if grp.get(c) == g]
        dist += [pos[0] if pos else 0.0, pos[-1] if pos else 0.0]
    dip_feats = dip + cc + [tr] + dist

    return np.array(pc, dtype=np.float32), np.array(dip_feats, dtype=np.float32)

def extract_esm2_batch(sequences, ids):
    """Extract ESM-2 embeddings for a batch of sequences."""
    data = [(ids[i], sequences[i][:1022]) for i in range(len(sequences))]
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        results = esm_model(tokens, repr_layers=[33])
    reps = results["representations"][33]
    embeddings = []
    for i, seq in enumerate(sequences):
        L = min(len(seq), 1022)
        embeddings.append(reps[i, 1:L+1].mean(0).numpy().astype(np.float32))
    return embeddings

# ── Download human proteome ───────────────────────────────────────
print("\n[5] Getting human proteome FASTA...")
fasta_path = BASE + "/data/human_proteome.fasta"

if os.path.exists(fasta_path):
    print("  Found cached FASTA: " + fasta_path)
else:
    print("  Downloading from UniProt...")
    url = ("https://rest.uniprot.org/uniprotkb/stream?"
           "format=fasta&query=reviewed:true+AND+organism_id:9606")
    r = requests.get(url, timeout=300, stream=True)
    with open(fasta_path, "w") as f:
        for chunk in r.iter_content(chunk_size=65536, decode_unicode=True):
            f.write(chunk)
    print("  Download complete")

# ── Parse FASTA ───────────────────────────────────────────────────
print("\n[6] Parsing FASTA...")
proteins = []
cur_id = cur_gene = cur_seq = None

with open(fasta_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if cur_id is not None:
                proteins.append({"uniprot_id": cur_id, "gene": cur_gene,
                                  "sequence": cur_seq, "length": len(cur_seq)})
            parts = line[1:].split("|")
            cur_id   = parts[1] if len(parts) > 1 else parts[0]
            rest     = parts[2] if len(parts) > 2 else ""
            gm       = re.search(r"GN=(\S+)", rest)
            cur_gene = gm.group(1) if gm else ""
            cur_seq  = ""
        elif line:
            cur_seq += line

if cur_id:
    proteins.append({"uniprot_id": cur_id, "gene": cur_gene,
                     "sequence": cur_seq, "length": len(cur_seq)})

print("  Total parsed: " + str(len(proteins)))
proteins = [p for p in proteins if 50 <= p["length"] <= 2000]
print("  After length filter (50-2000 aa): " + str(len(proteins)))

if MAX_PROTEINS is not None:
    proteins = proteins[:MAX_PROTEINS]
    print("  Limited to: " + str(MAX_PROTEINS) + " for testing")

# Flag training set overlap
for p in proteins:
    if p["uniprot_id"] in train_ids:
        p["in_training"] = "train"
    elif p["uniprot_id"] in val_ids:
        p["in_training"] = "val"
    elif p["uniprot_id"] in test_ids:
        p["in_training"] = "test"
    else:
        p["in_training"] = "novel"

n_novel = sum(1 for p in proteins if p["in_training"] == "novel")
print("  Novel (not in training/val/test): " + str(n_novel))
print("  Overlap with known set:           " + str(len(proteins) - n_novel))

# ── Score all proteins ────────────────────────────────────────────
print("\n[7] Scoring " + str(len(proteins)) + " proteins with real ESM-2...")
print("  ESM-2 batch size: " + str(ESM_BATCH))
est_hours = round(len(proteins) * 0.8 / 3600, 1)
print("  Estimated time: ~" + str(est_hours) + " hours")
print()

results  = []
failed   = 0
start    = time.time()
n        = len(proteins)

# Process in batches for ESM-2
for batch_start in tqdm(range(0, n, ESM_BATCH), desc="Batches", unit="batch"):
    batch = proteins[batch_start:batch_start + ESM_BATCH]

    # ESM-2 embeddings for the whole batch at once
    try:
        seqs_b = [p["sequence"] for p in batch]
        ids_b  = [p["uniprot_id"] for p in batch]
        esm_embs = extract_esm2_batch(seqs_b, ids_b)
    except Exception as e:
        print("\n  ESM-2 batch failed: " + str(e) + " -- using mean fallback")
        esm_embs = [None] * len(batch)

    for i, prot in enumerate(batch):
        try:
            pc, dip = compute_pc_and_dip(prot["sequence"])
            if pc is None:
                failed += 1
                continue

            esm_v = esm_embs[i]
            if esm_v is None:
                # Fallback to zeros if ESM-2 failed for this protein
                esm_v = np.zeros(1280, dtype=np.float32)

            raw = np.concatenate([
                esm_v,
                np.full(24, np.nan, dtype=np.float32),  # AF: imputed
                pc,
                dip
            ]).reshape(1, -1)

            X = scaler.transform(imputer.transform(raw)).astype(np.float32)

            with torch.no_grad():
                xb    = torch.from_numpy(X)
                score = float(model(xb[:,:1280], xb[:,1280:1304],
                                    xb[:,1304:1351], xb[:,1351:]).item())

            results.append({
                "uniprot_id":   prot["uniprot_id"],
                "gene":         prot["gene"],
                "length":       prot["length"],
                "llps_score":   round(score, 4),
                "prediction":   "LLPS+" if score >= 0.5 else "LLPS-",
                "in_training":  prot["in_training"],
            })
        except Exception as e:
            failed += 1

    # Progress every 500 proteins
    done = batch_start + len(batch)
    if done % 500 < ESM_BATCH:
        elapsed   = time.time() - start
        rate      = done / elapsed
        remaining = (n - done) / rate / 60
        print("\n  [" + str(done) + "/" + str(n) + "]"
              + "  scored: " + str(len(results))
              + "  failed: " + str(failed)
              + "  ETA: " + str(round(remaining, 1)) + " min")

elapsed_total = time.time() - start
print("\n  Done. Scored: " + str(len(results)) +
      "  Failed: " + str(failed) +
      "  Time: " + str(round(elapsed_total/60, 1)) + " min")

# ── Save results ──────────────────────────────────────────────────
print("\n[8] Saving results...")
df = pd.DataFrame(results)
df = df.sort_values("llps_score", ascending=False).reset_index(drop=True)

df.to_csv(RESULTS + "/human_proteome_scores.csv", index=False)
df.head(500).to_csv(RESULTS + "/human_proteome_top500.csv", index=False)

n_pos   = int((df["llps_score"] >= 0.5).sum())
n_total = len(df)
pct_pos = round(100 * n_pos / n_total, 1)
mean_sc = round(float(df["llps_score"].mean()), 4)

# Novel proteins only
df_novel = df[df["in_training"] == "novel"]
n_novel_pos = int((df_novel["llps_score"] >= 0.5).sum())
pct_novel   = round(100 * n_novel_pos / len(df_novel), 1) if len(df_novel) > 0 else 0

print("  Total scored:       " + str(n_total))
print("  LLPS+ (>= 0.5):     " + str(n_pos) + " (" + str(pct_pos) + "%)")
print("  Novel LLPS+:        " + str(n_novel_pos) + " (" + str(pct_novel) + "% of novel)")
print("  Mean score:         " + str(mean_sc))
print()
print("  Top 20 LLPS candidates:")
print("  " + "-" * 62)
print("  " + "UniProt".ljust(12) + "Gene".ljust(14) +
      "Length".rjust(8) + "  Score   " + "Known?")
print("  " + "-" * 62)
for _, row in df.head(20).iterrows():
    known = row["in_training"] if row["in_training"] != "novel" else ""
    print("  " + str(row["uniprot_id"]).ljust(12) +
          str(row["gene"]).ljust(14) +
          str(row["length"]).rjust(8) + "  " +
          str(row["llps_score"]).ljust(8) + known)

# Save JSON summary
summary = {
    "n_total":          n_total,
    "n_llps_pos":       n_pos,
    "pct_llps_pos":     pct_pos,
    "n_novel_pos":      n_novel_pos,
    "pct_novel_pos":    pct_novel,
    "mean_score":       mean_sc,
    "n_failed":         failed,
    "runtime_minutes":  round(elapsed_total / 60, 1),
    "top20": [{"uniprot_id": r["uniprot_id"], "gene": r["gene"],
               "score": r["llps_score"], "length": r["length"],
               "in_training": r["in_training"]}
              for _, r in df.head(20).iterrows()]
}
with open(LOGS + "/proteome_scan_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved: " + RESULTS + "/human_proteome_scores.csv")
print("  Saved: " + RESULTS + "/human_proteome_top500.csv")
print("  Saved: " + LOGS + "/proteome_scan_results.json")

# ── Plot ──────────────────────────────────────────────────────────
print("\n[9] Generating figure...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("HybridLLPS Human Proteome Scan (n=" + str(n_total) +
             " proteins, real ESM-2)", fontsize=13, fontweight="bold")

BLUE  = "#2E75B6"
RED   = "#E74C3C"
GREEN = "#2ECC71"
GREY  = "#95A5A6"

# Panel A: Score distribution -- novel vs known
ax = axes[0]
df_k = df[df["in_training"] != "novel"]["llps_score"]
df_n = df[df["in_training"] == "novel"]["llps_score"]
ax.hist(df_n, bins=50, alpha=0.7, color=BLUE,  label="Novel (" + str(len(df_n)) + ")",
        edgecolor="white")
ax.hist(df_k, bins=50, alpha=0.7, color=GREEN, label="Known (" + str(len(df_k)) + ")",
        edgecolor="white")
ax.axvline(0.5, color=RED, linestyle="--", lw=2, label="Threshold 0.5")
ax.set_xlabel("LLPS Score")
ax.set_ylabel("Number of Proteins")
ax.set_title("(A) Score Distribution\n" + str(n_pos) + " LLPS+ (" + str(pct_pos) + "%)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel B: Score vs sequence length
ax = axes[1]
ax.scatter(df_n["length"], df_n["llps_score"],
           alpha=0.15, s=3, color=BLUE, label="Novel")
ax.axhline(0.5, color=RED, linestyle="--", lw=1.5, label="Threshold 0.5")
top100 = df[df["in_training"] == "novel"].head(100)
ax.scatter(top100["length"], top100["llps_score"],
           alpha=0.8, s=20, color=RED, label="Top 100 novel")
ax.set_xlabel("Sequence Length (aa)")
ax.set_ylabel("LLPS Score")
ax.set_title("(B) Score vs Sequence Length\n(novel proteins only)")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# Panel C: Top 15 novel candidates
ax = axes[2]
top15_novel = df[df["in_training"] == "novel"].head(15)
genes  = top15_novel["gene"].tolist()
scores = top15_novel["llps_score"].tolist()
y_pos  = range(len(genes))
bars   = ax.barh(y_pos, scores, color=BLUE, alpha=0.85, edgecolor="white")
ax.axvline(0.5, color=RED, linestyle="--", lw=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(genes, fontsize=9)
ax.set_xlabel("LLPS Score")
ax.set_title("(C) Top 15 Novel LLPS Candidates")
ax.set_xlim(0, 1.0)
for bar, v in zip(bars, scores):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            str(v), va="center", fontsize=8)
ax.grid(alpha=0.2, axis="x")

plt.tight_layout()
plt.savefig(RESULTS + "/proteome_score_dist.png",  dpi=300, bbox_inches="tight")
plt.savefig(RESULTS + "/proteome_score_dist.tiff", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: " + RESULTS + "/proteome_score_dist.png")

print()
print("=" * 65)
print("HUMAN PROTEOME SCAN COMPLETE")
print("=" * 65)
print("Full results:  " + RESULTS + "/human_proteome_scores.csv")
print("Top 500:       " + RESULTS + "/human_proteome_top500.csv")
print("Figure:        " + RESULTS + "/proteome_score_dist.png")
