"""
07 COMPUTE DIPEPTIDE FEATURES
======================================================================
Dipeptide Frequency and CTD Feature Computation (410 features)
======================================================================

Computes 410 sequence-based features capturing local amino acid grammar.

Dipeptide frequencies (400):
  All 20×20 consecutive amino acid pair frequencies, normalised by (L-1).
  Captures what amino acids tend to appear next to each other — a direct
  measure of local sequence context that single-residue frequencies miss.

CTD features (10):
  Residues grouped into three physicochemical classes:
    Class 1 = Polar charged:   RKDEQN
    Class 2 = Polar/special:   GASTPHY
    Class 3 = Hydrophobic:     CVLIMFW
  Composition (3):   fraction of each class
  Transition (1):    fraction of positions where class changes between neighbours
  Distribution (6):  first and last position of each class, normalised by length

Inputs
------
data/splits/train.csv, val.csv, test.csv

Outputs
-------
features/traditional/train_trad.npy  shape (1896, 410)
features/traditional/val_trad.npy    shape (406,  410)
features/traditional/test_trad.npy   shape (407,  410)

Usage
-----
python 07_compute_dipeptide_features.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

SPLIT_DIR  = os.path.expanduser("~/llps_project/data/splits/")
OUTPUT_DIR = os.path.expanduser("~/llps_project/features/traditional/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_all = pd.concat([
    pd.read_csv(os.path.join(SPLIT_DIR, 'train.csv')),
    pd.read_csv(os.path.join(SPLIT_DIR, 'val.csv')),
    pd.read_csv(os.path.join(SPLIT_DIR, 'test.csv'))
])
print(f"Total proteins: {len(df_all)}")

def extract_traditional(uid, seq):
    """
    Traditional sequence features — simple but surprisingly powerful.
    These are fast to compute and interpretable.

    Dipeptide composition: instead of just counting single amino acids,
    we count pairs of consecutive amino acids (AA, AC, AD... YY = 400 features).
    This captures local sequence context — e.g. YG (tyrosine-glycine) is a
    hallmark of FUS-like LLPS domains. Single AA composition misses this.

    Why include these if we already have ESM-2?
    ESM-2 captures deep evolutionary context but is a black box.
    Dipeptide features are interpretable and may capture patterns
    ESM-2 encodes implicitly but can't be directly interrogated.
    Having both improves both accuracy and explainability.
    """
    feat   = {'uniprot_id': uid}
    seq    = seq.upper()
    L      = len(seq)
    AAs    = 'ACDEFGHIKLMNPQRSTVWY'

    # ── 1. Dipeptide composition (400 features) ───────────────────────────────
    # Count all possible pairs of consecutive amino acids
    # Normalized by (L-1) to get fractions
    dipep_counts = {}
    for a1 in AAs:
        for a2 in AAs:
            dipep_counts[f'dp_{a1}{a2}'] = 0

    for i in range(L - 1):
        pair = seq[i] + seq[i+1]
        key  = f'dp_{pair}'
        if key in dipep_counts:
            dipep_counts[key] += 1

    # Normalize
    total_pairs = L - 1
    for key in dipep_counts:
        feat[key] = dipep_counts[key] / total_pairs if total_pairs > 0 else 0.0

    # ── 2. CTD features (Composition/Transition/Distribution) ─────────────────
    # CTD is a classic bioinformatics feature set that groups amino acids
    # into 3 classes based on a property, then computes:
    # C = fraction in each class
    # T = fraction of transitions between classes
    # D = position of first/25%/50%/75%/100% occurrence of each class
    # We use hydrophobicity grouping (most informative for LLPS)

    # Hydrophobicity groups (Dubchak et al 1995):
    # Group 1 (polar)      : R, K, E, D, Q, N
    # Group 2 (neutral)    : G, A, S, T, P, H, Y
    # Group 3 (hydrophobic): C, L, V, I, M, F, W
    hydro_groups = {
        'R':1,'K':1,'E':1,'D':1,'Q':1,'N':1,
        'G':2,'A':2,'S':2,'T':2,'P':2,'H':2,'Y':2,
        'C':3,'L':3,'V':3,'I':3,'M':3,'F':3,'W':3
    }

    # Composition: fraction of each group
    group_seq = [hydro_groups.get(aa, 2) for aa in seq]
    for g in [1, 2, 3]:
        feat[f'ctd_C{g}'] = group_seq.count(g) / L

    # Transition: fraction of positions where group changes
    transitions = sum(1 for i in range(L-1) if group_seq[i] != group_seq[i+1])
    feat['ctd_T'] = transitions / (L - 1) if L > 1 else 0.0

    # Distribution: position of first occurrence of each group (normalized)
    for g in [1, 2, 3]:
        positions = [i/L for i, x in enumerate(group_seq) if x == g]
        feat[f'ctd_D{g}_first'] = positions[0]    if positions else 0.0
        feat[f'ctd_D{g}_last']  = positions[-1]   if positions else 0.0

    return feat

# ── Main loop ──────────────────────────────────────────────────────────────────
all_features = []
for _, row in tqdm(df_all.iterrows(), total=len(df_all)):
    feat = extract_traditional(row['uniprot_id'], row['sequence'])
    all_features.append(feat)

df_feat = pd.DataFrame(all_features)
print(f"\nFeature matrix shape: {df_feat.shape}")
print(f"Features: {df_feat.shape[1]-1}")
print(f"NaN count: {df_feat.isnull().sum().sum()}")

# Save
out_csv = os.path.join(OUTPUT_DIR, 'traditional_features.csv')
df_feat.to_csv(out_csv, index=False)

feature_cols = [c for c in df_feat.columns if c != 'uniprot_id']
for split_name in ['train', 'val', 'test']:
    df_split  = pd.read_csv(os.path.join(SPLIT_DIR, f'{split_name}.csv'))
    df_merged = df_split[['uniprot_id']].merge(df_feat, on='uniprot_id', how='left')
    arr = df_merged[feature_cols].values.astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, f'{split_name}_traditional.npy'), arr)
    print(f"Saved {split_name}: {arr.shape}")

print("\nDone!")
