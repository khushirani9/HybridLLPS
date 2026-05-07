"""
01 PARSE DATASET
======================================================================
Dataset Parsing and Label Assignment — PPMClab
======================================================================

Reads the PPMClab LLPS database (TSV format) and assigns binary LLPS labels.

Positive label (1) is assigned to proteins with any confirmed LLPS evidence:
  DE (Driver Exclusive), CE (Client Exclusive), C_D (both), D+, C+
Negative label (0) is assigned to proteins with only negative evidence:
  NP (non-phase-separating, ordered), ND (non-phase-separating, disordered)

Filters applied:
  - Sequence length: 50 to 2,000 amino acids
  - Standard amino acids only (removes B, Z, X, U, O)
  - Conflicting labels resolved conservatively: any strong positive overrides negative

Inputs
------
data/raw/datasets.tsv  (PPMClab database — download from https://github.com/PPMC-lab/llps-datasets)

Outputs
-------
data/processed/dataset_labeled.csv  (columns: uniprot_id, sequence, label)
data/processed/all_proteins.fasta  (input for CD-HIT in script 02)

Usage
-----
python 01_parse_dataset.py
"""

import pandas as pd
import re
import os

# ── paths ──────────────────────────────────────────────────────────────────────
RAW      = os.path.expanduser("~/llps_project/data/raw/datasets.tsv")
OUT_ALL  = os.path.expanduser("~/llps_project/data/processed/dataset_labeled.csv")
OUT_FASTA= os.path.expanduser("~/llps_project/data/processed/all_proteins.fasta")
os.makedirs(os.path.expanduser("~/llps_project/data/processed"), exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW, sep='\t')
print(f"Loaded {len(df)} proteins")
print(f"Columns: {df.columns.tolist()}")

# ── labeling logic ─────────────────────────────────────────────────────────────
# Strong positive tags — if ANY of these present, protein is POSITIVE
POSITIVE_TAGS = {'DE', 'CE', 'C_D', 'D+', 'C+'}
# Pure negative tags — only negative if NO positive tag present
NEGATIVE_TAGS = {'NP', 'ND', 'D-', 'C-'}

def assign_label(dataset_str):
    """
    Priority rule:
      - Any strong positive tag present → POSITIVE (label=1)
        (even if C- or D- also present from another database)
      - Only negative tags present → NEGATIVE (label=0)
      - Nothing recognizable → DROP (label=-1)
    """
    tags = set(str(dataset_str).split(';'))

    has_strong_positive = bool(tags & POSITIVE_TAGS)

    if has_strong_positive:
        return 1  # positive wins — confirmed by at least one trusted source
    
    has_negative = bool(tags & NEGATIVE_TAGS)
    if has_negative:
        return 0  # purely negative
    
    return -1  # unrecognized → drop
df['label'] = df['Datasets'].apply(assign_label)

# ── report before filtering ────────────────────────────────────────────────────
print(f"\nBefore filtering:")
print(f"  Positive (label=1) : {(df['label']==1).sum()}")
print(f"  Negative (label=0) : {(df['label']==0).sum()}")
print(f"  Conflicted (label=-1): {(df['label']==-1).sum()}")

# ── drop conflicted entries ────────────────────────────────────────────────────
df = df[df['label'] != -1].copy()
print(f"\nAfter dropping conflicts: {len(df)} proteins remain")

# ── drop rows with missing sequence ───────────────────────────────────────────
df = df[df['Full.seq'].notna() & (df['Full.seq'].str.strip() != '')]
print(f"After dropping missing sequences: {len(df)} proteins")

# ── sequence length filter (50–2000 residues) ──────────────────────────────────
df['seq_len'] = df['Full.seq'].str.len()
before = len(df)
df = df[(df['seq_len'] >= 50) & (df['seq_len'] <= 2000)]
print(f"After length filter (50-2000 aa): {len(df)} proteins (dropped {before - len(df)})")

# ── remove non-standard amino acids ───────────────────────────────────────────
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

def is_standard(seq):
    return set(seq.upper()) <= STANDARD_AA

before = len(df)
df = df[df['Full.seq'].apply(is_standard)]
print(f"After removing non-standard AA: {len(df)} proteins (dropped {before - len(df)})")

# ── final label counts ─────────────────────────────────────────────────────────
print(f"\nFinal dataset:")
print(f"  Positive (LLPS)     : {(df['label']==1).sum()}")
print(f"  Negative (non-LLPS) : {(df['label']==0).sum()}")
print(f"  Total               : {len(df)}")

# ── save CSV ───────────────────────────────────────────────────────────────────
df_out = df[['UniProt.Acc', 'Gene.Name', 'Datasets', 'label',
             'Frac.Order', 'Frac.Disorder', 'seq_len', 'Full.seq']].copy()
df_out.columns = ['uniprot_id', 'gene_name', 'datasets', 'label',
                  'frac_order', 'frac_disorder', 'seq_len', 'sequence']
df_out.to_csv(OUT_ALL, index=False)
print(f"\nSaved CSV → {OUT_ALL}")

# ── save FASTA ─────────────────────────────────────────────────────────────────
# FASTA format is needed for CD-HIT (redundancy removal in next step)
# Format: >UniProtID|label
# e.g.:   >O95613|1
with open(OUT_FASTA, 'w') as f:
    for _, row in df_out.iterrows():
        f.write(f">{row['uniprot_id']}|{row['label']}\n")
        # write sequence in 60-character lines (standard FASTA format)
        seq = row['sequence']
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + '\n')
print(f"Saved FASTA → {OUT_FASTA}")

