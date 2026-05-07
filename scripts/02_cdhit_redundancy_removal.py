"""
02 CDHIT REDUNDANCY REMOVAL
======================================================================
Sequence Redundancy Removal — CD-HIT at 40% Identity
======================================================================

Runs CD-HIT to remove redundant sequences at 40% identity threshold.

Why 40%? Sequences sharing >40% identity are considered structurally similar.
If a protein in training is 95% identical to one in test, the model can effectively
memorise the training case — this is called data leakage. CD-HIT removes all but one
representative from each such cluster.

CD-HIT parameters used: -c 0.40 -n 2 -T 8 -M 16000
CD-HIT must be installed: conda install -c bioconda cd-hit

Inputs
------
data/processed/all_proteins.fasta

Outputs
-------
data/processed/cdhit_40/cdhit_40  (cluster representatives)
data/processed/dataset_clean.csv  (2,709 proteins after redundancy removal)

Usage
-----
python 02_cdhit_redundancy_removal.py
"""

import pandas as pd
import os

CDHIT_OUT = os.path.expanduser("~/llps_project/data/processed/cdhit_90")
CSV_IN    = os.path.expanduser("~/llps_project/data/processed/dataset_labeled.csv")
CSV_OUT   = os.path.expanduser("~/llps_project/data/processed/dataset_clean.csv")

# ── read surviving UniProt IDs from CD-HIT FASTA output ───────────────────────
# Each header line looks like: >O95613|1
surviving_ids = []
with open(CDHIT_OUT) as f:
    for line in f:
        if line.startswith('>'):
            # strip '>' and split on '|' to get just the UniProt ID
            uniprot_id = line.strip().lstrip('>').split('|')[0]
            surviving_ids.append(uniprot_id)

print(f"Proteins surviving CD-HIT: {len(surviving_ids)}")

# ── filter original CSV to keep only surviving proteins ───────────────────────
df = pd.read_csv(CSV_IN)
df_clean = df[df['uniprot_id'].isin(surviving_ids)].copy()

print(f"\nAfter CD-HIT filtering:")
print(f"  Positive (LLPS)     : {(df_clean['label']==1).sum()}")
print(f"  Negative (non-LLPS) : {(df_clean['label']==0).sum()}")
print(f"  Total               : {len(df_clean)}")
print(f"  Removed             : {len(df) - len(df_clean)} redundant proteins")

df_clean.to_csv(CSV_OUT, index=False)
print(f"\nSaved → {CSV_OUT}")
