"""
03 TRAIN VAL TEST SPLIT
======================================================================
Stratified Train / Validation / Test Split
======================================================================

Splits the filtered dataset into three non-overlapping sets.

Split ratios: 70% train / 15% validation / 15% test
Stratified sampling ensures each split has the same positive:negative ratio (~25.6%).
Random seed = 42 for reproducibility.

Why three splits?
  Train:      model parameters are optimised on this
  Validation: used during training for early stopping — model never directly trains on it
  Test:       touched exactly once at the end for final honest performance measurement

Inputs
------
data/processed/dataset_clean.csv

Outputs
-------
data/splits/train.csv  (1896 proteins: 485+, 1411-)
data/splits/val.csv    (406 proteins: 104+, 302-)
data/splits/test.csv   (407 proteins: 104+, 303-)

Usage
-----
python 03_train_val_test_split.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ── paths ──────────────────────────────────────────────────────────────────────
CSV_IN   = os.path.expanduser("~/llps_project/data/processed/dataset_clean.csv")
SPLIT_DIR = os.path.expanduser("~/llps_project/data/splits/")
os.makedirs(SPLIT_DIR, exist_ok=True)

RANDOM_SEED = 42  # fixed seed → reproducible splits every time you run this

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)
print(f"Total proteins: {len(df)}")
print(f"Positives: {(df['label']==1).sum()}")
print(f"Negatives: {(df['label']==0).sum()}")

# ── split ──────────────────────────────────────────────────────────────────────
# Step 1: split off 15% as test set
# stratify=df['label'] ensures equal positive/negative ratio in each split
df_train_val, df_test = train_test_split(
    df,
    test_size=0.15,
    random_state=RANDOM_SEED,
    stratify=df['label']
)

# Step 2: split remaining 85% into train (70% of total) and val (15% of total)
# 15/85 = 0.176 gives us 15% of the total dataset as validation
df_train, df_val = train_test_split(
    df_train_val,
    test_size=0.176,
    random_state=RANDOM_SEED,
    stratify=df_train_val['label']
)

# ── report ─────────────────────────────────────────────────────────────────────
for name, split in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    pos = (split['label']==1).sum()
    neg = (split['label']==0).sum()
    print(f"\n{name} set: {len(split)} proteins")
    print(f"  Positive: {pos} ({pos/len(split)*100:.1f}%)")
    print(f"  Negative: {neg} ({neg/len(split)*100:.1f}%)")

# ── save ───────────────────────────────────────────────────────────────────────
df_train.to_csv(os.path.join(SPLIT_DIR, 'train.csv'), index=False)
df_val.to_csv(os.path.join(SPLIT_DIR,   'val.csv'),   index=False)
df_test.to_csv(os.path.join(SPLIT_DIR,  'test.csv'),  index=False)

print(f"\nSaved splits to {SPLIT_DIR}")
print("  train.csv / val.csv / test.csv")

# ── sanity check: no overlap between splits ────────────────────────────────────
# This is critical — if any protein appears in both train and test,
# your model has "seen" the test data and results are invalid
train_ids = set(df_train['uniprot_id'])
val_ids   = set(df_val['uniprot_id'])
test_ids  = set(df_test['uniprot_id'])

assert len(train_ids & test_ids) == 0,  "OVERLAP between train and test!"
assert len(train_ids & val_ids)  == 0,  "OVERLAP between train and val!"
assert len(val_ids   & test_ids) == 0,  "OVERLAP between val and test!"
print("\nSanity check passed — zero overlap between splits")
