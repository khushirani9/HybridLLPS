"""
08 COMBINE AND SCALE FEATURES
======================================================================
Feature Matrix Assembly, Imputation, and Standardisation
======================================================================

Combines the four feature arrays into one matrix and applies preprocessing.

Concatenation order (must be identical at all future inference):
  ESM-2 (1280) + AlphaFold (24) + Physicochemical (47) + Dipeptide (410) = 1,761 total

Preprocessing (fitted on training set only — never on val or test):
  SimpleImputer:    replaces NaN values with training set column means
                    (NaNs come from proteins without AlphaFold structures)
  StandardScaler:   scales each feature to mean=0, std=1 using training statistics

The fitted imputer.pkl and scaler.pkl must be used identically for any future
inference. Using different preprocessing would cause silent feature misalignment.

Inputs
------
features/esm/*.npy, features/alphafold/*.npy, features/physicochemical/*.npy, features/traditional/*.npy

Outputs
-------
data/final/train_X.npy, val_X.npy, test_X.npy  (scaled, shape: n×1761)
data/final/train_y.npy, val_y.npy, test_y.npy  (binary labels)
data/final/imputer.pkl  (keep this — needed for all inference)
data/final/scaler.pkl   (keep this — needed for all inference)

Usage
-----
python 08_combine_and_scale_features.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

SPLIT_DIR   = os.path.expanduser("~/llps_project/data/splits/")
FEAT_DIR    = os.path.expanduser("~/llps_project/features/")
OUTPUT_DIR  = os.path.expanduser("~/llps_project/data/final/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load all feature blocks ────────────────────────────────────────────────────
print("Loading feature blocks...")
splits = {}
for split_name in ['train', 'val', 'test']:
    esm  = np.load(os.path.join(FEAT_DIR, f'esm/{split_name}_esm.npy'))
    af   = np.load(os.path.join(FEAT_DIR, f'alphafold/{split_name}_alphafold.npy'))
    pc   = np.load(os.path.join(FEAT_DIR, f'physicochemical/{split_name}_physicochemical.npy'))
    trad = np.load(os.path.join(FEAT_DIR, f'traditional/{split_name}_traditional.npy'))
    df   = pd.read_csv(os.path.join(SPLIT_DIR, f'{split_name}.csv'))

    # Concatenate all features horizontally
    # axis=1 means we're adding columns (features), not rows (proteins)
    X = np.concatenate([esm, af, pc, trad], axis=1)
    y = df['label'].values

    splits[split_name] = {
        'X': X,
        'y': y,
        'ids': df['uniprot_id'].values
    }
    print(f"  {split_name}: X={X.shape}, y={y.shape}, "
          f"pos={y.sum()}, neg={(y==0).sum()}")

print(f"\nTotal features: {splits['train']['X'].shape[1]}")
print(f"  ESM-2        : 1280")
print(f"  AlphaFold    :   24")
print(f"  Physicochemical: 47")
print(f"  Traditional  :  410")
print(f"  ─────────────────")
print(f"  Total        : 1761")

# ── Step 1: Impute NaN values ──────────────────────────────────────────────────
# SimpleImputer replaces NaN with column mean
# CRITICAL: fit ONLY on training data, then apply to val and test
# If we fit on all data, we leak test information into the imputer
print("\nImputing NaN values (mean imputation)...")
print("Fitting imputer on TRAIN only — no data leakage")

nan_before = np.isnan(splits['train']['X']).sum()
print(f"NaN values in train before imputation: {nan_before}")

imputer = SimpleImputer(strategy='mean')
splits['train']['X'] = imputer.fit_transform(splits['train']['X'])
splits['val']['X']   = imputer.transform(splits['val']['X'])
splits['test']['X']  = imputer.transform(splits['test']['X'])

nan_after = np.isnan(splits['train']['X']).sum()
print(f"NaN values in train after imputation : {nan_after} ✓")

# ── Step 2: Normalize features ────────────────────────────────────────────────
# StandardScaler: for each feature, subtract mean and divide by std
# Result: every feature has mean=0 and std=1
# CRITICAL: fit ONLY on training data
print("\nNormalizing features (StandardScaler)...")
print("Fitting scaler on TRAIN only — no data leakage")

scaler = StandardScaler()
splits['train']['X'] = scaler.fit_transform(splits['train']['X'])
splits['val']['X']   = scaler.transform(splits['val']['X'])
splits['test']['X']  = scaler.transform(splits['test']['X'])

print(f"Train feature mean after scaling: {splits['train']['X'].mean():.6f} (should be ~0)")
print(f"Train feature std after scaling : {splits['train']['X'].std():.6f}  (should be ~1)")

# ── Step 3: Save final matrices ────────────────────────────────────────────────
print("\nSaving final feature matrices...")
for split_name in ['train', 'val', 'test']:
    X = splits[split_name]['X'].astype(np.float32)
    y = splits[split_name]['y'].astype(np.int64)
    ids = splits[split_name]['ids']

    np.save(os.path.join(OUTPUT_DIR, f'{split_name}_X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, f'{split_name}_y.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, f'{split_name}_ids.npy'), ids)
    print(f"  Saved {split_name}: X={X.shape}, y={y.shape}")

# Save imputer and scaler — needed later for mutation scoring
# When we score new sequences (mutations), we must apply the SAME
# imputer and scaler that was fitted on the training data
joblib.dump(imputer, os.path.join(OUTPUT_DIR, 'imputer.pkl'))
joblib.dump(scaler,  os.path.join(OUTPUT_DIR, 'scaler.pkl'))
print(f"\nSaved imputer → {OUTPUT_DIR}imputer.pkl")
print(f"Saved scaler  → {OUTPUT_DIR}scaler.pkl")

# ── Step 4: Final sanity check ─────────────────────────────────────────────────
print("\n" + "="*50)
print("FINAL DATASET SUMMARY")
print("="*50)
for split_name in ['train', 'val', 'test']:
    X = splits[split_name]['X']
    y = splits[split_name]['y']
    print(f"\n{split_name.upper()}:")
    print(f"  Shape    : {X.shape}")
    print(f"  Positive : {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Negative : {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")
    print(f"  NaN      : {np.isnan(X).sum()}")
    print(f"  Min/Max  : {X.min():.2f} / {X.max():.2f}")

print("\nDone! Ready for model training.")
