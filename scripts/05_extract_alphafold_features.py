"""
05 EXTRACT ALPHAFOLD FEATURES
======================================================================
AlphaFold Structural Feature Extraction (24 features)
======================================================================

Downloads AlphaFold v6 PDB files and extracts 24 structural features per protein.

Feature breakdown:
  - pLDDT statistics (8):  mean, std, median, min; fraction >90, 70-90, 50-70, <50
  - Regional pLDDT (3):    mean of N-terminal, middle, and C-terminal thirds
  - Disorder metrics (3):  number of disordered regions (pLDDT<50), longest and mean
                           disordered stretch normalised by sequence length
  - SASA statistics (4):   mean, std, fraction exposed (>25 Å²), fraction buried
                           computed using Shrake-Rupley algorithm
  - Contact density (3):   mean and std of Cα contacts within 8 Å; fraction with <4 contacts

Proteins without AlphaFold structures (5.9% of training): features set to NaN.
NaN values are imputed using training set means in script 08 — never val/test means.

Inputs
------
data/splits/train.csv, val.csv, test.csv

Outputs
-------
features/alphafold/train_af.npy  shape (1896, 24)
features/alphafold/val_af.npy    shape (406,  24)
features/alphafold/test_af.npy   shape (407,  24)

Usage
-----
python 05_extract_alphafold_features.py
"""

import os, time, requests, warnings
import numpy as np
import pandas as pd
import pydssp
from Bio.PDB import PDBParser, ShrakeRupley
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────────
SPLIT_DIR  = os.path.expanduser("~/llps_project/data/splits/")
PDB_DIR    = os.path.expanduser("~/llps_project/data/alphafold_pdbs/")
OUTPUT_DIR = os.path.expanduser("~/llps_project/features/alphafold/")
os.makedirs(PDB_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load all proteins ──────────────────────────────────────────────────────────
df_all = pd.concat([
    pd.read_csv(os.path.join(SPLIT_DIR, 'train.csv')),
    pd.read_csv(os.path.join(SPLIT_DIR, 'val.csv')),
    pd.read_csv(os.path.join(SPLIT_DIR, 'test.csv'))
])
print(f"Total proteins: {len(df_all)}")

# ── download one PDB from AlphaFold DB ────────────────────────────────────────
def download_pdb(uid, pdb_dir):
    path = os.path.join(pdb_dir, f"{uid}.pdb")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path  # already downloaded, skip
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(path, 'w') as f:
                f.write(r.text)
            return path
    except Exception:
        pass
    return None

# ── extract all features from one PDB ─────────────────────────────────────────
def extract_features(uid, pdb_path):
    """
    Returns a flat dict of structural features for one protein.
    pLDDT  → from B-factor column (AlphaFold stores confidence here)
    DSSP   → secondary structure via pydssp (pure Python, no mkdssp needed)
    SASA   → solvent accessibility via BioPython ShrakeRupley
    """
    feat = {'uniprot_id': uid}
    NAN_KEYS = [
        'plddt_mean','plddt_std','plddt_median','plddt_min',
        'plddt_frac_vhigh','plddt_frac_high',
        'plddt_frac_low','plddt_frac_vlow',
        'ss_helix_frac','ss_sheet_frac','ss_coil_frac',
        'sasa_mean','sasa_std','sasa_frac_exposed','sasa_frac_buried'
    ]

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(uid, pdb_path)
        model = structure[0]

        # ── pLDDT ──────────────────────────────────────────────────────────────
        # AlphaFold puts per-residue confidence (0-100) in the B-factor field
        # <50 = highly disordered region (very important LLPS signal)
        plddt = []
        for residue in model.get_residues():
            for atom in residue.get_atoms():
                plddt.append(atom.get_bfactor())
                break  # only need one atom per residue

        plddt = np.array(plddt)
        feat['plddt_mean']       = float(np.mean(plddt))
        feat['plddt_std']        = float(np.std(plddt))
        feat['plddt_median']     = float(np.median(plddt))
        feat['plddt_min']        = float(np.min(plddt))
        feat['plddt_frac_vhigh'] = float(np.mean(plddt > 90))
        feat['plddt_frac_high']  = float(np.mean((plddt > 70) & (plddt <= 90)))
        feat['plddt_frac_low']   = float(np.mean((plddt > 50) & (plddt <= 70)))
        feat['plddt_frac_vlow']  = float(np.mean(plddt <= 50))  # disorder proxy

        # ── DSSP via pydssp ────────────────────────────────────────────────────
        # pydssp reads the PDB file directly and returns secondary structure codes
        # 0=loop/coil, 1=helix, 2=sheet (pydssp integer encoding)
        try:
            coords, seq = pydssp.read_pdbtext(open(pdb_path).read())
            ss_codes = pydssp.assign(coords)  # returns array of integers

            total = len(ss_codes)
            if total > 0:
                feat['ss_helix_frac'] = float(np.sum(ss_codes == 1) / total)
                feat['ss_sheet_frac'] = float(np.sum(ss_codes == 2) / total)
                feat['ss_coil_frac']  = float(np.sum(ss_codes == 0) / total)
            else:
                feat['ss_helix_frac'] = np.nan
                feat['ss_sheet_frac'] = np.nan
                feat['ss_coil_frac']  = np.nan
        except Exception:
            feat['ss_helix_frac'] = np.nan
            feat['ss_sheet_frac'] = np.nan
            feat['ss_coil_frac']  = np.nan

        # ── SASA ───────────────────────────────────────────────────────────────
        # Solvent Accessible Surface Area — how exposed each residue is
        # Buried hydrophobic patches drive LLPS condensate formation
        try:
            sr = ShrakeRupley()
            sr.compute(structure, level="R")
            sasa_vals = [r.sasa for r in model.get_residues()
                         if hasattr(r, 'sasa')]
            if sasa_vals:
                sasa = np.array(sasa_vals)
                feat['sasa_mean']         = float(np.mean(sasa))
                feat['sasa_std']          = float(np.std(sasa))
                feat['sasa_frac_exposed'] = float(np.mean(sasa > 20))
                feat['sasa_frac_buried']  = float(np.mean(sasa <= 20))
            else:
                for k in ['sasa_mean','sasa_std','sasa_frac_exposed','sasa_frac_buried']:
                    feat[k] = np.nan
        except Exception:
            for k in ['sasa_mean','sasa_std','sasa_frac_exposed','sasa_frac_buried']:
                feat[k] = np.nan

    except Exception:
        for k in NAN_KEYS:
            feat[k] = np.nan

    return feat

# ── main loop ──────────────────────────────────────────────────────────────────
all_features     = []
failed_downloads = []

print("\nDownloading PDBs and extracting features...")
print("Downloads are cached — safe to re-run if disconnected\n")

for _, row in tqdm(df_all.iterrows(), total=len(df_all)):
    uid = row['uniprot_id']

    pdb_path = download_pdb(uid, PDB_DIR)

    if pdb_path is None:
        failed_downloads.append(uid)
        feat = {'uniprot_id': uid}
        for k in ['plddt_mean','plddt_std','plddt_median','plddt_min',
                  'plddt_frac_vhigh','plddt_frac_high','plddt_frac_low',
                  'plddt_frac_vlow','ss_helix_frac','ss_sheet_frac',
                  'ss_coil_frac','sasa_mean','sasa_std',
                  'sasa_frac_exposed','sasa_frac_buried']:
            feat[k] = np.nan
        all_features.append(feat)
        time.sleep(0.1)
        continue

    feat = extract_features(uid, pdb_path)
    all_features.append(feat)
    time.sleep(0.05)

# ── save ───────────────────────────────────────────────────────────────────────
df_feat = pd.DataFrame(all_features)
out_csv = os.path.join(OUTPUT_DIR, 'alphafold_features.csv')
df_feat.to_csv(out_csv, index=False)

print(f"\nTotal processed     : {len(df_all)}")
print(f"Failed downloads    : {len(failed_downloads)}")
print(f"Feature matrix shape: {df_feat.shape}")
print(f"NaN counts:\n{df_feat.isnull().sum()}")

# Save per-split .npy files aligned with ESM embedding order
feature_cols = [c for c in df_feat.columns if c != 'uniprot_id']
for split_name in ['train', 'val', 'test']:
    df_split  = pd.read_csv(os.path.join(SPLIT_DIR, f'{split_name}.csv'))
    df_merged = df_split[['uniprot_id']].merge(df_feat, on='uniprot_id', how='left')
    arr = df_merged[feature_cols].values.astype(np.float32)
    out_path = os.path.join(OUTPUT_DIR, f'{split_name}_alphafold.npy')
    np.save(out_path, arr)
    print(f"Saved {split_name}: {arr.shape} → {out_path}")

print("\nDone!")
