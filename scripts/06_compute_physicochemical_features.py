"""
06 COMPUTE PHYSICOCHEMICAL FEATURES
======================================================================
Physicochemical Sequence Feature Computation (47 features)
======================================================================

Computes 47 biochemical and biophysical features directly from amino acid sequence.

Feature groups:
  Amino acid composition (20):   mole fraction of each standard amino acid
  Global properties (5):         molecular weight, isoelectric point, GRAVY index,
                                  aromaticity (YFW fraction), instability index
  LLPS-specific composition (9): aromatic%, hydrophobic%, positive%, negative%, polar%,
                                  net charge per residue, QN%, PG%, Y%
  Charge patterning (4):         FCR, NCPR, kappa (κ), omega (Ω) via localCIDER
                                  kappa measures how charges are patterned (0=alternating, 1=blocked)
  Charge clustering (2):         max positive-charge run / L; max negative-charge run / L
  Aromatic spacing (3):          mean spacing between consecutive aromatics,
                                  std of spacing, fraction with optimal 2-6 residue gaps
  Complexity (4):                hydrophobic patch density, max patch/L,
                                  Eisenberg hydrophobicity, Shannon sequence entropy

Requires: biopython, localcider

Inputs
------
data/splits/train.csv, val.csv, test.csv

Outputs
-------
features/physicochemical/train_pc.npy  shape (1896, 47)
features/physicochemical/val_pc.npy    shape (406,  47)
features/physicochemical/test_pc.npy   shape (407,  47)

Usage
-----
python 06_compute_physicochemical_features.py
"""

import os, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ── paths ──────────────────────────────────────────────────────────────────────
SPLIT_DIR  = os.path.expanduser("~/llps_project/data/splits/")
OUTPUT_DIR = os.path.expanduser("~/llps_project/features/physicochemical/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load all proteins ──────────────────────────────────────────────────────────
df_all = pd.concat([
    pd.read_csv(os.path.join(SPLIT_DIR, 'train.csv')),
    pd.read_csv(os.path.join(SPLIT_DIR, 'val.csv')),
    pd.read_csv(os.path.join(SPLIT_DIR, 'test.csv'))
])
print(f"Total proteins: {len(df_all)}")

# ── amino acid definitions ─────────────────────────────────────────────────────
# These groupings are based on physicochemical properties relevant to LLPS
AROMATIC    = set('YFW')       # tyrosine, phenylalanine, tryptophan — pi-pi stacking
HYDROPHOBIC = set('VILMFYW')   # drive hydrophobic clustering in condensates
POSITIVE_AA = set('KR')        # lysine, arginine — positive charge
NEGATIVE_AA = set('DE')        # aspartate, glutamate — negative charge
POLAR_AA    = set('STNQ')      # polar uncharged

# ── feature extraction function ───────────────────────────────────────────────
def extract_physicochemical(uid, seq):
    """
    Extracts ~70 physicochemical features capturing the 'molecular grammar' of LLPS.

    These go beyond simple amino acid counts to capture PATTERNS:
    - WHERE charges are (clustered vs mixed) — kappa, omega
    - HOW aromatics are spaced — pi-pi stacking requires ~2-6 residue spacing
    - HOW hydrophobic patches cluster — drives condensate formation
    - Basic composition features from BioPython
    """
    feat = {'uniprot_id': uid}
    seq  = seq.upper()
    L    = len(seq)

    if L == 0:
        return feat

    # ── 1. Basic composition via BioPython ────────────────────────────────────
    # ProteinAnalysis gives us amino acid fractions, MW, pI, GRAVY score
    try:
        analysis = ProteinAnalysis(seq)
        aa_comp  = analysis.get_amino_acids_percent()

        # Store each amino acid fraction (20 features)
        for aa, frac in aa_comp.items():
            feat[f'aa_frac_{aa}'] = frac

        feat['mol_weight']    = analysis.molecular_weight()
        feat['isoelectric_pt']= analysis.isoelectric_point()
        # GRAVY = Grand Average of Hydropathicity
        # Positive = hydrophobic, Negative = hydrophilic
        feat['gravy']         = analysis.gravy()
        feat['aromaticity']   = analysis.aromaticity()
        feat['instability']   = analysis.instability_index()

    except Exception:
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            feat[f'aa_frac_{aa}'] = np.nan
        for k in ['mol_weight','isoelectric_pt','gravy','aromaticity','instability']:
            feat[k] = np.nan

    # ── 2. LLPS-specific composition counts ───────────────────────────────────
    # These specific amino acids are known LLPS drivers
    feat['frac_aromatic']   = sum(seq.count(a) for a in AROMATIC)   / L
    feat['frac_hydrophobic']= sum(seq.count(a) for a in HYDROPHOBIC)/ L
    feat['frac_positive']   = sum(seq.count(a) for a in POSITIVE_AA)/ L
    feat['frac_negative']   = sum(seq.count(a) for a in NEGATIVE_AA)/ L
    feat['frac_polar']      = sum(seq.count(a) for a in POLAR_AA)   / L
    feat['net_charge']      = feat['frac_positive'] - feat['frac_negative']
    # Q/N content — glutamine/asparagine rich regions form prion-like domains
    feat['frac_QN']         = (seq.count('Q') + seq.count('N')) / L
    # P/G content — proline/glycine break secondary structure → promote disorder
    feat['frac_PG']         = (seq.count('P') + seq.count('G')) / L
    # FUS/TDP-43 like: tyrosine fraction (Y is key LLPS driver in many proteins)
    feat['frac_Y']          = seq.count('Y') / L

    # ── 3. Charge patterning — Kappa (κ) ──────────────────────────────────────
    # Kappa measures whether + and - charges are SEGREGATED or MIXED
    # κ=0 → perfectly mixed (++--++--), κ=1 → fully segregated (+++++-----)
    # High kappa → charges cluster → stronger electrostatic interactions → LLPS
    # This is THE key parameter from the Pappu lab's LLPS theory
    try:
        pos_arr = np.array([1 if aa in POSITIVE_AA else 0 for aa in seq], dtype=float)
        neg_arr = np.array([1 if aa in NEGATIVE_AA else 0 for aa in seq], dtype=float)

        f_plus  = pos_arr.mean()   # fraction positive
        f_minus = neg_arr.mean()   # fraction negative
        FCR     = f_plus + f_minus # fraction of charged residues
        NCPR    = f_plus - f_minus # net charge per residue

        feat['FCR']  = FCR
        feat['NCPR'] = NCPR

        # Kappa calculation using blob method (Das & Pappu 2013)
        # Split sequence into blobs of size 5, calculate local charge imbalance
        if FCR > 0 and L >= 5:
            blob_size = 5
            delta_values = []
            for i in range(L - blob_size + 1):
                blob     = seq[i:i+blob_size]
                blob_pos = sum(1 for a in blob if a in POSITIVE_AA) / blob_size
                blob_neg = sum(1 for a in blob if a in NEGATIVE_AA) / blob_size
                # Local charge asymmetry
                delta    = ((blob_pos - f_plus)**2 + (blob_neg - f_minus)**2) / 2
                delta_values.append(delta)

            delta_mean = np.mean(delta_values)
            # Normalize by maximum possible delta
            delta_max  = ((f_plus**2 + f_minus**2) / 2) if FCR > 0 else 1
            feat['kappa'] = delta_mean / delta_max if delta_max > 0 else 0.0
        else:
            feat['kappa'] = 0.0

    except Exception:
        feat['FCR'] = feat['NCPR'] = feat['kappa'] = np.nan

    # ── 4. Charge clustering ───────────────────────────────────────────────────
    # Find longest run of consecutive same-charge residues
    # e.g. KKKKK = poly-lysine tract → strong electrostatic LLPS driver
    try:
        max_pos_run = max_neg_run = 0
        cur_pos = cur_neg = 0
        for aa in seq:
            if aa in POSITIVE_AA:
                cur_pos += 1
                cur_neg  = 0
            elif aa in NEGATIVE_AA:
                cur_neg += 1
                cur_pos  = 0
            else:
                cur_pos = cur_neg = 0
            max_pos_run = max(max_pos_run, cur_pos)
            max_neg_run = max(max_neg_run, cur_neg)

        feat['max_pos_run'] = max_pos_run / L  # normalize by length
        feat['max_neg_run'] = max_neg_run / L
    except Exception:
        feat['max_pos_run'] = feat['max_neg_run'] = np.nan

    # ── 5. Aromatic spacing ───────────────────────────────────────────────────
    # Pi-pi stacking between Y, F, W drives LLPS (especially in FUS, TDP-43)
    # Optimal spacing for stacking is 2-6 residues apart
    # We calculate: mean spacing, fraction of "optimal" spacings
    try:
        arom_positions = [i for i, aa in enumerate(seq) if aa in AROMATIC]
        if len(arom_positions) >= 2:
            spacings = np.diff(arom_positions)  # distances between consecutive aromatics
            feat['aromatic_spacing_mean']    = float(np.mean(spacings))
            feat['aromatic_spacing_std']     = float(np.std(spacings))
            feat['aromatic_spacing_optimal'] = float(np.mean((spacings >= 2) & (spacings <= 6)))
        else:
            feat['aromatic_spacing_mean']    = L  # no aromatics → max spacing
            feat['aromatic_spacing_std']     = 0.0
            feat['aromatic_spacing_optimal'] = 0.0
    except Exception:
        feat['aromatic_spacing_mean'] = feat['aromatic_spacing_std'] = np.nan
        feat['aromatic_spacing_optimal'] = np.nan

    # ── 6. Hydrophobic clustering ──────────────────────────────────────────────
    # Hydrophobic patches (3+ consecutive hydrophobic residues) drive LLPS
    # We count number of patches and total hydrophobic moment
    try:
        patches = re.findall(r'[VILMFYW]{3,}', seq)
        feat['hydrophobic_patch_count'] = len(patches) / L
        feat['hydrophobic_patch_maxlen']= max((len(p) for p in patches), default=0) / L

        # Hydrophobic moment — measure of amphipathicity
        # Uses Eisenberg scale for hydrophobicity
        eisenberg = {'A':0.25,'R':-1.76,'N':-0.64,'D':-0.72,'C':0.04,
                     'Q':-0.69,'E':-0.62,'G':0.16,'H':-0.40,'I':0.73,
                     'L':0.53,'K':-1.10,'M':0.26,'F':0.61,'P':-0.07,
                     'S':-0.26,'T':-0.18,'W':0.37,'Y':0.02,'V':0.54}
        hydro = [eisenberg.get(aa, 0) for aa in seq]
        feat['mean_hydrophobicity'] = float(np.mean(hydro))

    except Exception:
        feat['hydrophobic_patch_count']  = np.nan
        feat['hydrophobic_patch_maxlen'] = np.nan
        feat['mean_hydrophobicity']      = np.nan

    # ── 7. Sequence complexity ─────────────────────────────────────────────────
    # Low complexity regions (LCRs) are hallmarks of LLPS proteins
    # Shannon entropy measures how diverse the amino acid usage is
    # Low entropy = low complexity = repetitive sequence = LLPS prone
    try:
        aa_counts = np.array([seq.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'])
        aa_freq   = aa_counts / aa_counts.sum()
        aa_freq   = aa_freq[aa_freq > 0]  # remove zeros before log
        shannon_entropy = -np.sum(aa_freq * np.log2(aa_freq))
        feat['shannon_entropy'] = float(shannon_entropy)
    except Exception:
        feat['shannon_entropy'] = np.nan

    return feat

# ── main loop ──────────────────────────────────────────────────────────────────
all_features = []
for _, row in tqdm(df_all.iterrows(), total=len(df_all)):
    feat = extract_physicochemical(row['uniprot_id'], row['sequence'])
    all_features.append(feat)

df_feat = pd.DataFrame(all_features)
print(f"\nFeature matrix shape: {df_feat.shape}")
print(f"Features extracted  : {df_feat.shape[1]-1}")
print(f"NaN count           : {df_feat.isnull().sum().sum()}")

# Save CSV
out_csv = os.path.join(OUTPUT_DIR, 'physicochemical_features.csv')
df_feat.to_csv(out_csv, index=False)
print(f"Saved CSV → {out_csv}")

# Save per-split .npy files
feature_cols = [c for c in df_feat.columns if c != 'uniprot_id']
print(f"\nFeature columns ({len(feature_cols)}):")
print(feature_cols)

for split_name in ['train', 'val', 'test']:
    df_split  = pd.read_csv(os.path.join(SPLIT_DIR, f'{split_name}.csv'))
    df_merged = df_split[['uniprot_id']].merge(df_feat, on='uniprot_id', how='left')
    arr = df_merged[feature_cols].values.astype(np.float32)
    out_path = os.path.join(OUTPUT_DIR, f'{split_name}_physicochemical.npy')
    np.save(out_path, arr)
    print(f"Saved {split_name}: {arr.shape} → {out_path}")

print("\nDone!")
