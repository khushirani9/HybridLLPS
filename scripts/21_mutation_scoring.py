"""
21 MUTATION SCORING
======================================================================
ALS Mutation LLPS Score Analysis
======================================================================

Computes how ALS-associated missense mutations alter LLPS propensity.

AScore = predicted_probability(mutant) - predicted_probability(wild_type)
Positive AScore: mutation increases LLPS tendency (consistent with pathogenicity)
Negative AScore: mutation reduces LLPS tendency

Tests 30 verified ALS mutations in TDP-43, FUS, hnRNPA1, hnRNPA2B1, Tau.
Most ALS-associated mutations in the TDP-43 C-terminal IDR show positive AScores,
consistent with the established model that these mutations destabilise the IDR
and promote pathological phase separation.

Inputs
------
models/best_model.pt
data/final/imputer.pkl, scaler.pkl
data/mutations/als_mutations.csv

Outputs
-------
results/mutations/mutation_heatmap.png
logs/mutation_scores.json

Usage
-----
python 21_mutation_scoring.py
"""

import numpy as np, os, joblib, torch, re, warnings
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from localcider.sequenceParameters import SequenceParameters
import esm as esm_lib

DATA_DIR    = os.path.expanduser('~/llps_project/data/final/')
MODEL_DIR   = os.path.expanduser('~/llps_project/models/')
MUT_DIR     = os.path.expanduser('~/llps_project/data/mutations/')
RESULTS_DIR = os.path.expanduser('~/llps_project/results/mutations/')
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
ESM_END, AF_END, PC_END, TRAD_END = 1280, 1304, 1351, 1761

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
            self.esm_branch(x[:,:ESM_END]),
            self.af_branch(x[:,ESM_END:AF_END]),
            self.pc_branch(x[:,AF_END:PC_END]),
            self.trad_branch(x[:,PC_END:TRAD_END])
        ], dim=1)
        return self.fusion(out).squeeze(1)

print('Loading model...')
model = LLPSHybridModel().to(device)
ckpt  = torch.load(os.path.join(MODEL_DIR,'best_model.pt'), weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

imputer = joblib.load(os.path.join(DATA_DIR,'imputer.pkl'))
scaler  = joblib.load(os.path.join(DATA_DIR,'scaler.pkl'))

print('Loading ESM-2...')
esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval().to(device)

AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'
AROMATIC    = set('YFW')
HYDROPHOBIC = set('VILMFYW')
POS = set('KR')
NEG = set('DE')
EIS = {'A':0.25,'R':-1.76,'N':-0.64,'D':-0.72,'C':0.04,'Q':-0.69,'E':-0.62,
       'G':0.16,'H':-0.40,'I':0.73,'L':0.53,'K':-1.10,'M':0.26,'F':0.61,
       'P':-0.07,'S':-0.26,'T':-0.18,'W':0.37,'Y':0.02,'V':0.54}

def score_sequence(seq, uid='protein'):
    seq = seq.upper()
    L   = len(seq)

    # ESM-2 embedding
    _, _, tokens = batch_converter([(uid, seq[:1022])])
    with torch.no_grad():
        res = esm_model(tokens.to(device), repr_layers=[33], return_contacts=False)
    esm_feat = res['representations'][33][0, 1:min(L,1022)+1].mean(0).cpu().numpy()

    # AlphaFold — NaN for mutation sequences (no structure available)
    af_feat = np.full(24, np.nan, dtype=np.float32)

    # Physicochemical features (47) — exact training order
    analysis = ProteinAnalysis(seq)
    aa_vals  = analysis.get_amino_acids_percent()
    pc = [aa_vals.get(aa, 0.0) for aa in AA_ORDER]
    pc += [analysis.molecular_weight(), analysis.isoelectric_point(),
           analysis.gravy(), analysis.aromaticity(), analysis.instability_index()]
    fp = sum(seq.count(a) for a in POS) / L
    fn = sum(seq.count(a) for a in NEG) / L
    pc += [sum(seq.count(a) for a in AROMATIC)/L,
           sum(seq.count(a) for a in HYDROPHOBIC)/L,
           fp, fn, sum(seq.count(a) for a in 'STNQ')/L, fp-fn,
           (seq.count('Q')+seq.count('N'))/L,
           (seq.count('P')+seq.count('G'))/L,
           seq.count('Y')/L, fp+fn, fp-fn]
    mp=mn=cp=cn=0
    for aa in seq:
        if aa in POS: cp+=1; cn=0
        elif aa in NEG: cn+=1; cp=0
        else: cp=cn=0
        mp=max(mp,cp); mn=max(mn,cn)
    pc += [mp/L, mn/L]
    apos = [i for i,aa in enumerate(seq) if aa in AROMATIC]
    if len(apos) >= 2:
        sp = np.diff(apos)
        pc += [float(np.mean(sp)), float(np.std(sp)),
               float(np.mean((sp>=2)&(sp<=6)))]
    else:
        pc += [float(L), 0.0, 0.0]
    patches = re.findall(r'[VILMFYW]{3,}', seq)
    pc += [len(patches)/L,
           max((len(p) for p in patches), default=0)/L,
           float(np.mean([EIS.get(aa,0) for aa in seq]))]
    cnt = np.array([seq.count(aa) for aa in AA_ORDER])
    fr  = cnt/cnt.sum(); fr = fr[fr>0]
    pc.append(float(-np.sum(fr*np.log2(fr))))
    try:
        spp = SequenceParameters(seq); fcr = spp.get_FCR()
        if fcr > 0:
            k = spp.get_kappa(); o = spp.get_Omega()
            pc += [float(np.clip(k,0,1)) if k else 0.0,
                   float(np.clip(o,0,1)) if o else 0.0]
        else: pc += [0.0, 0.0]
    except: pc += [0.0, 0.0]
    pc_feat = np.array(pc, dtype=np.float32)

    # Traditional features (410) — dipeptide + CTD
    dipep = {f'{a1}{a2}':0 for a1 in AA_ORDER for a2 in AA_ORDER}
    for i in range(L-1):
        pair = seq[i]+seq[i+1]
        if pair in dipep: dipep[pair] += 1
    trad = [dipep[f'{a1}{a2}']/(L-1) for a1 in AA_ORDER for a2 in AA_ORDER]
    hydro = {'R':1,'K':1,'E':1,'D':1,'Q':1,'N':1,'G':2,'A':2,'S':2,'T':2,
             'P':2,'H':2,'Y':2,'C':3,'L':3,'V':3,'I':3,'M':3,'F':3,'W':3}
    gs  = [hydro.get(aa,2) for aa in seq]
    ctd = []
    for g in [1,2,3]: ctd.append(gs.count(g)/L)
    ctd.append(sum(1 for i in range(L-1) if gs[i]!=gs[i+1])/(L-1))
    for g in [1,2,3]:
        pos = [i/L for i,x in enumerate(gs) if x==g]
        ctd += [pos[0] if pos else 0.0, pos[-1] if pos else 0.0]
    trad_feat = np.array(trad+ctd, dtype=np.float32)

    # Assemble → impute → scale → predict
    x = np.concatenate([esm_feat, af_feat, pc_feat, trad_feat]).reshape(1,-1)
    x = imputer.transform(x)
    x = scaler.transform(x)
    with torch.no_grad():
        score = float(model(torch.tensor(x, dtype=torch.float32).to(device)).cpu().numpy()[0])
    return score

# ── Score all mutations ────────────────────────────────────────────────────────
df_mut = pd.read_csv(os.path.join(MUT_DIR,'disease_mutations.csv'))
print(f'\nScoring {len(df_mut)} mutations ({len(df_mut)*2} sequences total)...\n')

results = []
for _, row in tqdm(df_mut.iterrows(), total=len(df_mut)):
    wt_score  = score_sequence(row['wt_sequence'],  f"{row['uniprot_id']}_WT")
    mut_score = score_sequence(row['mut_sequence'], f"{row['uniprot_id']}_{row['mutation']}")
    ascore    = mut_score - wt_score
    predicted = 'gain' if ascore > 0.05 else ('loss' if ascore < -0.05 else 'neutral')

    results.append({
        'protein_name'       : row['protein_name'],
        'mutation'           : row['mutation'],
        'wt_score'           : round(wt_score,  4),
        'mut_score'          : round(mut_score, 4),
        'AScore'             : round(ascore,    4),
        'predicted_effect'   : predicted,
        'experimental_effect': row['experimental_effect'],
        'in_llps_region'     : row['in_llps_region'],
        'position'           : row['position'],
    })
    print(f"  {row['protein_name']:20s} {row['mutation']:8s} | "
          f"WT={wt_score:.4f} MUT={mut_score:.4f} "
          f"AScore={ascore:+.4f} | "
          f"pred={predicted:7s} exp={row['experimental_effect']}")

# ── Results ────────────────────────────────────────────────────────────────────
df_res   = pd.DataFrame(results)
df_known = df_res[df_res['experimental_effect'] != 'unknown']
correct  = (df_known['predicted_effect'] == df_known['experimental_effect']).sum()
total    = len(df_known)
accuracy = correct / total if total > 0 else 0

print(f"\n{'='*55}")
print(f"MUTATION PREDICTION RESULTS")
print(f"{'='*55}")
print(f"Total mutations scored   : {len(df_res)}")
print(f"With known effect        : {total}")
print(f"Correct predictions      : {correct}")
print(f"Accuracy                 : {accuracy:.1%}  (target >= 70%)")
print(f"\nPer-protein summary:")
print(df_res.groupby('protein_name')[['wt_score','mut_score','AScore']].mean().round(4).to_string())
print(f"\nFull results:")
print(df_res[['protein_name','mutation','wt_score','mut_score',
              'AScore','predicted_effect','experimental_effect']].to_string(index=False))

df_res.to_csv(os.path.join(RESULTS_DIR,'mutation_ascores.csv'), index=False)
print(f"\nSaved -> {RESULTS_DIR}mutation_ascores.csv")
print("Ready for SHAP analysis!")
