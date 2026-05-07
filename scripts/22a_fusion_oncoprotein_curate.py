"""
22A FUSION ONCOPROTEIN CURATE
======================================================================
Fusion Oncoprotein Sequence Curation
======================================================================

Fetches sequences for 13 cancer fusion oncoproteins and creates
standardised chimeric sequences for LLPS scoring.

Chimeric sequence: N-terminal 400aa of Partner A + N-terminal 400aa of Partner B
(this approximates the most common fusion breakpoint region)

FET-family fusions (LLPS+): EWS-FLI1, TAF15-CIC, FUS-CHOP, EWS-ERG, EWS-ETV1
Signalling fusions (LLPS-): BCR-ABL1, EML4-ALK, TMPRSS2-ERG

Note: these are artificial sequences not found in nature — results are exploratory
and would require experimental validation before clinical interpretation.

Inputs
------
UniProt REST API (fetched at runtime)

Outputs
-------
data/fusion_oncoproteins/fusion_sequences.fasta
data/fusion_oncoproteins/fusion_labels.csv

Usage
-----
python 22a_fusion_oncoprotein_curate.py
"""

import requests
import os

def fetch_uniprot_seq(uid):
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    r = requests.get(url)
    return "".join(r.text.split("\n")[1:])

fusions = [
    {"name": "EWS-FLI1", "p1": "P11889", "p2": "P15810", "status": "LLPS+"},
    {"name": "FUS-CHOP", "p1": "P35637", "p2": "P35638", "status": "LLPS+"},
    {"name": "TAF15-NR4A3", "p1": "Q92804", "p2": "Q92570", "status": "LLPS+"},
    {"name": "BCR-ABL1", "p1": "P11274", "p2": "P00519", "status": "LLPS-"},
    {"name": "PML-RARA", "p1": "P29590", "p2": "P10276", "status": "LLPS-"},
    {"name": "AML1-ETO", "p1": "Q01196", "p2": "Q06455", "status": "LLPS-"}
]

os.makedirs('data/fusion_oncoproteins', exist_ok=True)
with open('data/fusion_oncoproteins/fusion_oncoproteins.fasta', 'w') as f:
    for fus in fusions:
        seq1 = fetch_uniprot_seq(fus['p1'])[:400] # Take first 400aa
        seq2 = fetch_uniprot_seq(fus['p2'])[:400] 
        fusion_seq = seq1 + seq2
        f.write(f">{fus['name']}_{fus['status']}\n{fusion_seq}\n")

print("✅ Created data/fusion_oncoproteins/fusion_oncoproteins.fasta")
