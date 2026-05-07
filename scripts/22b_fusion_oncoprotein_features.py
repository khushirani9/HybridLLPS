"""
22B FUSION ONCOPROTEIN FEATURES
======================================================================
Fusion Oncoprotein Feature Extraction
======================================================================

Extracts all four feature branches for the chimeric fusion protein sequences.

AlphaFold features are set to NaN and imputed (no PDB available for artificial fusions).
ESM-2 encodes the chimeric sequence directly — mean pooling captures the blend
of LLPS propensity from both fusion partners.

Inputs
------
data/fusion_oncoproteins/fusion_sequences.fasta
models/best_model.pt, data/final/imputer.pkl, scaler.pkl

Outputs
-------
data/fusion_oncoproteins/fusion_X.npy
features/esm/fusion_esm.npy

Usage
-----
python 22b_fusion_oncoprotein_features.py
"""

# Save as scripts/27b_extract_fusion_esm.py
import torch
import esm
import numpy as np
from Bio import SeqIO
import os

print("🚀 Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval() 

fasta_path = "data/fusion_oncoproteins/fusion_oncoproteins.fasta"
output_path = "features/esm/fusion_oncoproteins.npy"
os.makedirs('features/esm', exist_ok=True)

embeddings = []
names = []

print("🧬 Extracting embeddings for fusions...")
with torch.no_grad():
    for record in SeqIO.parse(fasta_path, "fasta"):
        data = [("protein", str(record.seq))]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        
        # Mean pooling (excluding SOS/EOS tokens)
        seq_len = len(record.seq)
        mean_embedding = token_representations[0, 1 : seq_len + 1].mean(0).numpy()
        embeddings.append(mean_embedding)
        names.append(record.id)

np.save(output_path, np.array(embeddings))
np.save("features/esm/fusion_names.npy", np.array(names))
print(f"✅ Saved {len(embeddings)} embeddings to {output_path}")
