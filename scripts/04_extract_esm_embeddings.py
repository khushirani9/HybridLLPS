"""
04 EXTRACT ESM EMBEDDINGS
======================================================================
ESM-2 Protein Language Model Embedding Extraction (1,280 features)
======================================================================

Extracts 1,280-dimensional protein embeddings from ESM-2 650M
(esm2_t33_650M_UR50D) for all proteins in each data split.

How it works:
  1. Each sequence is tokenised and passed through the 33-layer transformer
  2. Residue-level representations from layer 33 are extracted
  3. Mean pooling across all residue positions gives one vector per protein
  4. Sequences longer than 1,022 residues are truncated (ESM-2 token limit)

ESM-2 model (~2.5 GB) downloads automatically from the ESM Hub on first run.
Runtime: approximately 4 hours on CPU for all 2,709 proteins.

Inputs
------
data/splits/train.csv, val.csv, test.csv

Outputs
-------
features/esm/train_esm.npy  shape (1896, 1280)
features/esm/val_esm.npy    shape (406,  1280)
features/esm/test_esm.npy   shape (407,  1280)

Usage
-----
python 04_extract_esm_embeddings.py
"""

import torch
import esm
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
SPLIT_DIR  = os.path.expanduser("~/llps_project/data/splits/")
OUTPUT_DIR = os.path.expanduser("~/llps_project/features/esm/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── device setup ──────────────────────────────────────────────────────────────
# This tells PyTorch to use GPU if available, otherwise CPU
# For 2709 proteins on RTX 2070, GPU will be ~20x faster than CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── load ESM-2 model ──────────────────────────────────────────────────────────
# This downloads ~2.5GB on first run, then caches locally
# esm2_t33_650M_UR50D = 33 layers, 650M parameters, trained on UniRef50
print("\nLoading ESM-2 model (downloads ~2.5GB on first run)...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

# Set model to evaluation mode — disables dropout layers
# We are NOT training ESM-2, just using it to extract features
model.eval()
model = model.to(device)
print("Model loaded successfully")

# ── load all proteins ──────────────────────────────────────────────────────────
# We extract embeddings for ALL proteins (train+val+test) now
# The split happens later when we load features for training
# This way we only run the expensive ESM-2 inference once
df_all = pd.concat([
    pd.read_csv(os.path.join(SPLIT_DIR, 'train.csv')),
    pd.read_csv(os.path.join(SPLIT_DIR, 'val.csv')),
    pd.read_csv(os.path.join(SPLIT_DIR, 'test.csv'))
])
print(f"\nTotal proteins to embed: {len(df_all)}")

# ── ESM-2 has a sequence length limit ─────────────────────────────────────────
# The model can handle up to 1022 residues at once (due to position embeddings)
# For longer sequences we truncate to 1022 — this covers >95% of our dataset
# since we already filtered to max 2000, but ESM has its own hard limit
MAX_LEN = 1022

# ── extraction loop ───────────────────────────────────────────────────────────
embeddings = {}  # dict: uniprot_id → numpy array of shape (1280,)

# We process ONE protein at a time to avoid VRAM overflow on RTX 2070
# Batch size of 1 is safe; we can try larger batches if it's too slow
BATCH_SIZE = 8  # process 8 proteins at once — good balance for 8GB VRAM

proteins = list(zip(df_all['uniprot_id'], df_all['sequence']))

print(f"\nExtracting embeddings (batch_size={BATCH_SIZE})...")
print("This will take ~15-20 minutes on RTX 2070\n")

# Process in batches
for i in tqdm(range(0, len(proteins), BATCH_SIZE)):
    batch = proteins[i : i + BATCH_SIZE]

    # Truncate sequences longer than MAX_LEN
    # ESM-2 adds special tokens so hard limit is 1022 residues
    batch_data = [(uid, seq[:MAX_LEN]) for uid, seq in batch]

    # batch_converter tokenizes sequences into integer tokens
    # batch_tokens shape: (batch_size, seq_len+2) — +2 for start/end tokens
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.to(device)

    # Forward pass through ESM-2
    # torch.no_grad() disables gradient calculation — saves memory, speeds up
    # repr_layers=[33] → extract from final layer (most informative)
    with torch.no_grad():
        results = model(
            batch_tokens,
            repr_layers=[33],
            return_contacts=False  # we don't need contact maps, saves memory
        )

    # results["representations"][33] shape: (batch, seq_len+2, 1280)
    # We extract per-protein mean embedding
    token_representations = results["representations"][33]

    for j, (uid, seq) in enumerate(batch_data):
        seq_len = len(seq)
        # Slice tokens 1 to seq_len+1 to exclude start [CLS] and end [EOS] tokens
        # Then mean pool across sequence length → shape (1280,)
        embedding = token_representations[j, 1:seq_len+1].mean(0)
        embeddings[uid] = embedding.cpu().numpy()  # move back to CPU, convert to numpy

    # Clear GPU cache periodically to prevent memory fragmentation
    if i % 100 == 0:
        torch.cuda.empty_cache()

print(f"\nExtracted {len(embeddings)} embeddings")
print(f"Embedding shape: {next(iter(embeddings.values())).shape}")

# ── save embeddings ────────────────────────────────────────────────────────────
# Save as .npy files — fast to load, compact storage
# One file per split for convenience during training
for split_name in ['train', 'val', 'test']:
    df_split = pd.read_csv(os.path.join(SPLIT_DIR, f'{split_name}.csv'))

    # Stack embeddings in the same order as the CSV rows
    split_embeddings = np.stack([embeddings[uid] for uid in df_split['uniprot_id']])
    split_labels     = df_split['label'].values

    out_path = os.path.join(OUTPUT_DIR, f'{split_name}_esm.npy')
    np.save(out_path, split_embeddings)
    print(f"Saved {split_name}: {split_embeddings.shape} → {out_path}")

# Also save a combined file with all IDs mapped to embeddings
np.save(os.path.join(OUTPUT_DIR, 'all_embeddings.npy'), embeddings)
print(f"\nDone! All embeddings saved to {OUTPUT_DIR}")
