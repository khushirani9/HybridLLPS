# Model Weights

Pre-trained HybridLLPS model weights are available on Zenodo.

## Files required

Place these three files in this directory before running predict.py:

- best_model.pt  (trained model checkpoint, ~15 MB)
- imputer.pkl    (fitted SimpleImputer - training set means)
- scaler.pkl     (fitted StandardScaler - training set statistics)

## Download

Zenodo DOI: [will be updated after upload]

## Quick usage

    python predict.py --fasta your_proteins.fasta
    python predict.py --sequence MSEYIRVTED...
