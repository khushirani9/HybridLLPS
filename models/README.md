# Model Weights

Pre-trained HybridLLPS model weights are available on Zenodo.

## Files required

Download and place in this directory:

- best_model.pt      trained model checkpoint (~11 MB)
- imputer.pkl        fitted SimpleImputer (1761 features)
- scaler.pkl         fitted StandardScaler (1761 features)

Also available:
- platt_scaler.pkl   post-hoc Platt calibration scaler

## Zenodo DOI

[DOI will be added after upload]

## Quick usage

    python predict.py --fasta your_proteins.fasta
    python predict.py --sequence MSEYIRVTED...

## Model performance

- Test AUROC:               0.9273 (95% CI: 0.894-0.955)
- Independent validation:   0.9760 (62 proteins, zero training overlap)
- 5-fold cross-validation:  0.931 +/- 0.008
- Calibration ECE:          0.0333 (after Platt scaling)
