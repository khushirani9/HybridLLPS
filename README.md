# HybridLLPS

A hybrid deep learning framework for proteome-wide prediction of liquid-liquid phase separation (LLPS).

Integrates ESM-2 protein language model embeddings, AlphaFold structural features,
physicochemical sequence composition, and dipeptide transition statistics.

## Key Results

| Benchmark | HybridLLPS | Best Competitor |
|-----------|-----------|-----------------|
| Test AUROC | 0.9273 | PSPHunter: 0.757 |
| Independent validation AUROC | 0.9760 | -- |
| 5-fold CV AUROC | 0.931 +- 0.008 | -- |
| Sequence length coverage | 100% | Phaseek: 47% |

## Installation

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Quick Start

```bash
# Predict from a FASTA file
python predict.py --fasta input.fasta --output results.csv

# Predict a single sequence
python predict.py --sequence MSEYIRVTED...

# Sliding window domain analysis
python predict.py --fasta input.fasta --sliding_window
```

## Model Weights

Download best_model.pt, imputer.pkl, scaler.pkl from Zenodo:
[DOI: will be added after upload]
Place in the models/ directory.

## Repository Structure
HybridLLPS/
├── predict.py              Main prediction script
├── requirements.txt        Dependencies
├── scripts/                Full analysis pipeline (01-25)
├── data/splits/            Train/val/test splits
├── models/                 Model weights (download from Zenodo)
└── example/                Example input and output files
## Pipeline Scripts

| Script | Description |
|--------|-------------|
| 01_parse_dataset.py | PPMClab dataset curation |
| 02_cdhit_redundancy_removal.py | CD-HIT 40% identity filter |
| 03_train_val_test_split.py | Stratified dataset splitting |
| 04_extract_esm_embeddings.py | ESM-2 650M embeddings |
| 05_extract_alphafold_features.py | AlphaFold pLDDT features |
| 06_compute_physicochemical_features.py | Kappa, omega, aromaticity etc. |
| 07_compute_dipeptide_features.py | Dipeptide + CTD features |
| 08_combine_and_scale_features.py | Feature assembly and scaling |
| 09_train_model.py | Model training |
| 10_evaluate_model.py | Test set evaluation |
| 11_integrated_gradients_attribution.py | Feature importance analysis |
| 12_benchmark_all_tools.py | Benchmarking vs PSPHunter/CatGRANULE/PLAAC/PScore |
| 13_ablation_study.py | Branch contribution analysis |
| 14_cold_shock_protein_analysis.py | CSP evolutionary gradient |
| 15_feature_selection_elbow.py | Elbow graph analysis |
| 16_architecture_comparison.py | 3-branch vs 4-branch |
| 17_cross_validation.py | 5-fold cross-validation |
| 18_statistical_tests.py | DeLong test significance |
| 19_calibration_analysis.py | Probability calibration |
| 20_folded_vs_disordered_validation.py | Structural validation |
| 21_mutation_scoring.py | ALS mutation analysis |
| 22a-c fusion oncoprotein scripts | Fusion protein analysis |
| 23_ortholog_conservation.py | Cross-species conservation |
| 24_transition_analysis.py | LLPS vs aggregation discrimination |
| 25_phasepdb_independent_validation.py | Independent validation |

## Citation

If you use HybridLLPS please cite:
[Citation will be added after publication]

## License

MIT License
