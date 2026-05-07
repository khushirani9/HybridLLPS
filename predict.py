"""
HybridLLPS Prediction Script
Usage:
    python predict.py --fasta input.fasta
    python predict.py --sequence MSEYIRVTED...
    python predict.py --fasta input.fasta --sliding_window

Requires: best_model.pt, imputer.pkl, scaler.pkl in models/ directory
Download model weights from Zenodo: [DOI to be added]
"""

import argparse, os, sys
print("HybridLLPS predictor - see README.md for full usage instructions")
print("Full predict.py implementation available in the paper repository")
print("Model weights: download from Zenodo (DOI in README)")

parser = argparse.ArgumentParser(description="HybridLLPS LLPS predictor")
parser.add_argument("--fasta", type=str, help="Input FASTA file")
parser.add_argument("--sequence", type=str, help="Single protein sequence")
parser.add_argument("--output", type=str, default="llps_predictions.csv")
parser.add_argument("--model_dir", type=str, default="models/")
parser.add_argument("--sliding_window", action="store_true")
args = parser.parse_args()

if not args.fasta and not args.sequence:
    parser.print_help()
    sys.exit(0)

print("Input received. Full inference code will be released with paper.")
print("Contact: khushirani9 on GitHub for early access.")
