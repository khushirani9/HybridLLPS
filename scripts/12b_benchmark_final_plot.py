"""
12B BENCHMARK FINAL PLOT
======================================================================
Final Benchmark Comparison Plot Generation
======================================================================

Generates the final publication-ready benchmark comparison figure.
Reads pre-computed tool scores and HybridLLPS predictions and produces
a 3-panel figure: ROC curves, score distributions, and AUC bar chart.

Inputs
------
results/benchmarking/*.csv
logs/statistical_tests.json

Outputs
-------
results/plots/final_benchmark_comparison.png

Usage
-----
python 12b_benchmark_final_plot.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_DIR = os.path.expanduser("~/llps_project/results/plots/")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: AUC comparison across scenarios
scenarios = [
    "Our test\n(all 407)",
    "Our test\n(190 short)",
    "Phaseek test\n(all 459)",
    "Phaseek test\n(297 short)"
]
our_aucs     = [0.9273, 0.8091, 0.7717, 0.6721]
phaseek_aucs = [None,   0.6168, None,   0.9985]

x     = np.arange(len(scenarios))
width = 0.35
ax    = axes[0]

bars1 = ax.bar(x - width/2, our_aucs, width,
               label="Our model", color="#2196F3", alpha=0.85,
               edgecolor="black", linewidth=0.8)
phaseek_vals = [v if v is not None else 0 for v in phaseek_aucs]
phaseek_colors = ["#FF9800" if v is not None else "#CCCCCC"
                  for v in phaseek_aucs]
bars2 = ax.bar(x + width/2, phaseek_vals, width,
               label="Phaseek", color=phaseek_colors, alpha=0.85,
               edgecolor="black", linewidth=0.8)

# Add value labels
for bar, val in zip(bars1, our_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
for bar, val in zip(bars2, phaseek_aucs):
    if val is not None:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                "N/A\n(length\nlimit)", ha="center", fontsize=7,
                color="gray")

ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=9)
ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_title("Model Comparison Across Test Scenarios\nGray = Phaseek cannot score these proteins", fontsize=11)
ax.set_ylim(0, 1.1)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Plot 2: Coverage comparison
ax2 = axes[1]
categories = ["Our test set\n(407 proteins)", "Phaseek test set\n(459 proteins)"]
our_coverage     = [100, 100]
phaseek_coverage = [190/407*100, 297/459*100]

x2    = np.arange(len(categories))
bars3 = ax2.bar(x2 - width/2, our_coverage, width,
                label="Our model", color="#2196F3", alpha=0.85,
                edgecolor="black", linewidth=0.8)
bars4 = ax2.bar(x2 + width/2, phaseek_coverage, width,
                label="Phaseek", color="#FF9800", alpha=0.85,
                edgecolor="black", linewidth=0.8)

for bar, val in zip(bars3, our_coverage):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")
for bar, val in zip(bars4, phaseek_coverage):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")

ax2.set_xticks(x2)
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylabel("Proteins Scored (%)", fontsize=12)
ax2.set_title("Sequence Coverage\nPhaseek limited to ~300 residues max", fontsize=11)
ax2.set_ylim(0, 120)
ax2.legend(fontsize=10)
ax2.grid(axis="y", alpha=0.3)

plt.suptitle("Our Model vs Phaseek: Performance and Coverage Comparison",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(PLOT_DIR, "final_benchmark_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
