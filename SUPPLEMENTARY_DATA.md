# Supplementary Data

## Human Proteome Scan Results

- results/proteome/human_proteome_scores_gpu.csv
  Full scores for all 19,869 reviewed human proteins
  Columns: uniprot_id, gene, length, llps_score, prediction, in_training

- results/proteome/novel_llps_FINAL_CLEAN.csv
  63 novel LLPS candidates after compositional bias filtering
  Tier 1 (score >= 0.80): 14 proteins
  Tier 2 (score 0.65-0.80): 18 proteins
  Tier 3 (score 0.50-0.65): 31 proteins

## Feature Correlation Validation

- results/feature_correlation/correlation_results.json
  Results from feature correlation removal + PCA analysis
  Confirms ablation results are not correlation artifacts

## Result Logs

- logs/
  JSON files with numerical results from all analyses

## Supplementary Tables

See Supplementary_Tables_S1_S7.xlsx (available on Zenodo)
