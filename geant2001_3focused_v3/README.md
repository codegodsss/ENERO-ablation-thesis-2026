# ENERO Thesis — 3 Analysis Scripts (v3 — Bug-fixed)
# Output: ~/ENERO-Thesis/ENERO-ablation-thesis-2026/analysis_output/

## Bugs fixed vs v2
  1. Division by zero   -> mi_safe / mi_safe.clip(lower=1e-9) guard (H1, H2, H4)
  2. pearsonr NaN crash -> safe_pearsonr() wrapper checks std < 1e-10 (H1, H2, H4)
  3. Empty colormap     -> vmin/vmax from quantile(0.05/0.95) instead of hardcoded (H4 fig_h4d)
  4. Missing columns    -> cost_drl/ls/enero default fallback + feat fillna(median) (H2, H4)

## Prerequisites
  python3 geant2001_analysis_v3.py       # -> geant2001_inference_results.csv
  python3 geant2001_deep_analysis_v3.py  # -> geant2001_deep_analysis.csv
                                         #    geant2001_convergence.csv

## Run
  python3 analysis_h2_ospf_vs_enero.py
  python3 analysis_h1_link_level.py
  python3 analysis_h4_failure_mode.py

## Figures (15 total x PDF+PNG = 30 files)
  H2: fig_h2a violin | fig_h2b CI bar | fig_h2c CDF | fig_h2d time-quality | fig_h2e scatter
  H1: fig_h1a heatmap | fig_h1b scatter | fig_h1c top10 | fig_h1d regression | fig_h1e bottleneck
  H4: fig_h4a quadrant | fig_h4b feature | fig_h4c convergence | fig_h4d contrib | fig_h4e case study
