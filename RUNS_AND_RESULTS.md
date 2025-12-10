# Runs & Results (R2)

## Sanity Baseline: `exp_sanity10`
- Setup: YOLOv10n, 5 epochs, 512 px, GPU: T1000 (4 GB), workers=0
- Validation: 4,099 images; 3,942 instances (fire)
- Metrics: Precision 0.5996; Recall 0.5761; mAP@0.5 0.5721; mAP@0.5:0.95 0.2947
- Interpretation: Reasonable for a short run; expect gains with 50 epochs, 640 px, class balancing.

## Next Experiments
E1 Full baseline (50 epochs, 640 px) → out/metrics_summary.json, out/pr_curve.png
E2 Temporal rules ablation → out/ablation_temporal.csv
E3 OVD veto ablation → out/ablation_ovd.csv
E4 INT8 on Orin NX → out/quantization.csv
E5 Cross‑site robustness → FA/hr & TTFD; error taxonomy
