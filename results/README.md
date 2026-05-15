[README.md](https://github.com/user-attachments/files/27798695/README.md)
# Results

All evaluation outputs, figures, and ablation results are saved in Google Drive at:

```
MyDrive/Research Dataset/results/
├── evaluation_*.json               ← overall test set metrics
├── per_class_results_*.json        ← per-class breakdown
├── ensemble_config_FINAL_*.json    ← final hyperparameter config
├── individual_model_full_metrics_*.json
├── plots/                          ← training and evaluation plots
├── ablation/                       ← 4-variant ablation study results
├── visualizations/                 ← annotated test images
├── paper_figures/                  ← 300 DPI publication-ready figures (Fig. 1–9)
└── demo_analysis/                  ← demo image inference results
```

## Summary of Final Test-Set Results (n=151, IoU@0.50)

| Metric | Value |
|--------|-------|
| Precision | 0.8892 |
| Recall | 0.9268 |
| F1-Score | 0.9076 |
| Avg Inference | 60.4 ms (~15 FPS) |
| Species Classification Accuracy | 99.70% |

### Per-Class

| Class | F1 | Precision | Recall |
|-------|----|-----------|--------|
| raw_apple | 0.8977 | 0.8608 | 0.9379 |
| raw_plum  | 0.9147 | 0.9104 | 0.9190 |

### By Scene Type

| Scene | F1 |
|-------|----|
| Simple | 0.9176 |
| Clustered | 0.8895 |
| Occluded | **0.9371** |
