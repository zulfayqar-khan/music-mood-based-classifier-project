# Test Set Evaluation Report

## Summary Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.6903 (69.03%) |
| Weighted F1 | 0.6845 |
| Macro F1 | 0.6444 |
| Macro ROC-AUC (OVR) | 0.9045 |
| 80% accuracy target met | No |

## Per-Class Classification Report

```
              precision    recall  f1-score   support

    acoustic       0.71      0.68      0.70      3150
 alternative       0.57      0.45      0.50      1800
       dance       0.69      0.81      0.74      6750
  electronic       0.73      0.71      0.72      2550
       heavy       0.66      0.61      0.64      1800
       vocal       0.76      0.45      0.56      1050

    accuracy                           0.69     17100
   macro avg       0.69      0.62      0.64     17100
weighted avg       0.69      0.69      0.68     17100

```

## Figures

- `reports/figures/confusion_matrix.png`: Normalised confusion matrix.
- `reports/figures/roc_auc_curves_*.png`: One-vs-rest ROC curves.