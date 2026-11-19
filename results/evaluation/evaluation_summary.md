# Final Holdout Test Evaluation Summary

- **Evaluated Model:** Support Vector Classifier (`svc`)
- **Pipeline Architecture:** `StandardScaler` + `SVC(C=10.0, gamma='scale', kernel='rbf', probability=True)`
- **Evaluation Timestamp:** 2026-11-16T18:58:15.298569+00:00
- **Holdout Test Partition:** 36 samples (Class 0: 12, Class 1: 14, Class 2: 10)

---

## 1. Primary Holdout Performance Metrics

| Metric | Holdout Test Score | Phase 3 5-Fold CV Mean | Generalization Note |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.9444** | 0.9862 | 34 of 36 holdout samples correctly classified (94.44%) |
| **Macro F1-Score** | **0.9407** | 0.9927 | Balanced performance across all 3 cultivars |
| **Macro Precision** | **0.9583** | 0.9865 | High precision across classes (95.83%) |
| **Macro Recall** | **0.9333** | 0.9878 | Strong recall across classes (93.33%) |
| **Weighted F1-Score** | **0.9432** | - | Frequency-weighted F1 across all samples |
| **ROC-AUC (OVR Macro)** | **1.0000** | - | Perfect multi-class ranking separation |

> [!NOTE]
> The holdout metrics report performance strictly on this 36-sample holdout test partition. ROC-AUC reflects ranking discrimination and should not be construed as calibrated probability certainty.

---

## 2. Confusion Matrix (Holdout Test Set)

| Actual \ Predicted | Predicted class_0 | Predicted class_1 | Predicted class_2 | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Actual class_0** | **12** | 0 | 0 | 12 |
| **Actual class_1** | 0 | **14** | 0 | 14 |
| **Actual class_2** | 0 | 2 | **8** | 10 |

---

## 3. Scikit-learn Classification Report

```text
              precision    recall  f1-score   support

     class_0       1.00      1.00      1.00        12
     class_1       0.88      1.00      0.93        14
     class_2       1.00      0.80      0.89        10

    accuracy                           0.94        36
   macro avg       0.96      0.93      0.94        36
weighted avg       0.95      0.94      0.94        36

```
