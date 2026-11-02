# Experiment Results

## Dataset and protocol
- Dataset: ULB Credit Card Fraud Detection, Kaggle.
- Duplicate removal: 1,081 rows removed from the 284,807-row input.
- Dataset after deduplication: **283,726 transactions**, including **473 fraud cases** (0.1667%).
- Stratified train/test split: **226,980 / 56,746**; seed 42.
- All resampling is training-only. Probability threshold: 0.50.
- Random Forest trained on a reproducible stratified subset of 60,000 training rows to keep compute practical; test split remains full.
- Random Forest uses 35 trees; XGBoost uses 100 estimators.

## Held-out metrics
Best model by PR-AUC: **Random Forest** (0.7304). XGBoost achieves the highest ROC-AUC (0.9781) and recall at threshold 0.5 (0.8421). Precision/recall tradeoffs at this cutoff are not production threshold recommendations.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Fit (s) |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9355 | 0.6105 | 0.7389 | 0.9144 | 0.7304 | 2.978 |
| XGBoost | 0.2192 | 0.8421 | 0.3478 | 0.9781 | 0.7176 | 4.064 |
| Logistic Regression (baseline) | 0.8462 | 0.5789 | 0.6875 | 0.9560 | 0.6920 | 0.504 |
| Logistic Regression (SMOTE) | 0.0530 | 0.8737 | 0.1000 | 0.9626 | 0.6750 | 3.811 |
| Logistic Regression (class weights) | 0.0564 | 0.8737 | 0.1059 | 0.9657 | 0.6719 | 4.564 |
| Logistic Regression (undersampling) | 0.0504 | 0.8737 | 0.0953 | 0.9571 | 0.5896 | 0.038 |

PR-AUC (average precision) is emphasized due to class imbalance. Threshold selection should be performed on a validation set using business costs of false positives and false negatives.
