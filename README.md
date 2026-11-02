# Credit Card Fraud Detection

A reproducible machine-learning project for detecting fraudulent credit-card transactions. Six approaches are evaluated: Logistic Regression baseline, Logistic Regression with SMOTE, Logistic Regression with class weights, Logistic Regression with random undersampling, Random Forest, and XGBoost.

## Dataset
Use the ULB Credit Card Fraud Detection dataset from Kaggle. Download `creditcard.csv` and place it in `data/`. The raw CSV is excluded from version control.

## Experiment design
- Stratified 80/20 train/test split; random seed 42.
- Scaling is fit on the training split only.
- SMOTE and undersampling are applied to training data only. The test data remains unmodified.
- Evaluation metrics: precision, recall, F1, ROC-AUC and PR-AUC.
- Baseline probability cutoff: 0.50. No cross-validation or threshold tuning is claimed.
- PR-AUC is highlighted because fraud is a rare class.\n- Random Forest uses a stratified 60,000-row subset of training data (full holdout used for evaluation); it uses 35 trees and XGBoost uses 100 estimators for resource-aware execution.

## Run
```bash
python -m venv .venv
pip install -r requirements.txt
python src/fraud_experiments.py
```
Download dataset before executing. Outputs are generated in `results/`, including metrics CSV, report, run metadata and plots. Current run metrics are included in `results/results.md` and `results/actual_metrics.csv`.

## Generated outputs
`results/actual_metrics.csv`, `results/results.csv`, `results/results.md`, `results/run_info.json` and plots under `results/figures/` (class distribution, amount-by-class, PR curve, ROC curve, metric comparison and confusion matrix).

## Limitations and future work
A random split is a baseline; temporal validation is more realistic. PCA-anonymized variables limit interpretability. Consider validation-based threshold selection using fraud operational costs, SHAP explanations, FastAPI inference and monitoring.

Author: Mars0981
