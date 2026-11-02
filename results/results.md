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

model,threshold,precision,recall,f1,roc_auc,pr_auc,fit_seconds
Random Forest,0.5,0.9354838709677419,0.6105263157894737,0.7388535031847133,0.9143906968706829,0.7304084383204384,2.9782866860000468
XGBoost,0.5,0.2191780821917808,0.8421052631578947,0.34782608695652173,0.9780640282282377,0.7176289441841218,4.064461455000128
Logistic Regression (baseline),0.5,0.8461538461538461,0.5789473684210527,0.6875,0.9560466717268892,0.6919665484964556,0.5039743389997966
Logistic Regression (SMOTE),0.5,0.053035143769968054,0.8736842105263158,0.1,0.9626184886409772,0.6750409387333339,3.8108697950001442
Logistic Regression (class weights),0.5,0.05638586956521739,0.8736842105263158,0.10593490746649649,0.9656548079701293,0.6719244530518994,4.564136225000311
Logistic Regression (undersampling),0.5,0.050425273390036454,0.8736842105263158,0.09534750143595634,0.9571076461696685,0.5895570356891764,0.03789134799990279

PR-AUC (average precision) is emphasized due to class imbalance. Threshold selection should be performed on a validation set using business costs of false positives and false negatives.
