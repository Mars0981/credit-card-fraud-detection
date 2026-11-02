# Experiment Results

Results from the included full-data experiment after duplicate removal. PR-AUC (average precision) is the primary metric because fraud cases are rare. Thresholds were selected from out-of-fold predictions on the training split; the test split remained held out.

| Model | CV PR-AUC mean | CV PR-AUC std | Test PR-AUC | Test ROC-AUC | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.853 | 0.026 | 0.826 | 0.979 | 0.987 | 0.789 | 0.877 |
| Random Forest | 0.800 | 0.030 | 0.782 | 0.965 | 0.866 | 0.747 | 0.802 |
| LR baseline | 0.753 | 0.031 | 0.695 | 0.958 | 0.835 | 0.747 | 0.789 |
| LR + SMOTE | 0.752 | 0.027 | 0.673 | 0.960 | 0.857 | 0.758 | 0.804 |
| LR + class weights | 0.754 | 0.026 | 0.672 | 0.966 | 0.857 | 0.758 | 0.804 |
| LR + undersampling | 0.602 | 0.116 | 0.600 | 0.955 | 0.658 | 0.768 | 0.709 |

**Best model in this run:** XGBoost, with test PR-AUC 0.8263 and F1 0.8772. Results are a snapshot from one experimental setup and may vary with data and environment.