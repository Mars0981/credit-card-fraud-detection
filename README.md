# Credit Card Fraud Detection

A machine-learning project for detecting fraudulent credit-card transactions in a highly imbalanced dataset. The project compares baseline classification with imbalance-aware approaches and tree-based models.

## Project objectives

- Explore and understand the class imbalance in transaction data.
- Establish a logistic-regression baseline.
- Evaluate class weighting, SMOTE, and random undersampling.
- Compare Random Forest and XGBoost models.
- Evaluate models with precision, recall, F1, ROC-AUC, and PR-AUC. PR-AUC and recall are especially useful when fraud is rare.

## Repository structure

```text
credit-card-fraud-detection/
├── data/                  # Place the dataset here (not committed)
├── results/
│   ├── figures/           # Generated evaluation plots
│   ├── results.csv        # Experiment metrics
│   ├── results.md         # Results summary
│   └── run_info.json      # Run metadata
├── src/
│   └── fraud_experiments.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

This project uses the well-known Credit Card Fraud Detection dataset (European cardholders, September 2013), commonly distributed through Kaggle. Download the dataset from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place the CSV at:

```text
data/creditcard.csv
```

The raw CSV is intentionally excluded from this repository because it is large. The `.gitignore` keeps local data files out of commits.

## Setup

Python 3.10+ is recommended.

```bash
git clone https://github.com/Mars0981/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run experiments

After downloading the dataset to `data/creditcard.csv`, run:

```bash
python src/fraud_experiments.py
```

Outputs are written to the `results/` directory, including metric summaries and figures.

## Evaluation approach

Fraud detection is an imbalanced binary-classification problem. Accuracy alone can be misleading because a model can achieve high accuracy by predicting the majority class. This project therefore emphasizes precision, recall, F1-score, ROC-AUC, and especially precision-recall AUC (PR-AUC).

## Current experiment snapshot

The recorded run contains 283,726 transactions after duplicate removal, including 473 fraud cases. On the held-out test set, the recorded XGBoost result is:

| Metric | Score |
|---|---:|
| PR-AUC | 0.8263 |
| ROC-AUC | 0.9791 |
| Precision | 0.9868 |
| Recall | 0.7895 |
| F1-score | 0.8772 |

These are results from the included experiment run, not a guarantee of performance on other data. See `results/results.csv` and `results/results.md` for the recorded comparisons.

## Limitations

- The dataset is historical and may not represent current fraud patterns.
- Performance depends on the train/test split, preprocessing, and decision threshold.
- Real-world deployment would require temporal validation, monitoring for data drift, cost-sensitive threshold selection, and careful protection of sensitive transaction data.

## License

No license has been added yet. Unless a license is added, standard copyright applies and reuse permissions are not explicitly granted.
