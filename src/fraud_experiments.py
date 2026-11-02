"""Reproducible credit-card fraud experiments.

Place the Kaggle ULB dataset at data/creditcard.csv, then run:
    python src/fraud_experiments.py

All resampling is applied to training data only. Test metrics use a held-out,
stratified split and a fixed 0.5 probability threshold.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50


def load_data() -> pd.DataFrame:
    """Load and validate the Kaggle dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\nDownload it from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place creditcard.csv inside data/."
        )
    frame = pd.read_csv(DATA_PATH)
    required = {"Class", "Amount", "Time"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if frame["Class"].isna().any() or not set(frame["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only binary values 0 and 1.")
    return frame.drop_duplicates().reset_index(drop=True)


def save_eda_plots(frame: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = frame["Class"].value_counts().reindex([0, 1], fill_value=0)
    plt.figure(figsize=(6, 4))
    plt.bar(["Legitimate", "Fraud"], counts.values)
    plt.title("Class distribution")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    frame.boxplot(column="Amount", by="Class", showfliers=False)
    plt.title("Transaction amount by class")
    plt.suptitle("")
    plt.xlabel("Class (0 = legitimate, 1 = fraud)")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "amount_by_class.png", dpi=160)
    plt.close()


def evaluate(name: str, model, x_train, x_test, y_train, y_test):
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    scores = model.predict_proba(x_test)[:, 1]
    predictions = (scores >= THRESHOLD).astype(int)
    row = {
        "model": name,
        "threshold": THRESHOLD,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "pr_auc": average_precision_score(y_test, scores),
        "fit_seconds": fit_seconds,
    }
    return row, scores


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_data()
    save_eda_plots(frame)

    y = frame["Class"].astype(int)
    x = frame.drop(columns=["Class"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Fit scaling on training data only. LR benefits from scaling; tree models
    # use the original feature values.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Resampling only occurs on the training split.
    smote = SMOTE(random_state=RANDOM_STATE)
    x_smote, y_smote = smote.fit_resample(x_train_scaled, y_train)
    undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
    x_under, y_under = undersampler.fit_resample(x_train_scaled, y_train)

    # Keep the full holdout for evaluation while fitting RF on a reproducible
    # stratified 60k subset of training rows to keep resource use practical.
    x_rf, _, y_rf, _ = train_test_split(
        x_train, y_train, train_size=60000, stratify=y_train, random_state=RANDOM_STATE
    )

    experiments = [
        ("Logistic Regression (baseline)",
         LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
         x_train_scaled, y_train, x_test_scaled),
        ("Logistic Regression (SMOTE)",
         LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
         x_smote, y_smote, x_test_scaled),
        ("Logistic Regression (class weights)",
         LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
         x_train_scaled, y_train, x_test_scaled),
        ("Logistic Regression (undersampling)",
         LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
         x_under, y_under, x_test_scaled),
        ("Random Forest",
         RandomForestClassifier(
             n_estimators=35, min_samples_leaf=2, class_weight="balanced_subsample",
             n_jobs=-1, random_state=RANDOM_STATE
         ),
         x_rf, y_rf, x_test),
        ("XGBoost",
         XGBClassifier(
             n_estimators=100, max_depth=4, learning_rate=0.05,
             subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
             scale_pos_weight=int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1),
             eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
         ),
         x_train, y_train, x_test),
    ]

    rows, scores_by_model = [], {}
    for name, model, train_features, train_labels, test_features in experiments:
        row, scores = evaluate(name, model, train_features, test_features, train_labels, y_test)
        rows.append(row)
        scores_by_model[name] = scores

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    results.to_csv(RESULTS_DIR / "actual_metrics.csv", index=False)
    results.to_csv(RESULTS_DIR / "results.csv", index=False)

    # Precision-recall and ROC curves.
    plt.figure(figsize=(8, 6))
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=160)
    plt.savefig(FIGURES_DIR / "pr_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, scores in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=160)
    plt.close()

    metric_columns = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    chart = results.set_index("model")[metric_columns]
    chart.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Score")
    plt.title("Model comparison (held-out test set)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
    plt.close()

    best_name = str(results.iloc[0]["model"])
    best_predictions = (scores_by_model[best_name] >= THRESHOLD).astype(int)
    matrix = confusion_matrix(y_test, best_predictions, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.title(f"Confusion matrix — {best_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for (i, j), value in np.ndenumerate(matrix):
        plt.text(j, i, str(value), ha="center", va="center")
    plt.xticks([0, 1], ["Legitimate", "Fraud"])
    plt.yticks([0, 1], ["Legitimate", "Fraud"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    summary = [
        "# Experiment Results", "",
        f"- Rows after duplicate removal: {len(frame):,}",
        f"- Fraud cases: {int(y.sum()):,} ({y.mean():.4%})",
        f"- Train/test rows: {len(y_train):,} / {len(y_test):,}",
        f"- Test split: stratified {TEST_SIZE:.0%}; random seed: {RANDOM_STATE}",
        f"- Probability threshold: {THRESHOLD:.2f}",
        f"- Best model by test PR-AUC: **{best_name}**", "",
        results.to_markdown(index=False, floatfmt=".4f"), "",
        "PR-AUC (average precision) is emphasized because the class distribution is highly imbalanced. "
        "The fixed 0.5 threshold is a baseline; threshold selection should be performed using validation "
        "data and the business costs of false positives and false negatives.",
    ]
    (RESULTS_DIR / "results.md").write_text("\n".join(summary), encoding="utf-8")
    info = {
        "dataset": "data/creditcard.csv",
        "rows_after_duplicate_removal": int(len(frame)),
        "fraud_count": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
        "resampling_scope": "training split only",\n        "random_forest_training_subset": 60000,\n        "random_forest_estimators": 35,\n        "xgboost_estimators": 100,
        "models": [name for name, *_ in experiments],
    }
    (RESULTS_DIR / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
