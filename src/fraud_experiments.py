"""Experiments for credit-card fraud detection.

Expected input: data/creditcard.csv
Outputs: results/results.csv, results/results.md, and results/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download creditcard.csv "
            "from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the data/ directory."
        )
    frame = pd.read_csv(DATA_PATH)
    required = {"Class", "Amount", "Time"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame


def save_exploration_plots(frame: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = frame["Class"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Legitimate", "Fraud"], [counts.get(0, 0), counts.get(1, 0)])
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


def evaluate(name: str, model, x_train, x_test, y_train, y_test) -> tuple[dict, np.ndarray]:
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    row = {
        "model": name,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "pr_auc": average_precision_score(y_test, scores),
    }
    return row, scores


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_data()
    save_exploration_plots(frame)

    y = frame["Class"].astype(int)
    x = frame.drop(columns=["Class"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight=None)),
        ("Logistic Regression (balanced)", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE
        )),
    ]

    rows, pr_curves, model_scores = [], [], {}
    for name, model in models:
        use_scaled = name.startswith("Logistic")
        train_x = x_train_scaled if use_scaled else x_train
        test_x = x_test_scaled if use_scaled else x_test
        row, scores = evaluate(name, model, train_x, test_x, y_train, y_test)
        rows.append(row)
        model_scores[name] = scores
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_curves.append((name, recall, precision))

    # Optional XGBoost experiment; the dependency is listed in requirements.txt.
    try:
        from xgboost import XGBClassifier

        negatives = int((y_train == 0).sum())
        positives = int((y_train == 1).sum())
        xgb = XGBClassifier(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            scale_pos_weight=negatives / max(positives, 1),
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        row, scores = evaluate("XGBoost", xgb, x_train, x_test, y_train, y_test)
        rows.append(row)
        model_scores["XGBoost"] = scores
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_curves.append(("XGBoost", recall, precision))
    except ImportError:
        print("xgboost is not installed; skipping XGBoost.")

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    results.to_csv(RESULTS_DIR / "results.csv", index=False)

    plt.figure(figsize=(8, 6))
    for name, recall, precision in pr_curves:
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curves.png", dpi=160)
    plt.close()

    best_name = str(results.iloc[0]["model"])
    best_scores = model_scores[best_name]
    best_predictions = (best_scores >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.title(f"Confusion matrix — {best_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for (i, j), value in np.ndenumerate(matrix):
        plt.text(j, i, str(value), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    summary = [
        "# Experiment results",
        "",
        f"- Transactions after duplicate removal: {len(frame):,}",
        f"- Fraud transactions: {int(y.sum()):,}",
        f"- Train rows: {len(y_train):,}; test rows: {len(y_test):,}",
        f"- Best model by PR-AUC: **{best_name}**",
        "",
        results.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The 0.5 probability threshold is a simple baseline. In production, select "
        "a threshold using validation data and the relative costs of false positives "
        "and false negatives.",
    ]
    (RESULTS_DIR / "results.md").write_text("\n".join(summary), encoding="utf-8")
    info = {
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "threshold": 0.5,
        "dataset": "data/creditcard.csv",
    }
    (RESULTS_DIR / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
