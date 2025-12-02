"""
Hyperparameter and threshold tuning for the Naive Bayes error predictor.

This is a lightweight grid search using only the Python standard library.
It searches over:
- Laplace smoothing alpha for categorical features.
- Variance floor for numeric features.
- Decision threshold on predicted probability.

Outputs the best setting by F1 and prints the top few configurations. Requires pandas for CSV loading.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Dict, List, Tuple

from model_train import (
    DATA_PATH,
    NUMERIC_FIELDS,
    NaiveBayes,
    build_dataset,
    load_table,
)


def metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for p, y in zip(y_pred, y_true) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(y_pred, y_true) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(y_pred, y_true) if p == 0 and y == 1)
    tn = len(y_true) - tp - fp - fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_search() -> None:
    print("Loading data...")
    headers, body = load_table(DATA_PATH)
    data, labels = build_dataset(headers, body)
    total = len(labels)
    pos = sum(labels)
    print(f"Rows: {total}, positives: {pos} ({pos/total:.2%})")

    # Train/validation split
    random.seed(42)
    idx = list(range(total))
    random.shuffle(idx)
    split = int(0.8 * total)
    train_idx, val_idx = idx[:split], idx[split:]
    train_rows = [data[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_rows = [data[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    alphas = [0.5, 1.0, 2.0]
    var_floors = [1e-9, 1e-6, 1e-4]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results: List[Tuple[float, Dict[str, float], Tuple[float, float, float]]] = []

    for alpha, var_floor in itertools.product(alphas, var_floors):
        model = NaiveBayes(NUMERIC_FIELDS, alpha=alpha, variance_floor=var_floor)
        model.fit(train_rows, train_labels)
        probs = [model.predict_proba(r) for r in val_rows]
        for th in thresholds:
            preds = [1 if p >= th else 0 for p in probs]
            m = metrics(val_labels, preds)
            results.append((m["f1"], m, (alpha, var_floor, th)))

    results.sort(key=lambda x: x[0], reverse=True)
    best_f1, best_metrics, (b_alpha, b_var, b_th) = results[0]

    print("\nBest configuration by F1:")
    print(
        f"  alpha={b_alpha}, variance_floor={b_var}, threshold={b_th} "
        f"| F1={best_f1:.4f}, precision={best_metrics['precision']:.4f}, "
        f"recall={best_metrics['recall']:.4f}, acc={best_metrics['accuracy']:.4f}"
    )
    print(
        f"  Confusion: TP={best_metrics['tp']} FP={best_metrics['fp']} "
        f"FN={best_metrics['fn']} TN={best_metrics['tn']}"
    )

    print("\nTop 5 configs:")
    for rank, (f1, m, (a, v, th)) in enumerate(results[:5], start=1):
        print(
            f"{rank}. alpha={a} var_floor={v} th={th} | "
            f"F1={f1:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"Acc={m['accuracy']:.4f}"
        )


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise SystemExit(f"Data file not found at {DATA_PATH}")
    run_search()
