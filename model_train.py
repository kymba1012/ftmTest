"""
Train a lightweight Naive Bayes model to predict the `error` column in
20251128_FtData.csv.

Usage:
    python3 model_train.py

This script:
- Reads the CSV file with pandas.
- Splits the data into train/test (80/20).
- Trains a mixed Naive Bayes model (categorical + Gaussian numeric).
- Reports accuracy, precision, recall, and example predictions.
- Persists the trained model to naive_bayes_model.pkl.
"""

from __future__ import annotations

import functools
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = Path("20251128_FtData.csv")
MODEL_PATH = Path("naive_bayes_model.pkl")


# --- CSV loader ------------------------------------------------------------- #
def load_table(path: Path) -> Tuple[List[str], List[List[Optional[str]]]]:
    """Return headers and row values (all as strings/None) from the CSV."""
    return _load_table_cached(path)


@functools.lru_cache(maxsize=1)
def _load_table_cached(path: Path) -> Tuple[List[str], List[List[Optional[str]]]]:
    """Cached CSV reader to avoid re-parsing on repeated training/tuning runs."""
    df = pd.read_csv(path, dtype=str)
    df = df.where(pd.notna(df), None)
    headers = list(df.columns)
    body: List[List[Optional[str]]] = df.values.tolist()
    return headers, body


def _normalize_strings(X):
    """Strip and lowercase string-like inputs; preserve shape for pipelines."""
    if isinstance(X, pd.DataFrame):
        X_out = X.copy()
        for col in X_out.columns:
            X_out[col] = (
                X_out[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
        return X_out
    # Assume ndarray-like; convert to DataFrame for consistent ops
    X_df = pd.DataFrame(X)
    return _normalize_strings(X_df)


def save_model(model: "NaiveBayes", path: Path = MODEL_PATH) -> None:
    if model.pipeline is None:
        raise RuntimeError("Model not fitted; cannot save.")
    joblib.dump(
        {
            "pipeline": model.pipeline,
            "feature_columns": model.feature_columns,
            "feature_names_": model.feature_names_,
            "numeric_fields": model.numeric_fields,
            "alpha": model.alpha,
            "variance_floor": model.variance_floor,
            "class_weight": model.class_weight,
        },
        path,
    )


def load_model(path: Path = MODEL_PATH) -> "NaiveBayes":
    payload = joblib.load(path)
    nb = NaiveBayes(
        payload["numeric_fields"],
        alpha=payload.get("alpha", 1.0),
        variance_floor=payload.get("variance_floor", 1e-9),
        class_weight=payload.get("class_weight", "balanced"),
    )
    nb.pipeline = payload["pipeline"]
    nb.feature_columns = payload["feature_columns"]
    nb.feature_names_ = payload.get("feature_names_", [])
    return nb


# --- Feature selection & model ---------------------------------------------- #
NUMERIC_FIELDS = {
    "submitted",
    "employee_id",
    "rt_units",
    "rt_charge_rate",
    "rt error",
    "ot_units",
    "ot_charge_rate",
    "ot error",
    "at_units",
    "at_charge_rate",
    "at error",
    "tt_units",
    "tt_charge_rate",
    "tt error",
    "sub_rate",
    "sub_charge_rate",
    "sub error",
    # Equipment fields
    "equipment_units",
    "equipment_charge_rate",
    "equip_total",
    "equip_rate_error",
    "equip_id_error",
    "truck_units_error",
    "equip_units error",
    "equip_error",
}



class NaiveBayes:
    """Wrapper around scikit-learn GaussianNB with OneHotEncoder for categoricals."""

    def __init__(self, numeric_fields: Iterable[str], alpha: float = 1.0, variance_floor: float = 1e-9, class_weight: Optional[str] = "balanced"):
        self.numeric_fields = set(numeric_fields)
        self.alpha = alpha
        self.variance_floor = variance_floor
        self.class_weight = class_weight
        self.pipeline: Optional[Pipeline] = None
        self.feature_columns: List[str] = []
        self.feature_names_: List[str] = []

    def _build_pipeline(self, df: pd.DataFrame) -> Pipeline:
        num_cols = [c for c in df.columns if c in self.numeric_fields]
        cat_cols = [c for c in df.columns if c not in self.numeric_fields]

        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "normalize",
                    FunctionTransformer(
                        _normalize_strings,
                        feature_names_out="one-to-one",
                        validate=False,
                    ),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        )

        return Pipeline(
            [
                ("prep", preprocessor),
                ("nb", GaussianNB(var_smoothing=self.variance_floor)),
            ]
        )

    def fit(self, rows: List[Dict[str, Optional[str]]], labels: List[int]) -> None:
        assert len(rows) == len(labels)
        df = pd.DataFrame(rows)
        if "error" in df.columns:
            df = df.drop(columns=["error"])

        # Coerce numerics once up-front; categoricals stay as strings for the pipeline
        for col in self.numeric_fields:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Track expected columns for later prediction alignment
        self.feature_columns = list(df.columns)

        self.pipeline = self._build_pipeline(df)

        sample_weight = None
        if self.class_weight is not None:
            sample_weight = compute_sample_weight(class_weight=self.class_weight, y=labels)

        fit_params = {}
        if sample_weight is not None:
            fit_params["nb__sample_weight"] = sample_weight

        self.pipeline.fit(df, labels, **fit_params)

        # Capture feature names for downstream inspection/debugging
        prep = self.pipeline.named_steps["prep"]
        num_cols = prep.transformers_[0][2]
        cat_cols = prep.transformers_[1][2]
        encoder: OneHotEncoder = prep.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(encoder.get_feature_names_out(cat_cols))
        self.feature_names_ = list(num_cols) + cat_names

    def _ensure_dataframe(self, rows: List[Dict[str, Optional[str]]]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        # Add missing expected columns with None
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = None
        # Coerce numeric columns
        for col in self.numeric_fields:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Keep only expected columns in the original order
        df = df[self.feature_columns]
        return df

    def predict_proba(self, row: Dict[str, Optional[str]]) -> float:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted")
        df = self._ensure_dataframe([row])
        prob = self.pipeline.predict_proba(df)[0, 1]
        return float(prob)

    def predict(self, row: Dict[str, Optional[str]], threshold: float = 0.5) -> int:
        return 1 if self.predict_proba(row) >= threshold else 0

    def predict_batch(self, rows: List[Dict[str, Optional[str]]], threshold: float = 0.5) -> List[int]:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted")
        df = self._ensure_dataframe(rows)
        probs = self.pipeline.predict_proba(df)[:, 1]
        return [1 if p >= threshold else 0 for p in probs]


# --- Pipeline ---------------------------------------------------------------- #
def build_dataset(headers: List[str], rows: List[List[Optional[str]]]):
    data = []
    labels = []
    error_idx = headers.index("error")
    for vals in rows:
        row_dict = {h: (vals[i] if i < len(vals) else None) for i, h in enumerate(headers)}
        label_raw = row_dict.get("error", "0")
        label = 1 if str(label_raw).strip() == "1" else 0
        labels.append(label)
        data.append(row_dict)
    return data, labels


def train_and_evaluate():
    print("Loading data...")
    headers, body = load_table(DATA_PATH)
    data, labels = build_dataset(headers, body)
    total = len(labels)
    pos = sum(labels)
    print(f"Total rows: {total}, positive errors: {pos} ({pos/total:.4%})")

    # Train/test split
    random.seed(42)
    indices = list(range(total))
    random.shuffle(indices)
    split = int(0.8 * total)
    train_idx, test_idx = indices[:split], indices[split:]

    train_rows = [data[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_rows = [data[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    model = NaiveBayes(NUMERIC_FIELDS)
    model.fit(train_rows, train_labels)
    save_model(model)

    # Evaluate
    preds = [model.predict(r, threshold=0.5) for r in test_rows]
    probs = [model.predict_proba(r) for r in test_rows]
    correct = sum(int(p == y) for p, y in zip(preds, test_labels))
    accuracy = correct / len(test_labels)
    tp = sum(1 for p, y in zip(preds, test_labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, test_labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, test_labels) if p == 0 and y == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (error=1): {precision:.4f}")
    print(f"Recall    (error=1): {recall:.4f}")
    print("Confusion matrix (pred rows x true cols):")
    print(f"  TP: {tp}  FP: {fp}")
    print(f"  FN: {fn}  TN: {len(test_labels) - tp - fp - fn}")

    print(f"\nSaved trained model to {MODEL_PATH}")

    # Show a few high-probability error cases
    ranked = sorted(
        zip(test_rows, test_labels, probs, preds),
        key=lambda x: x[2],
        reverse=True,
    )[:5]
    print("\nTop 5 predicted error=1 rows (prob, true_label, title, job, manager):")
    for row, y, p, pred in ranked:
        print(
            f"  p={p:.3f} pred={pred} true={y} "
            f"title={row.get('title')} job={row.get('job')} manager={row.get('manager')}"
        )


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise SystemExit(f"Data file not found at {DATA_PATH}")
    train_and_evaluate()
