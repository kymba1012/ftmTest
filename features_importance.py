import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from model_train import DATA_PATH, NUMERIC_FIELDS, NaiveBayes, build_dataset, load_table


def rank_features(model: NaiveBayes, X: pd.DataFrame, y: np.ndarray, sample_size: int = 5000):
    # Downsample for speed
    if sample_size and len(y) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y), size=sample_size, replace=False)
        X = X.iloc[idx].copy()
        y = y[idx]

    result = permutation_importance(
        model.pipeline,
        X,
        y,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    names = model.feature_names_
    scores = result.importances_mean
    ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    return ranked


if __name__ == "__main__":
    headers, body = load_table(DATA_PATH)
    data, labels = build_dataset(headers, body)
    df = pd.DataFrame(data)
    y = np.array(labels)
    X = df.drop(columns=["error"], errors="ignore")

    model = NaiveBayes(NUMERIC_FIELDS)
    model.fit(data, labels)

    ranked = rank_features(model, X, y)

    print("Top features (permutation importance):")
    for name, score in ranked[:10]:
        print(f"{name}: {score:.6f}")
