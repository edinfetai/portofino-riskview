from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import DEFAULT_TICKERS, download_fred_macro, download_market_prices
from feature_engineering import build_portfolio_features, realized_future_risk_label

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)


def random_portfolio(tickers: list[str], rng: np.random.Generator) -> dict[str, float]:
    """Create a random long-only portfolio using 3 to 6 assets."""
    n_assets = int(rng.integers(3, min(6, len(tickers)) + 1))
    selected = list(rng.choice(tickers, size=n_assets, replace=False))
    weights = rng.dirichlet(np.ones(n_assets))
    return {ticker: float(weight) for ticker, weight in zip(selected, weights)}


def create_training_dataset(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    n_samples: int = 1200,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate supervised training samples from rolling historical windows."""
    rng = np.random.default_rng(seed)
    usable_tickers = [col for col in prices.columns if prices[col].notna().sum() > 300]
    dates = prices.dropna(how="all").index

    candidate_dates = dates[260:-45]
    rows = []
    labels = []

    attempts = 0
    max_attempts = n_samples * 8

    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1
        as_of_date = pd.Timestamp(rng.choice(candidate_dates))
        weights = random_portfolio(usable_tickers, rng)

        try:
            features = build_portfolio_features(prices, macro, weights, lookback_days=63, as_of_date=as_of_date)
            label = realized_future_risk_label(prices, weights, as_of_date, horizon_days=21)
        except Exception:
            continue

        rows.append(features.iloc[0].to_dict())
        labels.append(label)

    if len(rows) < 200:
        raise RuntimeError("Too few training samples could be generated.")

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="risk_class")
    return X, y


def main() -> None:
    prices = download_market_prices(DEFAULT_TICKERS, start="2018-01-01")
    macro = download_fred_macro(start="2018-01-01")

    X, y = create_training_dataset(prices, macro, n_samples=1200)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = {}
    best_name = None
    best_score = -1.0
    best_model = None

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        score = f1_score(y_test, pred, average="macro")

        results[name] = {
            "macro_f1": float(score),
            "classification_report": classification_report(y_test, pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "labels": sorted(y.unique().tolist()),
        }

        if score > best_score:
            best_name = name
            best_score = score
            best_model = clf

    metadata = {
        "best_model_name": best_name,
        "best_macro_f1": float(best_score),
        "feature_columns": list(X.columns),
        "labels": sorted(y.unique().tolist()),
        "class_distribution": y.value_counts().to_dict(),
        "all_results": results,
    }

    joblib.dump(best_model, MODEL_DIR / "risk_model.pkl")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    X.to_csv("data/processed/training_features.csv", index=False)
    y.to_csv("data/processed/training_labels.csv", index=False)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
