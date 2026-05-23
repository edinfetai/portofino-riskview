from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from asset_resolver import resolve_extracted_portfolio
from data_loader import (
    data_freshness_summary,
    download_fred_macro,
    download_market_prices,
)
from feature_engineering import build_portfolio_features, stress_test
from nlp_utils import extract_portfolio_from_text, generate_risk_explanation
from vision_utils import extract_portfolio_from_image

MODEL_PATH = Path("model/risk_model.pkl")
METRICS_PATH = Path("model/metrics.json")

RISK_LABELS_DE = {
    "Low": "Niedrig",
    "Medium": "Mittel",
    "High": "Hoch",
}


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Die Datei model/risk_model.pkl fehlt. Führe zuerst train.py aus."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def portfolio_list_to_weights(portfolio_items: list[dict[str, Any]]) -> dict[str, float]:
    weights: dict[str, float] = {}

    for item in portfolio_items:
        ticker = str(item["ticker"]).upper().strip()
        weight = float(item["weight"])
        weights[ticker] = weight

    return weights


def _prepare_analysis_payload(
    extracted: dict[str, Any],
    question_source: str,
) -> dict[str, Any]:
    model = load_model()
    metrics = load_metrics()

    extracted = resolve_extracted_portfolio(extracted)
    weights = portfolio_list_to_weights(extracted["portfolio"])
    tickers = sorted(set(weights.keys()) | {"SPY"})

    prices = download_market_prices(
        tickers=tickers,
        period="2y",
        interval="1d",
        use_cache=False,
    )

    macro = download_fred_macro(
        start="2019-01-01",
        use_cache=False,
    )

    downloaded_symbols = set(str(column).upper() for column in prices.columns)
    required_symbols = set(weights.keys()) | {"SPY"}
    missing_symbols = sorted(required_symbols - downloaded_symbols)

    if missing_symbols:
        raise ValueError(
            "Für folgende aufgelöste Marktsymbole konnten keine Kursdaten geladen werden: "
            + ", ".join(missing_symbols)
            + ". Bitte verwende andere handelbare Symbole oder ein klareres Bild."
        )

    freshness = data_freshness_summary(
        prices=prices,
        macro=macro,
        market_source="yfinance + FRED",
        interval="1d",
    )

    features = build_portfolio_features(
        prices=prices,
        macro=macro,
        weights=weights,
    )

    feature_columns = metrics.get("feature_columns", list(features.columns))
    features = features[feature_columns]

    risk_class = str(model.predict(features)[0])
    risk_class_de = RISK_LABELS_DE.get(risk_class, risk_class)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = [str(value) for value in model.classes_]
        risk_probabilities = {
            label: round(float(prob), 4)
            for label, prob in zip(classes, probabilities)
        }
    else:
        risk_probabilities = {risk_class: 1.0}

    scenario = extracted.get("stress_scenario", {}) or {}
    shock_by_ticker = scenario.get("shock_by_ticker", {}) or {}
    stress_result = None

    if shock_by_ticker:
        clean_shocks = {
            str(ticker).upper(): float(shock)
            for ticker, shock in shock_by_ticker.items()
        }
        stress_result = stress_test(weights, clean_shocks)

    feature_dict = {
        key: round(float(value), 6)
        for key, value in features.iloc[0].to_dict().items()
    }

    explanation = generate_risk_explanation(
        extracted=extracted,
        risk_class=risk_class_de,
        risk_probabilities=risk_probabilities,
        features=feature_dict,
        stress_result=stress_result,
    )

    portfolio_weights = [
        {
            "ticker": ticker,
            "weight": round(float(weight), 6),
            "weight_pct": round(float(weight) * 100, 2),
        }
        for ticker, weight in weights.items()
    ]

    risk_probability_chart = [
        {
            "label": RISK_LABELS_DE.get(label, label),
            "raw_label": label,
            "value": round(float(value), 6),
            "value_pct": round(float(value) * 100, 2),
        }
        for label, value in sorted(
            risk_probabilities.items(),
            key=lambda item: {"Low": 0, "Medium": 1, "High": 2}.get(item[0], 99),
        )
    ]

    return {
        "source": question_source,
        "extracted": extracted,
        "prediction": {
            "risk_class": risk_class,
            "risk_class_de": risk_class_de,
            "risk_probabilities": risk_probabilities,
            "model_features": feature_dict,
            "stress_test_return": None if stress_result is None else round(float(stress_result), 6),
            "data_freshness": freshness,
        },
        "dashboard": {
            "portfolio_weights": portfolio_weights,
            "risk_probability_chart": risk_probability_chart,
            "stress_scenario": {
                "description": scenario.get("description"),
                "shocks": shock_by_ticker,
                "portfolio_reaction": None if stress_result is None else round(float(stress_result), 6),
            },
        },
        "explanation": explanation,
    }


def analyze_text_portfolio(user_text: str) -> dict[str, Any]:
    if not user_text or not user_text.strip():
        raise ValueError("Bitte gib zuerst eine Portfolio-Beschreibung ein.")

    extracted = extract_portfolio_from_text(user_text)
    return _prepare_analysis_payload(extracted, question_source="text")


def analyze_image_portfolio(
    image_path: str | Path,
    optional_question: str = "",
) -> dict[str, Any]:
    extracted = extract_portfolio_from_image(
        image_path=image_path,
        optional_user_question=optional_question or "",
    )
    return _prepare_analysis_payload(extracted, question_source="image")
