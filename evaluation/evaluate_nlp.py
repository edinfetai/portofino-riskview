from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp_utils import extract_portfolio_from_text
CASES_PATH = PROJECT_ROOT / "evaluation" / "test_cases" / "nlp_test_cases.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "evaluation" / "nlp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_TOLERANCE = 0.035
SHOCK_TOLERANCE = 0.02


def read_cases() -> list[dict[str, Any]]:
    if not CASES_PATH.exists():
        raise FileNotFoundError(f"Testfall-Datei fehlt: {CASES_PATH}")
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def normalize_portfolio(portfolio: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in portfolio:
        ticker = str(item.get("ticker", "") or "").upper().strip()
        if not ticker:
            continue
        result[ticker] = float(item.get("weight", 0.0) or 0.0)
    return result


def normalize_shocks(scenario: dict[str, Any] | None) -> dict[str, float]:
    scenario = scenario or {}
    shocks = scenario.get("shock_by_ticker", {}) or {}
    result: dict[str, float] = {}

    if isinstance(shocks, dict):
        for ticker, shock in shocks.items():
            result[str(ticker).upper().strip()] = float(shock)

    return result


def jaccard(expected: set[str], actual: set[str]) -> float:
    if not expected and not actual:
        return 1.0
    union = expected | actual
    if not union:
        return 0.0
    return len(expected & actual) / len(union)


def mean_absolute_weight_error(
    expected: dict[str, float],
    actual: dict[str, float],
) -> float | None:
    common = sorted(set(expected) & set(actual))
    if not common:
        return None
    return sum(abs(expected[ticker] - actual[ticker]) for ticker in common) / len(common)


def weights_within_tolerance(
    expected: dict[str, float],
    actual: dict[str, float],
) -> bool:
    if set(expected) != set(actual):
        return False
    return all(
        abs(expected[ticker] - actual.get(ticker, 999.0)) <= WEIGHT_TOLERANCE
        for ticker in expected
    )


def shocks_within_tolerance(
    expected: dict[str, float],
    actual: dict[str, float],
) -> bool:
    if set(expected) != set(actual):
        return False
    return all(
        abs(expected[ticker] - actual.get(ticker, 999.0)) <= SHOCK_TOLERANCE
        for ticker in expected
    )


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    expected_tickers = {str(ticker).upper() for ticker in case.get("expected_tickers", [])}
    expected_weights = {
        str(ticker).upper(): float(weight)
        for ticker, weight in case.get("expected_weights", {}).items()
    }
    expected_shocks = {
        str(ticker).upper(): float(shock)
        for ticker, shock in case.get("expected_shocks", {}).items()
    }

    row: dict[str, Any] = {
        "case_id": case["id"],
        "difficulty": case.get("difficulty", ""),
        "description": case.get("description", ""),
        "expect_success": bool(case.get("expect_success", True)),
        "success": False,
        "ticker_jaccard": 0.0,
        "ticker_exact_match": False,
        "weight_mae": None,
        "weights_within_tolerance": False,
        "stress_exact_match": False,
        "error": "",
        "extracted_tickers": "",
    }

    try:
        extracted = extract_portfolio_from_text(case["input_text"])
        actual_weights = normalize_portfolio(extracted.get("portfolio", []))
        actual_tickers = set(actual_weights)
        actual_shocks = normalize_shocks(extracted.get("stress_scenario"))

        row["success"] = True
        row["ticker_jaccard"] = round(jaccard(expected_tickers, actual_tickers), 4)
        row["ticker_exact_match"] = expected_tickers == actual_tickers
        row["weight_mae"] = mean_absolute_weight_error(expected_weights, actual_weights)
        row["weights_within_tolerance"] = weights_within_tolerance(expected_weights, actual_weights)
        row["stress_exact_match"] = shocks_within_tolerance(expected_shocks, actual_shocks)
        row["extracted_tickers"] = ", ".join(sorted(actual_tickers))
        row["raw_output"] = json.dumps(extracted, ensure_ascii=False)

    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def build_summary(frame: pd.DataFrame) -> dict[str, Any]:
    successful = frame[frame["success"] == True]
    weight_mae = successful["weight_mae"].dropna()

    return {
        "number_of_cases": int(len(frame)),
        "successful_calls": int(frame["success"].sum()),
        "call_success_rate": float(frame["success"].mean()) if len(frame) else 0.0,
        "ticker_exact_match_rate": float(successful["ticker_exact_match"].mean()) if len(successful) else 0.0,
        "average_ticker_jaccard": float(successful["ticker_jaccard"].mean()) if len(successful) else 0.0,
        "weights_within_tolerance_rate": float(successful["weights_within_tolerance"].mean()) if len(successful) else 0.0,
        "stress_exact_match_rate": float(successful["stress_exact_match"].mean()) if len(successful) else 0.0,
        "mean_weight_mae": float(weight_mae.mean()) if not weight_mae.empty else None,
        "weight_tolerance": WEIGHT_TOLERANCE,
        "shock_tolerance": SHOCK_TOLERANCE,
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    mae = summary["mean_weight_mae"]
    mae_text = "n/a" if mae is None else f"{mae:.4f}"

    return "\n".join([
        "# NLP Evaluation – Portofino RiskView",
        "",
        f"- Testfälle: **{summary['number_of_cases']}**",
        f"- Erfolgreiche Modellaufrufe: **{summary['successful_calls']}**",
        f"- API-/Call-Erfolgsquote: **{summary['call_success_rate']:.2%}**",
        f"- Exakter Ticker-Match: **{summary['ticker_exact_match_rate']:.2%}**",
        f"- Durchschnittlicher Ticker-Jaccard: **{summary['average_ticker_jaccard']:.4f}**",
        f"- Gewichtungen innerhalb Toleranz: **{summary['weights_within_tolerance_rate']:.2%}**",
        f"- Stress-Szenario exakt erkannt: **{summary['stress_exact_match_rate']:.2%}**",
        f"- Durchschnittlicher Gewichtsfehler (MAE): **{mae_text}**",
        "",
        "## Erzeugte Artefakte",
        "- `nlp_evaluation_cases.csv`",
        "- `nlp_evaluation_summary.json`",
        "- `nlp_evaluation_summary.md`",
        "",
    ])


def main() -> None:
    cases = read_cases()
    rows = [evaluate_case(case) for case in cases]
    frame = pd.DataFrame(rows)

    frame.to_csv(OUTPUT_DIR / "nlp_evaluation_cases.csv", index=False)

    summary = build_summary(frame)
    (OUTPUT_DIR / "nlp_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = summary_markdown(summary)
    (OUTPUT_DIR / "nlp_evaluation_summary.md").write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
