from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asset_resolver import resolve_extracted_portfolio
from vision_utils import extract_portfolio_from_image


CASES_PATH = PROJECT_ROOT / "evaluation" / "test_cases" / "vision_test_cases.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "evaluation" / "vision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_TOLERANCE = 0.055


def read_cases() -> list[dict[str, Any]]:
    if not CASES_PATH.exists():
        raise FileNotFoundError(f"Testfall-Datei fehlt: {CASES_PATH}")
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def normalize_portfolio(portfolio: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in portfolio:
        ticker = str(
            item.get("ticker")
            or item.get("resolved_symbol")
            or ""
        ).upper().strip()
        if not ticker:
            continue
        result[ticker] = float(item.get("weight", 0.0) or 0.0)
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


def evaluate_exact_case(
    *,
    row: dict[str, Any],
    case: dict[str, Any],
    resolved: dict[str, Any],
) -> None:
    expected_tickers = {str(ticker).upper() for ticker in case.get("expected_tickers", [])}
    expected_weights = {
        str(ticker).upper(): float(weight)
        for ticker, weight in case.get("expected_weights", {}).items()
    }

    actual_weights = normalize_portfolio(resolved.get("portfolio", []))
    actual_tickers = set(actual_weights)

    row["exact_case_applicable"] = True
    row["ticker_jaccard"] = round(jaccard(expected_tickers, actual_tickers), 4)
    row["ticker_exact_match"] = expected_tickers == actual_tickers
    row["weight_mae"] = mean_absolute_weight_error(expected_weights, actual_weights)
    row["weights_within_tolerance"] = weights_within_tolerance(expected_weights, actual_weights)
    row["coverage_passed"] = None
    row["resolved_tickers"] = ", ".join(sorted(actual_tickers))


def evaluate_coverage_case(
    *,
    row: dict[str, Any],
    case: dict[str, Any],
    resolved: dict[str, Any],
) -> None:
    resolution_meta = resolved.get("asset_resolution", {}) or {}
    resolved_count = int(resolution_meta.get("resolved_count", len(resolved.get("portfolio", []))) or 0)
    unresolved_count = int(resolution_meta.get("unresolved_count", 0) or 0)

    minimum_resolved_count = int(case.get("minimum_resolved_count", 0) or 0)
    maximum_unresolved_count = int(case.get("maximum_unresolved_count", 9999) or 9999)

    actual_weights = normalize_portfolio(resolved.get("portfolio", []))
    actual_tickers = set(actual_weights)

    row["exact_case_applicable"] = False
    row["ticker_jaccard"] = None
    row["ticker_exact_match"] = None
    row["weight_mae"] = None
    row["weights_within_tolerance"] = None
    row["coverage_passed"] = (
        resolved_count >= minimum_resolved_count
        and unresolved_count <= maximum_unresolved_count
    )
    row["minimum_resolved_count"] = minimum_resolved_count
    row["maximum_unresolved_count"] = maximum_unresolved_count
    row["resolved_tickers"] = ", ".join(sorted(actual_tickers))


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    image_path = PROJECT_ROOT / case["image_path"]
    mode = str(case.get("evaluation_mode", "exact") or "exact").lower()

    row: dict[str, Any] = {
        "case_id": case["id"],
        "evaluation_mode": mode,
        "difficulty": case.get("difficulty", ""),
        "description": case.get("description", ""),
        "image_path": str(case.get("image_path", "")),
        "image_exists": image_path.exists(),
        "success": False,
        "vision_success": False,
        "resolver_success": False,
        "exact_case_applicable": None,
        "ticker_jaccard": None,
        "ticker_exact_match": None,
        "weight_mae": None,
        "weights_within_tolerance": None,
        "coverage_passed": None,
        "resolved_count": None,
        "unresolved_count": None,
        "minimum_resolved_count": None,
        "maximum_unresolved_count": None,
        "error": "",
        "resolved_tickers": "",
    }

    if not image_path.exists():
        row["error"] = f"Bilddatei fehlt: {image_path}"
        return row

    try:
        extracted = extract_portfolio_from_image(
            image_path=image_path,
            optional_user_question=case.get("optional_question", ""),
        )
        row["vision_success"] = True

        resolved = resolve_extracted_portfolio(extracted)
        row["resolver_success"] = True

        resolution_meta = resolved.get("asset_resolution", {}) or {}
        row["resolved_count"] = int(
            resolution_meta.get("resolved_count", len(resolved.get("portfolio", []))) or 0
        )
        row["unresolved_count"] = int(resolution_meta.get("unresolved_count", 0) or 0)

        if mode == "coverage":
            evaluate_coverage_case(row=row, case=case, resolved=resolved)
            row["success"] = bool(row["coverage_passed"])
        else:
            evaluate_exact_case(row=row, case=case, resolved=resolved)
            row["success"] = True

        row["raw_output"] = json.dumps(
            {"extracted": extracted, "resolved": resolved},
            ensure_ascii=False,
        )

    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def safe_mean(series: pd.Series) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return 0.0
    return float(cleaned.astype(float).mean())


def build_summary(frame: pd.DataFrame) -> dict[str, Any]:
    successful = frame[frame["success"] == True]
    exact_cases = frame[frame["evaluation_mode"] == "exact"]
    successful_exact = exact_cases[exact_cases["success"] == True]
    coverage_cases = frame[frame["evaluation_mode"] == "coverage"]

    weight_mae = successful_exact["weight_mae"].dropna()

    return {
        "number_of_cases": int(len(frame)),
        "available_images": int(frame["image_exists"].sum()),
        "vision_success_rate": float(frame["vision_success"].mean()) if len(frame) else 0.0,
        "resolver_success_rate": float(frame["resolver_success"].mean()) if len(frame) else 0.0,
        "end_to_end_success_rate": float(frame["success"].mean()) if len(frame) else 0.0,
        "number_of_exact_cases": int(len(exact_cases)),
        "number_of_coverage_cases": int(len(coverage_cases)),
        "ticker_exact_match_rate_exact_cases": safe_mean(successful_exact["ticker_exact_match"]),
        "average_ticker_jaccard_exact_cases": safe_mean(successful_exact["ticker_jaccard"]),
        "weights_within_tolerance_rate_exact_cases": safe_mean(successful_exact["weights_within_tolerance"]),
        "mean_weight_mae_exact_cases": float(weight_mae.mean()) if not weight_mae.empty else None,
        "coverage_pass_rate_complex_cases": safe_mean(coverage_cases["coverage_passed"]),
        "weight_tolerance": WEIGHT_TOLERANCE,
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    mae = summary["mean_weight_mae_exact_cases"]
    mae_text = "n/a" if mae is None else f"{mae:.4f}"

    return "\n".join([
        "# Computer Vision Evaluation – Portofino RiskView",
        "",
        f"- Testbilder: **{summary['number_of_cases']}**",
        f"- Verfügbare Bilddateien: **{summary['available_images']}**",
        f"- Vision-Extraktion erfolgreich: **{summary['vision_success_rate']:.2%}**",
        f"- Asset Resolver erfolgreich: **{summary['resolver_success_rate']:.2%}**",
        f"- End-to-End-Erfolgsquote: **{summary['end_to_end_success_rate']:.2%}**",
        "",
        "## Kontrollierte Tabellenbilder mit exakt erwarteten Ticker-Sets",
        f"- Anzahl Exact-Cases: **{summary['number_of_exact_cases']}**",
        f"- Exakter Ticker-Match: **{summary['ticker_exact_match_rate_exact_cases']:.2%}**",
        f"- Durchschnittlicher Ticker-Jaccard: **{summary['average_ticker_jaccard_exact_cases']:.4f}**",
        f"- Gewichtungen innerhalb Toleranz: **{summary['weights_within_tolerance_rate_exact_cases']:.2%}**",
        f"- Durchschnittlicher Gewichtsfehler (MAE): **{mae_text}**",
        "",
        "## Komplexes reales Kreisdiagramm",
        f"- Anzahl Coverage-Cases: **{summary['number_of_coverage_cases']}**",
        f"- Coverage-Prüfung bestanden: **{summary['coverage_pass_rate_complex_cases']:.2%}**",
        "",
        "Hinweis: Für komplexe Portfolio-Screenshots mit vielen ETF-/Fondsnamen "
        "wird die Bildpipeline anhand der erfolgreichen Verarbeitung und "
        "Auflösungstiefe bewertet. Ein starres exaktes Ticker-Set ist hier "
        "weniger geeignet, weil Produktnamen und Börsenlistings mehrdeutig sein können.",
        "",
        "## Erzeugte Artefakte",
        "- `vision_evaluation_cases.csv`",
        "- `vision_evaluation_summary.json`",
        "- `vision_evaluation_summary.md`",
        "",
    ])


def main() -> None:
    cases = read_cases()
    rows = [evaluate_case(case) for case in cases]
    frame = pd.DataFrame(rows)

    frame.to_csv(OUTPUT_DIR / "vision_evaluation_cases.csv", index=False)

    summary = build_summary(frame)
    (OUTPUT_DIR / "vision_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = summary_markdown(summary)
    (OUTPUT_DIR / "vision_evaluation_summary.md").write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
