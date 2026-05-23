from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "training_features.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "training_labels.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "risk_model.pkl"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "evaluation" / "ml"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_labels(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("training_labels.csv ist leer.")
    if "risk_class" in frame.columns:
        return frame["risk_class"].astype(str)
    return frame.iloc[:, 0].astype(str)


def save_confusion_matrix_plot(
    matrix: list[list[int]] | Any,
    labels: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(matrix)

    ax.set_title("Konfusionsmatrix – ML-Risikoklassifikation")
    ax.set_xlabel("Vorhergesagte Klasse")
    ax.set_ylabel("Tatsächliche Klasse")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, str(value), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def extract_feature_importance(model: Any, feature_columns: list[str]) -> pd.DataFrame | None:
    estimator = model

    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        values = list(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if getattr(coef, "ndim", 1) == 2:
            values = list(abs(coef).mean(axis=0))
        else:
            values = list(abs(coef))
    else:
        return None

    if len(values) != len(feature_columns):
        return None

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": values,
    })
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = frame.head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.barh(plot_frame["feature"], plot_frame["importance"])
    ax.set_title("Wichtigste Modellfeatures")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_model_comparison(metrics: dict[str, Any]) -> pd.DataFrame | None:
    all_results = metrics.get("all_results", {})
    if not isinstance(all_results, dict) or not all_results:
        return None

    rows = []
    for model_name, result in all_results.items():
        rows.append({
            "model": model_name,
            "macro_f1": result.get("macro_f1"),
        })

    frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    frame.to_csv(OUTPUT_DIR / "ml_model_comparison.csv", index=False)
    return frame


def build_markdown_summary(
    summary: dict[str, Any],
    comparison: pd.DataFrame | None,
    importance: pd.DataFrame | None,
) -> str:
    lines = [
        "# ML Evaluation – Portofino RiskView",
        "",
        "## Zentrale Kennzahlen",
        f"- Test-Samples: **{summary['test_samples']}**",
        f"- Accuracy: **{summary['accuracy']:.4f}**",
        f"- Macro F1: **{summary['macro_f1']:.4f}**",
        f"- Weighted F1: **{summary['weighted_f1']:.4f}**",
        "",
        "## Klassenspezifische Ergebnisse",
        "",
    ]

    for label, values in summary["per_class"].items():
        lines.append(
            f"- **{label}** – Precision: {values['precision']:.4f}, "
            f"Recall: {values['recall']:.4f}, F1: {values['f1-score']:.4f}"
        )

    if comparison is not None and not comparison.empty:
        lines.extend([
            "",
            "## Modellvergleich aus dem Training",
            "",
        ])
        for _, row in comparison.iterrows():
            lines.append(f"- {row['model']}: Macro F1 = {float(row['macro_f1']):.4f}")

    if importance is not None and not importance.empty:
        lines.extend([
            "",
            "## Wichtigste Modellfeatures",
            "",
        ])
        for _, row in importance.head(8).iterrows():
            lines.append(f"- {row['feature']}: {float(row['importance']):.4f}")

    lines.extend([
        "",
        "## Erzeugte Artefakte",
        "- `ml_evaluation_summary.json`",
        "- `ml_classification_report.csv`",
        "- `ml_confusion_matrix.csv`",
        "- `ml_confusion_matrix.png`",
        "- optional `ml_feature_importance.csv` und `ml_feature_importance.png`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    for required_path in [FEATURES_PATH, LABELS_PATH, MODEL_PATH]:
        if not required_path.exists():
            raise FileNotFoundError(f"Pflichtdatei fehlt: {required_path}")

    X = pd.read_csv(FEATURES_PATH)
    y = load_labels(LABELS_PATH)
    model = joblib.load(MODEL_PATH)

    metrics_meta = {}
    if METRICS_PATH.exists():
        metrics_meta = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    feature_columns = metrics_meta.get("feature_columns", list(X.columns))
    feature_columns = [column for column in feature_columns if column in X.columns]
    X = X[feature_columns]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    predictions = model.predict(X_test)
    labels = sorted(y.unique().tolist())

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, predictions, labels=labels)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1-score": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in labels
        if label in report
    }

    summary = {
        "test_samples": int(len(y_test)),
        "labels": labels,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "reconstructed_split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
    }

    (OUTPUT_DIR / "ml_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "ml_classification_report.csv",
        index=True,
    )

    pd.DataFrame(confusion, index=labels, columns=labels).to_csv(
        OUTPUT_DIR / "ml_confusion_matrix.csv",
        index=True,
    )

    save_confusion_matrix_plot(
        confusion.tolist(),
        labels,
        OUTPUT_DIR / "ml_confusion_matrix.png",
    )

    comparison = save_model_comparison(metrics_meta)

    importance = extract_feature_importance(model, feature_columns)
    if importance is not None and not importance.empty:
        importance.to_csv(OUTPUT_DIR / "ml_feature_importance.csv", index=False)
        save_feature_importance_plot(
            importance,
            OUTPUT_DIR / "ml_feature_importance.png",
        )

    markdown = build_markdown_summary(summary, comparison, importance)
    (OUTPUT_DIR / "ml_evaluation_summary.md").write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
