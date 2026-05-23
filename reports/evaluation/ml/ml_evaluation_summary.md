# ML Evaluation – Portofino RiskView

## Zentrale Kennzahlen
- Test-Samples: **300**
- Accuracy: **0.7067**
- Macro F1: **0.6250**
- Weighted F1: **0.7056**

## Klassenspezifische Ergebnisse

- **High** – Precision: 0.4815, Recall: 0.4333, F1: 0.4561
- **Low** – Precision: 0.8182, Recall: 0.8232, F1: 0.8207
- **Medium** – Precision: 0.5926, Recall: 0.6038, F1: 0.5981

## Modellvergleich aus dem Training

- random_forest: Macro F1 = 0.6250
- gradient_boosting: Macro F1 = 0.6128
- logistic_regression: Macro F1 = 0.5469

## Wichtigste Modellfeatures

- volatility_63d: 0.1935
- market_beta: 0.1482
- max_drawdown_63d: 0.1188
- return_63d: 0.0994
- vix: 0.0864
- yield_curve_spread: 0.0580
- ten_year_yield: 0.0560
- two_year_yield: 0.0524

## Erzeugte Artefakte
- `ml_evaluation_summary.json`
- `ml_classification_report.csv`
- `ml_confusion_matrix.csv`
- `ml_confusion_matrix.png`
- optional `ml_feature_importance.csv` und `ml_feature_importance.png`
