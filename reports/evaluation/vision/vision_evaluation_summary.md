# Computer Vision Evaluation – Portofino RiskView

- Testbilder: **3**
- Verfügbare Bilddateien: **3**
- Vision-Extraktion erfolgreich: **100.00%**
- Asset Resolver erfolgreich: **100.00%**
- End-to-End-Erfolgsquote: **100.00%**

## Kontrollierte Tabellenbilder mit exakt erwarteten Ticker-Sets
- Anzahl Exact-Cases: **2**
- Exakter Ticker-Match: **100.00%**
- Durchschnittlicher Ticker-Jaccard: **1.0000**
- Gewichtungen innerhalb Toleranz: **100.00%**
- Durchschnittlicher Gewichtsfehler (MAE): **0.0000**

## Komplexes reales Kreisdiagramm
- Anzahl Coverage-Cases: **1**
- Coverage-Prüfung bestanden: **100.00%**

Hinweis: Für komplexe Portfolio-Screenshots mit vielen ETF-/Fondsnamen wird die Bildpipeline anhand der erfolgreichen Verarbeitung und Auflösungstiefe bewertet. Ein starres exaktes Ticker-Set ist hier weniger geeignet, weil Produktnamen und Börsenlistings mehrdeutig sein können.

## Erzeugte Artefakte
- `vision_evaluation_cases.csv`
- `vision_evaluation_summary.json`
- `vision_evaluation_summary.md`
