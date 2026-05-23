## Project Metadata

- Project title: Portofino RiskView
- Student: Edin Fetai
- GitHub repository URL: https://github.com/edinfetai/portofino-riskview
- Deployment URL: https://huggingface.co/spaces/fetaiedi/portofino-riskview
- Submission date: 07 June 2026, 18:00

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: ML Numeric Data
- Primary block 2: NLP

If a third block is selected, it is documented and graded separately as extra work.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement: Privatanlegerinnen und Privatanleger sehen ihre Depotpositionen häufig isoliert und erhalten selten eine verständliche Gesamteinschätzung zu Konzentration, Volatilität, Korrelation, Markt-Beta und möglichen Stressverlusten. Zusätzlich liegen Portfolios oft nicht nur als Text, sondern auch als Screenshots oder Tabellenbilder vor.
- Goal: Portofino RiskView soll Portfolios aus Freitext oder aus Portfolio-Screenshots automatisch strukturieren, aktuelle Markt- und Makrodaten laden, numerische Risikofeatures berechnen, eine ML-basierte Risikoklasse vorhersagen und das Ergebnis verständlich erklären.

- Success criteria:
  - Text- und Bildinputs werden in ein gemeinsames strukturiertes Portfolio-JSON überführt.
  - Erkannte Positionen werden in handelbare Marktsymbole aufgelöst.
  - Die Anwendung lädt aktuelle Marktpreise und Makrodaten.
  - Das ML-Modell klassifiziert das Portfolio in Low, Medium oder High.
  - Optional erkannte Stress-Szenarien werden berechnet und im Dashboard angezeigt.
  - Das Ergebnis wird mit Kennzahlen, Charts, technischer Extraktion und natürlichsprachlicher Erklärung dargestellt.

### 1.2 Integration Logic
- How the selected blocks interact:  
  Der NLP-Block verarbeitet Freitext-Portfolios und extrahiert Positionen, Gewichtungen, Nutzerfrage und optionale Stress-Szenarien. Der Computer-Vision-Block verarbeitet Portfolio-Screenshots und erzeugt dieselbe Art von strukturiertem Portfolio-JSON. Beide Inputs werden anschliessend durch den Asset Resolver vereinheitlicht. Danach lädt der ML-Block aktuelle Markt- und Makrodaten, berechnet numerische Features und klassifiziert das Risiko. Die berechneten ML-Ergebnisse werden anschliessend wieder vom NLP-Block genutzt, um eine verständliche Erklärung für das Frontend zu erzeugen.

- Data and output flow between blocks:

```text
Freitext-Eingabe
        ↓
NLP-Extraktion
        ↓
Strukturiertes Portfolio-JSON
        ↓
Asset Resolver
        ↓
Aktuelle Markt- und Makrodaten
        ↓
Feature Engineering
        ↓
ML-Risikoklassifikation
        ↓
Stress-Test, Dashboard und NLP-Erklärung


Portfolio-Screenshot
        ↓
Computer-Vision-Extraktion
        ↓
Strukturiertes Portfolio-JSON
        ↓
Asset Resolver
        ↓
Aktuelle Markt- und Makrodaten
        ↓
Feature Engineering
        ↓
ML-Risikoklassifikation
        ↓
Stress-Test, Dashboard und NLP-Erklärung
```

---

## 2. Block Documentation

### 2A. ML Numeric Data (selected)

#### 2A.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Yahoo Finance / `yfinance` | Numerische tägliche Marktpreis-Zeitreihen | Dynamisch; im Training ab 2018, in der App jeweils die neuesten verfügbaren Tagesdaten | Grundlage für Rendite, Volatilität, Drawdown, Beta und Korrelation |
| 2 | FRED-Makrodaten | Numerische Makro-Zeitreihen | Dynamisch; ausgewählte Zins- und Marktindikatoren | Ergänzung des Marktumfelds durch 10Y Yield, 2Y Yield, Yield Curve Spread und VIX |
| 3 | Projektintern erzeugter Trainingsdatensatz | Strukturierter ML-Datensatz | 1’200 Portfolios × 13 Features | Training und Vergleich der Risikoklassifikationsmodelle |

#### 2A.2 Preprocessing and Features
- Cleaning steps:
  - Marktdaten werden über `yfinance` geladen und auf angepasste Schlusskurse reduziert.
  - Fehlende Preisreihen werden entfernt oder bei unbrauchbaren Symbolen nicht verwendet.
  - FRED-Zeitreihen werden numerisch konvertiert, fehlende Werte werden bereinigt und per Forward-/Backward-Fill ergänzt.
  - Portfolio-Gewichte werden normalisiert, damit sie zusammen 100 % ergeben.
  - Trainingsfeatures und Labels werden in `data/processed/training_features.csv` und `data/processed/training_labels.csv` gespeichert.

- Preprocessing steps:
  - Für jedes Portfolio werden tägliche Renditen berechnet.
  - Aus den Einzelrenditen wird eine gewichtete Portfolio-Rendite gebildet.
  - Für das Training werden zufällige Long-only-Portfolios mit 3 bis 6 Assets erzeugt.
  - Die Labels `Low`, `Medium` und `High` werden aus zukünftiger 21-Tage-Volatilität und zukünftigem Drawdown abgeleitet.
  - Der Train/Test-Split ist stratified mit `test_size=0.25` und `random_state=42`.

- Feature engineering and selection:
  - Das finale Modell verwendet 13 Features:
    - `volatility_63d`
    - `return_63d`
    - `max_drawdown_63d`
    - `average_correlation`
    - `weighted_correlation`
    - `largest_position_weight`
    - `top_3_concentration`
    - `number_of_assets`
    - `market_beta`
    - `ten_year_yield`
    - `two_year_yield`
    - `yield_curve_spread`
    - `vix`
  - Die wichtigsten Features des finalen Random-Forest-Modells waren:
    - `volatility_63d`: 0.1935
    - `market_beta`: 0.1482
    - `max_drawdown_63d`: 0.1188
    - `return_63d`: 0.0994
  - Relevante Implementierung: `data_loader.py`, `feature_engineering.py`, `train.py`, `model/metrics.json`.

#### 2A.3 Model Selection
- Models tested:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting

- Why these models were chosen:
  - Logistic Regression wurde als einfache und interpretierbare Baseline verwendet.
  - Random Forest wurde getestet, weil Portfolio-Risiko wahrscheinlich durch nicht-lineare Zusammenhänge zwischen Volatilität, Konzentration, Drawdown, Beta und Makrodaten entsteht.
  - Gradient Boosting wurde als zusätzliches nicht-lineares Ensemble-Modell getestet.
  - Random Forest wurde als finales Modell gewählt, weil es im Modellvergleich den besten Macro F1 Score erreicht hat.

#### 2A.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline für Risikoklassifikation erstellen | Standardisiertes lineares Modell mit balancierter Klassengewichtung | Logistic Regression | Macro F1 = 0.5469 | Ausgangspunkt |
| 2 | Nicht-lineare Zusammenhänge testen | Baum-basiertes Boosting-Modell auf demselben Feature-Set | Gradient Boosting | Macro F1 = 0.6128 | +0.0659 |
| 3 | Robusteres Ensemble-Modell testen | Random Forest mit 300 Trees, begrenzter Tiefe und balancierter Klassengewichtung | Random Forest | Macro F1 = 0.6250 | +0.0122 |

#### 2A.5 Evaluation and Error Analysis
- Metrics used:
  - Accuracy
  - Macro F1
  - Weighted F1
  - Precision, Recall und F1 pro Klasse
  - Confusion Matrix
  - Feature Importance

- Final results:
  - Test-Samples: 300
  - Bestes Modell: Random Forest
  - Accuracy: 0.7067
  - Macro F1: 0.6250
  - Weighted F1: 0.7056
  - Klassenspezifische Ergebnisse:
    - `High`: Precision 0.4815, Recall 0.4333, F1 0.4561
    - `Low`: Precision 0.8182, Recall 0.8232, F1 0.8207
    - `Medium`: Precision 0.5926, Recall 0.6038, F1 0.5981
  - Ergebnisartefakte:
    - `reports/evaluation/ml/ml_evaluation_summary.json`
    - `reports/evaluation/ml/ml_classification_report.csv`
    - `reports/evaluation/ml/ml_confusion_matrix.csv`
    - `reports/evaluation/ml/ml_confusion_matrix.png`
    - `reports/evaluation/ml/ml_feature_importance.csv`
    - `reports/evaluation/ml/ml_feature_importance.png`

- Error patterns and likely causes:
  - Das Modell erkennt `Low`-Risk-Portfolios am zuverlässigsten.
  - Die Klasse `High` ist schwieriger zu erkennen, da sie im Trainingsdatensatz deutlich seltener vorkommt.
  - Die Klassenverteilung im Trainingsdatensatz war:
    - `Low`: 655
    - `Medium`: 426
    - `High`: 119
  - Dadurch kann das Modell bei sehr riskanten Portfolios eher zwischen `Medium` und `High` verwechseln.
  - Mögliche Verbesserungen wären zusätzliche High-Risk-Trainingsbeispiele, Oversampling oder eine stärkere Gewichtung der High-Klasse.

#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s):
  - Aus dem NLP-Block erhält das ML-Modell strukturierte Portfolios aus Freitext.
  - Aus dem Computer-Vision-Block erhält das ML-Modell strukturierte Portfolios aus Screenshots.
  - Der Asset Resolver liefert daraus verwendbare Ticker-Symbole und normalisierte Gewichte.

- Outputs provided to other block(s):
  - Das ML-Modell liefert die Risikoklasse `Low`, `Medium` oder `High`.
  - Zusätzlich werden Modellwahrscheinlichkeiten und numerische Portfoliofeatures ausgegeben.
  - Diese Werte werden im Frontend visualisiert und vom NLP-Erklärungsmodul in eine verständliche Analyse übersetzt.
  - Optional berechnet der Stress-Test die direkte Portfolioauswirkung eines erkannten Schockszenarios.


### 2B. NLP (selected)

#### 2B.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Nutzereingaben im Textmodus | Freitext | Dynamisch zur Laufzeit | Extraktion von Portfolio-Positionen, Gewichtungen, Nutzerfrage und optionalem Stress-Szenario |
| 2 | Projektinterner NLP-Testdatensatz `evaluation/test_cases/nlp_test_cases.json` | Kuratierte Text-Testfälle | 6 Testfälle | Systematische Evaluation der NLP-Extraktion |
| 3 | Strukturierte ML-Ergebnisse aus der Risikoanalyse | JSON-/Feature-Kontext | Dynamisch pro Analyse | Grundlage für die natürlichsprachliche Erklärung des Modellresultats |

#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing:
  - Der Nutzereingabetext wird zuerst auf leeren oder unbrauchbaren Inhalt geprüft.
  - Portfolioangaben können als Prozentwerte, Dezimalprozentwerte oder Marktwerte formuliert sein.
  - Die extrahierten Positionen werden in ein einheitliches JSON-Format überführt.
  - Ticker werden vereinheitlicht und später vom Asset Resolver weiterverarbeitet.
  - Erkannte Stress-Szenarien werden als strukturierte `shock_by_ticker`-Werte gespeichert.

- Prompt design or retrieval setup:
  - Der NLP-Block verwendet Prompt Engineering mit einem strikt erwarteten JSON-Output.
  - Der Extraktionsprompt ist darauf ausgelegt, folgende Informationen zu erkennen:
    - Portfolio-Positionen
    - Gewichtungen oder Marktwerte
    - Ticker-Symbole
    - Nutzerfrage
    - optionale Stress-Schocks
  - Für die Erklärung wird ein separater Prompt verwendet, der bereits berechnete ML-Ergebnisse erhält.
  - Die Erklärung soll keine neuen Zahlen oder Risikoklassen erfinden, sondern die vorhandenen Modellresultate verständlich formulieren.
  - Relevante Implementierung: `nlp_utils.py`, `analysis_service.py`.

#### 2B.3 Approach Selection
- Approach used (classical NLP, transformer, RAG, prompt engineering):
  - Das Projekt verwendet Prompt Engineering mit einem LLM für strukturierte Informationsextraktion und natürlichsprachliche Ergebnis-Erklärung.

- Alternatives considered:
  - Ein regelbasierter Parser wäre für einfache Eingaben wie `40 % AAPL` möglich, wäre aber weniger robust bei natürlicher Sprache, Marktwerten oder Stress-Szenarien.
  - Ein klassisches NLP-/NER-Modell hätte einen annotierten Trainingsdatensatz benötigt und müsste zusätzlich Gewichtungen und Schocks korrekt interpretieren.
  - RAG war für diesen Teil nicht notwendig, da die NLP-Aufgabe nicht primär aus Dokumentensuche besteht, sondern aus strukturierter Extraktion und Erklärung.
  - Prompt Engineering wurde gewählt, weil es flexible Nutzereingaben direkt in ein maschinenlesbares Format übersetzen kann.

#### 2B.4 Comparison and Iterations
| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Einfache Portfolioangaben aus Freitext extrahieren | Erste Promptversion für Ticker und Prozentgewichte | LLM-Extraktionsprompt | Manuelle Prüfung einfacher Beispiele | Funktionierte für klare Standardfälle |
| 2 | Robustere strukturierte Ausgabe erzeugen | Striktes JSON-Format, klarere Normalisierung von Gewichten und Marktwerten | Überarbeiteter Extraktionsprompt | Weniger Format- und Parsingprobleme | Stabilerer Input für die ML-Pipeline |
| 3 | Stress-Szenarien und Erklärung integrieren | Zusätzliche Extraktion von `shock_by_ticker` und separater Erklärungsprompt | Finale NLP-Pipeline | 6/6 Testfälle erfolgreich | Finale Version |

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy:
  - Die NLP-Komponente wurde mit 6 kuratierten Testfällen geprüft.
  - Die Testfälle decken verschiedene Eingabearten ab:
    - klare Prozentportfolios
    - breit diversifizierte Portfolios
    - explizite Stress-Szenarien
    - Marktwerte statt Prozentgewichte
    - Dezimalprozentangaben
    - stark risikoorientierte Portfolios mit Hebelprodukten
  - Bewertet wurden:
    - erfolgreiche Modellaufrufe
    - exakter Ticker-Match
    - durchschnittlicher Ticker-Jaccard
    - Gewichtungen innerhalb Toleranz
    - exakte Stress-Szenario-Erkennung
    - Mean Absolute Error der Gewichtungen

- Results:
  - Testfälle: 6
  - Erfolgreiche Modellaufrufe: 6
  - API-/Call-Erfolgsquote: 100.00 %
  - Exakter Ticker-Match: 100.00 %
  - Durchschnittlicher Ticker-Jaccard: 1.0000
  - Gewichtungen innerhalb Toleranz: 100.00 %
  - Stress-Szenario exakt erkannt: 100.00 %
  - Durchschnittlicher Gewichtsfehler, MAE: 0.0000
  - Ergebnisartefakte:
    - `reports/evaluation/nlp/nlp_evaluation_cases.csv`
    - `reports/evaluation/nlp/nlp_evaluation_summary.json`
    - `reports/evaluation/nlp/nlp_evaluation_summary.md`

- Error patterns and likely causes:
  - In den definierten Testfällen traten keine Extraktionsfehler auf.
  - Während der Entwicklung waren uneindeutige natürliche Formulierungen schwieriger als klare tickerbasierte Eingaben.
  - Marktwerte müssen korrekt in relative Gewichte umgerechnet werden; dies wurde in der Evaluation erfolgreich getestet.
  - Die NLP-Funktionalität hängt von einem gültigen API-Key ab. Bei fehlendem Kontingent oder ungültigem Key entsteht ein technischer Laufzeitfehler, nicht zwingend ein Modellfehler.
  - Mehrdeutige Fonds- oder ETF-Namen werden bewusst nicht nur durch NLP gelöst, sondern nachgelagert über den Asset Resolver behandelt.

#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s):
  - Vom ML-Block erhält das NLP-Erklärungsmodul die Risikoklasse, Modellwahrscheinlichkeiten, numerischen Portfoliofeatures und das Stress-Test-Ergebnis.
  - Aus der Analysepipeline erhält der Erklärungsprompt zusätzlich das strukturierte Portfolio-JSON.

- Outputs provided to other block(s):
  - Die NLP-Extraktion liefert strukturierte Portfoliopositionen, Gewichtungen und optional Stress-Szenarien an die ML-Pipeline.
  - Das extrahierte Portfolio wird vom Asset Resolver in handelbare Symbole überführt.
  - Die natürlichsprachliche Erklärung wird an das Frontend weitergegeben und macht die numerischen ML-Ergebnisse für Nutzende verständlich.


### 2C. Computer Vision (selected)

#### 2C.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Nutzende laden eigene Portfolio-Screenshots hoch | Bilddaten | Dynamisch zur Laufzeit | Extraktion von sichtbaren Positionen, Gewichtungen, Tickersymbolen und optionalen Identifikatoren |
| 2 | Kuratierte Beispielbilder im Ordner `frontend/examples/` | Bilddaten | 3 Beispielbilder | Demonstration und kontrollierte Bildanalyse in der App |
| 3 | Computer-Vision-Testfälle `evaluation/test_cases/vision_test_cases.json` | Kuratierte Evaluationsdaten | 3 Testbilder | Systematische Bewertung der Vision-Pipeline |

#### 2C.2 Preprocessing and Augmentation
- Image preprocessing:
  - Akzeptierte Bildformate sind PNG, JPG/JPEG und WEBP.
  - Das hochgeladene Bild wird in eine Data-URL umgewandelt und an das Vision-Modell übergeben.
  - Die Vision-Ausgabe wird validiert und normalisiert.
  - Sichtbare Produkt-, Fonds- und ETF-Namen werden als `raw_name` möglichst originalgetreu beibehalten.
  - Wenn sichtbar, werden zusätzlich `ticker`, `isin` und `wkn` extrahiert.
  - Prozentwerte werden in Dezimalwerte umgerechnet, z. B. 7.5 % zu 0.075.
  - Falls keine Prozentwerte, aber Marktwerte sichtbar sind, werden diese zur Normalisierung der Portfoliogewichte verwendet.
  - Relevante Implementierung: `vision_utils.py`, `asset_resolver.py`, `analysis_service.py`.

- Augmentation strategy:
  - Es wurde kein eigenes Computer-Vision-Modell trainiert, deshalb wurde keine klassische Bildaugmentation wie Rotation, Cropping oder Helligkeitsveränderung eingesetzt.
  - Stattdessen wurde die Robustheit über unterschiedlich aufgebaute Beispielbilder getestet:
    - ein komplexes Kreisdiagramm mit vielen ETF-/Fondsnamen,
    - ein Wachstumsportfolio als Tabelle,
    - ein aggressives Portfolio mit Hebelprodukten als Tabelle.
  - Diese Testbilder prüfen verschiedene reale Layouttypen: Diagramm, Tabelle, lange Produktnamen, klare Ticker und unterschiedliche Gewichtungsstrukturen.

#### 2C.3 Model Selection
- Vision model(s) used:
  - Multimodales LLM-basiertes Vision-Modul über die OpenAI API.
  - Das konkret verwendete Modell wird über `OPENAI_MODEL` konfiguriert und in der lokalen sowie deployten App mit `gpt-4.1-mini` genutzt.

- Why these model(s) were chosen:
  - Die Aufgabe ist keine klassische Bildklassifikation, sondern eine Kombination aus Texterkennung, Layoutverständnis und strukturierter Informationsextraktion.
  - Portfolio-Screenshots enthalten Tabellen, Diagramme, Prozentwerte, Produktnamen, Ticker und teilweise lange ETF-/Fondsbezeichnungen.
  - Ein multimodales Vision-Modell eignet sich dafür besser als ein reines CNN-Klassifikationsmodell, weil die Ausgabe direkt als strukturiertes JSON in die ML-Pipeline weitergegeben werden kann.
  - Zusätzlich wird die Vision-Ausgabe bewusst nicht direkt als finale Wahrheit genutzt, sondern durch den Asset Resolver validiert und in handelbare Symbole übersetzt.

#### 2C.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Einfache Portfolio-Screenshots auslesen | Grundlegender Vision-Prompt für Positionen und Gewichte | Vision-Prompt v1 | Manuelle Sichtprüfung | Funktionierte bei klaren Tabellenbildern |
| 2 | Lange Fonds- und ETF-Namen besser verarbeiten | Asset Resolver mit Aliasen und Fuzzy Matching ergänzt | Vision + Asset Resolver v2 | Weniger ungelöste Positionen | Höhere Robustheit bei langen Produktnamen |
| 3 | Allgemeine Bildpipeline für Tabellen und Kreisdiagramme verbessern | Originalgetreue `raw_name`-Extraktion, optionale Ticker-/ISIN-/WKN-Erkennung und Coverage-Evaluation für komplexe Bilder | Finale Vision-Pipeline | 100 % End-to-End-Erfolgsquote in der Evaluation | Finale Version |

#### 2C.5 Evaluation and Error Analysis
- Metrics and/or visual checks:
  - Vision-Extraktion erfolgreich
  - Asset Resolver erfolgreich
  - End-to-End-Erfolgsquote
  - Exakter Ticker-Match bei kontrollierten Tabellenbildern
  - Durchschnittlicher Ticker-Jaccard bei kontrollierten Tabellenbildern
  - Gewichtungen innerhalb Toleranz bei kontrollierten Tabellenbildern
  - Durchschnittlicher Gewichtsfehler, MAE
  - Coverage-Prüfung bei komplexem realistischem Kreisdiagramm

- Final results:
  - Testbilder: 3
  - Verfügbare Bilddateien: 3
  - Vision-Extraktion erfolgreich: 100.00 %
  - Asset Resolver erfolgreich: 100.00 %
  - End-to-End-Erfolgsquote: 100.00 %
  - Kontrollierte Tabellenbilder:
    - Anzahl Exact-Cases: 2
    - Exakter Ticker-Match: 100.00 %
    - Durchschnittlicher Ticker-Jaccard: 1.0000
    - Gewichtungen innerhalb Toleranz: 100.00 %
    - Durchschnittlicher Gewichtsfehler, MAE: 0.0000
  - Komplexes reales Kreisdiagramm:
    - Anzahl Coverage-Cases: 1
    - Coverage-Prüfung bestanden: 100.00 %
  - Ergebnisartefakte:
    - `reports/evaluation/vision/vision_evaluation_cases.csv`
    - `reports/evaluation/vision/vision_evaluation_summary.json`
    - `reports/evaluation/vision/vision_evaluation_summary.md`

- Error patterns and limitations:
  - Sehr lange ETF- oder Fondsnamen können in Screenshots klein, abgeschnitten oder schwer lesbar sein.
  - Bei komplexen Kreisdiagrammen ist ein starrer exakter Ticker-Match weniger sinnvoll, weil Produktnamen und Börsenlistings mehrdeutig sein können.
  - Einzelne Suchkandidaten können bei yfinance keine Preisdaten liefern oder historisch/delisted sein.
  - Deshalb testet der Resolver mehrere Kandidaten und akzeptiert nur belastbare Auflösungen.
  - Unscharfe, stark komprimierte oder abgeschnittene Bilder bleiben eine realistische Limitation.
  - Die finale Lösung reduziert diese Risiken durch Kombination aus Vision-Extraktion, Asset Resolver, Coverage-Evaluation und transparenter Fehlerbehandlung.

#### 2C.6 Integration with Other Block(s)
- Inputs received from other block(s):
  - Optionaler Zusatztext des Nutzers kann zusammen mit dem Bild an die Vision-Analyse übergeben werden.
  - Der Asset Resolver verarbeitet die von Computer Vision extrahierten Namen, Ticker, ISIN oder WKN.

- Outputs provided to other block(s):
  - Der Computer-Vision-Block liefert ein strukturiertes Portfolio-JSON mit Positionen und Gewichten.
  - Dieses JSON wird an den Asset Resolver und anschliessend an die ML-Pipeline weitergegeben.
  - Die ML-Pipeline berechnet daraus aktuelle Risikofeatures und eine Risikoklasse.
  - Die Resultate werden danach durch den NLP-Block verständlich erklärt und im Frontend dargestellt.

---

## 3. Deployment

- Deployment URL: https://huggingface.co/spaces/fetaiedi/portofino-riskview

- Main user flow:
  - Nutzende öffnen die Hugging-Face-App Portofino RiskView.
  - Sie wählen zwischen Textanalyse und Bildanalyse.
  - Bei der Textanalyse geben sie ein Portfolio als Freitext ein, z. B. mit Tickersymbolen, Gewichtungen und optionalem Stress-Szenario.
  - Bei der Bildanalyse laden sie einen Portfolio-Screenshot hoch oder wählen eines der Beispielbilder aus.
  - Die Anwendung extrahiert die Positionen, löst sie in Marktsymbole auf, lädt aktuelle Markt- und Makrodaten und berechnet numerische Risikofeatures.
  - Das trainierte ML-Modell klassifiziert das Portfolio in `Low`, `Medium` oder `High`.
  - Das Dashboard zeigt:
    - Risikoklasse und Modellwahrscheinlichkeiten,
    - zentrale Risikokennzahlen,
    - Portfolio-Gewichtungen,
    - Risikowahrscheinlichkeiten,
    - optionales Stress-Test-Ergebnis,
    - strukturierte Extraktion und technische Details,
    - verständliche KI-Erklärung.

- Screenshot or short demo:
  - Die finale App ist als Docker Space auf Hugging Face deployt.
  - Für die Abgabe werden Screenshots der folgenden Zustände ergänzt:
  - [Startansicht der App](docs/screenshots/01_start_screen.png)
  - [Textanalyse mit Ergebnis-Dashboard](docs/screenshots/02_text_analysis_result.png)
  - [Stress-Test-Beispiel](docs/screenshots/03_stress_test_result.png)
  - [Bildanalyse mit Bildvorschau](docs/screenshots/04_image_analysis_preview.png)
  - [Strukturierte Extraktion und technische Details](docs/screenshots/05_structured_extraction_details.png)
  - [Analysekennzahlen](docs/screenshots/06_analysis_metrics.png)
  
  - Deployment-Konfiguration:
    - `README.md` enthält den Hugging-Face-Header mit `sdk: docker` und `app_port: 7860`.
    - `Dockerfile` baut die Umgebung mit Python 3.11 und startet die App über `python app.py`.
    - `requirements.txt` definiert die benötigten Abhängigkeiten wie FastAPI, Uvicorn, OpenAI, pandas, scikit-learn, yfinance und matplotlib.
    - Der OpenAI API Key wird nicht im Code gespeichert, sondern als Hugging-Face-Secret `OPENAI_API_KEY` gesetzt.

---

## 4. Execution Instructions

- Environment setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- Data setup:
  - Die Anwendung verwendet mehrere Datenquellen:
    - aktuelle Marktpreisdaten über `yfinance`,
    - Makrodaten über FRED,
    - Nutzereingaben im Textmodus,
    - Portfolio-Screenshots im Bildmodus,
    - projektinterne Evaluationsdaten.
  - Für das Training werden Markt- und Makrodaten heruntergeladen und daraus ein strukturierter Trainingsdatensatz erzeugt.
  - Die Trainingsdaten werden gespeichert unter:
    - `data/processed/training_features.csv`
    - `data/processed/training_labels.csv`
  - Das trainierte Modell und die Modellmetadaten werden gespeichert unter:
    - `model/risk_model.pkl`
    - `model/metrics.json`
  - Für die Inferenz lädt die App aktuelle tägliche Markt- und Makrodaten dynamisch neu.

- Training command(s):

```powershell
.\.venv\Scripts\python.exe train.py
```

- Inference/run command(s):

```powershell
$env:OPENAI_API_KEY="OPENAI_API_KEY"
$env:OPENAI_MODEL="gpt-4.1-mini"
.\.venv\Scripts\python.exe app.py
```

  Danach lokal im Browser öffnen:

```text
http://127.0.0.1:7860
```

  Für Hugging Face wird der OpenAI API Key nicht im Code gespeichert, sondern als Secret gesetzt:

```text
OPENAI_API_KEY
```

  Zusätzlich wird das Modell als Variable gesetzt:

```text
OPENAI_MODEL = gpt-4.1-mini
```

- Reproducibility notes:
  - Das Projekt kann lokal vollständig über `requirements.txt`, `train.py` und `app.py` ausgeführt werden.
  - Das ML-Training ist durch `random_state=42`, einen festen Train/Test-Split und gespeicherte Modellmetadaten reproduzierbar.
  - Die Evaluation aller drei AI-Blöcke kann mit folgendem Befehl gestartet werden:

```powershell
$env:OPENAI_API_KEY="OPENAI_API_KEY"
$env:OPENAI_MODEL="gpt-4.1-mini"
.\.venv\Scripts\python.exe evaluation/run_all_evaluations.py
```

  - Die Evaluation erzeugt Reports in:
    - `reports/evaluation/ml/`
    - `reports/evaluation/nlp/`
    - `reports/evaluation/vision/`
  - Die ML-Ergebnisse sind reproduzierbar auf Basis der gespeicherten Trainings- und Evaluationsdaten.
  - Die NLP- und Computer-Vision-Ergebnisse benötigen einen gültigen OpenAI API Key und können bei Änderungen des externen Modells leicht variieren.
  - Die Inferenz verwendet aktuelle Tagesdaten. Deshalb können sich Dashboardwerte wie Volatilität, Beta, Drawdown oder VIX bei späterer Ausführung verändern.
  - Die Deployment-Version läuft als Docker Space auf Hugging Face über Port `7860`.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [x] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [x] A core section is done exceptionally well
- [x] Extended evaluation
- [ ] Ethics, bias, or fairness analysis
- [x] Creative or exceptional use case

Evidence for selected bonus items:
- **Third selected block implemented with strong quality:** Das Projekt nutzt nicht nur zwei, sondern alle drei AI-Blöcke. ML Numeric Data, NLP und Computer Vision sind technisch in einer gemeinsamen Pipeline integriert. Text- und Bildinputs werden beide in ein strukturiertes Portfolio-JSON überführt und danach durch dieselbe ML-Risikoanalyse verarbeitet.
- **More than two data sources used with clear added value:** Das Projekt kombiniert mehrere unterschiedliche Datenquellen: Marktpreisdaten von Yahoo Finance über `yfinance`, Makrodaten von FRED, Nutzereingaben im Textmodus, Portfolio-Screenshots, Beispielbilder und kuratierte Evaluationsdaten.
- **A core section is done exceptionally well:** Der ML-Block enthält Feature Engineering, Modellvergleich mit drei Modellen, quantitative Evaluation, Feature Importance und Fehleranalyse. Das finale Modell erreicht eine Accuracy von 0.7067, einen Macro F1 Score von 0.6250 und einen Weighted F1 Score von 0.7056.
- **Extended evaluation:** Für alle drei AI-Blöcke wurden eigene Evaluationsskripte und Reports erstellt. Die NLP-Evaluation erreichte 100.00 % bei Ticker-Match, Gewichtungserkennung und Stress-Szenario-Erkennung. Die Computer-Vision-Evaluation erreichte 100.00 % Vision-Erfolg, 100.00 % Asset-Resolver-Erfolg und 100.00 % End-to-End-Erfolg.
- **Creative or exceptional use case:** Portofino RiskView verbindet eine realistische Finance-Anwendung mit multimodaler Eingabe. Nutzende können entweder ein Portfolio in natürlicher Sprache eingeben oder ein Portfolio-Bild hochladen. Die App kombiniert daraus Live-Marktdaten, Makrodaten, ML-Risikoklassifikation, Stress-Test und verständliche KI-Erklärung in einem modernen Dashboard.