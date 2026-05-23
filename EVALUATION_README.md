# Portofino RiskView – Evaluation Suite

Dieses Paket ergänzt das Projekt um drei systematische Evaluationsbausteine:

1. **ML Numeric Data Evaluation**
2. **NLP Evaluation**
3. **Computer Vision Evaluation**

Die Dateien sind so aufgebaut, dass sie direkt in den bestehenden Projektordner
von **Portofino RiskView** kopiert werden können.

---

## Ordnerstruktur

```text
evaluation/
├── evaluate_ml.py
├── evaluate_nlp.py
├── evaluate_vision.py
├── run_all_evaluations.py
└── test_cases/
    ├── nlp_test_cases.json
    └── vision_test_cases.json
```

Die Ausgaben werden automatisch gespeichert in:

```text
reports/evaluation/
├── ml/
├── nlp/
└── vision/
```

---

## Voraussetzungen

Die Skripte erwarten die bestehenden Projektdateien im Hauptordner:

```text
model/risk_model.pkl
model/metrics.json
data/processed/training_features.csv
data/processed/training_labels.csv
nlp_utils.py
vision_utils.py
asset_resolver.py
```

Für NLP- und Vision-Evaluation wird ein aktiver OpenAI API Key benötigt.

---

## Startbefehle

### 1. ML Evaluation

```powershell
.\.venv\Scripts\python.exe evaluation/evaluate_ml.py
```

### 2. NLP Evaluation

```powershell
$env:OPENAI_API_KEY="DEIN_OPENAI_API_KEY"
$env:OPENAI_MODEL="gpt-4.1-mini"
.\.venv\Scripts\python.exe evaluation/evaluate_nlp.py
```

### 3. Computer Vision Evaluation

```powershell
$env:OPENAI_API_KEY="DEIN_OPENAI_API_KEY"
$env:OPENAI_MODEL="gpt-4.1-mini"
.\.venv\Scripts\python.exe evaluation/evaluate_vision.py
```

### 4. Alle nacheinander

```powershell
$env:OPENAI_API_KEY="DEIN_OPENAI_API_KEY"
$env:OPENAI_MODEL="gpt-4.1-mini"
.\.venv\Scripts\python.exe evaluation/run_all_evaluations.py
```

---

## Ergebnisse für die Dokumentation

Die Skripte erzeugen automatisch:

### ML
- Modellmetriken
- Konfusionsmatrix als CSV und PNG
- Modellvergleich aus `metrics.json`
- Feature Importance, falls verfügbar
- kurze Markdown-Zusammenfassung

### NLP
- Fallweise Extraktionsergebnisse
- Ticker-Match
- Gewichtsfehler
- Stress-Test-Match
- Erfolgsquote und Markdown-Zusammenfassung

### Vision
- Fallweise Bildauswertung
- Asset-Resolution-Status
- Ticker-Match
- Gewichtsfehler
- Erfolgsquote und Markdown-Zusammenfassung

Diese Dateien können später direkt für die Abschnitte
**2A.5, 2B.5 und 2C.5** der Projektdokumentation verwendet werden.
