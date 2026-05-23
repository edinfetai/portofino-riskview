# QuantView Frontend Rebuild

Diese Frontend-Version ersetzt die bisherige Gradio-Oberfläche durch eine dedizierte
FastAPI-HTML/CSS/JavaScript-Oberfläche im minimalistischen QuantView-Stil.

## Start lokal

```powershell
$env:OPENAI_API_KEY="DEIN_NEUER_API_KEY"
$env:OPENAI_MODEL="gpt-4.1-mini"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Danach im Browser öffnen:

```text
http://127.0.0.1:7860
```

## Wichtig

Die Dateien `data_loader.py`, `feature_engineering.py`, `nlp_utils.py`,
`vision_utils.py`, `asset_resolver.py`, `model/risk_model.pkl` und
`model/metrics.json` bleiben weiterhin erforderlich.
