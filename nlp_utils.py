from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# ---------------------------------------------------------------------
# OpenAI-Client und JSON-Parsing
# ---------------------------------------------------------------------
def _client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY fehlt. Setze den Schlüssel lokal als Umgebungsvariable "
            "oder später als Hugging-Face-Secret."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def call_llm_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """
    Ruft das LLM auf und erwartet strikt valides JSON ohne Markdown.
    """
    response = _client().responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw = getattr(response, "output_text", "") or ""
    raw = raw.strip()

    if not raw:
        raise ValueError("Das LLM hat keine Antwort zurückgegeben.")

    if raw.startswith("```"):
        raise ValueError(
            "Das LLM hat Markdown statt reinem JSON zurückgegeben: "
            f"{raw[:300]}"
        )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Das LLM hat kein gültiges JSON zurückgegeben. "
            f"Antwortauszug: {raw[:300]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Das LLM-JSON muss ein Objekt sein.")

    return parsed


# ---------------------------------------------------------------------
# Portfolio-Validierung und Normalisierung
# ---------------------------------------------------------------------
def validate_and_normalize_portfolio(parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Prüft extrahierte Portfolio-Daten und normalisiert Gewichte auf Summe 1.0.

    Unterstützt:
    - explizite Gewichtungen, z. B. 0.4 oder 40 %
    - Marktwerte, die vom Modell später als relative Gewichte normalisiert werden
    """
    if "portfolio" not in parsed or not isinstance(parsed["portfolio"], list):
        raise ValueError("Die KI-Ausgabe enthält keine gültige Portfolio-Liste.")

    if not parsed["portfolio"]:
        raise ValueError("Es wurden keine Portfolio-Positionen erkannt.")

    portfolio: list[dict[str, float | str]] = []

    for item in parsed["portfolio"]:
        if not isinstance(item, dict):
            raise ValueError("Eine Portfolio-Position ist nicht korrekt strukturiert.")

        ticker = str(item.get("ticker", "")).upper().strip()
        weight = item.get("weight")
        market_value = item.get("market_value")

        if not ticker:
            raise ValueError("Bei mindestens einer Portfolio-Position fehlt der Ticker.")

        if weight is not None:
            raw_value = float(weight)
        elif market_value is not None:
            raw_value = float(market_value)
        else:
            raise ValueError(
                f"Für {ticker} fehlt sowohl Gewichtung als auch Marktwert."
            )

        if raw_value <= 0:
            raise ValueError(
                f"Der Wert für {ticker} muss grösser als 0 sein."
            )

        portfolio.append(
            {
                "ticker": ticker,
                "weight": float(raw_value),
            }
        )

    total = sum(float(item["weight"]) for item in portfolio)

    if total <= 0:
        raise ValueError("Die Summe der erkannten Portfolio-Werte ist ungültig.")

    for item in portfolio:
        item["weight"] = float(item["weight"]) / total

    scenario = parsed.get("stress_scenario", {}) or {}
    if not isinstance(scenario, dict):
        scenario = {}

    description = scenario.get("description")
    shock_by_ticker = scenario.get("shock_by_ticker", {}) or {}

    if not isinstance(shock_by_ticker, dict):
        shock_by_ticker = {}

    clean_shocks: dict[str, float] = {}

    for ticker, shock in shock_by_ticker.items():
        clean_ticker = str(ticker).upper().strip()
        if not clean_ticker:
            continue

        shock_value = float(shock)

        # Schocks sollen als Dezimalwerte gespeichert werden.
        # -15 % = -0.15
        if abs(shock_value) > 1:
            shock_value = shock_value / 100

        clean_shocks[clean_ticker] = shock_value

    return {
        "portfolio": portfolio,
        "question": str(parsed.get("question", "") or "").strip(),
        "stress_scenario": {
            "description": description,
            "shock_by_ticker": clean_shocks,
        },
        "source": str(parsed.get("source", "text") or "text"),
        "extraction_notes": str(parsed.get("extraction_notes", "") or "").strip(),
    }


# ---------------------------------------------------------------------
# NLP-Extraktion aus Freitext
# ---------------------------------------------------------------------
def extract_portfolio_from_text(user_text: str) -> dict[str, Any]:
    """
    Extrahiert Portfolio-Positionen, eine Analysefrage und optionale Stress-Szenarien
    aus deutschem oder englischem Freitext.
    """
    if not user_text or not user_text.strip():
        raise ValueError("Bitte gib zuerst eine Portfolio-Beschreibung ein.")

    system_prompt = """
Du bist ein präziser Extraktionsassistent für Portfolioanalysen.

Deine Aufgabe:
- Erkenne Aktien- oder ETF-Ticker aus dem Nutzereingabetext.
- Extrahiere zugehörige Gewichtungen oder Marktwerte.
- Erkenne die Nutzerfrage bzw. den Analysewunsch.
- Erkenne optionale Stress-Test-Szenarien, falls sie im Text stehen.

Antworte ausschliesslich mit gültigem JSON.
Kein Markdown.
Keine Erklärung ausserhalb des JSON.
Erfinde keine Ticker.
Erfinde keine Positionen.
Wenn keine Gewichtung, Prozentangabe oder kein Marktwert erkennbar ist, lasse die Extraktion nicht frei raten.

Das JSON muss genau diese Struktur haben:
{
  "portfolio": [
    {
      "ticker": "AAPL",
      "weight": 0.4,
      "market_value": null
    }
  ],
  "question": "kurze Zusammenfassung der Nutzerfrage",
  "stress_scenario": {
    "description": "kurze Beschreibung oder null",
    "shock_by_ticker": {
      "AAPL": -0.10
    }
  },
  "source": "text",
  "extraction_notes": "kurzer Hinweis zur Extraktion oder leerer String"
}

Regeln:
- Ticker immer in Grossbuchstaben.
- Prozentwerte in Dezimalgewichte umwandeln: 40 % = 0.40.
- Wenn Werte als absolute Geldbeträge genannt werden, setze weight auf null und market_value auf den Zahlenwert.
- Wenn keine Stress-Schocks genannt werden, gib "shock_by_ticker": {} zurück.
- Stress-Schocks als Dezimalwerte speichern: -15 % = -0.15.
- Die Nutzerfrage soll auf Deutsch zusammengefasst werden.
""".strip()

    user_prompt = f"""
Extrahiere aus folgendem Text die Portfolio-Daten:

{user_text}

Beispiel:
"Mein Portfolio besteht aus 40 % AAPL, 30 % MSFT, 20 % NVDA und 10 % TSLA.
Analysiere mein Risiko und teste zusätzlich NVDA -15 % sowie TSLA -20 %."

Erwartetes Format:
{{
  "portfolio": [
    {{"ticker": "AAPL", "weight": 0.40, "market_value": null}},
    {{"ticker": "MSFT", "weight": 0.30, "market_value": null}},
    {{"ticker": "NVDA", "weight": 0.20, "market_value": null}},
    {{"ticker": "TSLA", "weight": 0.10, "market_value": null}}
  ],
  "question": "Das Portfolio-Risiko und zentrale Schwächen sollen analysiert werden.",
  "stress_scenario": {{
    "description": "NVDA fällt um 15 %, TSLA fällt um 20 %",
    "shock_by_ticker": {{"NVDA": -0.15, "TSLA": -0.20}}
  }},
  "source": "text",
  "extraction_notes": ""
}}
""".strip()

    parsed = call_llm_json(system_prompt, user_prompt)
    return validate_and_normalize_portfolio(parsed)


# ---------------------------------------------------------------------
# Erklärung der Risikoanalyse
# ---------------------------------------------------------------------
def generate_risk_explanation(
    extracted: dict[str, Any],
    risk_class: str,
    risk_probabilities: dict[str, float],
    features: dict[str, float],
    stress_result: float | None,
) -> str:
    """
    Erstellt eine deutschsprachige, nutzerfreundliche Erklärung der ML-Analyse.
    """
    system_prompt = """
Du bist ein seriöser deutschsprachiger Assistent für Portfolio-Risikoanalysen.

Deine Aufgabe:
- Erkläre die übergebenen numerischen Risikoergebnisse verständlich.
- Beziehe dich nur auf die gelieferten Daten.
- Berechne keine neue Risikoklasse.
- Erfinde keine Fakten.
- Gib keine Kauf-, Verkaufs- oder Halteempfehlungen.
- Gib keine konkrete Anlageberatung.
- Nutze eine ruhige, professionelle Sprache.

Die Antwort soll:
1. die erkannte Risikoklasse kurz einordnen,
2. die wichtigsten Risikotreiber nennen,
3. Diversifikation oder Konzentration kommentieren,
4. einen Stress-Test kurz erklären, falls vorhanden,
5. genau einen klaren Responsible-Use-Hinweis enthalten.

Antworte ausschliesslich mit gültigem JSON:
{
  "answer": "deutscher Erklärungstext"
}
""".strip()

    user_prompt = f"""
Erzeuge eine verständliche Risikoerklärung auf Deutsch.

Extrahierte Portfolio-Daten:
{json.dumps(extracted, ensure_ascii=False)}

ML-Risikoklasse:
{risk_class}

ML-Wahrscheinlichkeiten:
{json.dumps(risk_probabilities, ensure_ascii=False)}

Numerische Features:
{json.dumps(features, ensure_ascii=False)}

Stress-Test-Ergebnis:
{stress_result}
""".strip()

    parsed = call_llm_json(system_prompt, user_prompt)

    answer = str(parsed.get("answer", "") or "").strip()

    if not answer:
        raise ValueError("Die KI-Erklärung ist leer.")

    return answer
