from __future__ import annotations

"""
Vision-Extraktion für Portfolio-Screenshots.

Diese Version ist bewusst auf robuste nachgelagerte Asset-Auflösung ausgelegt:
- Fonds-/ETF-Namen sollen exakt wie sichtbar extrahiert werden.
- Keine Umwandlung in künstliche Snake-Case-Kürzel.
- Wenn sichtbar: Ticker, ISIN und WKN separat extrahieren.
- Gewichtungen oder Marktwerte werden sauber normalisiert.
- Zusätzliche User-Fragen können weiterhin an die Analyse weitergegeben werden.
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

MARKET_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,24}$")
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
WKN_RE = re.compile(r"^[A-Z0-9]{6}$")


# ---------------------------------------------------------------------
# OpenAI-Client und Bildvorbereitung
# ---------------------------------------------------------------------
def _client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY fehlt. Setze den Schlüssel lokal als Umgebungsvariable "
            "oder später als Hugging-Face-Secret."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Bilddatei nicht gefunden: {image_path}")

    suffix = path.suffix.lower().replace(".", "")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(suffix, "image/png")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


# ---------------------------------------------------------------------
# Normalisierung der Vision-Antwort
# ---------------------------------------------------------------------
def _as_clean_string(value: Any) -> str:
    return str(value or "").strip()


def _clean_raw_name(value: Any) -> str:
    raw = _as_clean_string(value)
    # Falls das Modell trotzdem Snake Case liefert, wird es lesbarer gemacht.
    # Der Asset Resolver kann damit robuster weiterarbeiten.
    if "_" in raw and " " not in raw:
        raw = raw.replace("_", " ")
    return re.sub(r"\s+", " ", raw).strip()


def _clean_symbol(value: Any) -> str | None:
    raw = _as_clean_string(value).upper()
    if raw and MARKET_SYMBOL_RE.fullmatch(raw) and "_" not in raw:
        return raw
    return None


def _clean_isin(value: Any) -> str | None:
    raw = _as_clean_string(value).upper().replace(" ", "")
    return raw if ISIN_RE.fullmatch(raw) else None


def _clean_wkn(value: Any) -> str | None:
    raw = _as_clean_string(value).upper().replace(" ", "")
    return raw if WKN_RE.fullmatch(raw) else None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None

    raw = str(value).strip()
    raw = raw.replace("’", "").replace("'", "")
    raw = raw.replace("CHF", "").replace("EUR", "").replace("USD", "")
    raw = raw.replace(" ", "")

    # Schweizer/Deutsche Formate: 1.234,56 -> 1234.56
    if "," in raw and "." in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        raw = raw.replace(",", ".")

    raw = raw.replace("%", "")

    try:
        return float(raw)
    except ValueError:
        return None


def _normalize_weight(value: Any) -> float | None:
    number = _to_float(value)
    if number is None:
        return None

    # Prozentwerte wie 40 oder 40.0 -> 0.40
    if number > 1.0 and number <= 100.0:
        return number / 100.0

    return number


def _normalize_shock(value: Any) -> float | None:
    number = _to_float(value)
    if number is None:
        return None
    if abs(number) > 1:
        return number / 100.0
    return number


def _normalize_stress_scenario(parsed: dict[str, Any]) -> dict[str, Any]:
    scenario = parsed.get("stress_scenario", {}) or {}
    if not isinstance(scenario, dict):
        scenario = {}

    raw_shocks = scenario.get("shock_by_ticker", {}) or {}
    shocks: dict[str, float] = {}

    if isinstance(raw_shocks, dict):
        for key, value in raw_shocks.items():
            ticker = _clean_symbol(key) or _as_clean_string(key).upper()
            shock = _normalize_shock(value)
            if ticker and shock is not None:
                shocks[ticker] = shock

    return {
        "description": scenario.get("description"),
        "shock_by_ticker": shocks,
    }


def normalize_image_extraction(
    parsed: dict[str, Any],
    *,
    optional_user_question: str = "",
) -> dict[str, Any]:
    """
    Validiert und normalisiert Vision-Ausgaben, ohne wertvolle Metadaten
    wie raw_name, ISIN oder WKN zu verlieren.
    """
    raw_portfolio = parsed.get("portfolio", [])
    if not isinstance(raw_portfolio, list) or not raw_portfolio:
        raise ValueError(
            "Im Bild konnten keine ausreichend klaren Portfolio-Positionen erkannt werden."
        )

    normalized_items: list[dict[str, Any]] = []
    discarded_items: list[str] = []

    for index, raw_item in enumerate(raw_portfolio, start=1):
        if not isinstance(raw_item, dict):
            discarded_items.append(f"Position {index}: ungültige Struktur")
            continue

        raw_name = _clean_raw_name(
            raw_item.get("raw_name")
            or raw_item.get("asset_name")
            or raw_item.get("name")
            or raw_item.get("label")
            or raw_item.get("ticker")
        )

        ticker = _clean_symbol(raw_item.get("ticker"))
        isin = _clean_isin(raw_item.get("isin"))
        wkn = _clean_wkn(raw_item.get("wkn"))

        weight = _normalize_weight(raw_item.get("weight"))
        market_value = _to_float(raw_item.get("market_value"))

        # Falls ein Prozentfeld anders benannt wurde, tolerieren wir das.
        if weight is None:
            for fallback_key in ("allocation", "percentage", "percent", "gewichtung"):
                weight = _normalize_weight(raw_item.get(fallback_key))
                if weight is not None:
                    break

        if not raw_name and not ticker and not isin and not wkn:
            discarded_items.append(f"Position {index}: kein Name/Ticker/ISIN/WKN")
            continue

        if weight is None and market_value is None:
            discarded_items.append(
                f"{raw_name or ticker or 'Position ' + str(index)}: keine Gewichtung/kein Marktwert"
            )
            continue

        normalized_items.append(
            {
                "raw_name": raw_name or ticker or isin or wkn or "",
                "ticker": ticker or "",
                "isin": isin,
                "wkn": wkn,
                "weight": weight,
                "market_value": market_value,
            }
        )

    if not normalized_items:
        raise ValueError(
            "Die sichtbaren Positionen enthielten keine ausreichend klaren Gewichte oder Marktwerte."
        )

    # Gewichtungsbasis bestimmen
    if all(item["weight"] is not None for item in normalized_items):
        raw_values = [float(item["weight"]) for item in normalized_items]
    elif all(item["market_value"] is not None for item in normalized_items):
        raw_values = [float(item["market_value"]) for item in normalized_items]
    else:
        # Gemischte Fälle: Gewichtungen bevorzugen, Marktwerte nur wenn notwendig.
        raw_values = []
        for item in normalized_items:
            if item["weight"] is not None:
                raw_values.append(float(item["weight"]))
            elif item["market_value"] is not None:
                raw_values.append(float(item["market_value"]))
            else:
                raw_values.append(0.0)

    total = sum(value for value in raw_values if value is not None)
    if total <= 0:
        raise ValueError("Die Summe der erkannten Portfolio-Werte ist ungültig.")

    for item, raw_value in zip(normalized_items, raw_values):
        item["weight"] = float(raw_value) / total

    extraction_notes = _as_clean_string(parsed.get("extraction_notes"))
    vision_notes = _as_clean_string(parsed.get("vision_notes"))

    if discarded_items:
        discard_note = " Ignoriert: " + "; ".join(discarded_items[:6])
        extraction_notes = (extraction_notes + discard_note).strip()

    question = _as_clean_string(parsed.get("question")) or _as_clean_string(optional_user_question)
    if not question:
        question = "Wie hoch ist das Risiko meines Portfolios basierend auf der erkannten Allokation?"

    return {
        "portfolio": normalized_items,
        "question": question,
        "stress_scenario": _normalize_stress_scenario(parsed),
        "source": "image",
        "extraction_notes": extraction_notes,
        "vision_notes": vision_notes,
    }


# ---------------------------------------------------------------------
# Vision-Aufruf
# ---------------------------------------------------------------------
def extract_portfolio_from_image(
    image_path: str | Path,
    optional_user_question: str = "",
) -> dict[str, Any]:
    if not image_path:
        raise ValueError("Bitte lade zuerst ein Portfolio-Bild hoch.")

    data_url = image_to_data_url(image_path)

    system_prompt = """
Du bist ein sehr präziser Vision-Assistent für Portfolio-Screenshots,
Depot-Tabellen, Kreisdiagramme und Allokationsübersichten.

Ziel:
Lies die sichtbaren Portfolio-Positionen so originalgetreu wie möglich aus dem Bild,
damit ein nachgelagerter Resolver die Namen zuverlässig in Marktsymbole übersetzen kann.

Extrem wichtig:
- Schreibe Produkt- und Fondsnamen exakt so ab, wie sie im Bild sichtbar sind.
- Erfinde keine Abkürzungen.
- Verwandle Namen NICHT in Snake Case, NICHT in Codes wie MSCI_EUROPE_SIZE_FACTOR.
- Verkürze lange ETF-Namen nicht unnötig.
- Wenn nur ein Index-/Produktname sichtbar ist, gib genau diesen sichtbaren Text als raw_name zurück.
- Wenn ein echter Ticker im Bild sichtbar ist, gib ihn zusätzlich in ticker zurück.
- Wenn eine ISIN oder WKN sichtbar ist, gib sie zusätzlich in isin bzw. wkn zurück.
- Wenn ISIN/WKN nicht sichtbar sind, setze null.
- Wenn Ticker nicht sichtbar ist, setze ticker auf null.
- Wenn der Name nicht sicher lesbar ist, nimm nur den sicher erkennbaren sichtbaren Teil,
  aber erfinde nichts dazu.
- Gewichtungen aus Prozentangaben müssen gelesen werden.
- Kreisdiagramm-Beschriftungen mit Prozentwerten sind gültige Portfolio-Positionen.
- Wenn statt Prozentwerten Marktwerte sichtbar sind, extrahiere market_value.
- Wenn weder Gewichtung noch Marktwert sichtbar ist, lasse die Position weg.

Antworte ausschliesslich mit gültigem JSON.
Kein Markdown.
Keine Erklärung ausserhalb des JSON.

Struktur:
{
  "portfolio": [
    {
      "raw_name": "iShares MSCI Europe Size Factor",
      "ticker": null,
      "isin": null,
      "wkn": null,
      "weight": 0.075,
      "market_value": null
    }
  ],
  "question": "kurze Analysefrage auf Deutsch",
  "stress_scenario": {
    "description": null,
    "shock_by_ticker": {}
  },
  "source": "image",
  "extraction_notes": "kurzer Hinweis zur Extraktion",
  "vision_notes": "kurzer Hinweis zur Bildlesbarkeit"
}

Regeln:
- Prozentwerte als Dezimalwerte ausgeben: 7,5 % = 0.075.
- Marktwerte als Zahl ohne Währungssymbol ausgeben.
- Kein Ticker-Raten anhand des Produktnamens.
- Stress-Szenarien nicht aus dem Bild erfinden.
- Falls die optionale Nutzerfrage explizite Stress-Schocks enthält, darfst du
  diese als stress_scenario übernehmen.
""".strip()

    user_prompt = f"""
Analysiere das Portfolio-Bild.

Optionale Zusatzfrage des Nutzers:
{optional_user_question or "Analysiere mein Portfolio-Risiko."}

Gib nur das geforderte JSON zurück.
""".strip()

    response = _client().responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        temperature=0,
    )

    raw = (getattr(response, "output_text", "") or "").strip()

    if not raw:
        raise ValueError("Das Vision-Modell hat keine Antwort zurückgegeben.")

    if raw.startswith("```"):
        raise ValueError(
            "Das Vision-Modell hat Markdown statt reinem JSON zurückgegeben: "
            f"{raw[:300]}"
        )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Das Vision-Modell hat kein gültiges JSON zurückgegeben. "
            f"Antwortauszug: {raw[:300]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Die Vision-Ausgabe muss ein JSON-Objekt sein.")

    return normalize_image_extraction(
        parsed,
        optional_user_question=optional_user_question,
    )
