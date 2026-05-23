from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Iterable

import yfinance as yf


# ---------------------------------------------------------------------
# Kuratierte Aliasbasis
# ---------------------------------------------------------------------
# Aliase bleiben sinnvoll für bekannte wiederkehrende Fondsnamen und für
# stark verkürzte Bild-Labels. Die generische Suche darunter übernimmt den
# Rest. "symbols" erlaubt mehrere alternative Listings; bevorzugt wird das
# erste Symbol, das aktuell Preisdaten liefert.
KNOWN_ASSET_ALIASES: dict[str, dict[str, Any]] = {
    # Bestehende Portofino-/Projektbeispiele
    # Häufige ausgeschriebene Aktiennamen aus Portfolio-Tabellen
    "nvidia corp": {
        "symbols": ["NVDA"],
        "display_name": "NVIDIA Corporation",
    },
    "nvidia corporation": {
        "symbols": ["NVDA"],
        "display_name": "NVIDIA Corporation",
    },
    "tesla inc": {
        "symbols": ["TSLA"],
        "display_name": "Tesla, Inc.",
    },
    "apple inc": {
        "symbols": ["AAPL"],
        "display_name": "Apple Inc.",
    },

    "amundi msci all country world ucits etf eur acc": {
        "symbols": ["ACWI.PA"],
        "display_name": "Amundi MSCI All Country World UCITS ETF EUR Acc",
    },
    "ishares core msci emerging markets imi ucits etf acc": {
        "symbols": ["IS3N.DE"],
        "display_name": "iShares Core MSCI Emerging Markets IMI UCITS ETF (Acc)",
    },
    "ishares core msci em imi ucits etf usd acc": {
        "symbols": ["IS3N.DE"],
        "display_name": "iShares Core MSCI EM IMI UCITS ETF USD (Acc)",
    },
    "ishares core msci emu ucits etf eur acc": {
        "symbols": ["SXR7.DE"],
        "display_name": "iShares Core MSCI EMU UCITS ETF EUR (Acc)",
    },
    "amundi euro government bond 7 10y ucits etf dist": {
        "symbols": ["MTDD.F"],
        "display_name": "Amundi Euro Government Bond 7-10Y UCITS ETF Dist",
    },
    "ubs etf ie cmci composite sf ucits etf usd a acc": {
        "symbols": ["UIQK.DE"],
        "display_name": "UBS ETF (IE) CMCI Composite SF UCITS ETF (USD) A-acc",
    },
    "ubs cmci composite sf ucits etf usd acc": {
        "symbols": ["UIQK.DE"],
        "display_name": "UBS CMCI Composite SF UCITS ETF USD Acc",
    },
    "spdr dow jones global real estate ucits etf": {
        "symbols": ["SPYJ.DE"],
        "display_name": "SPDR Dow Jones Global Real Estate UCITS ETF",
    },

    # Kreisdiagramm-/Fondsbild-Beispiele
    "msci europe size": {
        "symbols": ["IESZ.L"],
        "display_name": "iShares Edge MSCI Europe Size Factor UCITS ETF",
    },
    "msci europe size factor": {
        "symbols": ["IESZ.L"],
        "display_name": "iShares Edge MSCI Europe Size Factor UCITS ETF",
    },
    "ishares msci europe size factor": {
        "symbols": ["IESZ.L"],
        "display_name": "iShares Edge MSCI Europe Size Factor UCITS ETF",
    },
    "msci europe value": {
        "symbols": ["IEVL.L"],
        "display_name": "iShares Edge MSCI Europe Value Factor UCITS ETF",
    },
    "msci europe value factor": {
        "symbols": ["IEVL.L"],
        "display_name": "iShares Edge MSCI Europe Value Factor UCITS ETF",
    },
    "ishares msci europe value factor": {
        "symbols": ["IEVL.L"],
        "display_name": "iShares Edge MSCI Europe Value Factor UCITS ETF",
    },
    "ftse developed europe": {
        "symbols": ["VEUR.AS", "VEUR.L"],
        "display_name": "Vanguard FTSE Developed Europe UCITS ETF",
    },
    "vanguard ftse developed europe": {
        "symbols": ["VEUR.AS", "VEUR.L"],
        "display_name": "Vanguard FTSE Developed Europe UCITS ETF",
    },
    "comstage msci usa smallcap": {
        "symbols": ["X023.DE", "X023.F"],
        "display_name": "ComStage MSCI USA Small Cap TRN UCITS ETF",
    },
    "comstage msci usa small cap": {
        "symbols": ["X023.DE", "X023.F"],
        "display_name": "ComStage MSCI USA Small Cap TRN UCITS ETF",
    },
    "msci usa smallcap": {
        "symbols": ["X023.DE", "X023.F"],
        "display_name": "ComStage MSCI USA Small Cap TRN UCITS ETF",
    },
    "msci usa small cap": {
        "symbols": ["X023.DE", "X023.F"],
        "display_name": "ComStage MSCI USA Small Cap TRN UCITS ETF",
    },
    "dmsci usa midcap": {
        "symbols": ["EL41.F", "EL41.BE", "EL41.DU"],
        "display_name": "Deka MSCI USA MC UCITS ETF",
    },
    "deka msci usa midcap": {
        "symbols": ["EL41.F", "EL41.BE", "EL41.DU"],
        "display_name": "Deka MSCI USA MC UCITS ETF",
    },
    "deka msci usa mid cap": {
        "symbols": ["EL41.F", "EL41.BE", "EL41.DU"],
        "display_name": "Deka MSCI USA MC UCITS ETF",
    },
    "ishares s and p 500": {
        "symbols": ["SXR8.DE", "IUSA.L"],
        "display_name": "iShares Core S&P 500 UCITS ETF",
    },
    "s and p 500": {
        "symbols": ["SXR8.DE", "IUSA.L"],
        "display_name": "iShares Core S&P 500 UCITS ETF",
    },
    "vanguard ftse all world high dividend": {
        "symbols": ["VGWD.DE", "VHYL.L"],
        "display_name": "Vanguard FTSE All-World High Dividend Yield UCITS ETF",
    },
    "ftse all world high dividend": {
        "symbols": ["VGWD.DE", "VHYL.L"],
        "display_name": "Vanguard FTSE All-World High Dividend Yield UCITS ETF",
    },
    "arero weltfonds": {
        "symbols": ["HVJD.F"],
        "display_name": "ARERO - Der Weltfonds",
    },
    "arero weltfonds": {
        "symbols": ["HVJD.F"],
        "display_name": "ARERO - Der Weltfonds",
    },
    "dbx portfolio total return": {
        "symbols": ["DBX0.DE"],
        "display_name": "Xtrackers Portfolio UCITS ETF",
    },
    "db x portfolio total return": {
        "symbols": ["DBX0.DE"],
        "display_name": "Xtrackers Portfolio UCITS ETF",
    },
    "comstage msci pacific tr": {
        "symbols": ["X015.DE", "X015.F", "CBMP.SW"],
        "display_name": "ComStage MSCI Pacific ETF",
    },
    "msci pacific tr": {
        "symbols": ["X015.DE", "X015.F", "CBMP.SW"],
        "display_name": "ComStage MSCI Pacific ETF",
    },
    "ishares dj asia pacific select div 30": {
        "symbols": ["0MNU.IL", "0MNU.L", "IQQX.DE"],
        "display_name": "iShares Dow Jones Asia Pacific Select Dividend 30 UCITS ETF",
    },
    "ishares dj asia pacific select dividend 30": {
        "symbols": ["0MNU.IL", "0MNU.L", "IQQX.DE"],
        "display_name": "iShares Dow Jones Asia Pacific Select Dividend 30 UCITS ETF",
    },
    "dj asia pacific select div 30": {
        "symbols": ["0MNU.IL", "0MNU.L", "IQQX.DE"],
        "display_name": "iShares Dow Jones Asia Pacific Select Dividend 30 UCITS ETF",
    },
    "dj asia pacific select du 30": {
        "symbols": ["0MNU.IL", "0MNU.L", "IQQX.DE"],
        "display_name": "iShares Dow Jones Asia Pacific Select Dividend 30 UCITS ETF",
    },
}


# ---------------------------------------------------------------------
# Normalisierung
# ---------------------------------------------------------------------
STOPWORDS = {
    "etf", "ucits", "acc", "dist", "usd", "eur", "chf", "gbp",
    "tr", "trn", "dr", "the", "and", "a", "i", "ii", "iii",
    "share", "shares", "class", "fund", "fonds",
}

REPLACEMENTS = {
    "&": " and ",
    "_": " ",
    "-": " ",
    "/": " ",
    "s&p": "s and p",
    "s p": "s and p",
    "dj": "dow jones",
    "div": "dividend",
    "du": "dividend",
    "midcap": "mid cap",
    "smallcap": "small cap",
    "allworld": "all world",
}

ISIN_PATTERN = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b", re.IGNORECASE)
WKN_PATTERN = re.compile(r"\b[A-Z0-9]{6}\b", re.IGNORECASE)


def normalize_asset_text(text: str) -> str:
    """Macht OCR-, LLM- und Schreibvarianten besser vergleichbar."""
    value = str(text or "").strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))

    for source, target in REPLACEMENTS.items():
        value = value.replace(source, target)

    value = re.sub(r"[^a-z0-9\.\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def canonical_asset_text(text: str) -> str:
    """Reduziert wenig informative ETF-Zusätze für den Suchvergleich."""
    tokens = normalize_asset_text(text).split()
    cleaned = [token for token in tokens if token not in STOPWORDS]
    return " ".join(cleaned).strip()


def token_set(text: str) -> set[str]:
    return set(canonical_asset_text(text).split())


def token_overlap(left: str, right: str) -> float:
    a = token_set(left)
    b = token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        canonical_asset_text(left),
        canonical_asset_text(right),
    ).ratio()


def looks_like_market_symbol(value: str) -> bool:
    """
    Plausible Symbole: AAPL, SPY, SXR8.DE, VEUR.AS usw.
    Längere indexartige Strings mit Unterstrichen werden bewusst NICHT akzeptiert.
    """
    symbol = str(value or "").strip().upper()
    if not symbol or " " in symbol or "_" in symbol:
        return False
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,24}", symbol))


def extract_isin(value: str | None) -> str | None:
    match = ISIN_PATTERN.search(str(value or "").upper())
    return match.group(0).upper() if match else None


def extract_wkn(value: str | None) -> str | None:
    # WKNs sind nur ein Zusatzsignal. Sie dürfen nicht versehentlich als ISIN-
    # oder gewöhnliche Wortbestandteile interpretiert werden.
    raw = str(value or "").upper()
    match = WKN_PATTERN.search(raw)
    return match.group(0).upper() if match else None


# ---------------------------------------------------------------------
# yfinance-Validierung und Suche
# ---------------------------------------------------------------------
@lru_cache(maxsize=512)
def symbol_has_price_data(symbol: str) -> bool:
    """
    Kurzer Verfügbarkeitstest. Fehler werden bewusst abgefangen.
    """
    symbol = str(symbol or "").strip().upper()
    if not looks_like_market_symbol(symbol):
        return False

    try:
        prices = yf.download(
            symbol,
            period="1mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return prices is not None and not prices.empty
    except Exception:
        return False


def choose_best_working_symbol(symbols: Iterable[str]) -> str | None:
    cleaned = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    for symbol in cleaned:
        if symbol_has_price_data(symbol):
            return symbol
    # Falls Yahoo kurzfristig keine Daten liefert, aber ein kuratierter Kandidat
    # existiert, bleibt er als best effort erhalten.
    return cleaned[0] if cleaned else None


def query_variants(
    label: str,
    *,
    isin: str | None = None,
    wkn: str | None = None,
) -> list[str]:
    raw = str(label or "").strip()
    normalized = normalize_asset_text(raw)
    canonical = canonical_asset_text(raw)

    variants = [
        raw,
        normalized,
        canonical,
        canonical.replace("factor", "").strip(),
        canonical.replace("select dividend", "select").strip(),
        canonical.replace("dow jones", "dj").strip(),
    ]

    if isin:
        variants.insert(0, isin)
    if wkn:
        variants.insert(0, wkn)

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        v = re.sub(r"\s+", " ", str(variant or "").strip())
        key = v.lower()
        if len(v) >= 2 and key not in seen:
            seen.add(key)
            deduped.append(v)
    return deduped[:8]


def search_quotes(query: str) -> list[dict[str, Any]]:
    """
    Ruft yfinance.Search robust auf. Parameter sind bewusst konservativ,
    damit die Funktion auch bei Bibliotheksänderungen möglichst stabil bleibt.
    """
    try:
        search = yf.Search(
            query=query,
            max_results=10,
            news_count=0,
            lists_count=0,
            include_cb=False,
            include_nav_links=False,
            include_research=False,
            enable_fuzzy_query=True,
            recommended=0,
            raise_errors=False,
        )
        quotes = getattr(search, "quotes", []) or []
        return [quote for quote in quotes if isinstance(quote, dict)]
    except TypeError:
        # Fallback für ältere yfinance-Versionen
        try:
            search = yf.Search(query, max_results=10)
            quotes = getattr(search, "quotes", []) or []
            return [quote for quote in quotes if isinstance(quote, dict)]
        except Exception:
            return []
    except Exception:
        return []


def quote_display_name(quote: dict[str, Any]) -> str:
    return str(
        quote.get("longname")
        or quote.get("shortname")
        or quote.get("name")
        or quote.get("symbol")
        or ""
    ).strip()


def score_quote_candidate(query: str, quote: dict[str, Any]) -> float:
    symbol = str(quote.get("symbol", "") or "").strip().upper()
    name = quote_display_name(quote)
    quote_type = str(quote.get("quoteType", "") or "").upper()

    if not symbol or not looks_like_market_symbol(symbol):
        return 0.0

    overlap = token_overlap(query, name)
    similarity = text_similarity(query, name)

    score = 0.58 * overlap + 0.34 * similarity

    normalized_query = normalize_asset_text(query)
    normalized_name = normalize_asset_text(name)

    if symbol_has_price_data(symbol):
        score += 0.05

    if quote_type in {"ETF", "MUTUALFUND", "FUND"}:
        score += 0.04

    if any(term in normalized_query for term in ["etf", "ucits", "msci", "ftse", "bond", "index"]):
        if any(term in normalized_name for term in ["etf", "ucits", "msci", "ftse", "bond", "index"]):
            score += 0.03

    if any(symbol.endswith(suffix) for suffix in (".DE", ".F", ".L", ".AS", ".PA", ".SW", ".BE", ".DU", ".IL")):
        score += 0.01

    return round(min(score, 1.0), 6)


def collect_search_candidates(
    label: str,
    *,
    isin: str | None = None,
    wkn: str | None = None,
) -> list[tuple[float, dict[str, Any]]]:
    candidates: dict[str, tuple[float, dict[str, Any]]] = {}

    for variant in query_variants(label, isin=isin, wkn=wkn):
        for quote in search_quotes(variant):
            symbol = str(quote.get("symbol", "") or "").strip().upper()
            if not symbol:
                continue

            score = score_quote_candidate(label, quote)

            current = candidates.get(symbol)
            if current is None or score > current[0]:
                candidates[symbol] = (score, quote)

    return sorted(candidates.values(), key=lambda item: item[0], reverse=True)


# ---------------------------------------------------------------------
# Auflösung einer einzelnen Position
# ---------------------------------------------------------------------
def alias_resolution(label: str) -> dict[str, Any] | None:
    normalized = normalize_asset_text(label)
    canonical = canonical_asset_text(label)

    # Exakter Match
    exact = KNOWN_ASSET_ALIASES.get(normalized) or KNOWN_ASSET_ALIASES.get(canonical)
    if exact:
        symbol = choose_best_working_symbol(exact.get("symbols", []))
        if symbol:
            return {
                "resolved_symbol": symbol,
                "resolution_status": "resolved",
                "resolution_method": "curated_alias",
                "display_name": exact["display_name"],
                "resolution_score": 1.0,
            }

    # Robuster Fuzzy-Match gegen Aliasbasis
    fuzzy_matches: list[tuple[float, str, dict[str, Any]]] = []
    for alias_key, alias_value in KNOWN_ASSET_ALIASES.items():
        score = max(
            text_similarity(label, alias_key),
            token_overlap(label, alias_key),
        )

        alias_canonical = canonical_asset_text(alias_key)
        if canonical and alias_canonical:
            if canonical in alias_canonical or alias_canonical in canonical:
                score += 0.10

        if score >= 0.72:
            fuzzy_matches.append((score, alias_key, alias_value))

    if fuzzy_matches:
        score, _, alias_value = sorted(fuzzy_matches, key=lambda item: item[0], reverse=True)[0]
        symbol = choose_best_working_symbol(alias_value.get("symbols", []))
        if symbol:
            return {
                "resolved_symbol": symbol,
                "resolution_status": "resolved",
                "resolution_method": "fuzzy_curated_alias",
                "display_name": alias_value["display_name"],
                "resolution_score": round(min(score, 1.0), 4),
            }

    return None


def resolve_asset_label(
    label: str,
    *,
    isin: str | None = None,
    wkn: str | None = None,
) -> dict[str, Any]:
    """
    Löst einen erkannten Asset-Namen in ein Marktsymbol auf.
    """
    raw_label = str(label or "").strip()
    clean_isin = extract_isin(isin) or extract_isin(raw_label)
    clean_wkn = extract_wkn(wkn)

    if not raw_label and not clean_isin and not clean_wkn:
        return {
            "original_label": raw_label,
            "resolved_symbol": None,
            "resolution_status": "unresolved",
            "resolution_method": "empty_label",
            "display_name": "",
            "resolution_score": 0.0,
            "search_signals": {"isin": clean_isin, "wkn": clean_wkn},
        }

    # 1) Bereits sichtbares Symbol
    if looks_like_market_symbol(raw_label):
        return {
            "original_label": raw_label,
            "resolved_symbol": raw_label.upper(),
            "resolution_status": "resolved",
            "resolution_method": "already_symbol",
            "display_name": raw_label.upper(),
            "resolution_score": 1.0,
            "search_signals": {"isin": clean_isin, "wkn": clean_wkn},
        }

    # 2) Kuratierte Aliasbasis
    alias_hit = alias_resolution(raw_label)
    if alias_hit:
        return {
            "original_label": raw_label,
            **alias_hit,
            "search_signals": {"isin": clean_isin, "wkn": clean_wkn},
        }

    # 3) Generische Suche über Name / ISIN / WKN
    scored = collect_search_candidates(raw_label, isin=clean_isin, wkn=clean_wkn)
    accepted = [item for item in scored if item[0] >= 0.54]

    if accepted:
        best_score, best_quote = accepted[0]
        second_score = accepted[1][0] if len(accepted) > 1 else 0.0

        # Wenn zwei Ergebnisse sehr ähnlich sind und die Sicherheit nicht hoch ist,
        # lieber transparent unresolved statt falsch auflösen.
        ambiguous = (
            len(accepted) > 1
            and (best_score - second_score) < 0.035
            and best_score < 0.78
        )

        if not ambiguous:
            symbol = str(best_quote.get("symbol", "") or "").strip().upper()
            return {
                "original_label": raw_label,
                "resolved_symbol": symbol,
                "resolution_status": "resolved",
                "resolution_method": "yfinance_search_ranked",
                "display_name": quote_display_name(best_quote) or symbol,
                "resolution_score": round(float(best_score), 4),
                "search_signals": {"isin": clean_isin, "wkn": clean_wkn},
            }

    return {
        "original_label": raw_label,
        "resolved_symbol": None,
        "resolution_status": "unresolved",
        "resolution_method": "no_confident_match",
        "display_name": raw_label,
        "resolution_score": 0.0,
        "search_signals": {"isin": clean_isin, "wkn": clean_wkn},
    }


# ---------------------------------------------------------------------
# Portfolio-Auflösung
# ---------------------------------------------------------------------
def item_source_label(item: dict[str, Any]) -> str:
    return str(
        item.get("raw_name")
        or item.get("asset_name")
        or item.get("original_label")
        or item.get("name")
        or item.get("ticker")
        or ""
    ).strip()


def resolve_extracted_portfolio(extracted: dict[str, Any]) -> dict[str, Any]:
    """
    Ergänzt Portfolio-Positionen um aufgelöste Marktsymbole.

    Toleranzregel:
    - Sind mindestens 85 % des ursprünglichen Portfoliogewichts sauber auflösbar,
      wird weitergerechnet und die Gewichte der erkannten Positionen werden auf 100 %
      renormiert. Der Warnhinweis bleibt im JSON sichtbar.
    - Liegt die aufgelöste Gewichtssumme darunter, wird transparent abgebrochen.
    """
    portfolio = extracted.get("portfolio", []) or []
    if not isinstance(portfolio, list) or not portfolio:
        raise ValueError("Das extrahierte Portfolio enthält keine Positionen.")

    resolved_items: list[dict[str, Any]] = []
    unresolved_items: list[dict[str, Any]] = []

    for item in portfolio:
        if not isinstance(item, dict):
            continue

        source_label = item_source_label(item)
        resolution = resolve_asset_label(
            source_label,
            isin=str(item.get("isin", "") or "") or None,
            wkn=str(item.get("wkn", "") or "") or None,
        )

        weight = float(item.get("weight", 0.0) or 0.0)

        if resolution["resolution_status"] != "resolved":
            unresolved_items.append(
                {
                    "label": source_label or "Unbekannte Position",
                    "weight": weight,
                    "isin": item.get("isin"),
                    "wkn": item.get("wkn"),
                    "reason": resolution["resolution_method"],
                }
            )
            continue

        resolved_item = dict(item)
        resolved_item["original_label"] = source_label
        resolved_item["ticker"] = resolution["resolved_symbol"]
        resolved_item["resolved_symbol"] = resolution["resolved_symbol"]
        resolved_item["resolved_name"] = resolution["display_name"]
        resolved_item["resolution_method"] = resolution["resolution_method"]
        resolved_item["resolution_score"] = resolution["resolution_score"]
        resolved_item["search_signals"] = resolution.get("search_signals", {})
        resolved_items.append(resolved_item)

    if not resolved_items:
        raise ValueError(
            "Keine erkannte Position konnte belastbar in ein Marktsymbol aufgelöst werden."
        )

    resolved_weight = sum(float(item.get("weight", 0.0) or 0.0) for item in resolved_items)
    unresolved_weight = sum(float(item.get("weight", 0.0) or 0.0) for item in unresolved_items)

    if unresolved_items and resolved_weight < 0.85:
        names = "; ".join(item["label"] for item in unresolved_items)
        raise ValueError(
            "Für folgende erkannte Positionen konnte kein belastbares Marktsymbol "
            f"aufgelöst werden: {names}. Bitte nutze klarere Ticker, sichtbare ISIN/WKN "
            "oder ein deutlicheres Bild."
        )

    warnings: list[str] = []
    renormalized = False

    if unresolved_items:
        # Kleine Restmenge transparent entfernen und restliche Gewichte renormieren.
        if resolved_weight <= 0:
            raise ValueError("Die aufgelösten Portfolio-Gewichte sind ungültig.")

        for item in resolved_items:
            item["weight"] = float(item["weight"]) / resolved_weight

        renormalized = True
        warnings.append(
            "Ein kleiner Teil des Portfolios konnte nicht sicher aufgelöst werden. "
            "Die Analyse wurde mit den aufgelösten Positionen fortgeführt; deren Gewichte "
            "wurden auf 100 % renormiert."
        )

    enriched = dict(extracted)
    enriched["portfolio"] = resolved_items
    enriched["asset_resolution"] = {
        "status": "resolved_with_warnings" if unresolved_items else "resolved",
        "resolved_count": len(resolved_items),
        "unresolved_count": len(unresolved_items),
        "resolved_weight_before_renormalization": round(resolved_weight, 6),
        "unresolved_weight": round(unresolved_weight, 6),
        "renormalized": renormalized,
        "unresolved_items": unresolved_items,
        "warnings": warnings,
    }

    existing_notes = str(enriched.get("extraction_notes", "") or "").strip()
    if warnings:
        enriched["extraction_notes"] = " ".join([existing_notes, *warnings]).strip()

    return enriched
