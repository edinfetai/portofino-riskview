from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns."""
    return prices.sort_index().pct_change().dropna(how="all")


def max_drawdown(return_series: pd.Series) -> float:
    """Compute maximum drawdown from a return series."""
    cumulative = (1 + return_series.fillna(0)).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    return float(drawdowns.min())


def annualized_volatility(return_series: pd.Series) -> float:
    """Annualized volatility from daily returns."""
    return float(return_series.std() * np.sqrt(252))


def annualized_return(return_series: pd.Series) -> float:
    """Annualized return from daily returns."""
    return float(return_series.mean() * 252)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {str(ticker).upper(): max(0.0, float(weight)) for ticker, weight in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive value.")
    return {ticker: weight / total for ticker, weight in clean.items()}


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Calculate weighted portfolio returns."""
    weights = normalize_weights(weights)
    available = [ticker for ticker in weights if ticker in returns.columns]
    if not available:
        raise ValueError("None of the portfolio tickers are available in the price data.")

    weight_sum = sum(weights[ticker] for ticker in available)
    normalized_weights = {ticker: weights[ticker] / weight_sum for ticker in available}
    weighted = returns[available].mul(pd.Series(normalized_weights), axis=1)
    return weighted.sum(axis=1)


def portfolio_correlation_features(returns: pd.DataFrame, weights: dict[str, float]) -> tuple[float, float]:
    """Return average pairwise correlation and weighted average correlation."""
    weights = normalize_weights(weights)
    available = [ticker for ticker in weights if ticker in returns.columns]
    if len(available) < 2:
        return 0.0, 0.0

    corr = returns[available].corr().fillna(0)
    upper_values = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()

    avg_corr = float(upper_values.mean()) if len(upper_values) else 0.0

    w = np.array([weights[ticker] for ticker in available], dtype=float)
    w = w / w.sum()
    weighted_corr = float(w @ corr.values @ w)

    return avg_corr, weighted_corr


def concentration_features(weights: dict[str, float]) -> tuple[float, float, int]:
    """Largest position, top-3 concentration and number of assets."""
    weights = normalize_weights(weights)
    sorted_weights = np.sort(np.array(list(weights.values()), dtype=float))[::-1]

    largest_position = float(sorted_weights[0])
    top_3_concentration = float(sorted_weights[:3].sum())
    number_of_assets = int(len(sorted_weights))

    return largest_position, top_3_concentration, number_of_assets


def market_beta(portfolio_ret: pd.Series, market_ret: pd.Series) -> float:
    """Estimate beta against the market return series."""
    aligned = pd.concat([portfolio_ret.rename("portfolio"), market_ret.rename("market")], axis=1).dropna()
    if len(aligned) < 20 or aligned["market"].var() == 0:
        return 0.0

    covariance = aligned["portfolio"].cov(aligned["market"])
    variance = aligned["market"].var()
    return float(covariance / variance)


def latest_macro_values(macro: pd.DataFrame, as_of_date: pd.Timestamp) -> dict[str, float]:
    """Return latest known macro values at or before a date."""
    macro = macro.sort_index()
    eligible = macro.loc[macro.index <= as_of_date]
    if eligible.empty:
        row = macro.iloc[-1]
    else:
        row = eligible.iloc[-1]

    return {column: float(row[column]) for column in macro.columns}


def build_portfolio_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    weights: dict[str, float],
    lookback_days: int = 63,
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a one-row feature frame for a portfolio."""
    prices = prices.dropna(how="all").sort_index()
    returns = compute_returns(prices)

    if as_of_date is None:
        as_of_date = returns.index.max()
    else:
        as_of_date = pd.Timestamp(as_of_date)

    historical_returns = returns.loc[returns.index <= as_of_date].tail(lookback_days)
    if len(historical_returns) < 20:
        raise ValueError("Not enough return history to calculate portfolio features.")

    p_ret = portfolio_returns(historical_returns, weights)
    largest, top3, n_assets = concentration_features(weights)
    avg_corr, weighted_corr = portfolio_correlation_features(historical_returns, weights)

    if "SPY" in historical_returns.columns:
        beta = market_beta(p_ret, historical_returns["SPY"])
    else:
        beta = 0.0

    macro_values = latest_macro_values(macro, as_of_date)

    feature_dict = {
        "volatility_63d": annualized_volatility(p_ret),
        "return_63d": annualized_return(p_ret),
        "max_drawdown_63d": max_drawdown(p_ret),
        "average_correlation": avg_corr,
        "weighted_correlation": weighted_corr,
        "largest_position_weight": largest,
        "top_3_concentration": top3,
        "number_of_assets": n_assets,
        "market_beta": beta,
        **macro_values,
    }

    return pd.DataFrame([feature_dict])


def realized_future_risk_label(
    prices: pd.DataFrame,
    weights: dict[str, float],
    as_of_date: pd.Timestamp,
    horizon_days: int = 21,
) -> str:
    """Create a derived Low/Medium/High risk target from future volatility and drawdown."""
    returns = compute_returns(prices).sort_index()
    future_returns = returns.loc[returns.index > pd.Timestamp(as_of_date)].head(horizon_days)

    if len(future_returns) < 10:
        raise ValueError("Not enough future returns for target creation.")

    p_ret = portfolio_returns(future_returns, weights)
    future_vol = annualized_volatility(p_ret)
    future_dd = abs(max_drawdown(p_ret))

    if future_vol >= 0.35 or future_dd >= 0.12:
        return "High"
    if future_vol >= 0.18 or future_dd >= 0.06:
        return "Medium"
    return "Low"


def stress_test(weights: dict[str, float], shock_by_ticker: dict[str, float]) -> float:
    """Simple one-period stress test. Shocks are decimals, e.g. -0.15 for -15%."""
    weights = normalize_weights(weights)
    return float(sum(weights.get(ticker, 0.0) * shock_by_ticker.get(ticker, 0.0) for ticker in weights))
