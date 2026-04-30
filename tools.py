"""
Custom tools for the AI-Powered Stocks Analyser.

Provides yfinance-based financial data tools and the EXA news search tool.
All yfinance calls are wrapped with exponential-backoff retry logic.
"""

import json
import os
import time
from typing import Any, Callable

import pandas as pd
import yfinance as yf
from crewai.tools import tool
from crewai_tools import EXASearchTool
from curl_cffi import requests
from dotenv import load_dotenv

from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

session = requests.Session(impersonate="chrome")

os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY", "")
exa_search_tool = EXASearchTool()


# ── Retry helper ───────────────────────────────────────────────────────────────

def _with_retry(fn: Callable[[], Any], label: str, max_retries: int = 3, base_delay: float = 1.5) -> Any:
    """Call fn up to max_retries times with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn()
            if attempt > 1:
                logger.info("✅  '%s' succeeded on attempt %d.", label, attempt)
            return result
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
            if attempt < max_retries:
                logger.warning("⚠️   Attempt %d/%d failed for '%s': %s — retrying in %.1fs…", attempt, max_retries, label, exc, delay)
                time.sleep(delay)
            else:
                logger.error("❌  All %d attempts failed for '%s': %s", max_retries, label, exc)
    raise last_exc  # type: ignore[misc]


# ── Technical indicator helpers (pure pandas — no extra dependencies) ──────────

def _rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return round(float((100 - 100 / (1 + rs)).iloc[-1]), 2)


def _macd(prices: pd.Series):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return round(float(macd_line.iloc[-1]), 4), round(float(signal.iloc[-1]), 4), round(float(histogram.iloc[-1]), 4)


def _bollinger(prices: pd.Series, period: int = 20):
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return round(float(upper.iloc[-1]), 2), round(float(sma.iloc[-1]), 2), round(float(lower.iloc[-1]), 2)


# ── Financial Tools ────────────────────────────────────────────────────────────

@tool("Get current stock price")
def get_current_stock_price(symbol: str) -> str:
    """Get the current market price for a stock symbol.

    Args:
        symbol: Stock ticker (e.g. AAPL, RELIANCE.NS, SUZLON.BO).
    Returns:
        Current price as a string, or an error message.
    """
    symbol = symbol.strip()
    logger.info("📈  Fetching current stock price for: %s", symbol)
    try:
        def _fetch():
            time.sleep(0.5)
            info = yf.Ticker(symbol, session=session).info
            return info.get("regularMarketPrice") or info.get("currentPrice")
        price = _with_retry(_fetch, f"get_current_stock_price({symbol})")
        if price:
            return f"{price:.2f}"
        return f"Could not fetch current price for {symbol}. Symbol may be incorrect or delisted."
    except Exception as exc:
        logger.error("Error fetching price for %s: %s", symbol, exc)
        return f"Error fetching current price for {symbol}: {exc}"


@tool("Get company info")
def get_company_info(symbol: str) -> str:
    """Get company profile and key financial snapshot for a stock symbol.

    Args:
        symbol: Stock ticker (e.g. AAPL, RELIANCE.NS, SUZLON.BO).
    Returns:
        JSON string with company profile and key metrics, or an error message.
    """
    symbol = symbol.strip()
    logger.info("🏢  Fetching company info for: %s", symbol)
    try:
        raw = _with_retry(lambda: yf.Ticker(symbol, session=session).info, f"get_company_info({symbol})")
        if not raw:
            return f"Could not fetch company info for {symbol}."
        cleaned = {
            "Name": raw.get("shortName"),
            "Symbol": raw.get("symbol"),
            "Current Stock Price": f"{raw.get('regularMarketPrice') or raw.get('currentPrice')} {raw.get('currency', 'USD')}",
            "Market Cap": f"{raw.get('marketCap') or raw.get('enterpriseValue')} {raw.get('currency', 'USD')}",
            "Sector": raw.get("sector"),
            "Industry": raw.get("industry"),
            "Country": raw.get("country"),
            "EPS": raw.get("trailingEps"),
            "P/E Ratio": raw.get("trailingPE"),
            "52 Week Low": raw.get("fiftyTwoWeekLow"),
            "52 Week High": raw.get("fiftyTwoWeekHigh"),
            "Revenue Growth": raw.get("revenueGrowth"),
            "Gross Margins": raw.get("grossMargins"),
            "EBITDA": raw.get("ebitda"),
        }
        logger.debug("Company info fetched for %s: %s", symbol, cleaned.get("Name"))
        return json.dumps(cleaned, default=str)
    except Exception as exc:
        logger.error("Error fetching company info for %s: %s", symbol, exc)
        return f"Error fetching company info for {symbol}: {exc}"


@tool("Get income statements")
def get_income_statements(symbol: str) -> str:
    """Get annual income statements for a stock symbol.

    Args:
        symbol: Stock ticker (e.g. AAPL, RELIANCE.NS, SUZLON.BO).
    Returns:
        JSON string of income statement data, or an error message.
    """
    symbol = symbol.strip()
    logger.info("📊  Fetching income statements for: %s", symbol)
    try:
        def _fetch():
            fin = yf.Ticker(symbol, session=session).financials
            if fin is None or fin.empty:
                raise ValueError(f"No income statement data for {symbol}")
            return fin.to_json(orient="index")
        result = _with_retry(_fetch, f"get_income_statements({symbol})")
        logger.debug("Income statements fetched for %s.", symbol)
        return result
    except Exception as exc:
        logger.error("Error fetching income statements for %s: %s", symbol, exc)
        return f"Error fetching income statements for {symbol}: {exc}"


@tool("Get balance sheet")
def get_balance_sheet(symbol: str) -> str:
    """Get the annual balance sheet for a stock symbol, including assets, liabilities, and equity.

    Args:
        symbol: Stock ticker (e.g. AAPL, RELIANCE.NS, SUZLON.BO).
    Returns:
        JSON string of balance sheet data, or an error message.
    """
    symbol = symbol.strip()
    logger.info("🏦  Fetching balance sheet for: %s", symbol)
    try:
        def _fetch():
            bs = yf.Ticker(symbol, session=session).balance_sheet
            if bs is None or bs.empty:
                raise ValueError(f"No balance sheet data for {symbol}")
            return bs.to_json(orient="index")
        result = _with_retry(_fetch, f"get_balance_sheet({symbol})")
        logger.debug("Balance sheet fetched for %s.", symbol)
        return result
    except Exception as exc:
        logger.error("Error fetching balance sheet for %s: %s", symbol, exc)
        return f"Error fetching balance sheet for {symbol}: {exc}"


@tool("Get cash flow statement")
def get_cash_flow(symbol: str) -> str:
    """Get the annual cash flow statement for a stock symbol.
    Includes operating, investing, and financing cash flows, and free cash flow.

    Args:
        symbol: Stock ticker (e.g. AAPL, RELIANCE.NS, SUZLON.BO).
    Returns:
        JSON string of cash flow data, or an error message.
    """
    symbol = symbol.strip()
    logger.info("💵  Fetching cash flow statement for: %s", symbol)
    try:
        def _fetch():
            cf = yf.Ticker(symbol, session=session).cashflow
            if cf is None or cf.empty:
                raise ValueError(f"No cash flow data for {symbol}")
            return cf.to_json(orient="index")
        result = _with_retry(_fetch, f"get_cash_flow({symbol})")
        logger.debug("Cash flow statement fetched for %s.", symbol)
        return result
    except Exception as exc:
        logger.error("Error fetching cash flow for %s: %s", symbol, exc)
        return f"Error fetching cash flow for {symbol}: {exc}"


@tool("Get dividend history")
def get_dividend_history(symbol: str) -> str:
    """Get the dividend payment history for a stock symbol.

    Args:
        symbol: Stock ticker (e.g. AAPL, RELIANCE.NS, SUZLON.BO).
    Returns:
        JSON string of dividend history, or a message if no dividends found.
    """
    symbol = symbol.strip()
    logger.info("💰  Fetching dividend history for: %s", symbol)
    try:
        def _fetch():
            divs = yf.Ticker(symbol, session=session).dividends
            if divs is None or divs.empty:
                return None
            # Return last 10 dividends for conciseness
            recent = divs.tail(10)
            return recent.to_json(date_format="iso")
        result = _with_retry(_fetch, f"get_dividend_history({symbol})")
        if result is None:
            return f"No dividend history found for {symbol}. The company may not pay dividends."
        logger.debug("Dividend history fetched for %s.", symbol)
        return result
    except Exception as exc:
        logger.error("Error fetching dividend history for %s: %s", symbol, exc)
        return f"Error fetching dividend history for {symbol}: {exc}"


@tool("Get analyst recommendations")
def get_analyst_recommendations(symbol: str) -> str:
    """Get the latest analyst buy/hold/sell recommendations for a stock symbol.

    Args:
        symbol: Stock ticker (e.g. AAPL, RELIANCE.NS, SUZLON.BO).
    Returns:
        JSON string of analyst recommendations, or an error message.
    """
    symbol = symbol.strip()
    logger.info("🔍  Fetching analyst recommendations for: %s", symbol)
    try:
        def _fetch():
            ticker = yf.Ticker(symbol, session=session)
            rec = ticker.recommendations
            if rec is None or rec.empty:
                raise ValueError(f"No analyst recommendations for {symbol}")
            # Return most recent 10 recommendations
            recent = rec.tail(10)
            return recent.to_json(orient="records", date_format="iso")
        result = _with_retry(_fetch, f"get_analyst_recommendations({symbol})")
        logger.debug("Analyst recommendations fetched for %s.", symbol)
        return result
    except Exception as exc:
        logger.error("Error fetching analyst recommendations for %s: %s", symbol, exc)
        return f"Error fetching analyst recommendations for {symbol}: {exc}"


@tool("Get insider transactions")
def get_insider_transactions(symbol: str) -> str:
    """Get recent insider buying and selling activity for a stock symbol.
    Insider transactions can signal management confidence in the company's future.

    Args:
        symbol: Stock ticker (e.g. AAPL, RELIANCE.NS, SUZLON.BO).
    Returns:
        JSON string of insider transactions, or an error message.
    """
    symbol = symbol.strip()
    logger.info("👤  Fetching insider transactions for: %s", symbol)
    try:
        def _fetch():
            ticker = yf.Ticker(symbol, session=session)
            insiders = ticker.insider_transactions
            if insiders is None or insiders.empty:
                raise ValueError(f"No insider transaction data for {symbol}")
            return insiders.head(15).to_json(orient="records", date_format="iso")
        result = _with_retry(_fetch, f"get_insider_transactions({symbol})")
        logger.debug("Insider transactions fetched for %s.", symbol)
        return result
    except Exception as exc:
        logger.error("Error fetching insider transactions for %s: %s", symbol, exc)
        return f"Error fetching insider transactions for {symbol}: {exc}"


@tool("Get institutional holdings")
def get_institutional_holdings(symbol: str) -> str:
    """Get top institutional holders for a stock symbol.
    Shows which funds and institutions hold the stock and their stake sizes.

    Args:
        symbol: Stock ticker (e.g. AAPL, RELIANCE.NS, SUZLON.BO).
    Returns:
        JSON string of institutional holders, or an error message.
    """
    symbol = symbol.strip()
    logger.info("🏛️   Fetching institutional holdings for: %s", symbol)
    try:
        def _fetch():
            ticker = yf.Ticker(symbol, session=session)
            inst = ticker.institutional_holders
            if inst is None or inst.empty:
                raise ValueError(f"No institutional holder data for {symbol}")
            return inst.to_json(orient="records", date_format="iso")
        result = _with_retry(_fetch, f"get_institutional_holdings({symbol})")
        logger.debug("Institutional holdings fetched for %s.", symbol)
        return result
    except Exception as exc:
        logger.error("Error fetching institutional holdings for %s: %s", symbol, exc)
        return f"Error fetching institutional holdings for {symbol}: {exc}"


@tool("Get technical indicators")
def get_technical_indicators(symbol: str) -> str:
    """Calculate key technical indicators for a stock using 1 year of daily price data.

    Indicators calculated:
      - RSI (14-day): momentum — overbought >70, oversold <30
      - MACD (12/26/9): trend and momentum
      - Bollinger Bands (20-day, ±2σ): volatility bands
      - SMA 50 and SMA 200: trend direction
      - Price vs moving averages: above/below signal
      - Volume analysis: current vs 20-day average

    Args:
        symbol: Stock ticker (e.g. AAPL, RELIANCE.NS, SUZLON.BO).
    Returns:
        JSON string with all calculated indicators and plain-English signals.
    """
    symbol = symbol.strip()
    logger.info("📉  Calculating technical indicators for: %s", symbol)
    try:
        def _fetch():
            hist = yf.Ticker(symbol, session=session).history(period="1y")
            if hist is None or hist.empty:
                raise ValueError(f"No historical price data for {symbol}")
            return hist
        hist = _with_retry(_fetch, f"get_technical_indicators({symbol})")

        close = hist["Close"]
        volume = hist["Volume"]
        current_price = round(float(close.iloc[-1]), 2)

        # RSI
        rsi = _rsi(close)
        rsi_signal = "Overbought (potential sell)" if rsi > 70 else ("Oversold (potential buy)" if rsi < 30 else "Neutral")

        # MACD
        macd_line, signal_line, histogram = _macd(close)
        macd_signal = "Bullish (MACD above signal)" if macd_line > signal_line else "Bearish (MACD below signal)"

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = _bollinger(close)
        if current_price > bb_upper:
            bb_signal = "Price above upper band — potentially overbought"
        elif current_price < bb_lower:
            bb_signal = "Price below lower band — potentially oversold"
        else:
            bb_signal = "Price within bands — normal range"

        # Moving averages
        sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None
        sma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None

        ma_signal = []
        if sma50:
            ma_signal.append(f"{'Above' if current_price > sma50 else 'Below'} SMA50 ({sma50})")
        if sma200:
            ma_signal.append(f"{'Above' if current_price > sma200 else 'Below'} SMA200 ({sma200})")
        if sma50 and sma200:
            if sma50 > sma200:
                ma_signal.append("Golden Cross active (SMA50 > SMA200) — bullish long-term trend")
            else:
                ma_signal.append("Death Cross active (SMA50 < SMA200) — bearish long-term trend")

        # Volume
        avg_vol_20 = round(float(volume.rolling(20).mean().iloc[-1]), 0)
        current_vol = int(volume.iloc[-1])
        vol_ratio = round(current_vol / avg_vol_20, 2) if avg_vol_20 > 0 else None
        vol_signal = (
            f"{'Above' if vol_ratio and vol_ratio > 1 else 'Below'} average "
            f"({'%.1f' % ((vol_ratio - 1) * 100) if vol_ratio else 'N/A'}% vs 20-day avg)"
        )

        result = {
            "symbol": symbol,
            "current_price": current_price,
            "rsi_14": {"value": rsi, "signal": rsi_signal},
            "macd": {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
                "signal": macd_signal,
            },
            "bollinger_bands": {
                "upper": bb_upper,
                "middle": bb_middle,
                "lower": bb_lower,
                "signal": bb_signal,
            },
            "moving_averages": {
                "sma_50": sma50,
                "sma_200": sma200,
                "signals": ma_signal,
            },
            "volume": {
                "current": current_vol,
                "avg_20_day": avg_vol_20,
                "ratio_vs_avg": vol_ratio,
                "signal": vol_signal,
            },
        }

        logger.debug("Technical indicators calculated for %s.", symbol)
        return json.dumps(result, default=str)

    except Exception as exc:
        logger.error("Error calculating technical indicators for %s: %s", symbol, exc)
        return f"Error calculating technical indicators for {symbol}: {exc}"
