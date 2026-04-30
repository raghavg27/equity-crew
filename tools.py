"""
Custom tools for the AI-Powered Stocks Analyser.

Provides yfinance-based financial data tools and the EXA news search tool.
All yfinance calls are wrapped with exponential-backoff retry logic to handle
transient network errors and rate limits gracefully.
"""

import json
import os
import time
from typing import Any, Callable

import yfinance as yf
from crewai.tools import tool
from crewai_tools import EXASearchTool
from curl_cffi import requests
from dotenv import load_dotenv

from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# ── Shared HTTP session (impersonates Chrome to avoid bot-detection) ───────────
session = requests.Session(impersonate="chrome")

# ── EXA search tool ────────────────────────────────────────────────────────────
os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY", "")
exa_search_tool = EXASearchTool()


# ── Retry helper ───────────────────────────────────────────────────────────────

def _with_retry(
    fn: Callable[[], Any],
    label: str,
    max_retries: int = 3,
    base_delay: float = 1.5,
) -> Any:
    """
    Call *fn* up to *max_retries* times with exponential backoff.

    Args:
        fn:          Zero-argument callable that performs the actual work.
        label:       Human-readable description used in log messages.
        max_retries: Maximum number of attempts (default 3).
        base_delay:  Initial sleep duration in seconds; doubles each attempt.

    Returns:
        The return value of *fn* on the first successful attempt.

    Raises:
        The last exception raised by *fn* after all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("Attempt %d/%d for '%s'", attempt, max_retries, label)
            result = fn()
            if attempt > 1:
                logger.info("✅  '%s' succeeded on attempt %d.", label, attempt)
            return result
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
            if attempt < max_retries:
                logger.warning(
                    "⚠️   Attempt %d/%d failed for '%s': %s — retrying in %.1fs…",
                    attempt,
                    max_retries,
                    label,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "❌  All %d attempts failed for '%s': %s",
                    max_retries,
                    label,
                    exc,
                )

    raise last_exc  # type: ignore[misc]


# ── Financial Tools ────────────────────────────────────────────────────────────

@tool("Get current stock price")
def get_current_stock_price(symbol: str) -> str:
    """Use this function to get the current stock price for a given symbol.

    Args:
        symbol (str): The stock symbol (e.g. AAPL, RELIANCE.NS, SUZLON.BO).

    Returns:
        str: The current stock price as a string, or a descriptive error message.
    """
    symbol = symbol.strip()
    logger.info("📈  Fetching current stock price for: %s", symbol)

    def _fetch():
        time.sleep(0.5)  # Polite rate-limit buffer
        stock = yf.Ticker(symbol, session=session)
        price = stock.info.get("regularMarketPrice") or stock.info.get("currentPrice")
        return price

    try:
        price = _with_retry(_fetch, label=f"get_current_stock_price({symbol})")
        if price:
            logger.debug("Price for %s: %s", symbol, price)
            return f"{price:.2f}"
        logger.warning("No price found for symbol '%s'.", symbol)
        return f"Could not fetch current price for {symbol}. The symbol may be incorrect or delisted."
    except Exception as exc:
        logger.error("Error fetching price for %s: %s", symbol, exc)
        return f"Error fetching current price for {symbol}: {exc}"


@tool("Get company info")
def get_company_info(symbol: str) -> str:
    """Use this function to get company information and current financial snapshot for a given stock symbol.

    Args:
        symbol (str): The stock symbol (e.g. AAPL, RELIANCE.NS, SUZLON.BO).

    Returns:
        str: JSON string containing the company profile and key financial metrics,
             or a descriptive error message.
    """
    symbol = symbol.strip()
    logger.info("🏢  Fetching company info for: %s", symbol)

    def _fetch():
        return yf.Ticker(symbol, session=session).info

    try:
        raw = _with_retry(_fetch, label=f"get_company_info({symbol})")

        if not raw:
            logger.warning("Empty info returned for symbol '%s'.", symbol)
            return f"Could not fetch company info for {symbol}. The symbol may be incorrect."

        cleaned = {
            "Name": raw.get("shortName"),
            "Symbol": raw.get("symbol"),
            "Current Stock Price": (
                f"{raw.get('regularMarketPrice') or raw.get('currentPrice')} "
                f"{raw.get('currency', 'USD')}"
            ),
            "Market Cap": (
                f"{raw.get('marketCap') or raw.get('enterpriseValue')} "
                f"{raw.get('currency', 'USD')}"
            ),
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

        logger.debug("Company info fetched successfully for %s: %s", symbol, cleaned.get("Name"))
        return json.dumps(cleaned, default=str)

    except Exception as exc:
        logger.error("Error fetching company info for %s: %s", symbol, exc)
        return f"Error fetching company info for {symbol}: {exc}"


@tool("Get income statements")
def get_income_statements(symbol: str) -> str:
    """Use this function to get income statements for a given stock symbol.

    Args:
        symbol (str): The stock symbol (e.g. AAPL, RELIANCE.NS, SUZLON.BO).

    Returns:
        str: JSON string containing income statements, or a descriptive error message.
    """
    symbol = symbol.strip()
    logger.info("📊  Fetching income statements for: %s", symbol)

    def _fetch():
        stock = yf.Ticker(symbol, session=session)
        financials = stock.financials
        if financials is None or financials.empty:
            raise ValueError(f"No income statement data available for {symbol}")
        return financials.to_json(orient="index")

    try:
        result = _with_retry(_fetch, label=f"get_income_statements({symbol})")
        logger.debug("Income statements fetched successfully for %s.", symbol)
        return result
    except Exception as exc:
        logger.error("Error fetching income statements for %s: %s", symbol, exc)
        return f"Error fetching income statements for {symbol}: {exc}"
