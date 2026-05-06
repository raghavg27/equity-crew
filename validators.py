"""
Startup validation utilities for Equity Crew.

Validates:
  - Required environment variables are present and non-empty.
  - Stock symbol format is sane before passing to agents.
  - Live API connectivity for OpenRouter, EXA, and yfinance (optional --validate flag).
"""

import os
import re
import sys
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_ENV_VARS: dict[str, str] = {
    "OPENROUTER_API_KEY": (
        "OpenRouter API key for LLM access — get one at https://openrouter.ai"
    ),
    "EXA_API_KEY": (
        "EXA API key for real-time news search — get one at https://exa.ai"
    ),
}

# Accepts symbols like: AAPL, RELIANCE.NS, SUZLON.BO, ^NSEI, BRK-B, 005930.KS
_VALID_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-\^&]{1,20}$", re.IGNORECASE)

# ── Environment Variable Validation ───────────────────────────────────────────


def validate_env_vars() -> bool:
    """
    Check that all required environment variables are present and non-empty.

    Returns:
        True if all variables are set, False otherwise.
    """
    load_dotenv()
    missing: list[tuple[str, str]] = []

    for var, description in REQUIRED_ENV_VARS.items():
        value = os.getenv(var, "").strip()
        if not value:
            missing.append((var, description))
        else:
            logger.debug("Env var %s is set (%d chars)", var, len(value))

    if missing:
        logger.error("❌  Missing required environment variables:")
        for var, desc in missing:
            logger.error("    • %s — %s", var, desc)
        logger.error(
            "    ➜  Create or update your .env file in the project root and try again."
        )
        return False

    logger.info("✅  All required environment variables are present.")
    return True


# ── Stock Symbol Validation ───────────────────────────────────────────────────


def validate_stock_symbol(symbol: str) -> bool:
    """
    Perform a lightweight format-check on a stock symbol.

    Rules:
      - Must not be blank.
      - Maximum 20 characters.
      - Only letters, digits, dots, hyphens, carets, and ampersands.

    Args:
        symbol: The raw ticker string supplied by the user.

    Returns:
        True if the symbol looks valid, False otherwise.
    """
    symbol = symbol.strip()

    if not symbol:
        logger.error("❌  Stock symbol cannot be empty.")
        return False

    if not _VALID_SYMBOL_RE.match(symbol):
        logger.error(
            "❌  Stock symbol '%s' contains invalid characters or is too long.", symbol
        )
        logger.error(
            "    ➜  Valid examples: AAPL  RELIANCE.NS  SUZLON.BO  ^NSEI  BRK-B"
        )
        return False

    logger.info("✅  Stock symbol '%s' format looks valid.", symbol)
    return True


def resolve_stock_symbol(symbol: str) -> str | None:
    """
    Auto-detects if a stock symbol needs an exchange suffix (.NS or .BO)
    by testing it against yfinance.
    
    Returns the valid symbol string, or None if it cannot be found.
    """
    symbol = symbol.strip().upper()
    if not validate_stock_symbol(symbol):
        return None
        
    import yfinance as yf
    import warnings
    
    def _is_valid(sym: str) -> bool:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # A quick check using history. If it's empty, it's invalid.
                t = yf.Ticker(sym)
                hist = t.history(period="1d")
                return not hist.empty
            except Exception:
                return False

    logger.info("🔍  Validating symbol '%s' with yfinance...", symbol)
    
    if _is_valid(symbol):
        return symbol
        
    if "." not in symbol:
        logger.info("⚠️   Symbol '%s' not found. Testing with .NS suffix...", symbol)
        if _is_valid(f"{symbol}.NS"):
            logger.info("✅  Auto-corrected to %s", f"{symbol}.NS")
            return f"{symbol}.NS"
            
        logger.info("⚠️   Symbol '%s.NS' not found. Testing with .BO suffix...", symbol)
        if _is_valid(f"{symbol}.BO"):
            logger.info("✅  Auto-corrected to %s", f"{symbol}.BO")
            return f"{symbol}.BO"

    logger.error(
        "❌  Could not find market data for '%s' (tried bare, .NS, and .BO).\n"
        "    ➜  Check if the symbol is correct or if it was delisted.", symbol
    )
    return None


# ── Live API Connectivity Checks ──────────────────────────────────────────────


def _http_get(url: str, headers: dict[str, str], timeout: int = 10) -> int:
    """
    Perform a simple HTTP GET and return the HTTP status code.
    Returns -1 if a network error occurs.
    """
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        logger.debug("HTTP error reaching %s: %s", url, e)
        return -1


def check_openrouter_api() -> bool:
    """Test connectivity and auth against the OpenRouter API."""
    logger.info("🔌  Checking OpenRouter API...")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        logger.error("❌  OPENROUTER_API_KEY is not set.")
        return False

    status = _http_get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    if status == 200:
        logger.info("✅  OpenRouter API: connected and authenticated.")
        return True
    elif status == 401:
        logger.error(
            "❌  OpenRouter API: authentication failed (HTTP 401).\n"
            "    ➜  Your OPENROUTER_API_KEY is invalid or expired.\n"
            "    ➜  Visit https://openrouter.ai/keys to generate a new key."
        )
        return False
    elif status == -1:
        logger.error(
            "❌  OpenRouter API: could not reach the server.\n"
            "    ➜  Check your internet connection."
        )
        return False
    else:
        logger.warning(
            "⚠️   OpenRouter API: unexpected HTTP %d — connectivity may be partial.",
            status,
        )
        return True  # Non-fatal; let the run proceed


def check_exa_api() -> bool:
    """Test connectivity and auth against the EXA API."""
    logger.info("🔌  Checking EXA API...")
    api_key = os.getenv("EXA_API_KEY", "").strip()

    if not api_key:
        logger.error("❌  EXA_API_KEY is not set.")
        return False

    # EXA doesn't have a cheap /ping endpoint; a 401 means bad key, anything
    # reachable (including 404/405) means the server is up.
    status = _http_get(
        "https://api.exa.ai",
        headers={"x-api-key": api_key},
    )

    if status == 401:
        logger.error(
            "❌  EXA API: authentication failed (HTTP 401).\n"
            "    ➜  Your EXA_API_KEY is invalid or expired.\n"
            "    ➜  Visit https://exa.ai to verify your key."
        )
        return False
    elif status == -1:
        logger.error(
            "❌  EXA API: could not reach the server.\n"
            "    ➜  Check your internet connection."
        )
        return False
    else:
        logger.info("✅  EXA API: server reachable (HTTP %d).", status)
        return True


def check_yfinance() -> bool:
    """Test that yfinance can fetch live data."""
    logger.info("🔌  Checking yfinance data access...")
    try:
        import yfinance as yf

        ticker = yf.Ticker("AAPL")
        info = ticker.info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price:
            logger.info("✅  yfinance: working (AAPL test price = %s)", price)
        else:
            logger.warning(
                "⚠️   yfinance: no price returned for AAPL — data may be delayed."
            )
        return True
    except Exception as e:
        logger.error("❌  yfinance: error during connectivity test — %s", e)
        return False


def run_connectivity_checks() -> bool:
    """
    Run all API connectivity checks and return True only if all pass.
    Intended to be triggered by the --validate CLI flag.
    """
    logger.info("━" * 60)
    logger.info("Running API connectivity checks…")
    logger.info("━" * 60)

    results = {
        "OpenRouter": check_openrouter_api(),
        "EXA": check_exa_api(),
        "yfinance": check_yfinance(),
    }

    logger.info("━" * 60)
    all_ok = all(results.values())
    if all_ok:
        logger.info("✅  All connectivity checks passed. Ready to run.")
    else:
        failed = [name for name, ok in results.items() if not ok]
        logger.error(
            "❌  Some checks failed: %s\n    ➜  Fix the issues above before running the analyser.",
            ", ".join(failed),
        )
    logger.info("━" * 60)
    return all_ok
