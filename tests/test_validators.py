"""
Unit tests for validators.py

Covers:
  - validate_stock_symbol()  — regex format checks (no network)
  - validate_env_vars()      — env-var presence checks (no network)
  - resolve_stock_symbol()   — yfinance resolution (mocked)
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from validators import validate_env_vars, validate_stock_symbol, resolve_stock_symbol


# ── validate_stock_symbol ─────────────────────────────────────────────────────

class TestValidateStockSymbol:
    @pytest.mark.parametrize("symbol", [
        "AAPL",           # plain US ticker
        "RELIANCE.NS",    # NSE
        "SUZLON.BO",      # BSE
        "^NSEI",          # index with caret
        "BRK-B",          # hyphenated ticker
        "005930.KS",      # Korean exchange
    ])
    def test_valid_symbols(self, symbol):
        assert validate_stock_symbol(symbol) is True

    def test_empty_string_is_invalid(self):
        assert validate_stock_symbol("") is False

    def test_symbol_too_long_is_invalid(self):
        assert validate_stock_symbol("A" * 21) is False

    def test_exclamation_mark_is_invalid(self):
        assert validate_stock_symbol("AAPL!") is False

    def test_space_is_invalid(self):
        assert validate_stock_symbol("AA PL") is False

    def test_whitespace_only_is_invalid(self):
        assert validate_stock_symbol("   ") is False


# ── validate_env_vars ─────────────────────────────────────────────────────────

class TestValidateEnvVars:
    def test_returns_true_when_both_keys_set(self):
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "sk-or-test",
            "EXA_API_KEY": "exa-test",
        }):
            assert validate_env_vars() is True

    def test_returns_false_when_openrouter_key_missing(self):
        env = {"OPENROUTER_API_KEY": "", "EXA_API_KEY": "exa-test"}
        with patch.dict(os.environ, env):
            with patch("validators.load_dotenv"):   # prevent .env from overriding
                assert validate_env_vars() is False

    def test_returns_false_when_exa_key_missing(self):
        env = {"OPENROUTER_API_KEY": "sk-or-test", "EXA_API_KEY": ""}
        with patch.dict(os.environ, env):
            with patch("validators.load_dotenv"):
                assert validate_env_vars() is False

    def test_returns_false_when_both_keys_missing(self):
        env = {"OPENROUTER_API_KEY": "", "EXA_API_KEY": ""}
        with patch.dict(os.environ, env):
            with patch("validators.load_dotenv"):
                assert validate_env_vars() is False


# ── resolve_stock_symbol ──────────────────────────────────────────────────────

class TestResolveStockSymbol:
    def _make_ticker(self, has_data: bool):
        """Return a mock yfinance Ticker whose history() indicates data availability."""
        import pandas as pd
        mock = MagicMock()
        mock.history.return_value = (
            pd.DataFrame({"Close": [100.0]}) if has_data else pd.DataFrame()
        )
        return mock

    def test_valid_bare_symbol_returned_unchanged(self):
        # validators.py imports yfinance lazily inside the function body
        with patch("yfinance.Ticker", return_value=self._make_ticker(True)):
            result = resolve_stock_symbol("AAPL")
        assert result == "AAPL"

    def test_auto_appends_ns_suffix_when_bare_fails(self):
        def side_effect(sym, **_):
            # bare symbol has no data; .NS does
            return self._make_ticker("." in sym)

        with patch("yfinance.Ticker", side_effect=side_effect):
            result = resolve_stock_symbol("RELIANCE")
        assert result == "RELIANCE.NS"

    def test_returns_none_when_symbol_not_found(self):
        with patch("yfinance.Ticker", return_value=self._make_ticker(False)):
            result = resolve_stock_symbol("FAKEXYZ")
        assert result is None

    def test_invalid_format_returns_none(self):
        result = resolve_stock_symbol("INVALID SYMBOL!")
        assert result is None
