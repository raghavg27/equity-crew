"""
Unit tests for tools.py

Covers:
  - _rsi()        — RSI math properties (no I/O)
  - _macd()       — MACD math properties (no I/O)
  - _bollinger()  — Bollinger Band math properties (no I/O)
  - _with_retry() — retry / backoff logic (time.sleep mocked)
  - get_company_info  — yfinance call mocked
"""

import json
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from tools import _bollinger, _macd, _rsi, _with_retry, get_company_info


# ── RSI ───────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_output_is_float(self, sample_prices):
        assert isinstance(_rsi(sample_prices), float)

    def test_always_bounded_0_to_100(self, sample_prices):
        assert 0.0 <= _rsi(sample_prices) <= 100.0

    def test_pure_uptrend_gives_rsi_near_100(self):
        # Strictly increasing prices → all gains, zero losses → RSI = 100
        prices = pd.Series(range(20, 70), dtype=float)
        assert _rsi(prices) == 100.0

    def test_pure_downtrend_gives_rsi_near_0(self):
        # Strictly decreasing prices → zero gains, all losses → RSI = 0
        prices = pd.Series(range(70, 20, -1), dtype=float)
        assert _rsi(prices) == 0.0

    def test_sustained_uptrend_gives_high_rsi(self, trending_up_prices):
        assert _rsi(trending_up_prices) > 60

    def test_sustained_downtrend_gives_low_rsi(self, trending_down_prices):
        assert _rsi(trending_down_prices) < 40


# ── MACD ──────────────────────────────────────────────────────────────────────

class TestMACD:
    def test_returns_three_floats(self, sample_prices):
        result = _macd(sample_prices)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_histogram_equals_macd_minus_signal(self, sample_prices):
        macd_line, signal_line, histogram = _macd(sample_prices)
        # Histogram is defined as MACD line − signal line
        assert abs(histogram - (macd_line - signal_line)) < 1e-6

    def test_uptrend_macd_line_above_signal(self, trending_up_prices):
        macd_line, signal_line, _ = _macd(trending_up_prices)
        assert macd_line > signal_line

    def test_downtrend_macd_line_below_signal(self, trending_down_prices):
        macd_line, signal_line, _ = _macd(trending_down_prices)
        assert macd_line < signal_line


# ── Bollinger Bands ───────────────────────────────────────────────────────────

class TestBollinger:
    def test_returns_three_floats(self, sample_prices):
        result = _bollinger(sample_prices)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_band_ordering_upper_gt_middle_gt_lower(self, sample_prices):
        upper, middle, lower = _bollinger(sample_prices)
        assert upper > middle > lower

    def test_high_volatility_produces_wider_bands(self):
        rng = np.random.default_rng(99)
        low_vol  = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.1,  50)))
        high_vol = pd.Series(100.0 + np.cumsum(rng.normal(0, 5.0,  50)))
        u_lv, _, l_lv = _bollinger(low_vol)
        u_hv, _, l_hv = _bollinger(high_vol)
        assert (u_hv - l_hv) > (u_lv - l_lv)

    def test_middle_band_is_20day_sma(self, sample_prices):
        _, middle, _ = _bollinger(sample_prices)
        expected_sma = round(float(sample_prices.rolling(20).mean().iloc[-1]), 2)
        assert middle == expected_sma


# ── _with_retry ───────────────────────────────────────────────────────────────

class TestWithRetry:
    @patch("tools.time.sleep")
    def test_succeeds_first_try_no_sleep(self, mock_sleep):
        fn = MagicMock(return_value="ok")
        result = _with_retry(fn, "test_label")
        assert result == "ok"
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("tools.time.sleep")
    def test_retries_on_transient_failure_and_succeeds(self, mock_sleep):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return "recovered"

        result = _with_retry(flaky, "flaky_label")
        assert result == "recovered"
        assert calls["n"] == 2
        mock_sleep.assert_called_once()

    @patch("tools.time.sleep")
    def test_raises_after_all_retries_exhausted(self, mock_sleep):
        fn = MagicMock(side_effect=ValueError("always fails"))
        with pytest.raises(ValueError, match="always fails"):
            _with_retry(fn, "always_fail_label", max_retries=3)
        assert fn.call_count == 3

    @patch("tools.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        fn = MagicMock(side_effect=[RuntimeError("e1"), RuntimeError("e2"), "ok"])
        _with_retry(fn, "backoff_label", base_delay=2.0)
        # First sleep: 2.0 s, second sleep: 4.0 s
        assert mock_sleep.call_args_list == [call(2.0), call(4.0)]


# ── get_company_info (mocked yfinance) ────────────────────────────────────────

class TestGetCompanyInfo:
    _MOCK_INFO = {
        "shortName":          "Apple Inc.",
        "symbol":             "AAPL",
        "regularMarketPrice": 195.0,
        "currency":           "USD",
        "marketCap":          3_000_000_000_000,
        "sector":             "Technology",
        "industry":           "Consumer Electronics",
        "country":            "United States",
        "trailingEps":        6.43,
        "trailingPE":         30.3,
        "fiftyTwoWeekLow":    164.0,
        "fiftyTwoWeekHigh":   237.0,
        "revenueGrowth":      0.04,
        "grossMargins":       0.46,
        "ebitda":             132_000_000_000,
    }

    @patch("tools.yf.Ticker")
    def test_returns_valid_json(self, mock_ticker_class):
        mock_ticker_class.return_value.info = self._MOCK_INFO
        raw = get_company_info.func("AAPL")
        data = json.loads(raw)
        assert data["Name"] == "Apple Inc."
        assert data["Sector"] == "Technology"

    @patch("tools.yf.Ticker")
    def test_returns_error_string_on_empty_info(self, mock_ticker_class):
        mock_ticker_class.return_value.info = {}
        result = get_company_info.func("BADINPUT")
        # Should not raise; returns a human-readable fallback string
        assert isinstance(result, str)
        assert "Could not fetch" in result or "Error" in result

    @patch("tools.yf.Ticker")
    def test_handles_yfinance_exception_gracefully(self, mock_ticker_class):
        mock_ticker_class.side_effect = Exception("yfinance blew up")
        result = get_company_info.func("AAPL")
        assert "Error" in result
        assert "AAPL" in result
