"""
Shared fixtures and environment setup for the test suite.

Environment variables are injected here — before any test module imports —
so that module-level initialisation in tools.py / agents.py / tasks.py
(EXASearchTool, LLM) can succeed without real API credentials.
"""

import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so test files can import project modules.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Dummy credentials ─────────────────────────────────────────────────────────
# Must be set before any project module is imported (agents.py, tools.py read
# these at module level).  setdefault() is safe — real values win if present.
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key-ci")
os.environ.setdefault("EXA_API_KEY",        "test-exa-key-ci")

# ── Shared fixtures ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.Series:
    """250 trading days of normally-distributed synthetic close prices."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, 250)
    return pd.Series(100.0 * np.cumprod(1 + returns), name="Close")


@pytest.fixture
def trending_up_prices() -> pd.Series:
    """Strong uptrend — average +0.4 % per day."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0.004, 0.003, 250)
    return pd.Series(100.0 * np.cumprod(1 + returns), name="Close")


@pytest.fixture
def trending_down_prices() -> pd.Series:
    """Strong downtrend — average -0.4 % per day."""
    rng = np.random.default_rng(2)
    returns = rng.normal(-0.004, 0.003, 250)
    return pd.Series(100.0 * np.cumprod(1 + returns), name="Close")
