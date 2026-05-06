"""
Unit tests for tasks.py

Covers:
  - InvestmentRecommendation  — Pydantic schema validation (no LLM calls)
  - validate_recommendation() — guardrail logic (no LLM calls)

All tests work without any API credentials because we only instantiate
Pydantic models and call the pure-Python guardrail function.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tasks import InvestmentRecommendation, validate_recommendation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_rec(**overrides) -> InvestmentRecommendation:
    """Build a fully valid recommendation, applying optional field overrides."""
    defaults = dict(
        action="BUY",
        confidence=0.82,
        target_price=200.0,
        current_price=150.0,
        reasons=["Strong revenue growth year-over-year", "Expanding operating margins"],
        risks=["Elevated valuation vs sector peers"],
    )
    defaults.update(overrides)
    return InvestmentRecommendation(**defaults)


def _mock_task_output(rec) -> MagicMock:
    """Wrap a recommendation in a mock TaskOutput (only .pydantic is read)."""
    mock = MagicMock()
    mock.pydantic = rec
    return mock


# ── InvestmentRecommendation schema ───────────────────────────────────────────

class TestInvestmentRecommendationSchema:
    @pytest.mark.parametrize("action", ["BUY", "HOLD", "SELL"])
    def test_all_valid_actions_accepted(self, action):
        rec = _valid_rec(action=action)
        assert rec.action == action

    def test_invalid_action_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _valid_rec(action="MAYBE")

    def test_confidence_stored_as_float(self):
        rec = _valid_rec(confidence=0.75)
        assert rec.confidence == 0.75

    def test_confidence_non_numeric_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _valid_rec(confidence="high")

    def test_reasons_list_stored_correctly(self):
        reasons = ["Reason one", "Reason two", "Reason three"]
        rec = _valid_rec(reasons=reasons)
        assert rec.reasons == reasons

    def test_risks_list_stored_correctly(self):
        risks = ["Risk alpha", "Risk beta"]
        rec = _valid_rec(risks=risks)
        assert rec.risks == risks

    def test_target_and_current_price_stored(self):
        rec = _valid_rec(target_price=310.0, current_price=271.35)
        assert rec.target_price == 310.0
        assert rec.current_price == 271.35


# ── validate_recommendation guardrail ─────────────────────────────────────────

class TestValidateRecommendation:
    def test_valid_recommendation_passes(self):
        ok, result = validate_recommendation(_mock_task_output(_valid_rec()))
        assert ok is True
        assert isinstance(result, InvestmentRecommendation)

    def test_null_pydantic_fails(self):
        mock = MagicMock()
        mock.pydantic = None
        ok, msg = validate_recommendation(mock)
        assert ok is False
        assert isinstance(msg, str)

    def test_confidence_above_1_fails(self):
        rec = _valid_rec(confidence=1.5)
        ok, msg = validate_recommendation(_mock_task_output(rec))
        assert ok is False
        assert "confidence" in msg.lower()

    def test_confidence_below_0_fails(self):
        rec = _valid_rec(confidence=-0.1)
        ok, msg = validate_recommendation(_mock_task_output(rec))
        assert ok is False
        assert "confidence" in msg.lower()

    def test_only_one_reason_fails(self):
        rec = _valid_rec(reasons=["Single reason only"])
        ok, msg = validate_recommendation(_mock_task_output(rec))
        assert ok is False
        assert "reason" in msg.lower()

    def test_empty_reasons_fails(self):
        rec = _valid_rec(reasons=[])
        ok, msg = validate_recommendation(_mock_task_output(rec))
        assert ok is False

    def test_no_risks_fails(self):
        rec = _valid_rec(risks=[])
        ok, msg = validate_recommendation(_mock_task_output(rec))
        assert ok is False
        assert "risk" in msg.lower()

    def test_exactly_two_reasons_one_risk_passes(self):
        rec = _valid_rec(
            reasons=["Reason A", "Reason B"],
            risks=["Risk X"],
        )
        ok, _ = validate_recommendation(_mock_task_output(rec))
        assert ok is True

    def test_boundary_confidence_zero_passes(self):
        rec = _valid_rec(confidence=0.0)
        ok, _ = validate_recommendation(_mock_task_output(rec))
        assert ok is True

    def test_boundary_confidence_one_passes(self):
        rec = _valid_rec(confidence=1.0)
        ok, _ = validate_recommendation(_mock_task_output(rec))
        assert ok is True
