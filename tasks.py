"""
CrewAI task definitions for the AI-Powered Stocks Analyser.

Tasks:
  - get_company_financials  → income statements + fundamentals (Phase 1, parallel)
  - get_company_news        → latest news and sentiment    (Phase 1, parallel)
  - analyse                 → full financial + news analysis (Phase 2, sequential)
  - advise                  → structured investment recommendation (Phase 2, sequential)

Output files are written to the task_outputs/ directory.
"""

import os
from typing import Any, Literal, Tuple

from crewai import Task
from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel, Field

from agents import analyst, data_explorer, fin_expert, news_info_explorer
from logger import get_logger

logger = get_logger(__name__)

os.makedirs("task_outputs", exist_ok=True)

# ── Output Schema ──────────────────────────────────────────────────────────────


class InvestmentRecommendation(BaseModel):
    """Structured output model for the final investment recommendation."""

    action: Literal["BUY", "HOLD", "SELL"] = Field(
        description="Investment action to take"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 (no confidence) and 1.0 (very confident)"
    )
    target_price: float = Field(description="12-month target price in local currency")
    current_price: float = Field(description="Current market price of the stock")
    reasons: list[str] = Field(
        description="Key reasons supporting the recommendation (minimum 2)"
    )
    risks: list[str] = Field(
        description="Key risks an investor should be aware of (minimum 1)"
    )


# ── Guardrail ─────────────────────────────────────────────────────────────────


def validate_recommendation(result: TaskOutput) -> Tuple[bool, Any]:
    """
    Guardrail to ensure the investment recommendation meets minimum quality standards.

    Checks:
      - confidence is in [0.0, 1.0]
      - at least 2 reasons are provided
      - at least 1 risk is identified
    """
    rec = result.pydantic

    if not rec:
        logger.warning("Guardrail failed: recommendation object is empty.")
        return (False, "Recommendation output could not be parsed into the expected schema.")

    if not (0.0 <= rec.confidence <= 1.0):
        logger.warning(
            "Guardrail failed: confidence=%.2f is out of range [0, 1].", rec.confidence
        )
        return (False, "Confidence score must be between 0.0 and 1.0.")

    if len(rec.reasons) < 2:
        logger.warning(
            "Guardrail failed: only %d reason(s) provided (minimum 2).", len(rec.reasons)
        )
        return (False, "Please provide at least 2 reasons for the recommendation.")

    if len(rec.risks) < 1:
        logger.warning("Guardrail failed: no risks identified.")
        return (False, "Please identify at least 1 key risk.")

    logger.debug(
        "Guardrail passed: action=%s confidence=%.2f reasons=%d risks=%d",
        rec.action,
        rec.confidence,
        len(rec.reasons),
        len(rec.risks),
    )
    return (True, rec)


# ── Tasks ──────────────────────────────────────────────────────────────────────

logger.debug("Defining tasks…")

get_company_financials = Task(
    description=(
        "Get financial data like income statements and other fundamental ratios "
        "for stock: {stock}. Use the year 2026 as the current year."
    ),
    expected_output=(
        "Detailed information from the income statement and key financial ratios for {stock}. "
        "Indicate the current financial status and the trend over the reporting period."
    ),
    agent=data_explorer,
)

get_company_news = Task(
    description=(
        "Get the latest news and business information about company: {stock}. "
        "Use the year 2026 as the current year."
    ),
    expected_output=(
        "A comprehensive summary of the latest news, developments, and market sentiment "
        "surrounding the company."
    ),
    agent=news_info_explorer,
)

analyse = Task(
    description=(
        "Using the financial data and latest news gathered, make a thorough analysis "
        "of the stock. Cover financial health, valuation, growth prospects, and risks."
    ),
    expected_output=(
        "A comprehensive analysis of the stock outlining financial health, stock valuation, "
        "key risks, and relevant news developments."
    ),
    agent=analyst,
    context=[get_company_financials, get_company_news],
    output_file="task_outputs/financial_analysis.md",
)

advise = Task(
    description=(
        "Based on the analysis provided and the current stock price, make a clear investment "
        "recommendation. Provide the action (BUY / HOLD / SELL), confidence level, 12-month "
        "target price, key reasons, and key risks."
    ),
    expected_output=(
        "A structured investment recommendation with: action, confidence score, target price, "
        "current price, list of reasons, and list of risks."
    ),
    agent=fin_expert,
    context=[analyse],
    output_pydantic=InvestmentRecommendation,
    guardrail=validate_recommendation,
    output_file="task_outputs/investment_recommendation.md",
)

logger.debug("All tasks defined successfully.")
