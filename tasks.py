"""
CrewAI task definitions for the AI-Powered Stocks Analyser.

Tasks:
  - get_company_financials    → fundamentals + balance sheet + cash flow etc. (Phase 1, parallel)
  - get_company_news          → latest news and sentiment                      (Phase 1, parallel)
  - run_technical_analysis    → RSI, MACD, Bollinger Bands, MAs, volume       (Phase 1, parallel)
  - compare_with_peers        → sector peer valuation comparison               (Phase 1, parallel)
  - analyse                   → full combined analysis                         (Phase 2, sequential)
  - advise                    → structured investment recommendation            (Phase 2, sequential)
"""

import os
from typing import Any, Literal, Tuple

from crewai import Task
from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel, Field

from agents import analyst, data_explorer, fin_expert, news_info_explorer, sector_analyst, technical_analyst
from logger import get_logger

logger = get_logger(__name__)

os.makedirs("task_outputs", exist_ok=True)

# ── Output Schema ──────────────────────────────────────────────────────────────


class InvestmentRecommendation(BaseModel):
    """Structured output model for the final investment recommendation."""

    action: Literal["BUY", "HOLD", "SELL"] = Field(description="Investment action to take")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    target_price: float = Field(description="12-month target price in local currency")
    current_price: float = Field(description="Current market price of the stock")
    reasons: list[str] = Field(description="Key reasons supporting the recommendation (minimum 2)")
    risks: list[str] = Field(description="Key risks an investor should be aware of (minimum 1)")


# ── Guardrail ──────────────────────────────────────────────────────────────────


def validate_recommendation(result: TaskOutput) -> Tuple[bool, Any]:
    """Ensure the investment recommendation meets minimum quality standards."""
    rec = result.pydantic

    if not rec:
        logger.warning("Guardrail failed: recommendation object is empty.")
        return (False, "Recommendation could not be parsed into the expected schema.")

    if not (0.0 <= rec.confidence <= 1.0):
        logger.warning("Guardrail failed: confidence=%.2f out of range.", rec.confidence)
        return (False, "Confidence score must be between 0.0 and 1.0.")

    if len(rec.reasons) < 2:
        logger.warning("Guardrail failed: only %d reason(s) (minimum 2).", len(rec.reasons))
        return (False, "Please provide at least 2 reasons for the recommendation.")

    if len(rec.risks) < 1:
        logger.warning("Guardrail failed: no risks identified.")
        return (False, "Please identify at least 1 key risk.")

    logger.debug(
        "Guardrail passed: action=%s confidence=%.2f reasons=%d risks=%d",
        rec.action, rec.confidence, len(rec.reasons), len(rec.risks),
    )
    return (True, rec)


# ── Tasks ──────────────────────────────────────────────────────────────────────

logger.debug("Defining tasks…")

get_company_financials = Task(
    description=(
        "Gather comprehensive financial data for stock: {stock}. Use the year 2026 as the current year. "
        "Collect: company info, income statements, balance sheet, cash flow statement, "
        "dividend history, analyst recommendations, insider transactions, and institutional holdings."
    ),
    expected_output=(
        "A detailed financial profile of {stock} covering: income statement trends, balance sheet "
        "health (assets, liabilities, debt), cash flow (operating and free cash flow), dividend history, "
        "analyst consensus, insider activity, and top institutional holders. "
        "Highlight key trends and the overall financial health."
    ),
    agent=data_explorer,
)

get_company_news = Task(
    description=(
        "Get the latest news and business information about company: {stock}. "
        "Use the year 2026 as the current year. Look for recent earnings, management changes, "
        "regulatory developments, product launches, and market sentiment."
    ),
    expected_output=(
        "A comprehensive summary of the latest news and developments surrounding the company, "
        "including sentiment analysis (positive / negative / neutral) and potential market impact."
    ),
    agent=news_info_explorer,
)

run_technical_analysis = Task(
    description=(
        "Perform a complete technical analysis on stock: {stock}. "
        "Calculate and interpret RSI, MACD, Bollinger Bands, SMA50, SMA200, and volume trends. "
        "Identify key signals: trend direction, momentum, overbought/oversold conditions, "
        "and notable patterns (e.g. Golden Cross / Death Cross)."
    ),
    expected_output=(
        "A structured technical analysis report covering: current trend (bullish/bearish/sideways), "
        "momentum signals (RSI, MACD), volatility (Bollinger Bands), trend strength (moving averages), "
        "volume analysis, and an overall technical outlook with key price levels to watch."
    ),
    agent=technical_analyst,
    output_file="task_outputs/technical_analysis.md",
)

compare_with_peers = Task(
    description=(
        "Perform a relative valuation analysis for stock: {stock}. "
        "Using your knowledge of the market, identify 4-5 of the closest sector competitors or peers. "
        "Fetch their valuation metrics and build a side-by-side comparison covering: "
        "P/E ratio, P/B ratio, EV/EBITDA, ROE, revenue growth, gross margin, net margin, "
        "debt/equity ratio, and dividend yield. "
        "Conclude whether {stock} is trading at a premium, discount, or in line with peers, "
        "and explain whether that valuation gap is justified based on the company's fundamentals."
    ),
    expected_output=(
        "A peer comparison report containing: "
        "(1) a table of 4-5 peer companies with key metrics side by side, "
        "(2) a relative valuation verdict (premium / discount / in-line vs peers), "
        "(3) an explanation of whether the premium or discount is justified, "
        "(4) key takeaways for the investor."
    ),
    agent=sector_analyst,
    output_file="task_outputs/peer_comparison.md",
)

analyse = Task(
    description=(
        "Using the fundamental financial data, latest news, technical analysis, and peer comparison "
        "gathered, produce a thorough, balanced analysis of the stock. "
        "Cover: financial health, valuation vs peers, growth prospects, technical outlook, and key risks."
    ),
    expected_output=(
        "A comprehensive stock analysis covering: "
        "(1) financial health and fundamentals, "
        "(2) news and sentiment summary, "
        "(3) technical outlook, "
        "(4) peer / relative valuation assessment, "
        "(5) key risks and opportunities."
    ),
    agent=analyst,
    context=[get_company_financials, get_company_news, run_technical_analysis, compare_with_peers],
    output_file="task_outputs/financial_analysis.md",
)

advise = Task(
    description=(
        "Based on the comprehensive analysis provided, make a clear investment recommendation. "
        "Check the current stock price and provide a realistic 12-month target price. "
        "Your recommendation must include: action (BUY/HOLD/SELL), confidence score, "
        "target price, current price, key reasons (at least 2), and key risks (at least 1)."
    ),
    expected_output=(
        "A structured investment recommendation with: action, confidence score (0–1), "
        "12-month target price, current price, reasons list, and risks list."
    ),
    agent=fin_expert,
    context=[analyse],
    output_pydantic=InvestmentRecommendation,
    guardrail=validate_recommendation,
    output_file="task_outputs/investment_recommendation.md",
)

logger.debug("All 6 tasks defined successfully.")
