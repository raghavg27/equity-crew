import os
from typing import Any, Literal, Tuple

from crewai import Task
from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel, Field

from agents import analyst, data_explorer, fin_expert, news_info_explorer

os.makedirs("task_outputs", exist_ok=True)


class InvestmentRecommendation(BaseModel):
    action: Literal["BUY", "HOLD", "SELL"] = Field(description="Investment action")
    confidence: float = Field(description="Confidence score 0.0 to 1.0")
    target_price: float = Field(description="12-month target price")
    current_price: float = Field(description="Current stock price")
    reasons: list[str] = Field(description="Key reasons for recommendation")
    risks: list[str] = Field(description="Key risks to consider")


def validate_recommendation(result: TaskOutput) -> Tuple[bool, Any]:
    rec = result.pydantic
    if not rec or rec.confidence < 0 or rec.confidence > 1:
        return (False, "Confidence must be between 0.0 and 1.0")
    if len(rec.reasons) < 2:
        return (False, "Must provide at least 2 reasons for the recommendation")
    if len(rec.risks) < 1:
        return (False, "Must provide at least 1 risk")
    return (True, rec)


get_company_financials = Task(
    description="Get financial data like income statements and other fundamental ratios for stock: {stock}. "
    "Use the year 2026 as the current year.",
    expected_output="Detailed information from income statement, key ratios for {stock}. "
    "Indicate also about current financial status and trend over the period.",
    agent=data_explorer,
)

get_company_news = Task(
    description="Get latest news and business information about company: {stock}. "
    "Use the year 2026 as the current year.",
    expected_output="Latest news and business information about the company. Provide a summary also.",
    agent=news_info_explorer,
)

analyse = Task(
    description="Make thorough analysis based on given financial data and latest news of a stock.",
    expected_output="Comprehensive analysis of a stock outlining financial health, stock valuation, risks, and news.",
    agent=analyst,
    context=[get_company_financials, get_company_news],
    output_file="task_outputs/financial_analysis.md",
)

advise = Task(
    description="Make a recommendation about investing in a stock, based on analysis provided and current stock price. "
    "Explain the reasons.",
    expected_output="A structured investment recommendation with action, confidence, target price, reasons, and risks.",
    agent=fin_expert,
    context=[analyse],
    output_pydantic=InvestmentRecommendation,
    guardrail=validate_recommendation,
    output_file="task_outputs/investment_recommendation.md",
)
