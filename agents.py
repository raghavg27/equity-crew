"""
CrewAI agent definitions for the AI-Powered Stocks Analyser.

Agents:
  - data_explorer       → fetches financial statements and company fundamentals
  - news_info_explorer  → searches for latest news and market sentiment
  - technical_analyst   → calculates and interprets technical indicators
  - sector_analyst      → compares the stock against 4-5 industry peers on key valuation metrics
  - analyst             → synthesises all gathered data into a coherent analysis
  - fin_expert          → produces a final BUY / HOLD / SELL investment recommendation
"""

import os

from crewai import Agent
from crewai.llm import LLM
from dotenv import load_dotenv

from logger import get_logger
from tools import (
    exa_search_tool,
    get_analyst_recommendations,
    get_balance_sheet,
    get_cash_flow,
    get_company_info,
    get_current_stock_price,
    get_dividend_history,
    get_income_statements,
    get_insider_transactions,
    get_institutional_holdings,
    get_technical_indicators,
    get_valuation_metrics,
)

load_dotenv()

logger = get_logger(__name__)

# ── LLM ────────────────────────────────────────────────────────────────────────

logger.debug("Initialising LLM: openai/gpt-oss-120b:free via OpenRouter")

llm = LLM(
    model="openai/gpt-oss-120b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# ── Agents ─────────────────────────────────────────────────────────────────────

data_explorer = Agent(
    role="Fundamental Data Researcher",
    goal=(
        "Gather comprehensive financial data for a stock — income statements, balance sheet, "
        "cash flow, dividends, analyst recommendations, insider activity, and institutional holdings"
    ),
    llm=llm,
    verbose=True,
    backstory=(
        "You are an expert fundamental researcher who can gather detailed financial information "
        "about any publicly listed company. When using tools, use the stock symbol provided. "
        'Try with and without the ".NS" or ".BO" suffix and see what works.'
    ),
    tools=[
        get_company_info,
        get_income_statements,
        get_balance_sheet,
        get_cash_flow,
        get_dividend_history,
        get_analyst_recommendations,
        get_insider_transactions,
        get_institutional_holdings,
    ],
    max_iter=6,
    max_rpm=12,
    max_execution_time=540,
    respect_context_window=True,
)

news_info_explorer = Agent(
    role="News and Market Sentiment Researcher",
    goal="Gather the latest news, developments, and market sentiment about a company from the internet",
    llm=llm,
    verbose=True,
    backstory=(
        "You are an expert researcher who can find and summarise the latest news, regulatory updates, "
        "management changes, and market sentiment surrounding any company."
    ),
    tools=[exa_search_tool],
    max_iter=5,
    max_rpm=15,
    max_execution_time=600,
    respect_context_window=True,
)

technical_analyst = Agent(
    role="Technical Analyst",
    goal=(
        "Analyse price action and technical indicators for a stock and provide a clear "
        "technical outlook — trend direction, momentum, support/resistance levels, and signals"
    ),
    llm=llm,
    verbose=True,
    backstory=(
        "You are a seasoned technical analyst with deep expertise in reading charts and technical "
        "indicators. You interpret RSI, MACD, Bollinger Bands, and moving averages to determine "
        "whether a stock is in a bullish or bearish trend, overbought or oversold, and identify "
        "key price levels. You present your findings in a clear, structured manner."
    ),
    tools=[get_technical_indicators],
    max_iter=4,
    max_rpm=10,
    max_execution_time=300,
    respect_context_window=True,
)

sector_analyst = Agent(
    role="Sector & Peer Comparison Analyst",
    goal=(
        "Identify 4-5 key competitors or sector peers of the target stock and compare them "
        "side-by-side on valuation multiples, margins, growth, and returns to determine whether "
        "the target stock is overvalued, fairly valued, or undervalued relative to its peers"
    ),
    llm=llm,
    verbose=True,
    backstory=(
        "You are an expert equity research analyst specialising in relative valuation and sector analysis. "
        "Given a stock, you identify its closest 4-5 competitors from your knowledge of the market, "
        "fetch their key financial metrics, and build a side-by-side comparison table covering "
        "valuation multiples (P/E, P/B, EV/EBITDA), profitability (margins, ROE), growth (revenue YoY), "
        "and capital structure (debt/equity). You then conclude whether the target stock trades at a "
        "premium or discount to peers and why that premium/discount may or may not be justified."
    ),
    tools=[get_company_info, get_valuation_metrics],
    max_iter=6,
    max_rpm=12,
    max_execution_time=480,
    respect_context_window=True,
)

analyst = Agent(
    role="Senior Financial Analyst",
    goal=(
        "Combine fundamental financial data, latest news, and technical analysis into a "
        "comprehensive, balanced stock analysis"
    ),
    llm=llm,
    verbose=True,
    backstory=(
        "You are a senior financial analyst with expertise in combining quantitative data "
        "(financials, ratios), qualitative information (news, sentiment), and technical signals "
        "into a complete, well-structured investment analysis."
    ),
    max_iter=4,
    max_rpm=10,
    max_execution_time=300,
    respect_context_window=True,
)

fin_expert = Agent(
    role="Financial Expert & Investment Advisor",
    goal="Based on the full analysis of a stock, make a clear and well-reasoned investment recommendation",
    backstory=(
        "You are a senior investment advisor who synthesises fundamental analysis, news, and technical "
        "signals into a final BUY / HOLD / SELL recommendation. You always check the current stock "
        "price and provide a realistic 12-month target. "
        'When using tools, try with and without ".NS" or ".BO" suffix on the stock symbol.'
    ),
    llm=llm,
    verbose=True,
    tools=[get_current_stock_price],
    max_iter=5,
    max_rpm=8,
    max_execution_time=360,
    respect_context_window=True,
)

logger.debug("All 6 agents initialised successfully.")
