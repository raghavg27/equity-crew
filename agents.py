"""
CrewAI agent definitions for the AI-Powered Stocks Analyser.

Agents:
  - data_explorer       → fetches financial statements and company fundamentals
  - news_info_explorer  → searches for latest news and market sentiment
  - analyst             → synthesises financial data and news into a coherent analysis
  - fin_expert          → produces a final BUY / HOLD / SELL investment recommendation
"""

import os

from crewai import Agent
from crewai.llm import LLM
from dotenv import load_dotenv

from logger import get_logger
from tools import (
    exa_search_tool,
    get_company_info,
    get_current_stock_price,
    get_income_statements,
)

load_dotenv()

logger = get_logger(__name__)

# ── LLM Configuration ──────────────────────────────────────────────────────────

logger.debug("Initialising LLM with model: openai/gpt-oss-120b:free via OpenRouter")

llm = LLM(
    model="openai/gpt-oss-120b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# ── Agents ─────────────────────────────────────────────────────────────────────

logger.debug("Defining agents…")

data_explorer = Agent(
    role="Data Researcher",
    goal="Gather and provide financial data and company information about a stock",
    llm=llm,
    verbose=True,
    backstory=(
        "You are an expert researcher who can gather detailed information about a "
        "company or stock. When using tools, use the stock symbol and add a suffix "
        '".NS" to it. Try with and without the suffix and see what works.'
    ),
    tools=[get_company_info, get_income_statements],
    max_iter=5,
    max_rpm=12,
    max_execution_time=450,
    respect_context_window=True,
)

news_info_explorer = Agent(
    role="News and Info Researcher",
    goal="Gather and provide the latest news and information about a company from the internet",
    llm=llm,
    verbose=True,
    backstory=(
        "You are an expert researcher who can gather detailed information about a company "
        "from across the internet, including recent developments, regulatory news, and "
        "market sentiment."
    ),
    tools=[exa_search_tool],
    max_iter=5,
    max_rpm=15,
    max_execution_time=600,
    respect_context_window=True,
)

analyst = Agent(
    role="Data Analyst",
    goal="Consolidate financial data, stock information, and news into a comprehensive summary",
    llm=llm,
    verbose=True,
    backstory=(
        "You are an expert in analysing financial data, stock- and company-related "
        "current information, and making a comprehensive, balanced analysis."
    ),
    max_iter=4,
    max_rpm=10,
    max_execution_time=300,
    respect_context_window=True,
)

fin_expert = Agent(
    role="Financial Expert",
    goal="Based on financial analysis of a stock, make a clear investment recommendation",
    backstory=(
        "You are an expert financial advisor who provides clear investment recommendations. "
        "Consider the financial analysis, current information about the company, and the "
        "current stock price to make a BUY / HOLD / SELL recommendation with reasons. "
        'When using tools, try with and without the suffix ".NS" appended to the stock symbol.'
    ),
    llm=llm,
    verbose=True,
    tools=[get_current_stock_price],
    max_iter=5,
    max_rpm=8,
    max_execution_time=360,
    respect_context_window=True,
)

logger.debug("All agents initialised successfully.")
