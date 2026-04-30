import os

from crewai import Agent
from crewai.llm import LLM
from dotenv import load_dotenv

from tools import (
    exa_search_tool,
    get_company_info,
    get_current_stock_price,
    get_income_statements,
)

load_dotenv()

llm = LLM(
    model="openai/gpt-4.1-2025-04-14",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

data_explorer = Agent(
    role="Data Researcher",
    goal="Gather and provide financial data and company information about a stock",
    llm=llm,
    verbose=True,
    backstory=(
        "You are an expert researcher, who can gather detailed information about a company or stock. "
        'When using tools, use the stock symbol and add a suffix ".NS" to it. '
        'Try with and without the suffix and see what works.'
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
    backstory="You are an expert researcher, who can gather detailed information about a company.",
    tools=[exa_search_tool],
    max_iter=5,
    max_rpm=15,
    max_execution_time=600,
    respect_context_window=True,
)

analyst = Agent(
    role="Data Analyst",
    goal="Consolidate financial data, stock information, and provide a summary",
    llm=llm,
    verbose=True,
    backstory=(
        "You are an expert in analyzing financial data, stock/company-related current information, and "
        "making a comprehensive analysis."
    ),
    max_iter=4,
    max_rpm=10,
    max_execution_time=300,
    respect_context_window=True,
)

fin_expert = Agent(
    role="Financial Expert",
    goal="Considering financial analysis of a stock, make investment recommendations",
    backstory=(
        "You are an expert financial advisor who can provide investment recommendations. "
        "Consider the financial analysis, current information about the company, current stock price, "
        "and make recommendations about whether to buy/hold/sell a stock along with reasons. "
        'When using tools, try with and without the suffix ".NS" to the stock symbol and see what works.'
    ),
    llm=llm,
    verbose=True,
    tools=[get_current_stock_price],
    max_iter=5,
    max_rpm=8,
    max_execution_time=360,
    respect_context_window=True,
)
