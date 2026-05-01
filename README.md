<div align="center">

# 🤖 AI-Powered Stock Analyser

### Multi-Agent Financial Intelligence System Built with CrewAI

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-FF6B6B?style=for-the-badge)](https://crewai.com)
[![yfinance](https://img.shields.io/badge/yfinance-Market%20Data-4CAF50?style=for-the-badge)](https://pypi.org/project/yfinance/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20Gateway-7C3AED?style=for-the-badge)](https://openrouter.ai)
[![EXA](https://img.shields.io/badge/EXA-Neural%20Search-0EA5E9?style=for-the-badge)](https://exa.ai)

*A production-grade, multi-agent AI system that performs institutional-quality stock research — combining fundamental analysis, real-time news intelligence, technical chart analysis, and sector peer comparison — fully automated.*

</div>

---

## 📌 What This Project Does

This system orchestrates **5 specialised AI agents** that work in parallel and sequentially to produce a complete equity research report for any publicly listed stock. Given a ticker symbol, it:

1. **Fetches deep financial data** — income statements, balance sheet, cash flow, dividends, insider transactions, institutional holdings, and analyst recommendations
2. **Searches the internet for real-time news** — using semantic neural search to surface the most relevant recent developments
3. **Calculates technical indicators from scratch** — RSI, MACD, Bollinger Bands, SMA50/200, and volume analysis, all implemented using pure pandas (no TA library dependency)
4. **Benchmarks against sector peers** — identifies 4-5 competitors and builds a side-by-side valuation comparison (P/E, P/B, EV/EBITDA, ROE, margins, growth) to determine if the stock is over/under-valued relative to its sector
5. **Synthesises all four data streams** into a single structured analysis report
6. **Outputs a validated investment recommendation** — BUY / HOLD / SELL with a confidence score, 12-month target price, key reasons, and risks

**Sample output (SUZLON.BO, run on 30 Apr 2026):**
```json
{
  "action": "BUY",
  "confidence": 0.82,
  "target_price": 1190.0,
  "current_price": 958.95,
  "reasons": [
    "Strong earnings growth and margin expansion (EBITDA margin up to 12.2%) with a robust order book and export rebound.",
    "Solid balance sheet with low leverage (DE 0.32×) and sufficient cash to fund strategic initiatives."
  ],
  "risks": [
    "High valuation premium (P/E ~62×) could compress if FY27 growth or margin targets are not met."
  ]
}
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI  (main.py)                              │
│  python main.py --stock SUZLON.BO  |  --validate  |  --log-level    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │         Startup Validation         │
          │  ✓ Env vars  ✓ Symbol format       │
          │  ✓ API connectivity (--validate)    │
          └─────────────────┬─────────────────┘
                            │
          ╔═════════════════▼══════════════════════╗
          ║     PHASE 1 — Parallel (4 crews)       ║
          ╚═══════╤═══════╤══════════╤═════════╤═══╝
                  │       │          │         │
     ┌────────────▼──┐ ┌──▼──────┐ ┌─▼──────┐ ┌─▼──────────────┐
     │ Financial Crew│ │  News   │ │Tech    │ │  Peer Crew     │
     │               │ │  Crew   │ │Crew    │ │                │
     │ data_explorer │ │news_exp.│ │tech_   │ │ sector_analyst │
     │               │ │         │ │analyst │ │                │
     │ 8 yfinance    │ │ EXA     │ │RSI,MACD│ │ get_company_   │
     │ tools         │ │ search  │ │BB,SMA  │ │ info +         │
     │               │ │         │ │Volume  │ │ get_valuation_ │
     │               │ │         │ │        │ │ metrics        │
     └───────┬───────┘ └──┬──────┘ └──┬─────┘ └───────┬────────┘
             │             │           │               │
          ╔══▼═════════════▼═══════════▼═══════════════▼══╗
          ║     PHASE 2 — Sequential (1 crew)             ║
          ╚═════════════════════╤══════════════════════════╝
                                │
                    ┌───────────▼────────────┐
                    │  analyst  (Task 5)      │
                    │  Combines all 4        │
                    │  Phase 1 outputs       │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ fin_expert (Task 6)     │
                    │ BUY/HOLD/SELL rec       │
                    │ + Pydantic schema       │
                    │ + guardrail check       │
                    └───────────┬────────────┘
                                │
               ┌────────────────▼────────────────┐
               │           Output Files           │
               │  peer_comparison.md              │
               │  technical_analysis.md           │
               │  financial_analysis.md           │
               │  investment_recommendation.md    │
               │  logs/analyser_YYYYMMDD.log      │
               └─────────────────────────────────┘
```

---

## 🧠 Agent Design

The system uses **6 specialised agents**, each with a distinct role, tailored backstory, tool access, and rate/execution limits:

| Agent | Role | Tools | Key Design Decision |
|---|---|---|---|
| `data_explorer` | Fundamental Data Researcher | 8 yfinance tools | Larger `max_execution_time` (540s) — collecting 8 data sources in sequence takes time |
| `news_info_explorer` | News & Sentiment Researcher | EXA neural search | Semantic search surfaces contextually relevant news even when the ticker isn't explicitly mentioned |
| `technical_analyst` | Technical Chart Analyst | `get_technical_indicators` | Isolated agent keeps chart signals separate from fundamentals — prevents anchoring bias in the synthesis |
| `sector_analyst` | Sector & Peer Comparison Analyst | `get_company_info` + `get_valuation_metrics` | Uses LLM's market knowledge to identify peers dynamically — no hardcoded peer maps, adapts to any stock globally |
| `analyst` | Senior Financial Analyst | None (synthesis only) | Intentionally no tools — forced to reason from structured Phase 1 context, not re-fetch data |
| `fin_expert` | Investment Advisor | `get_current_stock_price` | Fetches live price last so the target price ratio reflects market conditions at recommendation time |

---

## 🛠️ Tool Design

### Fundamental Data Tools (8 tools via `yfinance`)

All tools follow the same pattern: input validation → `_with_retry()` → structured JSON output → descriptive error message on failure.

```python
# Retry with exponential backoff — no external dependency
def _with_retry(fn, label, max_retries=3, base_delay=1.5):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            delay = base_delay * (2 ** (attempt - 1))  # 1.5s → 3s → 6s
            if attempt < max_retries:
                time.sleep(delay)
            else:
                raise
```

| Tool | Data Source | Key Data Points |
|---|---|---|
| `get_company_info` | `ticker.info` | P/E, EPS, market cap, 52-week range, margins, EBITDA |
| `get_income_statements` | `ticker.financials` | Revenue, gross profit, operating income, net income (4 years) |
| `get_balance_sheet` | `ticker.balance_sheet` | Total assets, liabilities, equity, cash, debt |
| `get_cash_flow` | `ticker.cashflow` | Operating CF, investing CF, free cash flow |
| `get_dividend_history` | `ticker.dividends` | Last 10 dividend payments |
| `get_analyst_recommendations` | `ticker.recommendations` | Most recent buy/hold/sell consensus |
| `get_insider_transactions` | `ticker.insider_transactions` | Recent insider buying/selling activity |
| `get_institutional_holdings` | `ticker.institutional_holders` | Top institutional shareholders and stake sizes |
| `get_valuation_metrics` | `ticker.info` | Compact snapshot: P/E, P/B, EV/EBITDA, ROE, margins, D/E, dividend yield — designed for peer comparison |

### Technical Analysis Tool (implemented from scratch with `pandas`)

Rather than adding a TA library dependency, all indicators are implemented using vectorised pandas operations:

**RSI (14-day Relative Strength Index)**
```python
# Standard Wilder smoothing via rolling mean
delta = prices.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rsi   = 100 - (100 / (1 + gain / loss))
# Signal: > 70 → overbought, < 30 → oversold
```

**MACD (12/26/9)**
```python
ema12     = prices.ewm(span=12, adjust=False).mean()
ema26     = prices.ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
signal    = macd_line.ewm(span=9, adjust=False).mean()
histogram = macd_line - signal
# Signal: MACD above signal → bullish crossover
```

**Bollinger Bands (20-day, ±2σ)**
```python
sma   = prices.rolling(20).mean()
std   = prices.rolling(20).std()
upper = sma + 2 * std
lower = sma - 2 * std
# Price above upper → overbought; below lower → oversold
```

**Moving Averages + Golden/Death Cross**
```python
sma50  = prices.rolling(50).mean()
sma200 = prices.rolling(200).mean()
# Golden Cross: SMA50 crosses above SMA200 → long-term bullish signal
# Death Cross:  SMA50 crosses below SMA200 → long-term bearish signal
```

**Volume Analysis**
```python
avg_vol_20 = volume.rolling(20).mean()
vol_ratio  = current_volume / avg_vol_20
# > 1.0 → above average volume (confirms price moves)
```

---

## ⚙️ Engineering Highlights

### Parallel Execution with `ThreadPoolExecutor`

Phase 1 runs **4 independent crews concurrently**, cutting wall-clock time significantly:

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    financial_future = executor.submit(run_crew_task, financial_crew, inputs, "Financial")
    news_future      = executor.submit(run_crew_task, news_crew,      inputs, "News")
    technical_future = executor.submit(run_crew_task, technical_crew, inputs, "Technical")
    peer_future      = executor.submit(run_crew_task, peer_crew,      inputs, "Peers")

    financial_result = financial_future.result()
    news_result      = news_future.result()
    technical_result = technical_future.result()
    peer_result      = peer_future.result()
```

At the end of every run, the system reports estimated time saved:
```
Phase 1 (parallel):   142.3s
Phase 2 (sequential):  89.1s
Total:                231.4s   (vs ~373.7s sequential — 38% faster)
```

### Structured Output with Pydantic + Guardrails

The final recommendation is enforced as a strict Pydantic schema with a custom guardrail function:

```python
class InvestmentRecommendation(BaseModel):
    action:        Literal["BUY", "HOLD", "SELL"]
    confidence:    float    # must be 0.0–1.0
    target_price:  float
    current_price: float
    reasons:       list[str]  # minimum 2 required
    risks:         list[str]  # minimum 1 required

def validate_recommendation(result: TaskOutput) -> Tuple[bool, Any]:
    # guardrail — CrewAI will retry the task if this returns False
    if not (0.0 <= rec.confidence <= 1.0): return (False, "...")
    if len(rec.reasons) < 2:               return (False, "...")
    if len(rec.risks) < 1:                 return (False, "...")
    return (True, rec)
```

If the LLM output fails validation, CrewAI automatically retries the task with the error message as feedback — ensuring output quality without manual intervention.

### Production-Grade Error Handling

The system catches all known failure modes and converts them into actionable, user-friendly messages:

```
❌  Authentication error: the LLM API rejected your credentials.
    ➜  Check that OPENROUTER_API_KEY in your .env file is correct.
    ➜  Run  python main.py --validate  to test your API keys.
    ➜  Visit https://openrouter.ai/keys to verify or rotate your key.
```

Handled explicitly: 401 auth failures, 429 rate limits, agent timeouts, keyboard interrupts, and generic exceptions (full traceback written to the log file only).

### Structured Logging (Console + File)

```python
# Console: clean, timestamped INFO messages
23:15:00 [INFO    ] ✅  OpenRouter API: connected and authenticated.

# File (logs/analyser_YYYYMMDD.log): full DEBUG context
2026-04-30 23:15:00 [DEBUG   ] stocks_analyser.tools:get_company_info:98 — Company info fetched for SUZLON.BO: Suzlon Energy Ltd
```

Two handlers on one logger — operators see clean output, debug logs preserve full context for troubleshooting.

---

## 📁 Project Structure

```
ai-powered-stocks-analyser/
│
├── main.py           # CLI entry point — orchestrates crews, parallel execution, error handling
├── agents.py         # 6 CrewAI agent definitions with roles, backstories, tools, and limits
├── tasks.py          # 6 task definitions with descriptions, expected outputs, and guardrails
├── tools.py          # 11 tools: 8 fundamental + 1 valuation metrics + 1 technical + 1 EXA search
├── logger.py         # Centralised logging — console (INFO) + rotating file (DEBUG)
├── validators.py     # Startup checks: env vars, symbol format, live API connectivity
│
├── config/
│   ├── agents.yaml   # Agent configuration templates
│   └── tasks.yaml    # Task configuration templates
│
├── task_outputs/     # Generated reports (gitignored)
│   ├── peer_comparison.md
│   ├── technical_analysis.md
│   ├── financial_analysis.md
│   └── investment_recommendation.md
│
├── logs/             # Daily debug log files (gitignored)
│   └── analyser_YYYYMMDD.log
│
├── requirements.txt
├── .env              # API keys (gitignored)
└── .gitignore
```

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (free tier available — used for LLM access)
- An [EXA](https://exa.ai) API key (used for real-time news search)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/raghavg27/ai-powered-stocks-analyser.git
cd ai-powered-stocks-analyser

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# OPENROUTER_API_KEY=sk-or-v1-...
# EXA_API_KEY=...
```

### Running the Analyser

```bash
# Validate API keys and connectivity before your first run
python main.py --validate

# Analyse any stock — NSE, BSE, NYSE, NASDAQ
python main.py --stock RELIANCE.NS    # NSE (India)
python main.py --stock SUZLON.BO      # BSE (India)
python main.py --stock AAPL           # NYSE/NASDAQ (US)
python main.py --stock ^NSEI          # Nifty 50 Index

# Enable verbose debug output
python main.py --stock TCS.NS --log-level DEBUG
```

### Output

Three markdown reports are generated in `task_outputs/`:

| File | Contents |
|---|---|
| `peer_comparison.md` | Side-by-side peer valuation table, premium/discount verdict, key takeaways |
| `technical_analysis.md` | RSI, MACD, Bollinger Bands, MAs, volume analysis, technical outlook |
| `financial_analysis.md` | Full fundamental + news + technical + peer synthesis |
| `investment_recommendation.md` | Structured JSON: action, confidence, target, reasons, risks |

A debug log is written to `logs/analyser_YYYYMMDD.log` after every run.

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Agent Framework** | [CrewAI](https://crewai.com) | Multi-agent orchestration, task chaining, context passing |
| **LLM Gateway** | [OpenRouter](https://openrouter.ai) | Unified API for multiple LLM providers |
| **LLM** | GPT-class model (via OpenRouter) | Agent reasoning, synthesis, and recommendation |
| **Market Data** | [yfinance](https://pypi.org/project/yfinance/) | Financial statements, price history, insider data |
| **News Search** | [EXA](https://exa.ai) | Neural semantic search for real-time news |
| **HTTP** | [curl_cffi](https://pypi.org/project/curl-cffi/) | Chrome-impersonating session to bypass bot detection on yfinance |
| **Output Validation** | [Pydantic v2](https://docs.pydantic.dev/) | Strict schema enforcement on LLM output |
| **Concurrency** | `concurrent.futures.ThreadPoolExecutor` | Parallel Phase 1 crew execution |
| **Logging** | Python `logging` | Dual-handler: console (INFO) + file (DEBUG) |
| **Env Management** | `python-dotenv` | Secure API key loading |

---

## 🔄 How Parallel Execution Works

The pipeline is split into two phases specifically to maximise parallelism while respecting task dependencies:

```
                    ┌──────────────────────────────┐
                    │      Sequential Baseline      │
                    │  T1──T2──T3──T4──T5           │
                    │  ←─────── ~373s ────────→    │
                    └──────────────────────────────┘

                    ┌──────────────────────────────┐
                    │  This System (Parallel Phase 1)│
                    │  T1 ─┐                        │
                    │  T2 ─┼── (concurrent) ──T4──T5│
                    │  T3 ─┘                        │
                    │  ←── ~231s ──→                │
                    └──────────────────────────────┘
```

Tasks 1, 2, and 3 have no dependency on each other — they can start simultaneously. Tasks 4 and 5 depend on the combined output of Tasks 1–3, so they run sequentially after Phase 1 completes.

---

## 💡 Design Decisions & Engineering Rationale

**Why CrewAI?** CrewAI provides a clean abstraction for multi-agent pipelines — role-based agents, task context passing, structured output, and built-in retry logic. It handles the complexity of chaining LLM calls with tool use so the code focuses on the domain logic.

**Why OpenRouter instead of OpenAI directly?** OpenRouter provides a unified gateway to multiple LLMs (GPT, Claude, Gemini, Llama, etc.) under one API. This makes it trivial to swap models by changing a single config string — useful for cost/quality tradeoffs.

**Why identify peers dynamically via LLM rather than a hardcoded map?** A static peer mapping would cover only stocks we've pre-configured and would quickly go stale as companies enter/exit sectors. The LLM has broad market knowledge and can identify the most relevant competitors for any stock globally — from Nifty50 to NASDAQ — without maintenance overhead. The `get_valuation_metrics` tool then fetches live data for whichever peers the agent selects.

**Why the guardrail pattern?** LLMs can hallucinate or produce malformed structured output. The Pydantic schema + guardrail function creates a closed feedback loop — if the output is invalid, CrewAI feeds the validation error back to the agent as a correction prompt and retries automatically.

**Why separate console and file log handlers?** Operators running interactively want a clean, scannable INFO stream. Debugging a failure at 2am requires full context (module, function, line). Two handlers on one logger satisfies both needs without changing code.

---

## 📈 Potential Extensions

- **Streamlit Web UI** — Interactive dashboard with live progress, charts, and recommendation cards
- **Watchlist mode** — `--watchlist RELIANCE.NS,TCS.NS,INFY.NS` for batch analysis
- **Result caching** — Skip re-fetching data fetched within the last N hours
- **Historical tracking** — SQLite log of past recommendations vs actual price outcomes
- **Alerting** — Email/Telegram notification when a high-confidence BUY is detected
- **PDF report generation** — Formatted research report via Jinja2 + WeasyPrint

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using CrewAI, yfinance, and OpenRouter
</div>
