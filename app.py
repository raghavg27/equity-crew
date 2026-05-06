"""
Streamlit Web UI for the AI-Powered Stocks Analyser.

Run from the project root:
    streamlit run app.py
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv

# Ensure CWD is project root so task_outputs/ and templates/ paths resolve correctly
os.chdir(Path(__file__).parent)
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Stock Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.badge {
    font-size: 2.6rem;
    font-weight: 800;
    padding: .25rem 1rem;
    border-radius: 8px;
    display: inline-block;
    letter-spacing: .06em;
}
.BUY  { background: #14532d; color: #4ade80; }
.HOLD { background: #78350f; color: #fbbf24; }
.SELL { background: #7f1d1d; color: #f87171; }

.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.25rem 1.75rem;
    margin: .75rem 0;
}

[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: .75rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

_INIT: dict = dict(
    state="idle",   # idle | complete | error
    stock=None,
    rec=None,
    pdf=None,
    p1=None,
    p2=None,
    err=None,
)
for _k, _v in _INIT.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_rec() -> dict | None:
    """Parse investment_recommendation.md into a dict."""
    path = "task_outputs/investment_recommendation.md"
    if not os.path.exists(path):
        return None
    txt = open(path).read().strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(txt)
    except Exception:
        return None


def _read_md(filename: str) -> str:
    """Return markdown file contents, or a fallback message."""
    path = f"task_outputs/{filename}"
    if not os.path.exists(path):
        return "*Report not yet available.*"
    return open(path).read()


def _price_chart(symbol: str):
    """Build a dark-themed 1-year price chart with SMA50/200. Returns a Figure."""
    try:
        hist = yf.Ticker(symbol).history(period="1y")
        if hist.empty:
            return None
        hist["SMA50"]  = hist["Close"].rolling(50).mean()
        hist["SMA200"] = hist["Close"].rolling(200).mean()

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(hist.index, hist["Close"],  label="Close",    color="#60a5fa", lw=1.8)
        ax.plot(hist.index, hist["SMA50"],  label="SMA 50",   color="#fb923c", lw=1.2, ls="--")
        ax.plot(hist.index, hist["SMA200"], label="SMA 200",  color="#f87171", lw=1.2, ls="--")

        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, ls=":", alpha=0.35, color="#334155")
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
        ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=9)
        ax.set_xlabel("Date",  color="#94a3b8", fontsize=9)
        ax.set_ylabel("Price", color="#94a3b8", fontsize=9)
        fig.tight_layout()
        return fig
    except Exception:
        return None


def _run_pipeline(symbol: str, status) -> tuple:
    """
    Execute the full 2-phase analysis pipeline.
    Writes incremental progress to the st.status container.
    Returns (resolved_symbol, phase1_seconds, phase2_seconds, pdf_path).
    """
    from validators import resolve_stock_symbol, validate_env_vars
    from report_generator import generate_pdf_report

    # ── Preflight ─────────────────────────────────────────────────────────
    status.write("🔑  Checking API keys…")
    if not validate_env_vars():
        raise EnvironmentError(
            "Missing API keys. Add **OPENROUTER_API_KEY** and **EXA_API_KEY** "
            "to `.env` (local) or Streamlit Cloud secrets."
        )

    status.write(f"🔍  Resolving **{symbol}** via yfinance…")
    resolved = resolve_stock_symbol(symbol)
    if not resolved:
        raise ValueError(
            f"No market data found for **{symbol}**. "
            "Check the ticker — append `.NS` for NSE or `.BO` for BSE."
        )

    inp = {"stock": resolved}

    # ── Phase 1: parallel data collection ────────────────────────────────
    # Lazy import so heavy Crew/Agent setup only runs after preflight passes
    from main import (
        financial_crew, news_crew, technical_crew, peer_crew,
        analysis_crew, run_crew_task,
    )

    status.write("---")
    status.write("**⚡ Phase 1 — Parallel data collection** *(4 agents running simultaneously)*")
    t1 = time.time()

    crews = {
        "Financial Data":     financial_crew,
        "News & Sentiment":   news_crew,
        "Technical Analysis": technical_crew,
        "Peer Comparison":    peer_crew,
    }
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(run_crew_task, c, inp, n): n for n, c in crews.items()}
        for future in as_completed(futures):
            name = futures[future]
            elapsed = time.time() - t1
            try:
                future.result()
                status.write(f"  ✅  {name} — {elapsed:.0f}s")
            except Exception as exc:
                status.write(f"  ❌  {name} failed")
                raise RuntimeError(f"{name}: {exc}") from exc

    p1 = time.time() - t1
    status.write(f"Phase 1 complete in **{p1:.0f}s**")

    # ── Phase 2: synthesis ────────────────────────────────────────────────
    status.write("---")
    status.write("**🧠 Phase 2 — Synthesis & recommendation** *(sequential)*")
    t2 = time.time()
    analysis_crew.kickoff(inputs=inp)
    p2 = time.time() - t2
    status.write(f"  ✅  Recommendation generated in {p2:.0f}s")

    # ── PDF ───────────────────────────────────────────────────────────────
    status.write("---")
    status.write("📄  Generating PDF report…")
    pdf = generate_pdf_report(resolved)
    if pdf:
        status.write("  ✅  PDF ready")

    return resolved, p1, p2, pdf


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 AI Stock Analyser")
    st.markdown(
        "Six specialist AI agents built with **CrewAI** run in two phases "
        "to deliver a complete investment analysis — fundamentals, news, "
        "technicals, peer comparison — synthesised into a single "
        "**BUY / HOLD / SELL** recommendation."
    )
    st.divider()

    with st.form("form"):
        sym = st.text_input(
            "Stock ticker",
            placeholder="AAPL · RELIANCE.NS · SUZLON.BO",
            help="US tickers need no suffix. For NSE append .NS, for BSE append .BO.",
        )
        st.caption("Try: `AAPL` · `MSFT` · `NVDA` · `TCS.NS` · `INFY.NS`")
        go = st.form_submit_button(
            "🚀  Analyse", use_container_width=True, type="primary"
        )

    if st.session_state.state == "complete":
        if st.button("↩  New analysis", use_container_width=True):
            for k, v in _INIT.items():
                st.session_state[k] = v
            st.rerun()

    # Show warning if API keys are missing
    missing = [k for k in ("OPENROUTER_API_KEY", "EXA_API_KEY") if not os.getenv(k)]
    if missing:
        st.divider()
        st.warning(
            f"⚠️ Missing: `{'`, `'.join(missing)}`\n\n"
            "Add them to `.env` (local) or Streamlit Cloud secrets."
        )

    st.divider()
    st.markdown("**Architecture**")
    st.markdown(
        "- Phase 1 (parallel): 4 specialist crews  \n"
        "- Phase 2 (sequential): synthesis + recommendation  \n"
        "- Output validated via Pydantic guardrails  \n"
        "- PDF report with price chart"
    )
    st.divider()
    st.caption("**Stack:** CrewAI · OpenRouter · yfinance · EXA · WeasyPrint")


# ── Run analysis ──────────────────────────────────────────────────────────────

if go and sym.strip():
    # Reset any previous run
    for k, v in _INIT.items():
        st.session_state[k] = v

    with st.status(
        f"Analysing **{sym.upper()}** — this typically takes 3–5 minutes…",
        expanded=True,
    ) as sb:
        try:
            resolved, p1, p2, pdf = _run_pipeline(sym.strip(), sb)
            st.session_state.update(
                state="complete",
                stock=resolved,
                rec=_load_rec(),
                pdf=pdf,
                p1=p1,
                p2=p2,
            )
            sb.update(
                label=f"✅  Analysis complete — {resolved}  ·  {(p1 + p2) / 60:.1f} min total",
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            st.session_state.update(state="error", err=str(exc))
            sb.update(label="❌  Analysis failed", state="error")
            st.error(f"**Error:** {exc}")


# ── Landing page ──────────────────────────────────────────────────────────────

elif st.session_state.state == "idle":
    st.markdown("# AI-Powered Stock Analyser")
    st.markdown(
        "Enter any stock ticker in the sidebar to run a full multi-agent investment analysis. "
        "Six AI agents work in two phases to produce a **BUY / HOLD / SELL** recommendation "
        "with a 12-month price target — in 3–5 minutes."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.info(
        "**📋 Fundamental**\n"
        "Income statements, balance sheet, cash flow, dividends, "
        "insider transactions, institutional holdings"
    )
    c2.info(
        "**📰 News & Sentiment**\n"
        "Real-time semantic search via EXA — earnings, management "
        "changes, regulatory filings, market sentiment"
    )
    c3.info(
        "**📈 Technical**\n"
        "RSI · MACD · Bollinger Bands · SMA 50/200 · "
        "Golden/Death Cross · Volume analysis"
    )
    c4.info(
        "**🏭 Peer Comparison**\n"
        "Side-by-side vs 4–5 sector peers: "
        "P/E · P/B · EV/EBITDA · ROE · Margins · Yield"
    )

    st.divider()

    col_how, col_sample = st.columns([3, 2])

    with col_how:
        st.markdown("### How it works")
        st.markdown("""
**Phase 1 — Parallel** *(~40% faster than sequential)*

Four specialist agents run simultaneously:
- **Fundamental Researcher** → financial statements, insider & institutional data
- **News Researcher** → latest developments and sentiment via EXA neural search
- **Technical Analyst** → RSI, MACD, Bollinger Bands, price levels
- **Sector Analyst** → relative valuation vs 4–5 closest competitors

**Phase 2 — Sequential**

With all data gathered, two synthesis agents produce the final output:
- **Senior Financial Analyst** → combines all four data streams into a cohesive analysis
- **Investment Advisor** → generates the final BUY/HOLD/SELL with 12-month target price

Every recommendation is validated against a **Pydantic schema** — confidence score,
target price, at least 2 reasons, at least 1 risk — with automatic LLM retry on failure.
""")

    with col_sample:
        st.markdown("### Sample output")
        st.json({
            "action": "BUY",
            "confidence": 0.82,
            "target_price": 1190.0,
            "current_price": 958.95,
            "reasons": [
                "Strong FCF and improving EBITDA margins support continued growth investment.",
                "Expanding order book in renewables aligns with upcoming policy tailwinds.",
            ],
            "risks": [
                "Elevated debt-to-equity ratio limits financial flexibility in a rising rate environment."
            ],
        })


# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state.state == "complete":
    rec   = st.session_state.rec
    stock = st.session_state.stock

    if not rec:
        st.warning(
            "Analysis complete but the recommendation could not be parsed. "
            "Check `task_outputs/investment_recommendation.md`."
        )
        st.stop()

    action  = rec.get("action", "HOLD")
    conf    = rec.get("confidence", 0.0)
    cur     = rec.get("current_price", 0.0)
    tgt     = rec.get("target_price", 0.0)
    upside  = (tgt - cur) / cur * 100 if cur else 0.0
    upcol   = "#4ade80" if upside >= 0 else "#f87171"
    upsign  = "+" if upside >= 0 else ""
    reasons = rec.get("reasons", [])
    risks   = rec.get("risks", [])

    # ── Recommendation banner ─────────────────────────────────────────────
    st.markdown(f"""
<div class="card" style="padding:1.5rem 2rem">
  <div style="display:flex;gap:2.5rem;align-items:center;flex-wrap:wrap">
    <div>
      <div style="font-size:.72rem;color:#94a3b8;margin-bottom:.25rem;letter-spacing:.09em">RECOMMENDATION</div>
      <span class="badge {action}">{action}</span>
    </div>
    <div>
      <div style="font-size:.72rem;color:#94a3b8;margin-bottom:.25rem;letter-spacing:.09em">CONFIDENCE</div>
      <div style="font-size:2.2rem;font-weight:700;color:#f8fafc">{conf * 100:.0f}%</div>
    </div>
    <div>
      <div style="font-size:.72rem;color:#94a3b8;margin-bottom:.25rem;letter-spacing:.09em">CURRENT PRICE</div>
      <div style="font-size:2.2rem;font-weight:700;color:#f8fafc">{cur:,.2f}</div>
    </div>
    <div>
      <div style="font-size:.72rem;color:#94a3b8;margin-bottom:.25rem;letter-spacing:.09em">12-MONTH TARGET</div>
      <div style="font-size:2.2rem;font-weight:700;color:{upcol}">{tgt:,.2f}</div>
      <div style="font-size:.9rem;color:{upcol};margin-top:.1rem">{upsign}{upside:.1f}% upside</div>
    </div>
    <div style="margin-left:auto;text-align:right">
      <div style="font-size:.85rem;color:#64748b">Analysed by 6 AI agents</div>
      <div style="font-size:1rem;font-weight:600;color:#94a3b8;margin-top:.2rem">{stock}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Investment thesis & risks ─────────────────────────────────────────
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Investment thesis**")
        for r in reasons:
            st.markdown(f"- {r}")
    with cr:
        st.markdown("**Key risks**")
        for r in risks:
            st.markdown(f"- ⚠️ {r}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_fin, tab_tech, tab_peers, tab_chart = st.tabs([
        "📊 Financial Analysis",
        "📈 Technical Analysis",
        "🏭 Peer Comparison",
        "📉 Price Chart",
    ])

    with tab_fin:
        st.markdown(_read_md("financial_analysis.md"))

    with tab_tech:
        st.markdown(_read_md("technical_analysis.md"))

    with tab_peers:
        st.markdown(_read_md("peer_comparison.md"))

    with tab_chart:
        with st.spinner("Loading chart…"):
            fig = _price_chart(stock)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning("Price chart unavailable.")

    # ── Footer: timing stats + PDF download ──────────────────────────────
    st.divider()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    p1 = st.session_state.p1 or 0
    p2 = st.session_state.p2 or 0
    c1.metric("Phase 1 (parallel)",   f"{p1:.0f}s")
    c2.metric("Phase 2 (synthesis)",  f"{p2:.0f}s")
    c3.metric("Total time",           f"{(p1 + p2) / 60:.1f} min")

    pdf_path = st.session_state.pdf
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            c4.download_button(
                label="⬇️  Download Full PDF Report",
                data=f.read(),
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
