"""
Entry point for the AI-Powered Stocks Analyser.

Orchestrates four CrewAI crews in two phases:
  Phase 1 (parallel):    financial data + news + technical analysis + peer comparison
  Phase 2 (sequential):  analysis → investment recommendation

Usage:
    python main.py                        # analyse RELIANCE (default)
    python main.py --stock SUZLON.BO      # analyse a specific stock
    python main.py --validate             # check API keys and connectivity only
    python main.py --log-level DEBUG      # verbose output (also written to logs/)
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from crewai import Crew, Process

from logger import get_logger
from report_generator import generate_pdf_report
from validators import (
    run_connectivity_checks,
    validate_env_vars,
    validate_stock_symbol,
)

logger = get_logger(__name__)

# ── Crew Definitions ───────────────────────────────────────────────────────────
# Imported here (after logger/validators) to ensure any import-time logging
# in agents.py / tasks.py uses the already-configured logger.

from agents import analyst, data_explorer, fin_expert, news_info_explorer, sector_analyst, technical_analyst  # noqa: E402
from tasks import advise, analyse, compare_with_peers, get_company_financials, get_company_news, run_technical_analysis  # noqa: E402

financial_crew = Crew(
    agents=[data_explorer],
    tasks=[get_company_financials],
    verbose=True,
    process=Process.sequential,
    cache=True,
    max_rpm=15,
)

news_crew = Crew(
    agents=[news_info_explorer],
    tasks=[get_company_news],
    verbose=True,
    process=Process.sequential,
    cache=True,
    max_rpm=15,
)

technical_crew = Crew(
    agents=[technical_analyst],
    tasks=[run_technical_analysis],
    verbose=True,
    process=Process.sequential,
    cache=True,
    max_rpm=15,
)

peer_crew = Crew(
    agents=[sector_analyst],
    tasks=[compare_with_peers],
    verbose=True,
    process=Process.sequential,
    cache=True,
    max_rpm=12,
)

analysis_crew = Crew(
    agents=[analyst, fin_expert],
    tasks=[analyse, advise],
    verbose=True,
    process=Process.sequential,
    cache=True,
    max_rpm=15,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def run_crew_task(crew: Crew, inputs: dict, task_name: str):
    """
    Execute a single crew and return its result.

    Args:
        crew:      The Crew instance to run.
        inputs:    Dictionary of template variables for the tasks.
        task_name: Human-readable label used in log messages.

    Returns:
        The CrewOutput object returned by crew.kickoff().

    Raises:
        RuntimeError: Propagated from CrewAI on execution failure.
    """
    logger.info("🚀  Starting: %s", task_name)
    try:
        result = crew.kickoff(inputs=inputs)
        logger.info("✅  Completed: %s", task_name)
        return result
    except Exception as exc:
        logger.error("❌  %s failed: %s", task_name, exc)
        raise


def run_parallel_execution(stock_input: dict) -> tuple[float, float]:
    """
    Run the full two-phase analysis pipeline.

    Phase 1 — financial data gathering and news gathering run in parallel.
    Phase 2 — analysis and recommendation run sequentially (depend on Phase 1).

    Args:
        stock_input: Dict with key 'stock' (the ticker symbol string).

    Returns:
        Tuple of (phase1_duration_seconds, phase2_duration_seconds).
    """
    logger.info("━" * 60)
    logger.info("Starting AI-Powered Financial Analysis")
    logger.info("Stock: %s", stock_input["stock"])
    logger.info("━" * 60)

    # ── Phase 1: Parallel data gathering ──────────────────────────────────────
    logger.info("🔄  Phase 1: Financial Data, News, Technical Analysis & Peer Comparison (parallel)…")
    phase1_start = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        financial_future = executor.submit(
            run_crew_task, financial_crew, stock_input, "Financial Data Gathering"
        )
        news_future = executor.submit(
            run_crew_task, news_crew, stock_input, "News Gathering"
        )
        technical_future = executor.submit(
            run_crew_task, technical_crew, stock_input, "Technical Analysis"
        )
        peer_future = executor.submit(
            run_crew_task, peer_crew, stock_input, "Peer Comparison"
        )

        # Retrieve results — any exception raised inside the thread is re-raised here
        financial_result = financial_future.result()
        news_result = news_future.result()
        technical_result = technical_future.result()
        peer_result = peer_future.result()

    phase1_duration = time.time() - phase1_start
    logger.info("✅  Phase 1 complete in %.2f seconds.", phase1_duration)
    logger.debug("Financial result type: %s", type(financial_result))
    logger.debug("News result type: %s", type(news_result))
    logger.debug("Technical result type: %s", type(technical_result))
    logger.debug("Peer comparison result type: %s", type(peer_result))

    # ── Phase 2: Sequential analysis ──────────────────────────────────────────
    logger.info("🔄  Phase 2: Analysis & Recommendation…")
    phase2_start = time.time()

    analysis_result = analysis_crew.kickoff(inputs=stock_input)

    phase2_duration = time.time() - phase2_start
    logger.info("✅  Phase 2 complete in %.2f seconds.", phase2_duration)
    logger.debug("Analysis result type: %s", type(analysis_result))

    return phase1_duration, phase2_duration


# ── CLI ────────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="AI-Powered Stock Analyser — multi-agent financial analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                        # analyse RELIANCE (default)\n"
            "  python main.py --stock SUZLON.BO      # analyse Suzlon on BSE\n"
            "  python main.py --stock AAPL           # analyse Apple Inc.\n"
            "  python main.py --validate             # test API keys without running analysis\n"
            "  python main.py --log-level DEBUG      # enable verbose debug output\n"
        ),
    )
    parser.add_argument(
        "--stock",
        default="RELIANCE.NS",
        metavar="SYMBOL",
        help=(
            "Stock ticker symbol to analyse. "
            "Append .NS for NSE or .BO for BSE (e.g. RELIANCE.NS, SUZLON.BO). "
            "Default: RELIANCE.NS"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run API connectivity checks and exit without performing any analysis.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        dest="log_level",
        help="Console log verbosity. File logs are always at DEBUG level. Default: INFO",
    )
    return parser


def _set_console_log_level(level_name: str) -> None:
    """Adjust the console handler's log level after argument parsing."""
    import logging

    root = logging.getLogger("stocks_analyser")
    level = getattr(logging, level_name.upper(), logging.INFO)
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Apply user-specified console log level before anything else
    _set_console_log_level(args.log_level)

    logger.info("━" * 60)
    logger.info("AI-Powered Stocks Analyser")
    logger.info("━" * 60)

    # ── Step 1: Validate environment variables ─────────────────────────────────
    if not validate_env_vars():
        logger.error(
            "Startup aborted: missing environment variables.\n"
            "    ➜  Add the missing keys to your .env file and re-run."
        )
        sys.exit(1)

    # ── Step 2: --validate flag — connectivity check only ─────────────────────
    if args.validate:
        success = run_connectivity_checks()
        sys.exit(0 if success else 1)

    # ── Step 3: Validate stock symbol format ───────────────────────────────────
    if not validate_stock_symbol(args.stock):
        logger.error(
            "Startup aborted: invalid stock symbol '%s'.\n"
            "    ➜  Example valid symbols: RELIANCE.NS  SUZLON.BO  AAPL  ^NSEI",
            args.stock,
        )
        sys.exit(1)

    # ── Step 4: Run the analysis ───────────────────────────────────────────────
    stock_input = {"stock": args.stock}
    start_time = time.time()

    try:
        phase1_time, phase2_time = run_parallel_execution(stock_input)

    except RuntimeError as exc:
        # Check for common authentication failure
        exc_str = str(exc)
        if "401" in exc_str or "authentication" in exc_str.lower() or "user not found" in exc_str.lower():
            logger.error(
                "❌  Authentication error: the LLM API rejected your credentials.\n"
                "    ➜  Check that OPENROUTER_API_KEY in your .env file is correct.\n"
                "    ➜  Run  python main.py --validate  to test your API keys.\n"
                "    ➜  Visit https://openrouter.ai/keys to verify or rotate your key."
            )
        elif "rate limit" in exc_str.lower() or "429" in exc_str:
            logger.error(
                "❌  Rate limit exceeded.\n"
                "    ➜  You've hit the API's request-per-minute or daily limit.\n"
                "    ➜  Wait a few minutes and try again, or use a different model."
            )
        elif "timeout" in exc_str.lower():
            logger.error(
                "❌  A task timed out.\n"
                "    ➜  The agent exceeded its max_execution_time.\n"
                "    ➜  Try again, or increase max_execution_time in agents.py."
            )
        else:
            logger.error(
                "❌  Analysis failed with an unexpected error.\n"
                "    ➜  Details: %s\n"
                "    ➜  Check logs/ for the full DEBUG-level stack trace.",
                exc,
            )
        sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("⏹️   Analysis interrupted by user (Ctrl+C).")
        sys.exit(130)

    except Exception as exc:
        logger.exception(
            "❌  Unexpected error during analysis: %s\n"
            "    ➜  Full traceback has been written to logs/.",
            exc,
        )
        sys.exit(1)

    # ── Step 5: Print summary ──────────────────────────────────────────────────
    total_time = time.time() - start_time
    estimated_sequential = phase1_time * 2 + phase2_time
    time_saved = max(0.0, estimated_sequential - total_time)

    logger.info("━" * 60)
    logger.info("🎉  Financial analysis complete!")
    logger.info("━" * 60)
    logger.info("  Stock analysed      : %s", args.stock)
    logger.info("  Phase 1 (parallel)  : %.2f seconds", phase1_time)
    logger.info("  Phase 2 (sequential): %.2f seconds", phase2_time)
    logger.info("  Total execution time: %.2f seconds (%.2f minutes)", total_time, total_time / 60)
    logger.info("  Time saved (est.)   : %.2f seconds vs. sequential", time_saved)
    if estimated_sequential > 0:
        pct = (time_saved / estimated_sequential) * 100
        logger.info("  Efficiency gain     : %.1f%%", pct)
    logger.info("━" * 60)
    logger.info("  📄  Peers     → task_outputs/peer_comparison.md")
    logger.info("  📄  Technical → task_outputs/technical_analysis.md")
    logger.info("  📄  Analysis  → task_outputs/financial_analysis.md")
    logger.info("  📄  Advice    → task_outputs/investment_recommendation.md")
    
    # Generate the Rich PDF Report
    pdf_path = generate_pdf_report(args.stock)
    if pdf_path:
        logger.info("  📊  PDF Report→ %s", pdf_path)
        
    logger.info("  📋  Full logs → logs/analyser_%s.log", time.strftime("%Y%m%d"))
    logger.info("━" * 60)


if __name__ == "__main__":
    main()
