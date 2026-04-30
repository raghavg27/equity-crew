import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from crewai import Crew, Process

from agents import analyst, data_explorer, fin_expert, news_info_explorer
from tasks import advise, analyse, get_company_financials, get_company_news

# Create separate crews for parallel execution
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

# Analysis crew for sequential tasks that depend on parallel results
analysis_crew = Crew(
    agents=[analyst, fin_expert],
    tasks=[analyse, advise],
    verbose=True,
    process=Process.sequential,
    cache=True,
    max_rpm=15,
)


def run_crew_task(crew, inputs, task_name):
    """Helper function to run a crew task."""
    print(f"🚀 Starting {task_name}...")
    result = crew.kickoff(inputs=inputs)
    print(f"✅ Completed {task_name}")
    return result


def run_parallel_execution(stock_input):
    """Run financial analysis with parallel execution."""
    print("🚀 Starting Enhanced Financial Analysis with Parallel Execution...")

    # Phase 1: Run financial data gathering and news gathering in parallel
    print("\n🔄 Phase 1: Running Financial Data & News Gathering in Parallel...")
    parallel_start = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks to run in parallel
        financial_future = executor.submit(
            run_crew_task, financial_crew, stock_input, "Financial Data Gathering"
        )
        news_future = executor.submit(
            run_crew_task, news_crew, stock_input, "News Gathering"
        )

        # Wait for both tasks to complete
        financial_result = financial_future.result()
        news_result = news_future.result()

    parallel_end = time.time()
    parallel_time = parallel_end - parallel_start
    print(f"✅ Phase 1 completed in {parallel_time:.2f} seconds")

    # Phase 2: Run analysis and recommendation sequentially (they depend on Phase 1 results)
    print("\n🔄 Phase 2: Running Analysis & Recommendation...")
    analysis_start = time.time()

    # The analysis crew will use the context from the completed tasks
    analysis_result = analysis_crew.kickoff(inputs=stock_input)

    analysis_end = time.time()
    analysis_time = analysis_end - analysis_start
    print(f"✅ Phase 2 completed in {analysis_time:.2f} seconds")

    return parallel_time, analysis_time


def main():
    """Main function to run the financial analysis with configurable execution mode."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Financial Analysis with Configurable Execution Mode"
    )
    parser.add_argument(
        "--stock",
        default="RELIANCE",
        help="Stock symbol to analyze (default: RELIANCE)",
    )

    args = parser.parse_args()

    # Record start time
    start_time = time.time()

    # Scenario: Analyze specified stock
    print(f"\n📋 Stock Analysis: {args.stock}")
    stock_input = {"stock": args.stock}

    parallel_time, analysis_time = run_parallel_execution(stock_input)

    # Calculate and display results
    end_time = time.time()
    execution_time = end_time - start_time

    print("\n🎉 Financial analysis completed!")
    print(f"⏱️  Phase 1 (Parallel): {parallel_time:.2f} seconds")
    print(f"⏱️  Phase 2 (Sequential): {analysis_time:.2f} seconds")
    print(
        f"⏱️  Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
    )

    # Show potential time savings
    estimated_sequential_time = parallel_time * 2 + analysis_time
    time_saved = estimated_sequential_time - execution_time
    print(f"💡 Estimated time saved by parallel execution: {time_saved:.2f} seconds")
    try:
        percent_saved = (
            (time_saved / estimated_sequential_time) * 100
            if estimated_sequential_time > 0
            else 0
        )
        print(f"💡 Time saved by parallel execution: {percent_saved:.2f}%")
    except Exception:
        print("⚠️ Could not calculate time saved in percentage.")


if __name__ == "__main__":
    main()
