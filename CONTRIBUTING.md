# Contributing to AI-Powered Stocks Analyser

Thank you for your interest in contributing to the AI-Powered Stocks Analyser! This project is designed to demonstrate high-level AI engineering patterns, and we welcome contributions that maintain or improve its architectural integrity and reliability.

## 🚀 Development Workflow

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/raghavg27/ai-powered-stocks-analyser.git
    cd ai-powered-stocks-analyser
    ```
2.  **Set up the environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Run tests to ensure everything is working**:
    ```bash
    pytest
    ```

---

## 🛠️ How to Add a New Financial Tool

Tools are the capabilities provided to agents. To add a new one:

1.  **Define the function in [tools.py](tools.py)**:
    *   Decorate with `@tool("Tool Name")`.
    *   Write a comprehensive docstring including `Args` and `Returns`. This is what the LLM uses to decide when to call the tool.
    *   Use the `_with_retry` helper for network requests to handle transient failures.

    ```python
    @tool("Get ESG Scores")
    def get_esg_scores(symbol: str) -> str:
        """Fetch environmental, social, and governance scores for a stock.
        
        Args:
            symbol: Stock ticker (e.g. MSFT, RELIANCE.NS).
        Returns:
            JSON string of ESG metrics or an error message.
        """
        # Your implementation here...
    ```

2.  **Assign to an Agent in [agents.py](agents.py)**:
    *   Import your new tool.
    *   Add it to the `tools` list of the appropriate `Agent`.

---

## 🤖 How to Add a New Agent

Agents are autonomous workers with specific roles. To add one:

1.  **Define the Agent in [agents.py](agents.py)**:
    *   Give it a clear `role`, `goal`, and `backstory`.
    *   Assign it the necessary `tools`.
    *   Set appropriate `max_iter` and `max_execution_time`.

    ```python
    esg_analyst = Agent(
        role="ESG Specialist",
        goal="Evaluate the sustainability and ethical impact of a company",
        backstory="You are an expert in ESG frameworks...",
        tools=[get_esg_scores],
        llm=llm,
        verbose=True
    )
    ```

2.  **Define a Task in [tasks.py](tasks.py)**:
    *   Create a `Task` that uses your new agent.
    *   Specify the `description` and `expected_output`.
    *   Add a Pydantic guardrail if the output needs to be structured.

3.  **Integrate into the Pipeline in [main.py](main.py)**:
    *   Add the agent/task to a `Crew`.
    *   If it can run in parallel, add it to Phase 1 in `run_parallel_execution`.

---

## ✅ Quality Standards

*   **Pydantic Guardrails**: Any agent producing final data for the UI should have its output validated by a Pydantic schema in `tasks.py`.
*   **Logging**: Use the central `logger` for all significant events.
*   **Testing**: 
    *   New tools MUST have unit tests in `tests/test_tools.py`.
    *   Always use mocks for network calls (yfinance, EXA) and LLM calls.
*   **Docstrings**: Maintain institutional-quality docstrings for all new functions and classes.

## 🤝 Pull Request Process

1.  Create a feature branch from `main`.
2.  Ensure all tests pass (`pytest`).
3.  Update documentation if you've changed the system architecture or added major features.
4.  Open a Pull Request with a detailed summary of your changes and the "Why" behind them.
