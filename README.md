# AI-Powered Stocks Analyser

AI-Powered Stocks Analyser is a multi-agent system built using [CrewAI](https://www.crewai.com/) to perform comprehensive financial analysis on stocks. The tool uses multiple specialized AI agents working together to gather financial data, fetch the latest news, analyze the gathered information, and provide expert investment recommendations.

## Features

- **Multi-Agent Architecture**: Utilizes specialized agents (Data Explorer, News Info Explorer, Financial Analyst, Financial Expert) to handle different aspects of stock analysis.
- **Parallel Execution**: Speeds up the analysis process by gathering financial data and recent news in parallel using `ThreadPoolExecutor`.
- **Comprehensive Analysis**: Combines quantitative financial data with qualitative news sentiment to provide a holistic view of the stock's potential.
- **Command-Line Interface**: Easy to use from the command line with configurable stock symbols.

## How It Works

The analysis is broken down into two main phases to optimize execution time:

1. **Phase 1: Parallel Data Gathering**
   - **Financial Crew**: A Data Explorer agent gathers the latest financial data and metrics for the specified stock.
   - **News Crew**: A News Info Explorer agent searches for the latest news and market sentiment surrounding the company.
   - *These two tasks run concurrently to save time.*

2. **Phase 2: Sequential Analysis & Recommendation**
   - **Analysis Crew**: A Financial Analyst agent reviews the gathered data and news to form a coherent analysis. Then, a Financial Expert agent reviews the analysis to provide a final investment recommendation (e.g., Buy, Hold, Sell) and strategy.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd ai-powered-stocks-analyser
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The agents rely on LLMs and search tools to function. You will need to set up your environment variables. 
Create a `.env` file in the root directory and add the necessary API keys based on the LLM provider and search tools (like EXA) configured in your agents:

```env
OPENAI_API_KEY=your_openai_api_key_here
EXA_API_KEY=your_exa_api_key_here
# Add other necessary API keys here
```

## Usage

Run the main script to start the analysis. By default, it analyzes the "RELIANCE" stock.

```bash
python main.py
```

To analyze a specific stock, use the `--stock` argument:

```bash
python main.py --stock AAPL
```

### Output

The script will display the progress of each phase in the console, followed by the detailed analysis and final recommendation. At the end, it will output the execution time and the estimated time saved by using parallel execution.

## Project Structure

- `main.py`: The entry point of the application that orchestrates the crews and manages parallel execution.
- `agents.py`: Defines the different CrewAI agents (Data Explorer, News Explorer, Analyst, Expert).
- `tasks.py`: Defines the tasks assigned to each agent.
- `tools/` & `tools.py`: Contains custom tools used by the agents to fetch data (e.g., `yfinance` tools, Exa search).
- `config/`: Contains configuration files for agents and tasks.
- `requirements.txt`: Lists all Python dependencies required to run the project.
