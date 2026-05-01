import base64
import io
import json
import os
import time
from datetime import datetime

import markdown
import matplotlib.pyplot as plt
import yfinance as yf
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

from logger import get_logger
from tools import get_valuation_metrics

logger = get_logger(__name__)

def _generate_price_chart(symbol: str) -> str | None:
    """Generates a 1-year price chart with 50/200 MAs and returns it as a base64 string."""
    try:
        logger.info("📈  Generating price chart for %s...", symbol)
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            logger.warning("⚠️   No price data available for chart generation.")
            return None

        # Calculate MAs
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA200'] = hist['Close'].rolling(window=200).mean()

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(hist.index, hist['Close'], label='Close Price', color='#2b6cb0', linewidth=1.5)
        plt.plot(hist.index, hist['SMA50'], label='50-Day SMA', color='#ed8936', linewidth=1.2, linestyle='--')
        plt.plot(hist.index, hist['SMA200'], label='200-Day SMA', color='#e53e3e', linewidth=1.2, linestyle='--')
        
        plt.title(f"{symbol} - 1 Year Price History", fontsize=14, pad=15)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Price", fontsize=10)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Adjust layout
        plt.tight_layout()

        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close()
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64

    except Exception as e:
        logger.error("❌  Failed to generate price chart: %s", e)
        return None


def _read_markdown_as_html(filepath: str) -> str:
    """Reads a markdown file and converts it to an HTML string."""
    if not os.path.exists(filepath):
        return f"<p><em>{filepath} not found.</em></p>"
    
    with open(filepath, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    return markdown.markdown(md_text, extensions=['extra', 'nl2br'])


def generate_pdf_report(symbol: str) -> str | None:
    """
    Compiles the analysis results into a rich PDF report.
    Reads outputs from task_outputs/, fetches a chart and metrics, and renders a PDF.
    
    Returns the path to the generated PDF.
    """
    logger.info("📄  Compiling rich PDF report for %s...", symbol)
    
    try:
        # 1. Parse Recommendation JSON
        rec_path = "task_outputs/investment_recommendation.md"
        recommendation = {}
        if os.path.exists(rec_path):
            with open(rec_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Remove markdown json codeblock ticks if present
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                
                try:
                    recommendation = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning("⚠️   Failed to parse recommendation JSON: %s", e)
                    # Fallback so it doesn't crash the template
                    recommendation = {
                        "action": "UNKNOWN",
                        "confidence": 0,
                        "target_price": "N/A",
                        "current_price": "N/A",
                        "reasons": ["Could not parse JSON"],
                        "risks": ["Could not parse JSON"]
                    }

        # 2. Get Valuation Metrics
        metrics_json = get_valuation_metrics.func(symbol)
        metrics = {}
        try:
            metrics = json.loads(metrics_json)
        except Exception:
            pass
            
        # 3. Generate Chart
        chart_base64 = _generate_price_chart(symbol)
        
        # 4. Read Markdown Files
        fin_html = _read_markdown_as_html("task_outputs/financial_analysis.md")
        tech_html = _read_markdown_as_html("task_outputs/technical_analysis.md")
        peer_html = _read_markdown_as_html("task_outputs/peer_comparison.md")
        
        # 5. Render Template
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('report_template.html')
        
        html_out = template.render(
            symbol=symbol,
            company_name=metrics.get("name", symbol),
            date=datetime.now().strftime("%d %b %Y, %H:%M"),
            recommendation=recommendation,
            metrics=metrics,
            chart_img_base64=chart_base64,
            financial_analysis_html=fin_html,
            technical_analysis_html=tech_html,
            peer_comparison_html=peer_html
        )
        
        # 6. Save PDF
        output_path = f"task_outputs/{symbol.replace('^', '').replace('.', '_')}_Report_{time.strftime('%Y%m%d')}.pdf"
        HTML(string=html_out).write_pdf(output_path)
        
        logger.info("✅  PDF Report successfully generated: %s", output_path)
        return output_path

    except Exception as e:
        logger.error("❌  Error generating PDF report: %s", e)
        return None
