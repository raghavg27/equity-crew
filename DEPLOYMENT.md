# Deployment Guide

This guide provides instructions for deploying the AI-Powered Stocks Analyser to various environments, ensuring that the multi-agent system and PDF generation features work reliably.

## 🐳 Docker Deployment (Recommended)

Docker is the most reliable way to run the application, as it bundles all system-level dependencies required for PDF generation (WeasyPrint).

### Prerequisites
- Docker and Docker Compose installed.

### Steps
1.  **Environment Setup**:
    Ensure your `.env` file is populated with your API keys:
    ```env
    OPENROUTER_API_KEY=your_key_here
    EXA_API_KEY=your_key_here
    ```

2.  **Launch the System**:
    ```bash
    docker-compose up --build
    ```

3.  **Access the App**:
    Open your browser to `http://localhost:8501` to use the Streamlit interface.

---

## ☁️ Streamlit Cloud Deployment

Streamlit Cloud is ideal for hosting the project for free.

### Steps
1.  **Push to GitHub**: Ensure your code is in a GitHub repository.
2.  **Deploy**:
    - Sign in to [Streamlit Cloud](https://share.streamlit.io).
    - Click **"New app"**.
    - Select your repository, branch (`main`), and main file path (`app.py`).
3.  **Configure Secrets**:
    This is critical. In the app settings on Streamlit Cloud, go to **Secrets** and add:
    ```toml
    OPENROUTER_API_KEY = "sk-or-v1-..."
    EXA_API_KEY = "..."
    ```
4.  **System Dependencies**: The project includes a `packages.txt` file. Streamlit Cloud will automatically detect this and install the necessary system libraries (libpango, libcairo, etc.) for WeasyPrint.

---

## 💻 Manual Local Deployment

If you prefer to run the app directly on your machine:

### 1. Install System Libraries
WeasyPrint requires certain libraries to be installed on your OS:

- **macOS**: `brew install cairo pango gdk-pixbuf libffi`
- **Ubuntu/Debian**: `sudo apt-get install python3-dev libffi-dev liboa-dev libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf2.0-0 shared-mime-info`
- **Windows**: Follow the [WeasyPrint Windows installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows).

### 2. Set up Python Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 📁 Troubleshooting Persistence

By default, the application writes reports to `task_outputs/` and logs to `logs/`. 
- **Docker**: These directories are mapped as volumes in `docker-compose.yml`, so reports will persist on your host machine even after the container stops.
- **Streamlit Cloud**: Files written to disk are ephemeral and will be lost when the app reboots. Users should download the **PDF Report** immediately after generation.
