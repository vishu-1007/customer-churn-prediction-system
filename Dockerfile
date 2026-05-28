# ── Base image ───────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ─────────────────────────────────────────────────
LABEL maintainer="Goli Raghu Sharan Teja"
LABEL project="Customer Churn Prediction"
LABEL version="1.0.0"

# ── Set working directory ────────────────────────────────────
WORKDIR /app

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ───────────────────────────────────────
COPY configs/    ./configs/
COPY src/        ./src/
COPY app/        ./app/
COPY models/     ./models/
COPY outputs/    ./outputs/

# ── Expose Streamlit port ────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Launch app ───────────────────────────────────────────────
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
