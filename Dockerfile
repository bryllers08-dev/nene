# ─────────────────────────────────────────────────────────────────────────────
# EXPIRYRANGE LSTM Bot — Railway Dockerfile
# Base: python:3.11-slim  (keeps image small)
# TensorFlow CPU-only build (no GPU needed, smaller image)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps needed by TensorFlow + scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy requirements first (Docker layer cache — only reinstalls on change)
COPY requirements.txt .

# Install Python deps
# tensorflow-cpu is smaller and sufficient — no GPU on Railway
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        tensorflow-cpu \
        scikit-learn \
        websockets \
        numpy \
        supabase

# Copy bot
COPY main_lstm_bot.py .

# ── Data directory ────────────────────────────────────────────────────────────
# Railway Volume should be mounted at /app/data
# This line just ensures the folder exists even without a volume
RUN mkdir -p /app/data/symbol_data

# ── Environment defaults ──────────────────────────────────────────────────────
# All of these are overridden by Railway ENV vars — these are just safe defaults
ENV PERSIST_DIR=/app/data \
    DATA_DIR=/app/data/symbol_data \
    PORT=8080 \
    DERIV_APP_ID=1089 \
    SUPABASE_URL=https://qtkjixwiahghisqqxrcz.supabase.co \
    SUPABASE_BUCKET=Supabot \
    BASE_STAKE=1.0 \
    TARGET_PROFIT=45.0 \
    STOP_LOSS=10.0 \
    LSTM_THRESHOLD=0.72 \
    LSTM_THRESHOLD_FLOOR=0.60 \
    LSTM_SEQ_LEN=60 \
    MAX_IDLE_MINUTES=30 \
    RETRAIN_HOURS=4 \
    UPLOAD_INTERVAL_SECS=300

# Expose health server port
EXPOSE 8080

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Railway pings / every 30s — gives 5min grace period for LSTM training
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# ── Run ───────────────────────────────────────────────────────────────────────
CMD ["python", "-u", "main_lstm_bot.py"]
