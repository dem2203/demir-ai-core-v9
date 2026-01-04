FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları (Minimal - Playwright YOK)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları kopyala
COPY . .

# Başlat - DEMIR AI v11 Live Trading
CMD ["python", "run_live_trading.py"]