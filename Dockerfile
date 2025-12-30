FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları + Playwright için gerekli dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    # Playwright Chromium dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright browserları indir
RUN playwright install chromium

# Cache busting - change date to force rebuild (2024-12-31-00:39)
ARG CACHE_BUST=2024123100039

# Kodları kopyala
COPY . .

# Çalıştırma izni
RUN chmod +x start.sh

# Port
EXPOSE 8501

# Başlat
CMD ["./start.sh"]