FROM python:3.11-slim

WORKDIR /app

COPY . .

# Gerekli kütüphaneleri yükle
RUN pip install --no-cache-dir -r requirements.txt

# start.sh dosyasına çalıştırma izni ver
RUN chmod +x start.sh

# Çift motoru çalıştır (Bot + Dashboard)
CMD ["./start.sh"]