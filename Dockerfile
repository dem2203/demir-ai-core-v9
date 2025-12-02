# Hafif Python sürümü
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Dosyaları kopyala
COPY . .

# Gerekli kütüphaneleri yükle
RUN pip install --no-cache-dir -r requirements.txt

# BOTU BAŞLAT (Arayüzü değil, Ana Motoru çalıştır)
CMD ["python", "main.py"]