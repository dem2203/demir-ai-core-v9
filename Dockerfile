# Hafif Python sürümü
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Dosyaları kopyala
COPY . .

# Kütüphaneleri yükle
RUN pip install --no-cache-dir -r requirements.txt

# start.sh dosyasına "Çalıştırılma İzni" ver (Çok Önemli!)
RUN chmod +x start.sh

# Çift motoru başlat
CMD ["./start.sh"]