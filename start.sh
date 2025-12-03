#!/bin/bash

# Railway Deployment Script

echo ">> 🚀 STARTING DEMIR AI CORE v9.0"
echo ">> Environment: $ENVIRONMENT"

# 1. Streamlit Dashboard'ı Arka Planda Başlat
echo ">> 📡 Launching Dashboard..."
streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0 --server.headless true &

# 2. Ana Bot Motorunu Başlat
# Bot, dashboard'un veri okuyabilmesi için JSON dosyaları üretir.
echo ">> 🧠 Launching AI Engine..."
python main.py