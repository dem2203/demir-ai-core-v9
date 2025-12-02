#!/bin/bash

# Railway'den gelen PORT'u al, yoksa 8501 yap
PORT="${PORT:-8501}"

echo "-----------------------------------"
echo "🚀 DEMIR AI SYSTEM STARTING..."
echo "🔌 PORT DETECTED: $PORT"
echo "-----------------------------------"

# 1. Botu Arka Planda Başlat (& işareti ile)
python main.py &

# 2. Dashboard'u Ön Planda Başlat (Railway Portuna Bağla)
streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0