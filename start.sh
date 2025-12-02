#!/bin/bash

# 1. Botu arka planda başlat (& işareti arka plana atar)
echo "🚀 Starting AI Bot Engine..."
python main.py &

# 2. Dashboard'u ön planda başlat (Railway'in verdiği PORT'u dinle)
echo "📊 Starting Dashboard..."
streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0