@echo off
title DEMIR AI v10 - LAUNCHER
echo Starting DEMIR AI v10 System...
echo --------------------------------

echo 1. Starting Web Dashboard...
start "DEMIR AI - DASHBOARD" cmd /k "uvicorn src.dashboard.server:app --reload"

echo 2. Starting Telegram Bot...
start "DEMIR AI - TELEGRAM BOT" cmd /k "python src/v10/telegram_bot.py"

echo 3. Starting Main Engine...
start "DEMIR AI - ENGINE" cmd /k "python -m src.v10.engine"

echo.
echo All systems launched in separate windows!
echo Dashboard: http://localhost:8000
echo Telegram: /analiz BTC
echo.
pause
