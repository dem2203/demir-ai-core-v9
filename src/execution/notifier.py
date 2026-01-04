# -*- coding: utf-8 -*-
"""
DEMIR AI v11 - NOTIFIER
=======================
Sinyalleri Telegram'a iletir.
Bot polling'inden bağımsız çalışır (API çağrısı).

Author: DEMIR AI Team
Date: 2026-01-04
"""
import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("NOTIFIER")

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_message(text: str):
    """Basit metin mesajı gönder."""
    if not TOKEN or not CHAT_ID:
        logger.error("TELEGRAM_TOKEN or CHAT_ID not set!")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            logger.error(f"Telegram API Error: {response.text}")
    except Exception as e:
        logger.error(f"Failed to send message: {e}")

def send_signal_alert(signal: dict):
    """
    Sinyal objesini formatla ve gönder.
    
    signal = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "price": 95000,
        "confidence": 0.82,
        "stop_loss": 94000,
        "take_profit": 96500,
        "size_usd": 150,
        "risk_ratio": 1.5,
        "time": ...
    }
    """
    emoji = "🟢" if signal['side'] == "BUY" else "🔴"
    
    msg = (
        f"{emoji} **SİNYAL ALARMI** {emoji}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🪙 **{signal['symbol']}**\n"
        f"↕️ Yön: **{signal['side']}**\n"
        f"💵 Fiyat: `{signal['price']}`\n"
        f"🎯 Güven: **%{signal['confidence']*100:.1f}**\n\n"
        f"🛑 SL: `{signal['stop_loss']}`\n"
        f"💰 TP: `{signal['take_profit']}`\n"
        f"🎲 Büyüklük: `${signal['size_usd']:.2f}`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🤖 *Demir AI v11 Live*"
    )
    
    send_message(msg)
    logger.info(f"Signal sent for {signal['symbol']}")
