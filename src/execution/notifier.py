# -*- coding: utf-8 -*-
"""
DEMIR AI v11.1 - NOTIFIER (Türkçe Açıklamalı)
=============================================
Sinyalleri Telegram'a iletir.
Her değerin yanında Türkçe açıklama bulunur.

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

def _get_whale_explanation(score: float) -> str:
    """Whale skoru için Türkçe açıklama."""
    if score > 0.05:
        return "Balinalar ALIYOR 🐋📈"
    elif score < -0.05:
        return "Balinalar SATIYOR 🐋📉"
    elif score > 0:
        return "Hafif alım baskısı"
    elif score < 0:
        return "Hafif satış baskısı"
    else:
        return "Nötr"

def _get_liq_explanation(score: float) -> str:
    """Liquidation skoru için Türkçe açıklama."""
    if score > 0.05:
        return "Short squeeze riski var ↑"
    elif score < -0.05:
        return "Long squeeze riski var ↓"
    elif score > 0:
        return "Shortlar baskı altında"
    elif score < 0:
        return "Longlar baskı altında"
    else:
        return "Dengeli"

def _get_sent_explanation(score: float) -> str:
    """Sentiment skoru için Türkçe açıklama."""
    if score > 0.05:
        return "Piyasa iyimser (Greed)"
    elif score < -0.05:
        return "Piyasa korkulu (Fear)"
    elif score > 0:
        return "Hafif iyimserlik"
    elif score < 0:
        return "Hafif korku"
    else:
        return "Nötr duyarlılık"

def _get_side_explanation(side: str) -> str:
    """Yön açıklaması."""
    if side == "BUY":
        return "AL - Fiyat yükselecek beklentisi"
    else:
        return "SAT - Fiyat düşecek beklentisi"

def send_signal_alert(signal: dict):
    """
    Sinyal objesini formatla ve Türkçe açıklamalarla gönder.
    """
    emoji = "🟢" if signal['side'] == "BUY" else "🔴"
    side_exp = _get_side_explanation(signal['side'])
    
    # Risk hesaplama
    risk_pct = abs(signal['price'] - signal['stop_loss']) / signal['price'] * 100
    reward_pct = abs(signal['take_profit'] - signal['price']) / signal['price'] * 100
    
    # Temel mesaj
    msg = (
        f"{emoji} **SİNYAL ALARMI** {emoji}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 **{signal['symbol']}**\n"
        f"↕️ **{signal['side']}** - {side_exp}\n"
        f"💵 Fiyat: `{signal['price']:,.2f}`\n"
        f"🎯 Güven: **%{signal['confidence']*100:.1f}** _(Model bu tahminden %{signal['confidence']*100:.0f} emin)_\n"
        f"\n"
    )
    
    # Advanced modül verileri varsa ekle
    if signal.get('advanced'):
        adv = signal['advanced']
        
        whale_score = adv.get('whale_score', 0)
        liq_score = adv.get('liq_score', 0)
        sent_score = adv.get('sentiment_score', 0)
        boost = adv.get('boost', 0)
        
        msg += (
            f"📊 **GELİŞMİŞ ANALİZ:**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🐋 Whale: `{whale_score:+.2f}` → _{_get_whale_explanation(whale_score)}_\n"
            f"💥 Liq: `{liq_score:+.2f}` → _{_get_liq_explanation(liq_score)}_\n"
            f"😊 Sent: `{sent_score:+.2f}` → _{_get_sent_explanation(sent_score)}_\n"
            f"⚡ Toplam Boost: `{boost:+.2f}` → _Eşik {'düştü' if boost > 0 else 'yükseldi' if boost < 0 else 'değişmedi'}_\n"
            f"\n"
        )
    
    # Risk yönetimi
    msg += (
        f"🛡️ **RİSK YÖNETİMİ:**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛑 Stop Loss: `{signal['stop_loss']:,.2f}` → _Buraya düşerse %{risk_pct:.1f} zararla çık_\n"
        f"💰 Take Profit: `{signal['take_profit']:,.2f}` → _Buraya çıkarsa %{reward_pct:.1f} karla çık_\n"
        f"🎲 Pozisyon: `${signal['size_usd']:.2f}` → _Önerilen işlem büyüklüğü_\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 *Demir AI v11.1 Live*"
    )
    
    send_message(msg)
    logger.info(f"Signal sent for {signal['symbol']}")
