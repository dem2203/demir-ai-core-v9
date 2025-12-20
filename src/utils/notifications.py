# -*- coding: utf-8 -*-
"""
DEMIR AI - Telegram Notification Manager V2
============================================
4 Bildirim Kategorisi:
1. TEKNİK SİNYAL - Matematiksel indikatör bazlı
2. CANLI VERİ TAHMİNİ - Web scraping bazlı (Whale, Liq, OB)
3. ANİ HAREKET UYARISI - Volatilite ve risk tespiti
4. AI GÖZ ANALİZİ - Düşünen yapay zeka perspektifi
"""
import logging
import requests
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from src.config.settings import Config

logger = logging.getLogger("NOTIFICATION_MANAGER")


class NotificationManager:
    """
    DEMIR AI V2.0 - 4-Katmanlı Bildirim Sistemi
    """
    
    def __init__(self):
        self.telegram_token = Config.TELEGRAM_TOKEN
        self.telegram_chat_id = Config.TELEGRAM_CHAT_ID
        self.telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage" if self.telegram_token else None
        
        if self.telegram_token and self.telegram_chat_id:
            logger.info("✅ Telegram V2 Ready (4 Notification Types)")
        else:
            logger.warning("⚠️ Telegram credentials not configured")

    # =========================================
    # TEMEL MESAJ GÖNDERME
    # =========================================
    
    async def send_message_raw(self, text: str):
        """Temel mesaj gönderme - Tüm metodların temeli."""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        try:
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                lambda: requests.post(self.telegram_url, data=payload, timeout=10)
            )
        except Exception as e:
            logger.error(f"Telegram Error: {e}")

    # =========================================
    # 1️⃣ TEKNİK SİNYAL
    # =========================================
    
    async def send_technical_signal(
        self,
        symbol: str,
        direction: str,  # "LONG" or "SHORT"
        entry: float,
        tp1: float,
        tp2: float,
        sl: float,
        strong_indicators: List[Dict],  # [{"name": "RSI", "value": 78, "reason": "Aşırı Alım"}]
        confidence: float
    ):
        """
        Teknik/Matematiksel indikatör bazlı sinyal.
        Sadece %70+ güven veren indikatörler gösterilir.
        """
        dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
        
        # Risk/Reward hesapla
        risk = abs(entry - sl)
        reward1 = abs(tp1 - entry)
        rr1 = round(reward1 / risk, 1) if risk > 0 else 0
        reward2 = abs(tp2 - entry)
        rr2 = round(reward2 / risk, 1) if risk > 0 else 0
        
        # Güçlü indikatörleri listele
        indicator_lines = ""
        for ind in strong_indicators:
            indicator_lines += f"• {ind['name']}: {ind['value']:.0f}% ({ind['reason']})\n"
        
        msg = f"""🎯 *TEKNİK SİNYAL - {symbol}*
━━━━━━━━━━━━━━━━━━
📊 Yön: *{dir_emoji}*
💰 Entry: `${entry:,.2f}`
🎯 TP1: `${tp1:,.2f}` (R:R {rr1})
🎯 TP2: `${tp2:,.2f}` (R:R {rr2})
🛡️ SL: `${sl:,.2f}`
━━━━━━━━━━━━━━━━━━
✅ *Güçlü Sinyaller (%70+):*
{indicator_lines}━━━━━━━━━━━━━━━━━━
🧠 AI Güven: *{confidence:.0f}%*
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}"""
        
        await self.send_message_raw(msg)
        logger.info(f"📊 Technical Signal sent: {symbol} {direction} {confidence:.0f}%")

    # =========================================
    # 2️⃣ CANLI VERİ TAHMİNİ
    # =========================================
    
    async def send_live_prediction(
        self,
        symbol: str,
        prediction: str,  # "YUKARI" or "ASAGI"
        target_price: float,
        timeframe: str,  # "1-4 saat", "24 saat", etc.
        data_sources: Dict,  # {"whale": {...}, "orderbook": {...}, "liq": {...}, ...}
        confidence: float
    ):
        """
        Canlı veri bazlı tahmin.
        Whale, Order Book, Liquidation, Kurumsal akış kullanır.
        """
        pred_emoji = "📈 YUKARI" if prediction == "YUKARI" else "📉 AŞAĞI"
        
        # Veri kaynakları
        whale = data_sources.get('whale', {})
        orderbook = data_sources.get('orderbook', {})
        liq = data_sources.get('liq', {})
        funding = data_sources.get('funding', {})
        institutional = data_sources.get('institutional', {})
        
        source_lines = ""
        
        # Whale
        if whale.get('net_flow'):
            flow = whale['net_flow']
            flow_type = "ALIŞ" if flow > 0 else "SATIŞ"
            source_lines += f"🐋 Whale: ${abs(flow)/1e6:.1f}M NET {flow_type}\n"
        
        # Order Book
        if orderbook.get('imbalance'):
            imb = orderbook['imbalance']
            imb_type = "BID ağırlıklı" if imb > 1 else "ASK ağırlıklı"
            source_lines += f"📊 Order Book: {imb:.1f}x {imb_type}\n"
        
        # Liquidation
        if liq.get('nearest_level'):
            liq_dir = liq.get('direction', '')
            liq_amount = liq.get('amount', 0)
            source_lines += f"💧 Liq Zones: ${liq['nearest_level']:,.0f} ({liq_dir}, ${liq_amount/1e6:.0f}M)\n"
        
        # Funding
        if funding.get('rate'):
            rate = funding['rate']
            risk = "Short Squeeze riski" if rate < -0.01 else "Long Squeeze riski" if rate > 0.05 else ""
            source_lines += f"📉 Funding: {rate:.3f}% {risk}\n"
        
        # Institutional
        if institutional.get('exchanges'):
            exchanges = institutional['exchanges']
            source_lines += f"🏦 Kurumsal: {', '.join(exchanges)}\n"
        
        if not source_lines:
            source_lines = "• Veri toplanıyor...\n"
        
        msg = f"""🐋 *CANLI VERİ TAHMİNİ - {symbol}*
━━━━━━━━━━━━━━━━━━
{pred_emoji}
🎯 Hedef: `${target_price:,.2f}`
⏱️ Süre: {timeframe}

━━━ VERİ KAYNAKLARI ━━━
{source_lines}━━━━━━━━━━━━━━━━━━
🧠 AI Güven: *{confidence:.0f}%*
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}"""
        
        await self.send_message_raw(msg)
        logger.info(f"🐋 Live Prediction sent: {symbol} {prediction} {confidence:.0f}%")

    # =========================================
    # 3️⃣ ANİ HAREKET UYARISI
    # =========================================
    
    async def send_sudden_alert(
        self,
        symbol: str,
        alert_type: str,  # "FIRSAT" or "RİSK"
        direction: str,  # "YUKARI" or "ASAGI"
        potential_price: float,
        potential_pct: float,
        triggers: List[Dict],  # [{"name": "Bollinger Squeeze", "value": "0.8%", "status": "PATLAMA YAKLAŞIYOR"}]
        expected_time: str,  # "15-60 dakika"
        warning: str = None  # Optional risk warning
    ):
        """
        Ani hareket öncesi uyarı.
        Volatilite spike, squeeze, cascade tespiti.
        """
        type_emoji = "💡 FIRSAT" if alert_type == "FIRSAT" else "⚠️ RİSK"
        dir_emoji = "📈 YUKARI" if direction == "YUKARI" else "📉 AŞAĞI"
        pct_sign = "+" if potential_pct > 0 else ""
        
        # Tetikleyiciler
        trigger_lines = ""
        for t in triggers:
            trigger_lines += f"• {t['name']}: {t['value']} ({t['status']})\n"
        
        if not trigger_lines:
            trigger_lines = "• Çoklu sinyal uyumu\n"
        
        msg = f"""⚡ *ANİ HAREKET UYARISI!*
━━━━━━━━━━━━━━━━━━
🚨 {type_emoji}
📍 Coin: *{symbol}*
{dir_emoji}
🎯 Potansiyel: `${potential_price:,.2f}` ({pct_sign}{potential_pct:.1f}%)

━━━ TETİKLEYİCİLER ━━━
{trigger_lines}━━━━━━━━━━━━━━━━━━
⏱️ Beklenen Süre: {expected_time}"""
        
        if warning:
            msg += f"\n⚠️ Dikkat: _{warning}_"
        
        msg += f"\n⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        
        await self.send_message_raw(msg)
        logger.info(f"⚡ Sudden Alert sent: {symbol} {alert_type} {direction}")

    # =========================================
    # 4️⃣ AI GÖZ ANALİZİ
    # =========================================
    
    async def send_ai_vision(
        self,
        symbol: str,
        overview: str,  # Genel görünüm paragrafı
        bull_scenario: Dict,  # {"probability": 40, "condition": "$96k kırılırsa", "target": "$98k"}
        bear_scenario: Dict,  # {"probability": 60, "condition": "$94k kırılırsa", "target": "$91k"}
        recommendation: str,  # "BEKLE", "AL", "SAT"
        recommendation_reason: str
    ):
        """
        Düşünen yapay zeka perspektifi.
        Piyasayı bir analist gibi değerlendirir.
        """
        # Öneriye göre emoji
        rec_map = {
            "BEKLE": "⏳ BEKLE",
            "AL": "🟢 AL",
            "SAT": "🔴 SAT",
            "DİKKATLİ AL": "🟡 DİKKATLİ AL",
            "DİKKATLİ SAT": "🟡 DİKKATLİ SAT"
        }
        rec_text = rec_map.get(recommendation, f"📊 {recommendation}")
        
        msg = f"""🧠 *AI PİYASA ANALİZİ - {symbol}*
━━━━━━━━━━━━━━━━━━
📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}

*Genel Görünüm:*
{overview}

*Bull Senaryosu ({bull_scenario.get('probability', 0)}%):*
{bull_scenario.get('condition', 'N/A')} → {bull_scenario.get('target', 'N/A')}

*Bear Senaryosu ({bear_scenario.get('probability', 0)}%):*
{bear_scenario.get('condition', 'N/A')} → {bear_scenario.get('target', 'N/A')}

━━━━━━━━━━━━━━━━━━
*AI Tavsiyesi:*
{rec_text}
_{recommendation_reason}_
━━━━━━━━━━━━━━━━━━"""
        
        await self.send_message_raw(msg)
        logger.info(f"🧠 AI Vision sent: {symbol} → {recommendation}")

    # =========================================
    # YARDIMCI: Model Update Bildirimi
    # =========================================
    
    async def send_model_update(
        self,
        model_type: str,  # "LSTM" or "RL"
        symbol: str,
        accuracy: float,
        loss: float,
        samples: int
    ):
        """Model eğitim tamamlandı bildirimi."""
        msg = f"""🧠 *MODEL GÜNCELLENDİ*
━━━━━━━━━━━━━━━━━━
📊 Model: {model_type}
📍 Coin: {symbol}
✅ Accuracy: {accuracy:.1%}
📉 Loss: {loss:.4f}
📚 Samples: {samples:,}
━━━━━━━━━━━━━━━━━━
⏰ Sonraki eğitim: 24 saat
"""
        await self.send_message_raw(msg)
        logger.info(f"🧠 Model Update sent: {model_type} {symbol}")