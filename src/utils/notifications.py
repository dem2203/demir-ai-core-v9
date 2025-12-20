# -*- coding: utf-8 -*-
"""
DEMIR AI - Telegram Notification Manager V3 (INSTITUTIONAL GRADE)
=================================================================
17 Veri Kaynağı + 13 Tetikleyici Entegre

4 Bildirim Kategorisi:
1. TEKNİK SİNYAL - Matematiksel indikatör bazlı
2. CANLI VERİ TAHMİNİ - 17 kaynak entegre
3. ANİ HAREKET UYARISI - 13 tetikleyici
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
    DEMIR AI V3.0 - Kurumsal Seviye Bildirim Sistemi
    """
    
    def __init__(self):
        self.telegram_token = Config.TELEGRAM_TOKEN
        self.telegram_chat_id = Config.TELEGRAM_CHAT_ID
        self.telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage" if self.telegram_token else None
        
        if self.telegram_token and self.telegram_chat_id:
            logger.info("✅ Telegram V3 Ready (17 Sources + 13 Triggers)")
        else:
            logger.warning("⚠️ Telegram credentials not configured")

    # =========================================
    # TEMEL MESAJ GÖNDERME
    # =========================================
    
    async def send_message_raw(self, text: str):
        """Temel mesaj gönderme."""
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
        direction: str,
        entry: float,
        tp1: float,
        tp2: float,
        sl: float,
        strong_indicators: List[Dict],
        confidence: float
    ):
        """Teknik/Matematiksel indikatör bazlı sinyal."""
        dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
        
        risk = abs(entry - sl)
        rr1 = round(abs(tp1 - entry) / risk, 1) if risk > 0 else 0
        rr2 = round(abs(tp2 - entry) / risk, 1) if risk > 0 else 0
        
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
        logger.info(f"📊 Technical Signal: {symbol} {direction}")

    # =========================================
    # 2️⃣ CANLI VERİ TAHMİNİ (17 KAYNAK)
    # =========================================
    
    async def send_live_prediction(self, symbol: str = "BTCUSDT"):
        """
        17 kaynaktan toplanan verilerle TAKTİKSEL tahmin gönder.
        ✅ Hangi kaynaklar GÜÇLÜ sinyal veriyor?
        ✅ Şimdi giriş yapmalı mı?
        ✅ Hedefler neler?
        """
        try:
            from src.brain.institutional_aggregator import get_aggregator
            agg = get_aggregator()
            snapshot = await agg.get_live_snapshot(symbol)
            
            # Yön analizi - Her kaynak için güç seviyesi belirleme
            bullish_signals = []  # (score, text) tuples
            bearish_signals = []
            neutral_signals = []
            
            # Fiyat bilgisi
            current_price = snapshot.current_price or 0
            
            # 1. Whale (GÜÇLÜ sinyal kaynağı)
            if snapshot.whale_net_flow > 1000000:
                bullish_signals.append((1.5, f"🐋 Whale: +${snapshot.whale_net_flow/1e6:.1f}M *BÜYÜK ALIM*"))
            elif snapshot.whale_net_flow > 500000:
                bullish_signals.append((1.0, f"🐋 Whale: +${snapshot.whale_net_flow/1e6:.1f}M net alım"))
            elif snapshot.whale_net_flow < -1000000:
                bearish_signals.append((1.5, f"🐋 Whale: ${abs(snapshot.whale_net_flow)/1e6:.1f}M *BÜYÜK SATIŞ*"))
            elif snapshot.whale_net_flow < -500000:
                bearish_signals.append((1.0, f"🐋 Whale: ${abs(snapshot.whale_net_flow)/1e6:.1f}M net satış"))
            
            # 2. Order Book (ÇOK GÜÇLÜ - anlık alım satım baskısı)
            if snapshot.orderbook_imbalance > 2.0:
                bullish_signals.append((2.0, f"📊 Order Book: *{snapshot.orderbook_imbalance:.1f}x BID DUVARI* 🔥"))
            elif snapshot.orderbook_imbalance > 1.3:
                bullish_signals.append((1.0, f"📊 Order Book: {snapshot.orderbook_imbalance:.2f}x BID ağırlıklı"))
            elif snapshot.orderbook_imbalance < 0.5:
                bearish_signals.append((2.0, f"📊 Order Book: *{1/snapshot.orderbook_imbalance:.1f}x ASK DUVARI* 🔥"))
            elif snapshot.orderbook_imbalance < 0.7:
                bearish_signals.append((1.0, f"📊 Order Book: {1/snapshot.orderbook_imbalance:.2f}x ASK ağırlıklı"))
            
            # 3. CVD - Cumulative Volume Delta (ÇOK GÜÇLÜ)
            if snapshot.cvd_trend == "BULLISH" and snapshot.cvd_value > 0:
                bullish_signals.append((1.5, f"📊 CVD: BULLISH ({snapshot.cvd_value:,.0f})"))
            elif snapshot.cvd_trend == "BEARISH" and snapshot.cvd_value < 0:
                bearish_signals.append((1.5, f"📊 CVD: BEARISH ({snapshot.cvd_value:,.0f})"))
            
            # 4. Taker Buy/Sell Ratio (GÜÇLÜ - anlık momentum)
            if snapshot.taker_buy_ratio > 0.65:
                bullish_signals.append((1.5, f"💥 Taker: *%{snapshot.taker_buy_ratio*100:.0f} ALICI* - Momentum!"))
            elif snapshot.taker_buy_ratio > 0.55:
                bullish_signals.append((1.0, f"📊 Taker: %{snapshot.taker_buy_ratio*100:.0f} alıcı"))
            elif snapshot.taker_buy_ratio < 0.35:
                bearish_signals.append((1.5, f"💥 Taker: *%{(1-snapshot.taker_buy_ratio)*100:.0f} SATICI* - Baskı!"))
            elif snapshot.taker_buy_ratio < 0.45:
                bearish_signals.append((1.0, f"📊 Taker: %{(1-snapshot.taker_buy_ratio)*100:.0f} satıcı"))
            
            # 5. Funding Rate (Kontrarian sinyal)
            if snapshot.funding_rate > 0.05:
                bearish_signals.append((1.0, f"💸 Funding: +{snapshot.funding_rate:.3f}% *(Long Squeeze Riski)*"))
            elif snapshot.funding_rate < -0.03:
                bullish_signals.append((1.0, f"💸 Funding: {snapshot.funding_rate:.3f}% *(Short Squeeze Riski)*"))
            
            # 6. Long/Short Ratio (Kontrarian sinyal)
            if snapshot.long_short_ratio > 2.0:
                bearish_signals.append((1.5, f"⚠️ L/S Ratio: {snapshot.long_short_ratio:.2f} *ÇOK FAZLA LONG*"))
            elif snapshot.long_short_ratio > 1.5:
                bearish_signals.append((0.5, f"📊 L/S Ratio: {snapshot.long_short_ratio:.2f} (Kalabalık long)"))
            elif snapshot.long_short_ratio < 0.5:
                bullish_signals.append((1.5, f"🚀 L/S Ratio: {snapshot.long_short_ratio:.2f} *SHORT SQUEEZE HAZIR*"))
            elif snapshot.long_short_ratio < 0.7:
                bullish_signals.append((0.5, f"📊 L/S Ratio: {snapshot.long_short_ratio:.2f} (Kalabalık short)"))
            
            # 7. Fear & Greed Index (GÜÇLÜ kontrarian)
            if snapshot.fear_greed_index < 20:
                bullish_signals.append((1.5, f"😱 Fear&Greed: *{snapshot.fear_greed_index}* _(Extreme Fear = ALIM FIRSATI)_"))
            elif snapshot.fear_greed_index < 30:
                bullish_signals.append((1.0, f"😱 Fear&Greed: {snapshot.fear_greed_index} (Fear)"))
            elif snapshot.fear_greed_index > 80:
                bearish_signals.append((1.5, f"🤑 Fear&Greed: *{snapshot.fear_greed_index}* _(Extreme Greed = DİKKAT)_"))
            elif snapshot.fear_greed_index > 70:
                bearish_signals.append((1.0, f"🤑 Fear&Greed: {snapshot.fear_greed_index} (Greed)"))
            
            # 8. Options Put/Call
            if snapshot.put_call_ratio > 1.5:
                bearish_signals.append((1.0, f"📈 Put/Call: {snapshot.put_call_ratio:.2f} (Hedge ağırlıklı)"))
            elif snapshot.put_call_ratio < 0.6:
                bullish_signals.append((1.0, f"📈 Put/Call: {snapshot.put_call_ratio:.2f} (Call ağırlıklı)"))
            
            # 9. Exchange Netflow
            if snapshot.exchange_netflow < -200:
                bullish_signals.append((1.0, f"🏦 Exchange: *${abs(snapshot.exchange_netflow):.0f}M OUTFLOW*"))
            elif snapshot.exchange_netflow > 200:
                bearish_signals.append((1.0, f"🏦 Exchange: ${snapshot.exchange_netflow:.0f}M inflow"))
            
            # 10. ETF Flow
            if snapshot.etf_flow_daily > 200:
                bullish_signals.append((1.5, f"📊 ETF: *+${snapshot.etf_flow_daily:.0f}M INFLOW* 🏦"))
            elif snapshot.etf_flow_daily < -200:
                bearish_signals.append((1.5, f"📊 ETF: *${abs(snapshot.etf_flow_daily):.0f}M OUTFLOW* ⚠️"))
            
            # SONUÇ HESAPLA
            bullish_score = sum(s[0] for s in bullish_signals)
            bearish_score = sum(s[0] for s in bearish_signals)
            total_score = bullish_score + bearish_score
            
            # TAHMİN ve GİRİŞ ÖNERİSİ
            if total_score < 2:
                prediction = "BEKLE"
                confidence = 40
                entry_advice = "⏳ *GİRİŞ YOK* - Yeterli sinyal yok, bekle."
                entry_emoji = "⏸️"
            elif bullish_score > bearish_score * 2:
                prediction = "GÜÇLÜ YUKARI"
                confidence = min(90, 60 + bullish_score * 8)
                entry_advice = f"✅ *LONG GİRİŞ UYGUNı*\n   📍 Entry: ${current_price:,.0f}\n   🎯 TP1: ${current_price*1.02:,.0f} (+2%)\n   🛡️ SL: ${current_price*0.985:,.0f} (-1.5%)"
                entry_emoji = "🟢"
            elif bullish_score > bearish_score:
                prediction = "YUKARI"
                confidence = min(75, 55 + (bullish_score - bearish_score) * 10)
                entry_advice = f"🔵 *LONG düşünülebilir* (Dikkatli)\n   📍 Entry bekle: ${current_price*0.995:,.0f} civarı"
                entry_emoji = "📈"
            elif bearish_score > bullish_score * 2:
                prediction = "GÜÇLÜ AŞAĞI"
                confidence = min(90, 60 + bearish_score * 8)
                entry_advice = f"✅ *SHORT GİRİŞ UYGUN*\n   📍 Entry: ${current_price:,.0f}\n   🎯 TP1: ${current_price*0.98:,.0f} (-2%)\n   🛡️ SL: ${current_price*1.015:,.0f} (+1.5%)"
                entry_emoji = "🔴"
            elif bearish_score > bullish_score:
                prediction = "AŞAĞI"
                confidence = min(75, 55 + (bearish_score - bullish_score) * 10)
                entry_advice = f"🔵 *SHORT düşünülebilir* (Dikkatli)\n   📍 Entry bekle: ${current_price*1.005:,.0f} civarı"
                entry_emoji = "📉"
            else:
                prediction = "NÖTR"
                confidence = 50
                entry_advice = "⏳ *GİRİŞ YOK* - Kararsız piyasa, bekle."
                entry_emoji = "↔️"
            
            # GÜÇLÜ SİNYALLER (score > 1.0)
            strong_bullish = [s[1] for s in bullish_signals if s[0] >= 1.0]
            strong_bearish = [s[1] for s in bearish_signals if s[0] >= 1.0]
            
            # Mesaj oluştur
            strong_signals_text = ""
            if strong_bullish:
                strong_signals_text += "🟢 *GÜÇLÜ YUKARI SİNYALLER:*\n"
                for sig in strong_bullish[:3]:
                    strong_signals_text += f"  • {sig}\n"
            if strong_bearish:
                strong_signals_text += "🔴 *GÜÇLÜ AŞAĞI SİNYALLER:*\n"
                for sig in strong_bearish[:3]:
                    strong_signals_text += f"  • {sig}\n"
            if not strong_signals_text:
                strong_signals_text = "• Henüz güçlü sinyal yok\n"
            
            msg = f"""🏦 *{symbol} - CANLI ANALİZ*
━━━━━━━━━━━━━━━━━━
{entry_emoji} *Tahmin: {prediction}*
🧠 Güven: *%{confidence:.0f}*
💰 Fiyat: ${current_price:,.2f}

━━━ GİRİŞ ÖNERİSİ ━━━
{entry_advice}

━━━ GÜÇLÜ SİNYALLER ━━━
{strong_signals_text}
━━━ VERİ ÖZETİ ━━━
🟢 Bullish: {len(bullish_signals)} sinyal (güç: {bullish_score:.1f})
🔴 Bearish: {len(bearish_signals)} sinyal (güç: {bearish_score:.1f})
📊 17 kaynaktan {len(bullish_signals) + len(bearish_signals)} aktif
━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}"""
            
            await self.send_message_raw(msg)
            logger.info(f"🏦 Live Prediction: {symbol} → {prediction} ({confidence:.0f}%)")
            
        except Exception as e:
            logger.error(f"Live prediction error: {e}")

    # =========================================
    # 3️⃣ ANİ HAREKET UYARISI (13 TETİKLEYİCİ)
    # =========================================
    
    async def send_sudden_alert(self, symbol: str = "BTCUSDT"):
        """
        13 tetikleyiciyi kontrol edip uyarı gönder.
        InstitutionalAggregator'dan veri çeker.
        """
        try:
            from src.brain.institutional_aggregator import get_aggregator
            agg = get_aggregator()
            alert = await agg.check_sudden_triggers(symbol)
            
            if not alert.should_alert:
                return  # Uyarı gerekmiyor
            
            # Tetikleyici listesi
            trigger_lines = ""
            for t in alert.triggers[:5]:  # Max 5 tetikleyici
                severity_emoji = {
                    "CRITICAL": "🔴",
                    "HIGH": "🟠",
                    "MEDIUM": "🟡",
                    "LOW": "⚪"
                }.get(t.severity, "⚪")
                
                direction_emoji = "📈" if t.direction == "BULLISH" else "📉" if t.direction == "BEARISH" else "↔️"
                
                trigger_lines += f"{severity_emoji} *{t.name}*: {t.value}\n   {direction_emoji} _{t.message}_\n\n"
            
            # Yön emojisi
            dir_emoji = "📈 YUKARI" if alert.dominant_direction == "BULLISH" else "📉 AŞAĞI" if alert.dominant_direction == "BEARISH" else "↔️ NÖTR"
            
            # Severity emojisi
            sev_emoji = {
                "CRITICAL": "🚨 KRİTİK",
                "HIGH": "⚠️ YÜKSEK",
                "MEDIUM": "🟡 ORTA",
                "LOW": "⚪ DÜŞÜK"
            }.get(alert.overall_severity, "⚪")
            
            msg = f"""⚡ *ANİ HAREKET UYARISI!*
━━━━━━━━━━━━━━━━━━
📍 Coin: *{symbol}*
{dir_emoji}
{sev_emoji} RİSK

━━━ AKTİF TETİKLEYİCİLER ({alert.active_trigger_count}/13) ━━━
{trigger_lines}━━━━━━━━━━━━━━━━━━
⏱️ Hemen dikkat edin!
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}"""
            
            await self.send_message_raw(msg)
            logger.info(f"⚡ Sudden Alert: {symbol} ({alert.active_trigger_count} triggers)")
            
        except Exception as e:
            logger.error(f"Sudden alert error: {e}")

    # =========================================
    # 4️⃣ AI GÖZ ANALİZİ
    # =========================================
    
    async def send_ai_vision(
        self,
        symbol: str,
        overview: str,
        bull_scenario: Dict,
        bear_scenario: Dict,
        recommendation: str,
        recommendation_reason: str
    ):
        """Düşünen yapay zeka perspektifi."""
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
        logger.info(f"🧠 AI Vision: {symbol} → {recommendation}")

    # =========================================
    # YARDIMCI: Model Update
    # =========================================
    
    async def send_model_update(self, model_type: str, symbol: str, accuracy: float, loss: float, samples: int):
        """Model eğitim tamamlandı bildirimi."""
        msg = f"""🧠 *MODEL GÜNCELLENDİ*
━━━━━━━━━━━━━━━━━━
📊 Model: {model_type}
📍 Coin: {symbol}
✅ Accuracy: {accuracy:.1%}
📉 Loss: {loss:.4f}
📚 Samples: {samples:,}
━━━━━━━━━━━━━━━━━━
⏰ Sonraki eğitim: 24 saat"""
        await self.send_message_raw(msg)
        logger.info(f"🧠 Model Update: {model_type} {symbol}")

    # =========================================
    # LEGACY COMPATIBILITY STUBS
    # Eski engine.py çağrıları için boş metodlar
    # =========================================
    
    async def check_telegram_commands(self, money_flow_analyzer=None):
        """Legacy stub - Telegram komutları devre dışı."""
        pass
    
    async def check_and_update_signals(self):
        """Legacy stub - Signal gate kontrolü devre dışı."""
        pass
    
    async def check_active_position_risks(self):
        """Legacy stub - Pozisyon risk kontrolü devre dışı."""
        pass
    
    async def send_signal(self, signal: dict, snapshot: dict = None):
        """Legacy stub - Eski sinyal gönderme devre dışı."""
        pass
    
    async def send_heartbeat(self, price_info: dict):
        """Legacy stub - Heartbeat devre dışı."""
        pass