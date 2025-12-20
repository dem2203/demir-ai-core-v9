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
        17 kaynaktan toplanan verilerle tahmin gönder.
        InstitutionalAggregator'dan veri çeker.
        """
        try:
            from src.brain.institutional_aggregator import get_aggregator
            agg = get_aggregator()
            snapshot = await agg.get_live_snapshot(symbol)
            
            # Yön analizi
            bullish_score = 0
            bearish_score = 0
            sources = []
            
            # 1. Whale
            if snapshot.whale_net_flow > 500000:
                bullish_score += 1
                sources.append(f"🐋 Whale: +${snapshot.whale_net_flow/1e6:.1f}M net alım")
            elif snapshot.whale_net_flow < -500000:
                bearish_score += 1
                sources.append(f"🐋 Whale: ${abs(snapshot.whale_net_flow)/1e6:.1f}M net satış")
            
            # 2. Order Book
            if snapshot.orderbook_imbalance > 1.3:
                bullish_score += 1
                sources.append(f"📊 Order Book: {snapshot.orderbook_imbalance:.2f}x BID ağırlıklı")
            elif snapshot.orderbook_imbalance < 0.7:
                bearish_score += 1
                sources.append(f"📊 Order Book: {1/snapshot.orderbook_imbalance:.2f}x ASK ağırlıklı")
            
            # 3. Liquidation
            if snapshot.liq_long_total > snapshot.liq_short_total * 1.5:
                bearish_score += 1
                sources.append(f"💧 Liq: Long ağırlıklı (${snapshot.liq_long_total/1e6:.0f}M)")
            elif snapshot.liq_short_total > snapshot.liq_long_total * 1.5:
                bullish_score += 1
                sources.append(f"💧 Liq: Short ağırlıklı (${snapshot.liq_short_total/1e6:.0f}M)")
            
            # 4. Funding
            if snapshot.funding_rate > 0.03:
                bearish_score += 1
                sources.append(f"💸 Funding: +{snapshot.funding_rate:.3f}% (Long squeeze riski)")
            elif snapshot.funding_rate < -0.02:
                bullish_score += 1
                sources.append(f"💸 Funding: {snapshot.funding_rate:.3f}% (Short squeeze riski)")
            
            # 5. OI Change
            if snapshot.oi_change_1h > 5:
                sources.append(f"📈 OI: +{snapshot.oi_change_1h:.1f}% (1h)")
            elif snapshot.oi_change_1h < -5:
                sources.append(f"📉 OI: {snapshot.oi_change_1h:.1f}% (1h)")
            
            # 6. Long/Short Ratio
            if snapshot.long_short_ratio > 1.5:
                bearish_score += 1
                sources.append(f"📊 L/S Ratio: {snapshot.long_short_ratio:.2f} (Kalabalık long)")
            elif snapshot.long_short_ratio < 0.7:
                bullish_score += 1
                sources.append(f"📊 L/S Ratio: {snapshot.long_short_ratio:.2f} (Kalabalık short)")
            
            # 7. CVD
            if snapshot.cvd_trend == "BULLISH":
                bullish_score += 1
                sources.append(f"📊 CVD: BULLISH ({snapshot.cvd_value:,.0f})")
            elif snapshot.cvd_trend == "BEARISH":
                bearish_score += 1
                sources.append(f"📊 CVD: BEARISH ({snapshot.cvd_value:,.0f})")
            
            # 8. Exchange Flow
            if snapshot.exchange_netflow < -100:
                bullish_score += 1
                sources.append(f"🏦 Exchange: Outflow (${abs(snapshot.exchange_netflow):.0f}M)")
            elif snapshot.exchange_netflow > 100:
                bearish_score += 1
                sources.append(f"🏦 Exchange: Inflow (${snapshot.exchange_netflow:.0f}M)")
            
            # 9-10. Stablecoin (eğer veri varsa)
            if snapshot.usdt_supply_change > 0:
                sources.append(f"💵 USDT: +${snapshot.usdt_supply_change/1e6:.0f}M basıldı")
            
            # 11. DeFi TVL
            if snapshot.defi_tvl_change_24h > 5:
                bullish_score += 0.5
                sources.append(f"🔗 DeFi TVL: +{snapshot.defi_tvl_change_24h:.1f}%")
            
            # 12. Options
            if snapshot.put_call_ratio > 1.3:
                bearish_score += 0.5
                sources.append(f"📈 Put/Call: {snapshot.put_call_ratio:.2f} (Hedge ağırlıklı)")
            elif snapshot.put_call_ratio < 0.7:
                bullish_score += 0.5
                sources.append(f"📈 Put/Call: {snapshot.put_call_ratio:.2f} (Call ağırlıklı)")
            
            # 13. CME Gap
            if not snapshot.cme_gap_filled and snapshot.cme_gap_price > 0:
                sources.append(f"📊 CME Gap: ${snapshot.cme_gap_price:,.0f} ({snapshot.cme_gap_direction})")
            
            # 14. Cross-Exchange
            if abs(snapshot.coinbase_premium) > 0.2:
                direction = "premium" if snapshot.coinbase_premium > 0 else "discount"
                sources.append(f"🏦 Coinbase: {abs(snapshot.coinbase_premium):.2f}% {direction}")
            
            # 15. ETF
            if abs(snapshot.etf_flow_daily) > 100:
                flow_type = "inflow" if snapshot.etf_flow_daily > 0 else "outflow"
                sources.append(f"📊 ETF: ${abs(snapshot.etf_flow_daily):.0f}M {flow_type}")
            
            # 16. Fear & Greed
            if snapshot.fear_greed_index < 25:
                bullish_score += 0.5
                sources.append(f"😱 Fear&Greed: {snapshot.fear_greed_index} (Extreme Fear)")
            elif snapshot.fear_greed_index > 75:
                bearish_score += 0.5
                sources.append(f"🤑 Fear&Greed: {snapshot.fear_greed_index} (Extreme Greed)")
            
            # 17. Taker
            if snapshot.taker_buy_ratio > 0.6:
                bullish_score += 1
                sources.append(f"📊 Taker: {snapshot.taker_buy_ratio*100:.0f}% alıcı")
            elif snapshot.taker_buy_ratio < 0.4:
                bearish_score += 1
                sources.append(f"📊 Taker: {(1-snapshot.taker_buy_ratio)*100:.0f}% satıcı")
            
            # Sonuç hesapla
            total = bullish_score + bearish_score
            if total == 0:
                prediction = "NÖTR"
                confidence = 50
            elif bullish_score > bearish_score:
                prediction = "YUKARI"
                confidence = min(85, 50 + (bullish_score - bearish_score) * 10)
            else:
                prediction = "AŞAĞI"
                confidence = min(85, 50 + (bearish_score - bullish_score) * 10)
            
            pred_emoji = "📈" if prediction == "YUKARI" else "📉" if prediction == "AŞAĞI" else "↔️"
            
            # Kaynak listesi
            source_text = "\n".join(sources[:10])  # Max 10 kaynak göster
            if not source_text:
                source_text = "• Veri toplanıyor..."
            
            msg = f"""🏦 *CANLI VERİ TAHMİNİ - {symbol}*
━━━━━━━━━━━━━━━━━━
{pred_emoji} *Tahmin: {prediction}*
🧠 AI Güven: *{confidence:.0f}%*

━━━ 17 VERİ KAYNAĞI ━━━
{source_text}

━━━ ÖZET ━━━
🟢 Bullish Sinyaller: {int(bullish_score)}
🔴 Bearish Sinyaller: {int(bearish_score)}
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