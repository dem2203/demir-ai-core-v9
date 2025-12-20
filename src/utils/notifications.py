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
        ✅ Data validation ile mock/fallback tespit
        """
        try:
            from src.brain.institutional_aggregator import get_aggregator
            from src.brain.data_validator import get_data_validator
            
            agg = get_aggregator()
            snapshot = await agg.get_live_snapshot(symbol)
            
            # ═══ DATA VALIDATION CHECK ═══
            validator = get_data_validator()
            validation = await validator.validate_live_snapshot(snapshot, symbol)
            
            if not validation.is_usable:
                logger.warning(f"⚠️ {symbol} verisi reddedildi: {validation.rejection_reason}")
                # Notify about data quality issue (optional)
                await self.send_message_raw(f"⚠️ *VERİ UYARISI - {symbol}*\n{validation.rejection_reason}\n_Bildirim gönderilmedi_")
                return
            
            # Log validation quality
            if validation.overall_quality.value != "VERIFIED":
                logger.info(f"📊 {symbol} veri kalitesi: {validation.overall_quality.value} ({validation.verification_rate:.0f}%)")
            
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
            
            # ═══════════════════════════════════════════════════════════════
            # 11-15. MAKRO & MİKRO VERİLER (DXY, VIX, BTC.D, MVRV, NUPL)
            # ═══════════════════════════════════════════════════════════════
            try:
                from src.brain.macro_micro_data import get_macro_micro_fetcher
                mm_fetcher = get_macro_micro_fetcher()
                combined = await mm_fetcher.get_combined_analysis(symbol)
                
                macro = combined.macro
                micro = combined.micro
                
                # 11. DXY (Dollar Index) - Ters korelasyon
                if macro.dxy > 0:
                    if macro.dxy_trend == "DOWN":
                        bullish_signals.append((1.5, f"💵 DXY: {macro.dxy:.1f} *DÜŞÜYOR* ({macro.dxy_change_24h:+.2f}%) - Crypto için BULLISH"))
                    elif macro.dxy_trend == "UP":
                        bearish_signals.append((1.5, f"💵 DXY: {macro.dxy:.1f} *YÜKSELİYOR* ({macro.dxy_change_24h:+.2f}%) - Crypto için BEARISH"))
                
                # 12. VIX (Volatility Index) - Risk Sentiment
                if macro.vix > 0:
                    if macro.vix_level == "EXTREME":
                        bearish_signals.append((2.0, f"📈 VIX: *{macro.vix:.1f}* _(Extreme Fear - Risk-Off)_ ⚠️"))
                    elif macro.vix_level == "HIGH":
                        bearish_signals.append((1.0, f"📈 VIX: {macro.vix:.1f} (Yüksek volatilite)"))
                    elif macro.vix_level == "LOW":
                        bullish_signals.append((1.0, f"📈 VIX: {macro.vix:.1f} (Düşük - Risk-On ortamı)"))
                
                # 13. Stock Market Correlation
                if macro.spx500_change_24h > 1:
                    bullish_signals.append((0.5, f"📊 S&P500: +{macro.spx500_change_24h:.1f}% (Risk-On)"))
                elif macro.spx500_change_24h < -1:
                    bearish_signals.append((0.5, f"📊 S&P500: {macro.spx500_change_24h:.1f}% (Risk-Off)"))
                
                # 14. BTC Dominance
                if micro.btc_dominance > 0:
                    if micro.btc_dominance > 55:
                        neutral_signals.append(f"₿ BTC.D: {micro.btc_dominance:.1f}% (Altcoin sezonu değil)")
                    elif micro.btc_dominance < 45:
                        bullish_signals.append((0.5, f"₿ BTC.D: {micro.btc_dominance:.1f}% (Altseason potansiyeli)"))
                
                # 15. On-Chain Metrics (MVRV + NUPL)
                if micro.mvrv_zscore != 0:
                    if micro.mvrv_zscore < 0:
                        bullish_signals.append((1.5, f"🔗 MVRV: *{micro.mvrv_zscore:.1f}* _(Undervalued - ALIM FIRSATI)_"))
                    elif micro.mvrv_zscore > 5:
                        bearish_signals.append((1.5, f"🔗 MVRV: *{micro.mvrv_zscore:.1f}* _(Overvalued - DİKKAT)_ ⚠️"))
                    
                if micro.nupl != 0:
                    if micro.nupl < 0:
                        bullish_signals.append((1.0, f"📊 NUPL: {micro.nupl:.2f} (Capitulation zone - Dip olabilir)"))
                    elif micro.nupl > 0.7:
                        bearish_signals.append((1.0, f"📊 NUPL: {micro.nupl:.2f} (Euphoria zone - Tepe yakın)"))
                
                # Stablecoin Dominance
                if micro.stablecoin_total_dominance > 8:
                    bearish_signals.append((0.5, f"💵 Stablecoin D: {micro.stablecoin_total_dominance:.1f}% (Yüksek - Bekleme modu)"))
                    
            except Exception as mm_err:
                logger.debug(f"Macro/Micro data error: {mm_err}")
            
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
    # TELEGRAM KOMUT HANDLER (AKTİF)
    # Kullanıcı coin yazınca detaylı analiz gönderir
    # =========================================
    
    async def check_telegram_commands(self, money_flow_analyzer=None):
        """
        Telegram'dan gelen mesajları kontrol et.
        BTC, ETH, SOL, LTC yazılırsa detaylı analiz gönder.
        """
        try:
            # Get updates from Telegram
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates?offset=-10&timeout=1"
            response = requests.get(url, timeout=5)
            
            if response.status_code != 200:
                return
            
            data = response.json()
            messages = data.get('result', [])
            
            if not hasattr(self, '_last_processed_update_id'):
                self._last_processed_update_id = 0
            
            for update in messages:
                update_id = update.get('update_id', 0)
                
                # Skip already processed
                if update_id <= self._last_processed_update_id:
                    continue
                
                self._last_processed_update_id = update_id
                
                message = update.get('message', {})
                text = message.get('text', '').upper().strip()
                chat_id = message.get('chat', {}).get('id')
                
                # Only process from our chat
                if str(chat_id) != str(self.telegram_chat_id):
                    continue
                
                # Check for coin analysis command
                await self._handle_coin_command(text)
                
        except Exception as e:
            logger.debug(f"Telegram command check error: {e}")
    
    async def _handle_coin_command(self, text: str):
        """Coin analiz komutlarını işle"""
        try:
            from src.brain.interactive_analyzer import get_interactive_analyzer
            
            analyzer = get_interactive_analyzer()
            
            # Parse coin from message
            symbol = analyzer.parse_coin_from_message(text)
            
            if symbol:
                logger.info(f"🔍 Interactive analysis requested: {symbol}")
                
                # Send "analyzing" message
                await self.send_message_raw(f"🔍 *{symbol} analiz ediliyor...*\n_Lütfen bekleyin (5-10 saniye)_")
                
                # Perform detailed analysis
                analysis = await analyzer.analyze_coin(symbol)
                
                # Send formatted result
                telegram_msg = analyzer.format_for_telegram(analysis)
                await self.send_message_raw(telegram_msg)
                
                logger.info(f"✅ Interactive analysis sent: {symbol} → {analysis.overall_direction}")
            
            # Check for help command
            elif text in ['/HELP', '/YARDIM', 'HELP', 'YARDIM', '/KOMUTLAR']:
                help_text = """🤖 *DEMIR AI - KOMUTLAR*
━━━━━━━━━━━━━━━━━━
📊 *Coin Analizi:*
Sadece coin ismini yaz:
• `BTC` - Bitcoin analizi
• `ETH` - Ethereum analizi
• `SOL` - Solana analizi
• `LTC` - Litecoin analizi

Veya komut ile:
• `/analiz BTC`
• `/analiz ETH`

━━━━━━━━━━━━━━━━━━
📡 *Otomatik Bildirimler:*
• Teknik Sinyal - Anlık (%70+)
• Canlı Tahmin - 15 dk
• Ani Hareket - 60 sn
• AI Analizi - 1 saat

━━━━━━━━━━━━━━━━━━
🔍 Detaylı analiz için coin yaz!"""
                await self.send_message_raw(help_text)
                
        except Exception as e:
            logger.error(f"Coin command handler error: {e}")
    
    # =========================================
    # LEGACY COMPATIBILITY STUBS
    # =========================================
    
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