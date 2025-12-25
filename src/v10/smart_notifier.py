# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - SMART NOTIFIER
==============================
Telegram'a net, anlaşılır trading sinyalleri gönderir.

FORMAT:
- Entry aralığı
- TP1, TP2, TP3
- SL
- R/R oranı
- Güven skoru
- Reasoning (neden bu sinyal?)
- Risk seviyesi
"""
import logging
import requests
from datetime import datetime
from typing import Optional

from src.config.settings import Config

logger = logging.getLogger("SMART_NOTIFIER")


class SmartNotifier:
    """
    DEMIR AI v10 - Akıllı Telegram Bildirim Sistemi
    
    Net, anlaşılır formatla trading sinyalleri gönderir.
    """
    
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self._error_count = 0
        logger.info("📱 Smart Notifier initialized")
    
    def _send_message(self, text: str) -> bool:
        """
        Telegram'a mesaj gönder.
        HATA YUTULMAZ - başarısız olursa loglar.
        """
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"❌ Telegram API error: {response.status_code} - {response.text}")
                self._error_count += 1
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            self._error_count += 1
            return False
    
    def send_trading_signal(self, signal) -> bool:
        """
        Trading sinyalini formatla ve gönder.
        
        Args:
            signal: TradingSignal from PredictorEngine
        """
        if not signal.is_valid:
            logger.debug(f"Signal not valid for {signal.symbol}: {signal.warnings}")
            return False
        
        # Emoji seçimi
        direction_emoji = "🟢" if signal.signal_type.value == "LONG" else "🔴"
        risk_emoji = {
            "LOW": "🟢",
            "MEDIUM": "🟡",
            "HIGH": "🔴"
        }.get(signal.risk_level.value, "⚪")
        
        # Reasons formatla
        reasons_text = "\n".join([f"✅ {r}" for r in signal.reasons[:6]])
        
        # Warnings
        warnings_text = ""
        if signal.warnings:
            warnings_text = "\n━━━ *UYARILAR* ━━━\n" + "\n".join([f"⚠️ {w}" for w in signal.warnings[:3]])
        
        # Ana mesaj
        message = f"""🎯 *{signal.signal_type.value} SİNYALİ - {signal.symbol}*
━━━━━━━━━━━━━━━━━━━━━━

{direction_emoji} *YÖN: {signal.signal_type.value}*
📍 *ENTRY:* ${signal.entry_low:,.0f} - ${signal.entry_high:,.0f}

🎯 *TP1:* ${signal.tp1:,.0f}
🎯 *TP2:* ${signal.tp2:,.0f}
🎯 *TP3:* ${signal.tp3:,.0f}
🛑 *SL:* ${signal.sl:,.0f}

⚖️ *R/R:* 1:{signal.risk_reward:.1f}
🧠 *Güven:* %{signal.confidence:.0f}
{risk_emoji} *Risk:* {signal.risk_level.value}
💰 *Potansiyel:* ${signal.potential_usd:,.0f}

━━━ *NEDEN BU SİNYAL?* ━━━
{reasons_text}
{warnings_text}

━━━ *VERİ KALİTESİ* ━━━
📡 Kaynak: {signal.data_sources_ok}/{signal.data_sources_total} OK

━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
📡 *DEMIR AI v10 - LIVE DATA*"""
        
        success = self._send_message(message)
        
        if success:
            logger.info(f"✅ Signal sent: {signal.symbol} {signal.signal_type.value} %{signal.confidence:.0f}")
        
        return success
    
    def send_market_summary(self, snapshots: dict) -> bool:
        """
        Tüm coinlerin özet durumunu gönder.
        """
        lines = ["📊 *PİYASA ÖZETİ*\n━━━━━━━━━━━━━━━━━━"]
        
        for symbol, snapshot in snapshots.items():
            if not snapshot.is_valid:
                lines.append(f"❌ {symbol}: VERİ HATASI")
                continue
            
            # Trend emoji
            trend_emoji = {
                "BULLISH": "📈",
                "BEARISH": "📉",
                "NEUTRAL": "➡️"
            }.get(snapshot.trend, "❓")
            
            # Price change emoji
            change = snapshot.price_change_24h
            change_emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            
            lines.append(
                f"\n{trend_emoji} *{symbol}*\n"
                f"💰 ${snapshot.price:,.0f} ({change_emoji}{change:+.1f}%)\n"
                f"📊 RSI: {snapshot.rsi_1h:.0f} | OB: {snapshot.bid_ask_ratio:.2f}x\n"
                f"💰 FR: {snapshot.funding_rate:.3f}% | 🐋: {snapshot.whale_net_flow:+.0f}"
            )
        
        lines.append(f"\n━━━━━━━━━━━━━━━━━━\n⏰ {datetime.now().strftime('%H:%M:%S')}")
        
        return self._send_message("\n".join(lines))
    
    def send_technical_analysis(self, snapshots: dict) -> bool:
        """
        Detaylı teknik analiz bildirimi gönder.
        Her coin için: RSI, EMA, Support/Resistance, Funding, Whale, Öneri
        """
        lines = ["📈 *TEKNİK ANALİZ RAPORU*", "━━━━━━━━━━━━━━━━━━━━━━━"]
        
        for symbol, snapshot in snapshots.items():
            if not snapshot.is_valid:
                lines.append(f"\n❌ *{symbol}*: Veri alınamadı")
                continue
            
            # === COIN HEADER ===
            trend_emoji = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}.get(snapshot.trend, "❓")
            change = snapshot.price_change_24h
            change_emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            
            lines.append(f"\n{trend_emoji} *{symbol}*")
            lines.append(f"💰 Fiyat: ${snapshot.price:,.0f} ({change_emoji}{change:+.2f}%)")
            
            # === RSI ANALİZİ ===
            rsi = snapshot.rsi_1h
            rsi_4h = snapshot.rsi_4h
            
            # -1 = veri yok (şeffaf göster)
            if rsi < 0:
                rsi_status = "❌ VERİ YOK"
                lines.append(f"📊 RSI: VERİ YOK")
            else:
                if rsi < 30:
                    rsi_status = "🔥 AŞIRI SATIM - Alım fırsatı olabilir"
                elif rsi < 40:
                    rsi_status = "🟢 Düşük - Potansiyel dip"
                elif rsi > 70:
                    rsi_status = "⚠️ AŞIRI ALIM - Satış baskısı olabilir"
                elif rsi > 60:
                    rsi_status = "🟡 Yüksek - Tepe yakınlaşıyor"
                else:
                    rsi_status = "➡️ Normal aralık"
                
                rsi_4h_str = f"{rsi_4h:.0f}" if rsi_4h >= 0 else "N/A"
                lines.append(f"📊 RSI: 1s:{rsi:.0f} | 4s:{rsi_4h_str} → {rsi_status}")
            
            # === EMA TREND ===
            if snapshot.ema_20 > 0 and snapshot.ema_50 > 0:
                if snapshot.price > snapshot.ema_20 > snapshot.ema_50:
                    ema_status = "🟢 BULLISH - EMA20 > EMA50, fiyat üstünde"
                elif snapshot.price < snapshot.ema_20 < snapshot.ema_50:
                    ema_status = "🔴 BEARISH - EMA20 < EMA50, fiyat altında"
                else:
                    ema_status = "🟡 KARARSIZ - Cross bekleniyor"
                lines.append(f"📉 Trend: {ema_status}")
            
            # === ORDER BOOK ===
            ratio = snapshot.bid_ask_ratio
            if ratio < 0:  # -1 = veri yok
                lines.append(f"📗 Order Book: ❌ VERİ YOK")
            elif ratio > 1.5:
                ob_status = f"🟢 Alıcı dominantı ({ratio:.2f}x BID)"
                lines.append(f"📗 Order Book: {ob_status}")
            elif ratio < 0.67:
                ob_status = f"🔴 Satıcı dominantı ({1/ratio:.2f}x ASK)"
                lines.append(f"📗 Order Book: {ob_status}")
            else:
                ob_status = f"⚪ Dengeli ({ratio:.2f}x)"
                lines.append(f"📗 Order Book: {ob_status}")
            
            # === SUPPORT / RESISTANCE (Kline-based pivots) ===
            if snapshot.support > 0:
                distance_pct = ((snapshot.price - snapshot.support) / snapshot.price) * 100
                lines.append(f"🟢 Destek: ${snapshot.support:,.0f} ({distance_pct:.1f}% uzakta)")
            if snapshot.resistance > 0:
                distance_pct = ((snapshot.resistance - snapshot.price) / snapshot.price) * 100
                lines.append(f"🔴 Direnç: ${snapshot.resistance:,.0f} ({distance_pct:.1f}% uzakta)")
            
            # Majör seviyeler (7 günlük)
            if snapshot.major_support > 0 and snapshot.major_support != snapshot.support:
                lines.append(f"🟩 Majör Destek: ${snapshot.major_support:,.0f} (7g)")
            if snapshot.major_resistance > 0 and snapshot.major_resistance != snapshot.resistance:
                lines.append(f"🟥 Majör Direnç: ${snapshot.major_resistance:,.0f} (7g)")
            
            # === FUNDING RATE ===
            fr = snapshot.funding_rate
            if fr <= -900:  # -999 = veri yok
                lines.append(f"💰 Funding: ❌ VERİ YOK")
            elif fr < -0.02:
                fr_status = "🚀 Negatif (Short squeeze potansiyeli)"
                lines.append(f"💰 Funding: {fr:.4f}% → {fr_status}")
            elif fr > 0.05:
                fr_status = "⚠️ Yüksek (Long squeeze riski)"
                lines.append(f"💰 Funding: {fr:.4f}% → {fr_status}")
            else:
                lines.append(f"💰 Funding: {fr:.4f}% → Normal")
            
            # === WHALE AKTİVİTESİ ===
            whale = snapshot.whale_net_flow
            large_buys = snapshot.large_buys
            large_sells = snapshot.large_sells
            
            if whale <= -900 or large_buys < 0:  # -999/-1 = veri yok
                lines.append(f"🐋 Whale: ❌ VERİ YOK")
            elif whale > 2:
                whale_status = f"🐋 Net ALIM (+{whale:.0f})"
                lines.append(f"Whale: {whale_status} | Büyük: {large_buys}↑ {large_sells}↓")
            elif whale < -2:
                whale_status = f"🐋 Net SATIM ({whale:.0f})"
                lines.append(f"Whale: {whale_status} | Büyük: {large_buys}↑ {large_sells}↓")
            else:
                whale_status = f"🐋 Dengeli ({whale:+.0f})"
                lines.append(f"Whale: {whale_status} | Büyük: {large_buys}↑ {large_sells}↓")
            
            # === ÖNERİ ===
            recommendation = self._generate_recommendation(snapshot)
            lines.append(f"📌 *Öneri:* {recommendation}")
            
            lines.append("─" * 25)
        
        # Footer
        lines.append(f"\n⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        lines.append("📡 *DEMIR AI v10 - LIVE DATA*")
        
        return self._send_message("\n".join(lines))
    
    def _generate_recommendation(self, snapshot) -> str:
        """Snapshot'tan öneri üret - VERİ YOKSA SKORU ETKİLEMEZ"""
        score = 0
        data_points = 0  # Kaç veri noktası kullandık
        
        # RSI (sadece geçerli ise)
        if snapshot.rsi_1h >= 0:
            data_points += 1
            if snapshot.rsi_1h < 30:
                score += 2
            elif snapshot.rsi_1h < 40:
                score += 1
            elif snapshot.rsi_1h > 70:
                score -= 2
            elif snapshot.rsi_1h > 60:
                score -= 1
        
        # Trend (sadece UNKNOWN değilse)
        if snapshot.trend not in ["UNKNOWN", ""]:
            data_points += 1
            if snapshot.trend == "BULLISH":
                score += 1
            elif snapshot.trend == "BEARISH":
                score -= 1
        
        # Order book (sadece geçerli ise)
        if snapshot.bid_ask_ratio >= 0:
            score += 1
        elif snapshot.bid_ask_ratio < 0.67:
            score -= 1
        
        # Funding (contrarian)
        if snapshot.funding_rate < -0.01:
            score += 1
        elif snapshot.funding_rate > 0.03:
            score -= 1
        
        # Whale
        if snapshot.whale_net_flow > 2:
            score += 1
        elif snapshot.whale_net_flow < -2:
            score -= 1
        
        # Karar
        if score >= 3:
            return "🚀 GÜÇLÜ LONG bölgesi"
        elif score >= 1:
            return "🟢 LONG eğilimli"
        elif score <= -3:
            return "💀 GÜÇLÜ SHORT bölgesi"
        elif score <= -1:
            return "🔴 SHORT eğilimli"
        else:
            return "⏸️ BEKLE - Net sinyal yok"
    
    def send_error_alert(self, error_message: str) -> bool:
        """
        Hata bildirimi gönder.
        HATA YUTULMAZ - kullanıcıya bildirilir.
        """
        message = f"""⚠️ *DEMIR AI - HATA*
━━━━━━━━━━━━━━━━━━

❌ {error_message}

━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
📡 *Lütfen kontrol edin!*"""
        
        return self._send_message(message)
    
    def send_startup_message(self) -> bool:
        """
        Bot başlatıldı mesajı.
        """
        message = """🚀 *DEMIR AI v10 - BAŞLATILDI*
━━━━━━━━━━━━━━━━━━━━━━

✅ Data Hub: AKTIF
✅ Predictor: AKTIF
✅ Notifier: AKTIF

📊 *Takip:* BTC, ETH, SOL, LTC
🎯 *Mod:* Prediktif Sinyal

━━━━━━━━━━━━━━━━━━━━━━
📡 *SADECE GERÇEK VERİ - MOCK YOK*"""
        
        return self._send_message(message)
    
    def send_data_quality_alert(self, symbol: str, errors: list) -> bool:
        """
        Veri kalitesi uyarısı gönder.
        """
        error_list = "\n".join([f"❌ {e}" for e in errors[:5]])
        
        message = f"""⚠️ *VERİ HATASI - {symbol}*
━━━━━━━━━━━━━━━━━━

{error_list}

━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}
📡 *Veri kaynakları kontrol ediliyor...*"""
        
        return self._send_message(message)


# Singleton
_notifier: Optional[SmartNotifier] = None

def get_notifier() -> SmartNotifier:
    global _notifier
    if _notifier is None:
        _notifier = SmartNotifier()
    return _notifier
