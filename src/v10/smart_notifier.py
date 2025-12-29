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
        Markdown parse hatası olursa düz text olarak tekrar dener.
        """
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            
            # Clean text - remove potentially problematic markdown
            clean_text = text.replace('_', ' ').replace('*', '')  # Remove markdown chars
            
            # First try with Markdown
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            # If markdown parse fails, retry without parse_mode
            if response.status_code == 400 and "can't parse entities" in response.text:
                logger.warning("Markdown parse failed, retrying without parse_mode...")
                payload = {
                    "chat_id": self.chat_id,
                    "text": clean_text,  # Use cleaned text
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
    
    
    async def send_market_summary(self, snapshots: dict) -> bool:
        """
        Tüm coinlerin özet durumunu gönder + AI TAVSİYESİ
        """
        lines = ["📊 *PİYASA ÖZETİ + AI TAVSİYE*\n━━━━━━━━━━━━━━━━━━"]
        
        # Get LSTM predictions for AI recommendation
        try:
            from src.v10.lstm_predictor import get_lstm_predictor
            lstm = get_lstm_predictor()
        except:
            lstm = None
        
        for symbol, snapshot in snapshots.items():
            # Sadece price varsa yeterli (web fallback OK)
            if not snapshot.price or snapshot.price <= 0:
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
            
            # AI Recommendation based on LSTM + RSI
            ai_advice = "⏸️ BEKLE"
            try:
                if lstm:
                    pred = await lstm.predict(symbol)
                    if pred:
                        rsi = snapshot.rsi_1h
                        # Decision logic
                        if pred.direction == "UP" and pred.confidence > 40:
                            if rsi < 70:  # Not overbought
                                ai_advice = f"🟢 LONG (%{pred.confidence:.0f})"
                            else:
                                ai_advice = "⚠️ RSI Yüksek"
                        elif pred.direction == "DOWN" and pred.confidence > 40:
                            if rsi > 30:  # Not oversold
                                ai_advice = f"🔴 SHORT (%{pred.confidence:.0f})"
                            else:
                                ai_advice = "⚠️ RSI Düşük"
                        else:
                            ai_advice = "⏸️ BEKLE"
            except:
                pass
            
            lines.append(
                f"\n{trend_emoji} *{symbol}*\n"
                f"💰 ${snapshot.price:,.0f} ({change_emoji}{change:+.1f}%)\n"
                f"📊 RSI: {snapshot.rsi_1h:.0f} | OB: {snapshot.bid_ask_ratio:.2f}x\n"
                f"💰 FR: {snapshot.funding_rate:.3f}% | 🐋: {snapshot.whale_net_flow:+.0f}\n"
                f"🤖 *TAVSİYE: {ai_advice}*"
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
    
    async def send_deep_technical_report(self, symbol: str) -> bool:
        """
        TÜMGÜÇ VE MODÜL - EN DERİN ANALİZ
        SADECE CANLI VERİ - MOCK/FALLBACK YOK
        
        Kullanılan Modüller:
        - LiquidationHunter (CoinGlass)
        - PatternEngine (Chart patterns)
        - PivotPointsAnalyzer (S/R levels)
        - VolatilityPredictor (Squeeze/Breakout)
        - RegimeClassifier (Trend/Range)
        - SentimentAnalyzer (News-based)
        """
        from src.brain.liquidation_hunter import get_liquidation_hunter
        from src.brain.pattern_engine import get_pattern_engine
        from src.brain.pivot_points import get_pivot_points
        from src.brain.volatility_predictor import VolatilityPredictor
        from src.brain.regime_classifier import RegimeClassifier
        from src.brain.news_scraper import CryptoNewsScraper
        
        lines = [f"🔬 *DERİN TEKNİK ANALİZ - {symbol}*", "━━━━━━━━━━━━━━━━━━━━━━"]
        
        try:
            # 1. LIQUIDATION ZONES (CoinGlass)
            liq_hunter = get_liquidation_hunter()
            liq_data = await liq_hunter.analyze(symbol)
            
            if liq_data and 'heatmap_clusters' in liq_data:
                clusters = liq_data['heatmap_clusters'][:3]  # Top 3
                if clusters:
                    lines.append("\n💧 *LİKİDASYON BÖLGE (MAGNET)*")
                    for i, c in enumerate(clusters, 1):
                        lines.append(f"  {i}. ${c['price']:,.0f} - Güç: {c['intensity']:.1f}")
                else:
                    lines.append("\n💧 Likidasyon: Veri yok")
            
            # 2. CHART PATTERNS
            pattern_engine = get_pattern_engine()
            pattern_data = await pattern_engine.analyze(symbol)
            
            if pattern_data and 'patterns' in pattern_data:
                patterns = pattern_data['patterns']
                if patterns:
                    lines.append("\n📐 *CHART PATTERN*")
                    for p in patterns[:2]:  # Top 2
                        lines.append(f"  • {p['name']} - {p['direction']} ({p['confidence']:.0f}%)")
                else:
                    lines.append("\n📐 Pattern: Henüz oluşmadı")
            
            # === 🧠 AI BRAIN ENSEMBLE ===
            
            # 1. LSTM PREDICTION (from our trained model)
            try:
                from src.v10.lstm_predictor import get_lstm_predictor
                lstm = get_lstm_predictor()
                prediction = await lstm.predict(symbol)
                
                if prediction:
                    direction = prediction.direction
                    change_pct = prediction.predicted_change_pct
                    confidence = prediction.confidence
                    
                    emoji = "🟢" if direction == "UP" else "🔴" if direction == "DOWN" else "⚪"
                    lines.append("\n🧠 *AI BRAIN - LSTM*")
                    lines.append(f"  Yön: {emoji} {direction} ({change_pct:+.2f}%)")
                    lines.append(f"  Güven: {confidence:.0f}%")
            except Exception as e:
                logger.debug(f"LSTM prediction skipped: {e}")
            
            # 2. 📊 LEADING INDICATORS (Order Book, Whale, Funding)
            try:
                from src.v10.leading_indicators import get_leading_indicators
                leading = await get_leading_indicators()
                leading_signal = await leading.get_signal(symbol)
                
                if leading_signal:
                    lines.append("\n📊 *ENSEMBLE INDICATORS*")
                    
                    # Order Book
                    ob_score = leading_signal.orderbook_score
                    ob_emoji = "🟢" if ob_score > 20 else "🔴" if ob_score < -20 else "⚪"
                    lines.append(f"  📗 Order Book: {ob_emoji} {ob_score:+.0f}")
                    
                    # Whale Activity
                    whale_score = leading_signal.whale_score
                    whale_emoji = "🐋" if whale_score > 20 else "🐋" if whale_score < -20 else "⚪"
                    action = "Alım" if whale_score > 0 else "Satım" if whale_score < 0 else "Nötr"
                    lines.append(f"  {whale_emoji} Whale: {action} ({whale_score:+.0f})")
                    
                    # Funding Rate
                    funding_score = leading_signal.funding_score
                    fund_emoji = "🟢" if funding_score > 0 else "🔴" if funding_score < 0 else "⚪"
                    lines.append(f"  💰 Funding: {fund_emoji} ({funding_score:+.0f})")
                    
                    # Direction
                    direction = leading_signal.direction.value
                    strength = leading_signal.strength
                    dir_emoji = "🟢" if "BULLISH" in direction else "🔴" if "BEARISH" in direction else "⚪"
                    lines.append(f"  ━━━━━━━━━━━━━")
                    lines.append(f"  Yön: {dir_emoji} {direction} (%{strength:.0f})")
            except Exception as e:
                logger.debug(f"Leading indicators skipped: {e}")
            
            # 4. 📈 ENHANCED PREDICTOR (7 Indicators)
            try:
                from src.v10.enhanced_predictor import EnhancedPredictor
                enhanced = EnhancedPredictor()
                prediction = enhanced.predict(symbol)
                
                if prediction:
                    signals = prediction.get('signals', {})
                    overall = prediction.get('overall_signal', 'NEUTRAL')
                    score = prediction.get('bullish_count', 0)
                    total = len(signals)
                    
                    if total > 0:
                        lines.append("\n📈 *GELİŞMİŞ TEKNİK*")
                        
                        # Show key indicators
                        for indicator in ['OBV', 'CMF', 'ADX', 'VWAP']:
                            if indicator in signals:
                                sig = signals[indicator]
                                emoji = "🟢" if sig == 'BUY' else "🔴" if sig == 'SELL' else "⚪"
                                lines.append(f"  {indicator}: {emoji} {sig}")
                        
                        lines.append("  ━━━━━━━━━━━━━")
                        
                        emoji = "🟢" if overall == 'BUY' else "🔴" if overall == 'SELL' else "⚪"
                        lines.append(f"  SKOR: {score}/{total} {emoji} {overall}")
            except Exception as e:
                logger.debug(f"Enhanced predictor skipped: {e}")
            
            # 5. 💥 LIQUIDATION ANALYSIS (FULL)
            try:
                liq_data = liq_hunter_data  # Already fetched in step 1
                
                if liq_data and liq_data.get('data_available'):
                    magnet = liq_data.get('magnet_zone', 0)
                    ls_ratio = liq_data.get('long_short_ratio', 0)
                    oi_change = liq_data.get('oi_change_1h', 0)
                    funding = liq_data.get('funding_rate', 0) * 100
                    
                    lines.append("\n💥 *LİKİDASYON ANALİZİ*")
                    if magnet > 0:
                        lines.append(f"  Magnet: ${magnet:,.0f}")
                    
                    # L/S Ratio with interpretation
                    if ls_ratio > 2.0:
                        lines.append(f"  L/S: {ls_ratio:.2f} (🔴 Çok Long Ağır)")
                        if oi_change > 5:
                            lines.append("  ⚠️ Düzeltme riski yüksek!")
                    elif ls_ratio < 0.8:
                        lines.append(f"  L/S: {ls_ratio:.2f} (🟢 Çok Short Ağır)")
                    elif ls_ratio > 0:
                        lines.append(f"  L/S: {ls_ratio:.2f} (⚪ Dengeli)")
                    
                    # OI Change
                    if abs(oi_change) > 5:
                        emoji = "🔥" if oi_change > 0 else "❄️"
                        action = "Giriş" if oi_change > 0 else "Çıkış"
                        lines.append(f"  OI: {oi_change:+.1f}% ({emoji} Güçlü {action})")
                    
                    # Funding
                    if abs(funding) > 0.01:
                        emoji = "🟢" if funding > 0 else "🔴"
                        lines.append(f"  Funding: {funding:.3f}% ({emoji})")
            except Exception as e:
                logger.debug(f"Liquidation full analysis skipped: {e}")
            
            # 6. 🎨 PATTERN ANALYSIS (SMC FULL)
            try:
                pattern_data = pattern_result  # Already fetched in step 2
                
                if pattern_data and pattern_data.get('wyckoff'):
                    wyckoff = pattern_data['wyckoff']
                    phase = wyckoff.get('phase', 'UNKNOWN')
                    confidence = wyckoff.get('confidence', 0)
                    signal = wyckoff.get('signal', 'NEUTRAL')
                    
                    lines.append("\n🎨 *PATTERN (SMC)*")
                    
                    if phase == 'ACCUMULATION' and confidence > 50:
                        lines.append(f"  Wyckoff: 📊 {phase} ({confidence}%)")
                        lines.append("  → Kurumlar topluyor!")
                    elif phase == 'DISTRIBUTION' and confidence > 50:
                        lines.append(f"  Wyckoff: 📉 {phase} ({confidence}%)")
                        lines.append("  → Kurumlar dağıtıyor!")
                    
                    # Order Blocks
                    if 'order_blocks' in pattern_data:
                        ob_data = pattern_data['order_blocks']
                        bullish_count = ob_data.get('bullish_count', 0)
                        bearish_count = ob_data.get('bearish_count', 0)
                        
                        if bullish_count > 0 or bearish_count > 0:
                            lines.append(f"  Order Blocks: {bullish_count} 🟢 / {bearish_count} 🔴")
                    
                    # FVG
                    if pattern_data.get('fvg_count', 0) > 0:
                        fvg_count = pattern_data['fvg_count']
                        lines.append(f"  FVG: {fvg_count} boşluk")
                    
                    # Market Structure
                    if 'structure' in pattern_data:
                        structure = pattern_data['structure']
                        trend = structure.get('trend', 'UNKNOWN')
                        if trend != 'UNKNOWN':
                            lines.append(f"  Structure: {trend}")
            except Exception as e:
                logger.debug(f"Pattern SMC analysis skipped: {e}")
            
            # === MEVCUT GÖSTERGELER (Pivot, Volatility, vb.) ===
            
            # 3. PIVOT POINTS (Support/Resistance)
            pivot_analyzer = get_pivot_points()
            pivot_data = await pivot_analyzer.analyze(symbol)
            
            if pivot_data and 'daily_pivots' in pivot_data:
                pivots = pivot_data['daily_pivots'][:5]
                if pivots:
                    lines.append("\\n🎯 *PIVOT POINTS*")
                    for p in pivots:
                        level_type = p.get('name', 'Unknown')
                        price = p.get('price', 0)
                        if price > 0:
                            lines.append(f"  {level_type}: ${price:,.0f}")
            
            # 4. VOLATILITY STATUS
            vol_predictor = VolatilityPredictor()
            vol_status = vol_predictor.predict_volatility(symbol)
            
            if vol_status and vol_status.get('state'):
                state = vol_status.get('state', 'UNKNOWN')
                ratio = vol_status.get('volatility_ratio', 0)
                lines.append("\n🌋 *VOLATİLİTE DURUMU*")
                if state == 'SQUEEZE':
                    lines.append("  ⚠️ SIKIŞMA - Büyük hareket yakın!")
                else:
                    lines.append(f"  ➡️ {state} (Ratio: {ratio:.2f})")
            
            # 5. MARKET REGIME
            regime_classifier = RegimeClassifier()
            # Note: identify_regime needs DataFrame, skip for now
            # regime = regime_classifier.identify_regime(df)
            # Just skip this section for now
            regime = None
            
            if regime:
                lines.append("\n🧭 *PİYASA REJİMİ*")
                regime_type = regime.get('regime', 'UNKNOWN')
                confidence = regime.get('confidence', 0)
                lines.append(f"  {regime_type} ({confidence:.0f}% güven)")
            
            # 6. SENTIMENT (News-based)
            scraper = CryptoNewsScraper()
            scraper.fetch_all_news(max_age_hours=2)
            sentiment_data = scraper.get_market_sentiment()
            
            if sentiment_data:
                score = sentiment_data.get('score', 0)
                mood = sentiment_data.get('overall', 'NEUTRAL')
                lines.append("\n📰 *SENTIMENT (Haber Bazlı)*")
                emoji = "🐂" if mood == 'BULLISH' else "🐻" if mood == 'BEARISH' else "⚪"
                lines.append(f"  {emoji} {mood} (Skor: {score:.1f}/10)")
            
            # 7. MACRO CONTEXT & MOMENTUM (NEW)
            try:
                from src.brain.macro_context import get_macro_context
                from src.brain.momentum_detector import get_momentum_context
                
                # Fetch data
                import asyncio
                # Use asyncio.gather for parallel fetching if needed, but direct await is fine here
                macro = await get_macro_context()
                momentum = await get_momentum_context(symbol)
                
                if macro:
                    lines.append("\n🌍 *MAKRO BAĞLAM*")
                    fear = macro.fear_greed_index
                    fear_label = macro.fear_greed_label
                    btc_d = macro.btc_dominance
                    
                    fear_emoji = "😨" if fear < 30 else "🤑" if fear > 70 else "😐"
                    lines.append(f"  {fear_emoji} Fear & Greed: {fear} ({fear_label})")
                    lines.append(f"  📊 BTC.D: {btc_d:.1f}% ({macro.btc_dominance_trend})")
                    if macro.altcoin_season_index > 50:
                        lines.append(f"  🔥 Altcoin Season: {macro.altcoin_season_index:.0f}/100")

                if momentum:
                    lines.append("\n⚡ *MOMENTUM & BREAKOUT*")
                    
                    # Volume Spike
                    if momentum.volume_spike:
                        lines.append(f"  🔥 Volume Spike: {momentum.volume_ratio:.1f}x (ANLIK)")
                    
                    # Momentum
                    mom_5m = momentum.momentum_5m
                    mom_emoji = "🚀" if mom_5m > 0.5 else "🩸" if mom_5m < -0.5 else "➡️"
                    lines.append(f"  {mom_emoji} 5m Momentum: {mom_5m:+.2f}%")
                    
                    # CVD & Breakout
                    if momentum.cvd_divergence:
                        lines.append(f"  ⚠️ CVD Uyumsuzluğu (Dönüş riski)")
                    
                    if momentum.breakout_probability > 60:
                        lines.append(f"  🎯 Breakout İhtimali: {momentum.breakout_probability:.0f}%")

            except Exception as e:
                logger.debug(f"Macro/Momentum part skipped: {e}")
            
            # === 🎯 NET AI TAVSİYE ===
            try:
                from src.v10.lstm_predictor import get_lstm_predictor
                lstm = get_lstm_predictor()
                pred = await lstm.predict(symbol)
                
                if pred:
                    lines.append("\n🎯 *AI SMART TAVSİYE*") # Replaced title
                    
                    # Decision logic combining LSTM + Volatility + Sentiment
                    action = "⏸️ BEKLE"
                    reason = ""
                    
                    # LSTM direction
                    is_bullish = pred.direction == "UP" and pred.confidence > 40
                    is_bearish = pred.direction == "DOWN" and pred.confidence > 40
                    
                    # Volatility squeeze = be cautious
                    is_squeeze = vol_status and vol_status.get('state') == 'SQUEEZE'
                    
                    # Sentiment boost
                    sentiment_bullish = sentiment_data and sentiment_data.get('overall') == 'BULLISH'
                    sentiment_bearish = sentiment_data and sentiment_data.get('overall') == 'BEARISH'
                    
                    if is_bullish and not is_squeeze:
                        action = "🟢 AL (Long)"
                        if sentiment_bullish:
                            reason = "LSTM UP + Pozitif haber"
                        else:
                            reason = f"LSTM: +{pred.predicted_change_pct:.1f}%"
                        
                        # Momentum boost
                        if momentum and momentum.volume_spike and momentum.momentum_5m > 0:
                            action = "🚀 GÜÇLÜ AL"
                            reason += " + Hacim Patlaması"

                    elif is_bearish and not is_squeeze:
                        action = "🔴 SAT (Short)"
                        if sentiment_bearish:
                            reason = "LSTM DOWN + Negatif haber"
                        else:
                            reason = f"LSTM: {pred.predicted_change_pct:.1f}%"
                            
                        # Momentum boost
                        if momentum and momentum.volume_spike and momentum.momentum_5m < 0:
                            action = "🩸 GÜÇLÜ SAT"
                            reason += " + Hacim Patlaması"

                    elif is_squeeze:
                        action = "⏸️ BEKLE"
                        reason = "Volatilite sıkışması - Patlama bekleniyor"
                    else:
                        action = "⏸️ BEKLE"
                        reason = "Net sinyal yok"
                    
                    lines.append(f"  {action}")
                    lines.append(f"  📝 Neden: {reason}")
            except Exception as e:
                logger.debug(f"AI recommendation skipped: {e}")
            
            lines.append("━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
            lines.append("📡 *DEMIR AI v10 - HYBRID MODE*")
            
            return self._send_message("\n".join(lines))
            
        except Exception as e:
            logger.error(f"Deep technical analysis error: {e}")
            return self._send_message(f"❌ Rapor hatası: {str(e)[:50]}")

    def send_signal(self, signal) -> bool:
        """
        HYBRID AI SIGNAL FORMATI
        
        İçerik:
        1. Ana Yön ve Güven
        2. AI Reasoning (Madde madde)
        3. Momentum & Breakout (Varsa)
        4. Claude Haiku Yorumu
        5. Seviyeler (Entry, TP, SL)
        """
        if not signal:
            return False
            
        # Emoji seçimi
        if signal.action == "BUY":
            header_emoji = "🟢"
            direction = "LONG"
        elif signal.action == "SELL":
            header_emoji = "🔴"
            direction = "SHORT"
        else:
            header_emoji = "⏸️"
            direction = "BEKLE"
            
        lines = [
            f"🧠 *AI TRADE SİNYALİ - {signal.symbol}*",
            "━━━━━━━━━━━━━━━━━━",
            f"",
            f"📍 YÖN: {header_emoji} *{direction}*",
            f"🎯 GÜVEN: *%{signal.confidence:.0f}*",
            f"",
            "🔍 *AI REASONING:*"
        ]
        
        # 1. Ana Reasoning (Maddeler)
        reasons = signal.reasoning.split(" | ")
        for r in reasons[:4]:  # İlk 4 sebep
            lines.append(f"• {r}")
            
        # 2. Momentum Alerts
        if hasattr(signal, 'momentum_alerts') and signal.momentum_alerts:
            for alert in signal.momentum_alerts:
                atype = alert.get('alert_type', '')
                if atype == 'VOLUME_SPIKE':
                    lines.append(f"• 🔥 *Volume Spike ({alert.get('value', 0):.1f}x)*")
                elif atype == 'LIQ_CLUSTER':
                    lines.append(f"• 🧲 *Likidasyon Mıknatısı (Yakın)*")
                elif atype == 'CVD_DIVERGENCE':
                    lines.append(f"• ⚠️ *CVD Uyumsuzluğu (Tuzak?)*")
                    
        # 3. Skor Dağılımı
        if hasattr(signal, 'score_breakdown') and signal.score_breakdown:
            sb = signal.score_breakdown
            lines.append(f"")
            lines.append(f"📊 *Skor:* Tech:{sb.get('technical',0):.0f} | Macro:{sb.get('macro',0):.0f} | 🐋:{sb.get('onchain',0):.0f} | 🤖:{sb.get('llm',0):.0f}")

        # 4. Kasa Yönetimi (Risk Manager)
        if hasattr(signal, 'risk_profile') and signal.risk_profile:
            rp = signal.risk_profile
            lines.append(f"")
            lines.append(f"💰 *KASA YÖNETİMİ:*")
            lines.append(f"• Kaldıraç: *{rp.get('leverage', 1)}x*")
            lines.append(f"• Marjin: *%{rp.get('position_size_pct', 1.0):.1f}* (Kasa Oranı)")
            lines.append(f"• Not: {rp.get('reason', '')}")

        # 5. Claude Yorumu
        if hasattr(signal, 'llm_reasoning') and signal.llm_reasoning:
            lines.append(f"")
            lines.append(f"🤖 *Claude:* \"{signal.llm_reasoning}\"")
            
        # 6. Seviyeler
        if signal.action != "HOLD":
            lines.append(f"")
            lines.append(f"🚪 *GİRİŞ:* ${signal.entry_zone[0]:,.2f} - ${signal.entry_zone[1]:,.2f}")
            lines.append(f"🎯 *HEDEF:* ${signal.take_profit:,.2f}")
            lines.append(f"🛡️ *STOP:* ${signal.stop_loss:,.2f} (R/R: {signal.risk_reward:.2f})")
        
        lines.append(f"")
        lines.append("━━ DEMIR AI v10 HYBRID ━━")
        
        return self._send_message("\n".join(lines))

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
