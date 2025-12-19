# -*- coding: utf-8 -*-
"""
DEMIR AI - AI REASONING ENGINE
Phase 127: Gerçek Yapay Zeka

Tüm modülleri birleştirip DÜŞÜNEN bir sistem.
Sadece voting değil - AKIL YÜRÜTME.

ÖZELLİKLER:
1. Tüm modül verilerini toplar
2. Aralarındaki korelasyonları bulur
3. Çelişkileri tespit eder
4. Insan-like piyasa yorumu üretir
5. "Yarın ne olacak?" tahmini yapar
"""
import logging
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("AI_REASONING_ENGINE")


@dataclass
class MarketContext:
    """Piyasa bağlamı - tüm veriler bir arada."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Fiyat & Hacim
    symbol: str = "BTCUSDT"
    current_price: float = 0
    price_change_1h: float = 0
    price_change_24h: float = 0
    volume_ratio: float = 1.0  # vs ortalama
    
    # Whale & Kurumsal
    whale_bias: str = "NEUTRAL"  # LONG/SHORT/NEUTRAL
    whale_confidence: float = 50
    whale_activity: str = ""  # "Büyük alım var" gibi
    
    # Funding & Derivatives
    funding_rate: float = 0
    funding_signal: str = "NEUTRAL"  # Contrarian yorumu
    open_interest_change: float = 0
    long_short_ratio: float = 1.0
    
    # Sentiment
    fear_greed_index: int = 50
    fear_greed_label: str = "Neutral"
    sentiment_signal: str = "NEUTRAL"
    
    # Teknik
    rsi: float = 50
    macd_signal: str = "NEUTRAL"
    bb_position: str = "MIDDLE"  # UPPER/MIDDLE/LOWER
    trend_1h: str = "NEUTRAL"
    trend_4h: str = "NEUTRAL"
    trend_1d: str = "NEUTRAL"
    
    # SMC
    nearest_order_block: str = ""
    liquidity_hunt_risk: str = "LOW"
    
    # Fibonacci
    nearest_support: float = 0
    nearest_resistance: float = 0
    fib_position: str = ""  # "0.618 seviyesinde" gibi
    
    # Liquidation
    liq_magnet_direction: str = "NEUTRAL"
    liq_magnet_price: float = 0
    
    # Session
    session_name: str = ""
    session_volatility: str = "NORMAL"
    
    # Order Flow
    taker_buy_ratio: float = 0.5
    order_flow_bias: str = "NEUTRAL"


@dataclass
class MarketPrediction:
    """AI'ın piyasa tahmini."""
    direction: str  # YUKARI/ASAGI/YATAY
    confidence: float  # 0-100
    timeframe: str  # "1-4 saat", "24 saat", etc.
    target_price: float
    stop_loss: float
    rr_ratio: float
    
    # Reasoning
    primary_reason: str
    supporting_reasons: List[str]
    contradicting_factors: List[str]
    risk_factors: List[str]
    
    # Narrative
    narrative: str  # İnsan-like açıklama
    action_recommendation: str  # "Bekle", "Al", "Sat" gibi


class AIReasoningEngine:
    """
    AI AKIL YÜRÜTME MOTORU
    
    Tüm modülleri birleştirip gerçek zeka üretir.
    Voting sistemi değil - MANTIK.
    """
    
    def __init__(self):
        self.context = None
        self.last_analysis = None
        self.analysis_history = []
        
        logger.info("✅ AI Reasoning Engine initialized - True Intelligence Mode")
    
    async def gather_all_intelligence(self, symbol: str = 'BTCUSDT') -> MarketContext:
        """
        TÜM modüllerden veri topla ve bir bağlam oluştur.
        """
        ctx = MarketContext(symbol=symbol)
        
        # 1. Temel Fiyat Verisi
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={'symbol': symbol},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                ctx.current_price = float(data['lastPrice'])
                ctx.price_change_24h = float(data['priceChangePercent'])
                ctx.volume_ratio = float(data['volume']) / float(data['quoteVolume']) if float(data['quoteVolume']) > 0 else 1
        except Exception as e:
            logger.debug(f"Price fetch failed: {e}")
        
        # 2. Whale Intelligence
        try:
            from src.brain.whale_intelligence import WhaleIntelligence
            whale = WhaleIntelligence()
            whale_data = await whale.analyze_whale_sentiment(symbol)
            
            ctx.whale_bias = whale_data.get('bias', 'NEUTRAL')
            ctx.whale_confidence = whale_data.get('confidence', 50)
            
            # Activity description
            if ctx.whale_bias == 'LONG':
                ctx.whale_activity = "Balinalar ALIYOR - Büyük oyuncular yukarı bekliyor"
            elif ctx.whale_bias == 'SHORT':
                ctx.whale_activity = "Balinalar SATIYOR - Kurumsal çıkış sinyali"
            else:
                ctx.whale_activity = "Balinalar bekleme modunda"
        except Exception as e:
            logger.debug(f"Whale intel failed: {e}")
        
        # 3. Funding Rate (Contrarian)
        try:
            from src.brain.coinglass_funding import get_funding_rate
            funding = await get_funding_rate(symbol)
            
            if funding:
                ctx.funding_rate = funding.get('rate', 0)
                
                # Contrarian logic
                if ctx.funding_rate > 0.02:
                    ctx.funding_signal = "SHORT_OPPORTUNITY"
                elif ctx.funding_rate < -0.02:
                    ctx.funding_signal = "LONG_OPPORTUNITY"
                else:
                    ctx.funding_signal = "NEUTRAL"
        except Exception as e:
            logger.debug(f"Funding fetch failed: {e}")
        
        # 4. Fear & Greed
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            if resp.status_code == 200:
                data = resp.json()['data'][0]
                ctx.fear_greed_index = int(data['value'])
                ctx.fear_greed_label = data['value_classification']
                
                # Contrarian
                if ctx.fear_greed_index <= 25:
                    ctx.sentiment_signal = "EXTREME_FEAR_BUY"
                elif ctx.fear_greed_index >= 75:
                    ctx.sentiment_signal = "EXTREME_GREED_SELL"
                else:
                    ctx.sentiment_signal = "NEUTRAL"
        except Exception as e:
            logger.debug(f"Fear/Greed fetch failed: {e}")
        
        # 5. Fibonacci Levels
        try:
            from src.brain.fibonacci_analyzer import FibonacciAnalyzer
            fib = FibonacciAnalyzer()
            fib_data = await fib.calculate_levels(symbol)
            
            if fib_data:
                ctx.nearest_support = fib_data.get('nearest_support', {}).get('price', 0)
                ctx.nearest_resistance = fib_data.get('nearest_resistance', {}).get('price', 0)
                
                # Position description
                if fib_data.get('nearest_support', {}).get('level'):
                    ctx.fib_position = f"Fib {fib_data['nearest_support']['level']} desteğinde"
        except Exception as e:
            logger.debug(f"Fibonacci failed: {e}")
        
        # 6. Liquidation Magnet
        try:
            from src.brain.liquidation_hunter import LiquidationHunter
            hunter = LiquidationHunter()
            liq_data = await hunter.calculate_liquidation_levels(symbol)
            
            if liq_data and 'error' not in liq_data:
                ctx.liq_magnet_direction = liq_data.get('magnet_direction', 'NEUTRAL')
                ctx.liq_magnet_price = liq_data.get('magnet_price', 0)
        except Exception as e:
            logger.debug(f"Liquidation failed: {e}")
        
        # 7. Session & Trader Mindset
        try:
            from src.brain.trader_mindset import get_trader_mindset
            mindset = get_trader_mindset()
            
            session = mindset.get_current_session()
            ctx.session_name = session.get('name', 'Unknown')
            ctx.session_volatility = session.get('volatility_expected', 'NORMAL')
            
            # Order flow
            order_flow = mindset.analyze_order_flow(symbol)
            if order_flow.get('available'):
                ctx.taker_buy_ratio = order_flow.get('taker_buy_ratio', 0.5)
                ctx.order_flow_bias = order_flow.get('flow', 'NEUTRAL')
        except Exception as e:
            logger.debug(f"Trader mindset failed: {e}")
        
        # 8. LSTM Prediction
        try:
            from src.brain.models.lstm_trend import LSTMTrendPredictor
            lstm = LSTMTrendPredictor(symbol=symbol)
            # LSTM tahminini al (eğer model yüklüyse)
        except Exception as e:
            logger.debug(f"LSTM failed: {e}")
        
        self.context = ctx
        return ctx
    
    def analyze_correlations(self, ctx: MarketContext) -> Dict:
        """
        Modüller arasındaki korelasyonları analiz et.
        Hangi sinyaller birbirini destekliyor/çelişiyor?
        """
        correlations = {
            'strong_bullish': [],
            'strong_bearish': [],
            'contradictions': [],
            'neutral': []
        }
        
        # --- BULLISH FACTORS ---
        
        # Whale + Fear/Greed combo
        if ctx.whale_bias == 'LONG' and ctx.fear_greed_index <= 30:
            correlations['strong_bullish'].append(
                "🐋😱 Balinalar alıyor + Piyasa korkuyor = GÜÇLÜ ALIM FIRSATI"
            )
        
        # Funding contrarian
        if ctx.funding_rate < -0.02:
            correlations['strong_bullish'].append(
                f"📊 Funding rate negatif ({ctx.funding_rate:.3f}%) = Short'lar fazla, squeeze riski"
            )
        
        # Liquidation magnet up
        if ctx.liq_magnet_direction == 'UP':
            correlations['strong_bullish'].append(
                f"🎯 Tasfiye mıknatısı yukarıda (${ctx.liq_magnet_price:,.0f}) = Fiyat oraya çekilebilir"
            )
        
        # Order flow
        if ctx.taker_buy_ratio > 0.55:
            correlations['strong_bullish'].append(
                f"💹 Order flow: %{ctx.taker_buy_ratio*100:.0f} alıcı baskısı = Aktif talep var"
            )
        
        # --- BEARISH FACTORS ---
        
        # Whale selling + Greed
        if ctx.whale_bias == 'SHORT' and ctx.fear_greed_index >= 70:
            correlations['strong_bearish'].append(
                "🐋😊 Balinalar satıyor + Piyasa açgözlü = SATIŞ ZAMANI"
            )
        
        # High funding (overleveraged longs)
        if ctx.funding_rate > 0.05:
            correlations['strong_bearish'].append(
                f"📊 Funding rate çok yüksek ({ctx.funding_rate:.3f}%) = Long'lar aşırı kalabalık"
            )
        
        # Liquidation magnet down
        if ctx.liq_magnet_direction == 'DOWN':
            correlations['strong_bearish'].append(
                f"🎯 Tasfiye mıknatısı aşağıda (${ctx.liq_magnet_price:,.0f}) = Düşüş baskısı"
            )
        
        # Order flow selling
        if ctx.taker_buy_ratio < 0.45:
            correlations['strong_bearish'].append(
                f"💹 Order flow: %{(1-ctx.taker_buy_ratio)*100:.0f} satıcı baskısı = Satış baskısı"
            )
        
        # --- CONTRADICTIONS ---
        
        # Whale vs Fear/Greed contradiction
        if ctx.whale_bias == 'LONG' and ctx.fear_greed_index >= 75:
            correlations['contradictions'].append(
                "⚠️ Balinalar alıyor AMA piyasa zaten aşırı açgözlü - dikkat!"
            )
        
        if ctx.whale_bias == 'SHORT' and ctx.fear_greed_index <= 25:
            correlations['contradictions'].append(
                "⚠️ Balinalar satıyor AMA piyasa zaten aşırı korkuyor - panik satışı mı?"
            )
        
        # Funding vs Whale contradiction
        if ctx.funding_rate > 0.03 and ctx.whale_bias == 'LONG':
            correlations['contradictions'].append(
                "⚠️ Funding yüksek ama balinalar hala alıyor - kısa vadeli düzeltme sonrası devam mı?"
            )
        
        return correlations
    
    def generate_prediction(self, ctx: MarketContext, correlations: Dict) -> MarketPrediction:
        """
        Tüm verileri sentezleyerek TAHMİN üret.
        """
        bullish_count = len(correlations['strong_bullish'])
        bearish_count = len(correlations['strong_bearish'])
        contradiction_count = len(correlations['contradictions'])
        
        # Direction decision
        if bullish_count > bearish_count + 1:
            direction = "YUKARI"
            confidence = min(90, 50 + bullish_count * 10 - contradiction_count * 5)
        elif bearish_count > bullish_count + 1:
            direction = "AŞAĞI"
            confidence = min(90, 50 + bearish_count * 10 - contradiction_count * 5)
        else:
            direction = "YATAY"
            confidence = 40
        
        # Targets
        if direction == "YUKARI":
            target = ctx.nearest_resistance if ctx.nearest_resistance > ctx.current_price else ctx.current_price * 1.025
            stop = ctx.nearest_support if ctx.nearest_support > 0 else ctx.current_price * 0.985
        elif direction == "AŞAĞI":
            target = ctx.nearest_support if ctx.nearest_support < ctx.current_price else ctx.current_price * 0.975
            stop = ctx.nearest_resistance if ctx.nearest_resistance > 0 else ctx.current_price * 1.015
        else:
            target = ctx.current_price
            stop = ctx.current_price
        
        # R:R
        risk = abs(ctx.current_price - stop)
        reward = abs(target - ctx.current_price)
        rr = reward / risk if risk > 0 else 0
        
        # Primary reason
        if direction == "YUKARI" and correlations['strong_bullish']:
            primary_reason = correlations['strong_bullish'][0]
        elif direction == "AŞAĞI" and correlations['strong_bearish']:
            primary_reason = correlations['strong_bearish'][0]
        else:
            primary_reason = "Karma sinyaller - net yön yok"
        
        # Risk factors
        risk_factors = []
        if contradiction_count > 0:
            risk_factors.extend(correlations['contradictions'])
        if ctx.session_volatility == "HIGH":
            risk_factors.append(f"⚡ {ctx.session_name} session'ı yüksek volatilite bekliyor")
        
        # Action recommendation
        if confidence >= 70 and rr >= 2:
            action = f"✅ {direction} pozisyon düşünülebilir (R:R {rr:.1f})"
        elif confidence >= 60:
            action = f"👀 {direction} yönünde izle, daha iyi giriş bekle"
        else:
            action = "⏸️ Bekle - net sinyal yok"
        
        # Generate narrative
        narrative = self._generate_narrative(ctx, correlations, direction, confidence)
        
        return MarketPrediction(
            direction=direction,
            confidence=confidence,
            timeframe="1-4 saat",
            target_price=target,
            stop_loss=stop,
            rr_ratio=rr,
            primary_reason=primary_reason,
            supporting_reasons=correlations['strong_bullish'] if direction == "YUKARI" else correlations['strong_bearish'],
            contradicting_factors=correlations['contradictions'],
            risk_factors=risk_factors,
            narrative=narrative,
            action_recommendation=action
        )
    
    def _generate_narrative(self, ctx: MarketContext, correlations: Dict, 
                           direction: str, confidence: float) -> str:
        """
        İnsan-like piyasa yorumu oluştur.
        """
        lines = []
        
        # Session context
        lines.append(f"🌍 Şu an {ctx.session_name} session'ı aktif.")
        
        # Main observation
        if direction == "YUKARI":
            lines.append(f"📈 Piyasa YUKARI yönlü sinyal veriyor (%{confidence:.0f} güven).")
        elif direction == "AŞAĞI":
            lines.append(f"📉 Piyasa AŞAĞI yönlü sinyal veriyor (%{confidence:.0f} güven).")
        else:
            lines.append(f"↔️ Piyasa kararsız - net yön yok.")
        
        # Key insights
        if ctx.whale_activity:
            lines.append(f"🐋 {ctx.whale_activity}")
        
        if ctx.fear_greed_index <= 30:
            lines.append(f"😱 Fear & Greed: {ctx.fear_greed_index} - Aşırı korku = Contrarian alım fırsatı")
        elif ctx.fear_greed_index >= 70:
            lines.append(f"😊 Fear & Greed: {ctx.fear_greed_index} - Aşırı açgözlülük = Dikkatli ol")
        
        if ctx.funding_signal == "LONG_OPPORTUNITY":
            lines.append("📊 Funding negatif - Short'lar kalabalık, squeeze olabilir")
        elif ctx.funding_signal == "SHORT_OPPORTUNITY":
            lines.append("📊 Funding çok yüksek - Long'lar aşırı, düzeltme riski")
        
        # Contradictions warning
        if correlations['contradictions']:
            lines.append(f"⚠️ DİKKAT: {len(correlations['contradictions'])} çelişkili sinyal var!")
        
        return "\n".join(lines)
    
    async def think(self, symbol: str = 'BTCUSDT') -> Optional[MarketPrediction]:
        """
        ANA DÜŞÜNME FONKSİYONU
        
        1. Tüm veriyi topla
        2. Korelasyonları bul
        3. Tahmin üret
        """
        try:
            # 1. Gather all intelligence
            ctx = await self.gather_all_intelligence(symbol)
            
            if ctx.current_price == 0:
                logger.error("No price data available")
                return None
            
            # 2. Analyze correlations
            correlations = self.analyze_correlations(ctx)
            
            # 3. Generate prediction
            prediction = self.generate_prediction(ctx, correlations)
            
            # Save to history
            self.last_analysis = {
                'timestamp': datetime.now(),
                'context': ctx,
                'correlations': correlations,
                'prediction': prediction
            }
            self.analysis_history.append(self.last_analysis)
            
            # Keep only last 24 analyses
            if len(self.analysis_history) > 24:
                self.analysis_history = self.analysis_history[-24:]
            
            logger.info(f"🧠 AI REASONING: {symbol} → {prediction.direction} %{prediction.confidence:.0f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"AI Reasoning failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def format_for_telegram(self, prediction: MarketPrediction, symbol: str) -> str:
        """
        Telegram için zengin format.
        """
        ctx = self.context
        
        # Direction emoji
        if prediction.direction == "YUKARI":
            dir_emoji = "🟢📈"
        elif prediction.direction == "AŞAĞI":
            dir_emoji = "🔴📉"
        else:
            dir_emoji = "⚪↔️"
        
        # Confidence stars
        if prediction.confidence >= 80:
            stars = "⭐⭐⭐"
        elif prediction.confidence >= 65:
            stars = "⭐⭐"
        elif prediction.confidence >= 50:
            stars = "⭐"
        else:
            stars = ""
        
        msg = f"""
🧠 AI AKIL YÜRÜTME - {symbol}
━━━━━━━━━━━━━━━━━━━━━━
{dir_emoji} YÖN: {prediction.direction} {stars}
📊 Güven: %{prediction.confidence:.0f}
💰 Fiyat: ${ctx.current_price:,.2f}
━━━━━━━━━━━━━━━━━━━━━━
📝 AI YORUMU:
{prediction.narrative}
━━━━━━━━━━━━━━━━━━━━━━
🎯 ANA SEBEP:
{prediction.primary_reason}
"""
        
        # Add supporting reasons
        if prediction.supporting_reasons:
            msg += "\n✅ DESTEKLEYEN FAKTÖRLER:\n"
            for reason in prediction.supporting_reasons[:3]:
                msg += f"  • {reason}\n"
        
        # Add contradictions
        if prediction.contradicting_factors:
            msg += "\n⚠️ ÇELİŞKİLER:\n"
            for contra in prediction.contradicting_factors[:2]:
                msg += f"  • {contra}\n"
        
        # Add trade setup
        if prediction.rr_ratio >= 1.5:
            msg += f"""
━━━━━━━━━━━━━━━━━━━━━━
🎯 İŞLEM ÖNERİSİ:
▸ Giriş: ${ctx.current_price:,.2f}
▸ Hedef: ${prediction.target_price:,.2f}
▸ Stop: ${prediction.stop_loss:,.2f}
▸ R:R = 1:{prediction.rr_ratio:.1f}
"""
        
        msg += f"""
━━━━━━━━━━━━━━━━━━━━━━
🎯 AKSİYON: {prediction.action_recommendation}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
"""
        
        return msg.strip()


# Global instance
_reasoning_engine = None

def get_reasoning_engine() -> AIReasoningEngine:
    """Get or create reasoning engine instance."""
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = AIReasoningEngine()
    return _reasoning_engine
