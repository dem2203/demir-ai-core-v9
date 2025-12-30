# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - Premium Signal Generator (Enhanced)
===================================================
Profesyonel grade Telegram sinyalleri.

YENİ ÖZELLİKLER (Phase 22-25):
1. Confluence Filter - Min 3 faktör aynı yönde olmalı
2. Multi-Timeframe - 4H ve 1D trend onayı
3. Win Rate Tracking - Faktör bazında performans
4. Self-Learning - Claude'a trade sonuçları gönderilir
"""
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("PREMIUM_SIGNALS")


@dataclass
class PremiumSignal:
    """Premium sinyal sonucu"""
    symbol: str
    direction: str  # LONG, SHORT, BEKLE
    confidence: int  # 0-100
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    
    # Faktörler
    bullish_factors: List[str]
    bearish_factors: List[str]
    risk_factors: List[str]
    
    # Phase 22: Confluence
    confluence_score: int  # Aynı yönde kaç faktör var
    confluence_passed: bool
    
    # Phase 24: Multi-Timeframe
    trend_4h: str  # UP, DOWN, NEUTRAL
    trend_1d: str  # UP, DOWN, NEUTRAL
    trend_aligned: bool
    
    # Claude analizi
    claude_analysis: str
    
    # Meta
    data_sources_count: int
    timestamp: datetime


class PremiumSignalGenerator:
    """
    Premium Sinyal Üretici (Enhanced)
    
    YENİ FİLTRELER:
    - Confluence: Min 3 faktör aynı yönde
    - Multi-Timeframe: 4H + 1D trend aynı yönde
    - Win Rate: Faktör performansı takibi
    """
    
    # Minimum güven eşiği
    MIN_CONFIDENCE = 60
    
    # Phase 22: Confluence - minimum faktör sayısı
    MIN_CONFLUENCE = 3
    
    def __init__(self):
        self._llm_brain = None
        self._aggregator = None
        self._scrapers = None
        
        # Phase 25: Win Rate Tracking
        self.factor_stats = {}  # {"factor_name": {"wins": 0, "losses": 0}}
    
    async def _get_llm_brain(self):
        if self._llm_brain is None:
            from src.brain.llm_brain import LLMBrain
            self._llm_brain = LLMBrain()
        return self._llm_brain
    
    async def _get_aggregator(self):
        if self._aggregator is None:
            from src.brain.institutional_aggregator import get_aggregator
            self._aggregator = get_aggregator()
        return self._aggregator
    
    async def _get_scrapers(self):
        if self._scrapers is None:
            from src.brain.advanced_scrapers import AdvancedMarketScrapers
            self._scrapers = AdvancedMarketScrapers()
        return self._scrapers
    
    async def _get_trend(self, symbol: str, interval: str) -> str:
        """
        Phase 24: Trend hesapla (EMA 20 vs EMA 50)
        
        Args:
            symbol: BTCUSDT
            interval: 4h veya 1d
        
        Returns:
            UP, DOWN, NEUTRAL
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Son 60 mum al
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=60"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return "NEUTRAL"
                    data = await resp.json()
                    
                    closes = [float(k[4]) for k in data]
                    
                    if len(closes) < 50:
                        return "NEUTRAL"
                    
                    # EMA hesapla
                    ema20 = self._calculate_ema(closes, 20)
                    ema50 = self._calculate_ema(closes, 50)
                    
                    current_price = closes[-1]
                    
                    # Trend belirleme
                    if ema20 > ema50 and current_price > ema20:
                        return "UP"
                    elif ema20 < ema50 and current_price < ema20:
                        return "DOWN"
                    else:
                        return "NEUTRAL"
        except Exception as e:
            logger.debug(f"Trend calculation error: {e}")
            return "NEUTRAL"
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """EMA hesapla"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # SMA başlangıç
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    async def _get_past_performance(self) -> Dict:
        """
        Phase 23: Geçmiş trade sonuçlarını al (Self-Learning için)
        """
        try:
            from src.brain.paper_trading_manager import get_paper_trading_manager
            ptm = get_paper_trading_manager()
            
            # Son 10 kapanan trade'i al
            recent_trades = [t for t in ptm.trades if t.status != "OPEN"][-10:]
            
            if not recent_trades:
                return {}
            
            wins = len([t for t in recent_trades if t.status in ["TP1_HIT", "TP2_HIT"]])
            losses = len([t for t in recent_trades if t.status == "SL_HIT"])
            
            # Hangi faktörler çalıştı?
            winning_factors = []
            losing_factors = []
            
            for trade in recent_trades:
                if trade.status in ["TP1_HIT", "TP2_HIT"]:
                    winning_factors.extend(trade.bullish_factors + trade.bearish_factors)
                elif trade.status == "SL_HIT":
                    losing_factors.extend(trade.risk_factors)
            
            return {
                "recent_win_rate": wins / len(recent_trades) * 100 if recent_trades else 0,
                "wins": wins,
                "losses": losses,
                "winning_factors": winning_factors[:5],
                "losing_factors": losing_factors[:3]
            }
        except Exception as e:
            logger.debug(f"Past performance error: {e}")
            return {}
    
    async def generate(self, symbol: str = "BTCUSDT") -> Optional[PremiumSignal]:
        """Premium sinyal üret - Tüm filtrelerle."""
        try:
            # 1. Veri Toplama
            aggregator = await self._get_aggregator()
            scrapers = await self._get_scrapers()
            
            snapshot = await aggregator.get_live_snapshot(symbol)
            base_symbol = symbol.replace("USDT", "")
            futures_data = scrapers.get_liquidation_levels(base_symbol)
            fng_data = scrapers.get_fear_greed_index()
            
            if snapshot.current_price == 0:
                logger.warning(f"⚠️ {symbol}: Fiyat verisi yok!")
                return None
            
            current_price = snapshot.current_price
            
            # ========================================
            # Phase 24: Multi-Timeframe Trend Al
            # ========================================
            trend_4h = await self._get_trend(symbol, "4h")
            trend_1d = await self._get_trend(symbol, "1d")
            
            # 2. Faktör Analizi
            bullish_factors = []
            bearish_factors = []
            risk_factors = []
            
            bullish_score = 0.0
            bearish_score = 0.0
            
            # === WHALE FLOW ===
            if snapshot.whale_net_flow > 1000000:
                bullish_factors.append(f"🐋 Whale +${snapshot.whale_net_flow/1e6:.1f}M alım")
                bullish_score += 2.0
            elif snapshot.whale_net_flow < -1000000:
                bearish_factors.append(f"🐋 Whale ${abs(snapshot.whale_net_flow)/1e6:.1f}M satış")
                bearish_score += 2.0
            
            # === ORDER BOOK ===
            if snapshot.orderbook_imbalance > 2.0:
                bullish_factors.append(f"📊 Order Book {snapshot.orderbook_imbalance:.1f}x alım")
                bullish_score += 2.5
            elif snapshot.orderbook_imbalance < 0.5:
                bearish_factors.append(f"📊 Order Book {1/snapshot.orderbook_imbalance:.1f}x satış")
                bearish_score += 2.5
            elif snapshot.orderbook_imbalance > 1.3:
                bullish_factors.append(f"📊 Order Book {snapshot.orderbook_imbalance:.2f}x bid")
                bullish_score += 1.0
            elif snapshot.orderbook_imbalance < 0.7:
                bearish_factors.append(f"📊 Order Book {1/snapshot.orderbook_imbalance:.2f}x ask")
                bearish_score += 1.0
            
            # === CVD ===
            if snapshot.cvd_trend == "BULLISH" and snapshot.cvd_value > 0:
                bullish_factors.append("📈 CVD pozitif")
                bullish_score += 1.5
            elif snapshot.cvd_trend == "BEARISH" and snapshot.cvd_value < 0:
                bearish_factors.append("📉 CVD negatif")
                bearish_score += 1.5
            
            # === TAKER VOLUME ===
            if snapshot.taker_buy_ratio > 0.55:
                bullish_factors.append(f"💥 Taker %{snapshot.taker_buy_ratio*100:.0f} alıcı")
                bullish_score += 1.5
            elif snapshot.taker_buy_ratio < 0.45:
                bearish_factors.append(f"💥 Taker %{(1-snapshot.taker_buy_ratio)*100:.0f} satıcı")
                bearish_score += 1.5
            
            # === FUNDING RATE ===
            funding = futures_data.get('funding_rate', 0)
            if funding > 0.0005:
                risk_factors.append(f"⚠️ Funding +{funding*100:.4f}%")
                bearish_score += 1.0
            elif funding < -0.0001:
                bullish_factors.append("💸 Funding negatif")
                bullish_score += 1.0
            
            # === L/S RATIO ===
            ls_ratio = futures_data.get('long_short_ratio', 1.0)
            if ls_ratio > 2.0:
                risk_factors.append(f"⚠️ L/S {ls_ratio:.2f}")
                bearish_score += 1.5
            elif ls_ratio < 0.7:
                bullish_factors.append(f"🚀 L/S {ls_ratio:.2f}")
                bullish_score += 1.5
            
            # === FEAR & GREED ===
            fng_value = fng_data.get('value', 50)
            if fng_value < 25:
                bullish_factors.append(f"😱 F&G {fng_value}")
                bullish_score += 2.0
            elif fng_value > 75:
                risk_factors.append(f"🤑 F&G {fng_value}")
                bearish_score += 1.5
            elif fng_value < 35:
                bullish_factors.append(f"😨 F&G {fng_value}")
                bullish_score += 1.0
            
            # === Multi-Timeframe Trend ===
            if trend_4h == "UP":
                bullish_factors.append("📊 4H Trend UP")
                bullish_score += 1.5
            elif trend_4h == "DOWN":
                bearish_factors.append("📊 4H Trend DOWN")
                bearish_score += 1.5
            
            if trend_1d == "UP":
                bullish_factors.append("📊 1D Trend UP")
                bullish_score += 2.0
            elif trend_1d == "DOWN":
                bearish_factors.append("📊 1D Trend DOWN")
                bearish_score += 2.0
            
            # ========================================
            # Phase 22: Confluence Check
            # ========================================
            bullish_count = len(bullish_factors)
            bearish_count = len(bearish_factors)
            
            if bullish_score > bearish_score:
                confluence_score = bullish_count
                primary_direction = "LONG"
            elif bearish_score > bullish_score:
                confluence_score = bearish_count
                primary_direction = "SHORT"
            else:
                confluence_score = 0
                primary_direction = "BEKLE"
            
            confluence_passed = confluence_score >= self.MIN_CONFLUENCE
            
            # ========================================
            # Phase 24: Trend Alignment Check
            # ========================================
            if primary_direction == "LONG":
                trend_aligned = trend_4h in ["UP", "NEUTRAL"] and trend_1d in ["UP", "NEUTRAL"]
            elif primary_direction == "SHORT":
                trend_aligned = trend_4h in ["DOWN", "NEUTRAL"] and trend_1d in ["DOWN", "NEUTRAL"]
            else:
                trend_aligned = False
            
            # 3. Karar (Filtrelerle)
            data_sources = bullish_count + bearish_count + len(risk_factors)
            
            # Confluence ve Trend filter
            if not confluence_passed:
                direction = "BEKLE"
                confidence = 35
                logger.info(f"⚠️ {symbol}: Confluence fail ({confluence_score}/{self.MIN_CONFLUENCE})")
            elif not trend_aligned and primary_direction != "BEKLE":
                direction = "BEKLE"
                confidence = 40
                logger.info(f"⚠️ {symbol}: Trend not aligned (4H:{trend_4h}, 1D:{trend_1d})")
            elif bullish_score > bearish_score + 2.0:
                direction = "LONG"
                confidence = min(90, int(60 + (bullish_score - bearish_score) * 5))
            elif bearish_score > bullish_score + 2.0:
                direction = "SHORT"
                confidence = min(90, int(60 + (bearish_score - bullish_score) * 5))
            else:
                direction = "BEKLE"
                confidence = 40
            
            if confidence < self.MIN_CONFIDENCE:
                direction = "BEKLE"
            
            # 4. Entry/TP/SL
            if direction == "LONG":
                entry = current_price
                sl = current_price * 0.97
                tp1 = current_price * 1.03
                tp2 = current_price * 1.06
                rr = 2.0
            elif direction == "SHORT":
                entry = current_price
                sl = current_price * 1.03
                tp1 = current_price * 0.97
                tp2 = current_price * 0.94
                rr = 2.0
            else:
                entry = current_price
                sl = tp1 = tp2 = 0
                rr = 0
            
            # ========================================
            # Phase 23: Self-Learning Feedback
            # ========================================
            past_perf = await self._get_past_performance()
            
            # 5. Claude AI Analizi (with past performance)
            claude_analysis = ""
            if direction != "BEKLE":
                try:
                    llm = await self._get_llm_brain()
                    if llm.is_enabled:
                        # Past performance bilgisini ekle
                        onchain_data = {
                            'whale_flow': snapshot.whale_net_flow,
                            'funding_rate': funding,
                            'long_short_ratio': ls_ratio
                        }
                        
                        if past_perf:
                            onchain_data['past_win_rate'] = past_perf.get('recent_win_rate', 0)
                            onchain_data['winning_factors'] = str(past_perf.get('winning_factors', []))
                            onchain_data['losing_factors'] = str(past_perf.get('losing_factors', []))
                        
                        analysis = await llm.analyze(
                            symbol=symbol,
                            current_price=current_price,
                            technical_data={
                                'orderbook_score': snapshot.orderbook_imbalance,
                                'cvd_trend': snapshot.cvd_trend,
                                'taker_ratio': snapshot.taker_buy_ratio,
                                'trend_4h': trend_4h,
                                'trend_1d': trend_1d,
                                'confluence_score': confluence_score
                            },
                            macro_data={
                                'fear_greed_index': fng_value,
                                'btc_dominance': 0
                            },
                            onchain_data=onchain_data
                        )
                        if analysis and analysis.reasoning:
                            claude_analysis = analysis.reasoning
                except Exception as e:
                    logger.debug(f"Claude analysis skipped: {e}")
            
            if not claude_analysis:
                if direction == "LONG":
                    claude_analysis = f"Confluence: {confluence_score} faktör. Trend: 4H={trend_4h}, 1D={trend_1d}."
                elif direction == "SHORT":
                    claude_analysis = f"Confluence: {confluence_score} faktör. Trend: 4H={trend_4h}, 1D={trend_1d}."
                else:
                    claude_analysis = f"Confluence veya trend koşulları sağlanmadı."
            
            return PremiumSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                entry_price=entry,
                stop_loss=sl,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk_reward=rr,
                bullish_factors=bullish_factors,
                bearish_factors=bearish_factors,
                risk_factors=risk_factors,
                confluence_score=confluence_score,
                confluence_passed=confluence_passed,
                trend_4h=trend_4h,
                trend_1d=trend_1d,
                trend_aligned=trend_aligned,
                claude_analysis=claude_analysis,
                data_sources_count=data_sources,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Premium signal generation error: {e}")
            return None
    
    def format_telegram_message(self, signal: PremiumSignal) -> str:
        """Premium sinyal Telegram formatı - Enhanced."""
        if signal.direction == "LONG":
            direction_emoji = "🟢"
            direction_text = "LONG"
        elif signal.direction == "SHORT":
            direction_emoji = "🔴"
            direction_text = "SHORT"
        else:
            direction_emoji = "⏸️"
            direction_text = "BEKLE"
        
        conf_bars = "█" * (signal.confidence // 10) + "░" * (10 - signal.confidence // 10)
        
        # Confluence ve Trend göstergeleri
        confluence_status = "✅" if signal.confluence_passed else "❌"
        trend_status = "✅" if signal.trend_aligned else "❌"
        
        bullish_text = ""
        if signal.bullish_factors:
            for f in signal.bullish_factors[:5]:
                bullish_text += f"  • {f}\n"
        
        bearish_text = ""
        if signal.bearish_factors:
            for f in signal.bearish_factors[:5]:
                bearish_text += f"  • {f}\n"
        
        risk_text = ""
        if signal.risk_factors:
            for f in signal.risk_factors[:3]:
                risk_text += f"  • {f}\n"
        
        if signal.direction != "BEKLE":
            levels_text = f"""
📍 *TRADE PLAN:*
  Entry: ${signal.entry_price:,.2f}
  TP1: ${signal.take_profit_1:,.2f} ({((signal.take_profit_1/signal.entry_price-1)*100):+.1f}%)
  TP2: ${signal.take_profit_2:,.2f} ({((signal.take_profit_2/signal.entry_price-1)*100):+.1f}%)
  SL: ${signal.stop_loss:,.2f} ({((signal.stop_loss/signal.entry_price-1)*100):+.1f}%)
"""
        else:
            levels_text = "\n⏳ *Giriş yok - Filtreler geçmedi*\n"
        
        msg = f"""🧠 *DEMIR AI PREMIUM*
━━━━━━━━━━━━━━━━━━━━

{direction_emoji} *{signal.symbol}* → *{direction_text}*
🎯 Güven: *%{signal.confidence}* [{conf_bars}]
💰 Fiyat: ${signal.entry_price:,.2f}

📊 *FİLTRELER:*
  {confluence_status} Confluence: {signal.confluence_score}/3 faktör
  {trend_status} Trend: 4H={signal.trend_4h} | 1D={signal.trend_1d}
{levels_text}
"""
        
        if bullish_text:
            msg += f"🟢 *YUKARI:*\n{bullish_text}\n"
        
        if bearish_text:
            msg += f"🔴 *AŞAĞI:*\n{bearish_text}\n"
        
        if risk_text:
            msg += f"⚠️ *RİSK:*\n{risk_text}\n"
        
        msg += f"""━━━━━━━━━━━━━━━━━━━━
🤖 *CLAUDE:*
_{signal.claude_analysis[:180]}{'...' if len(signal.claude_analysis) > 180 else ''}_

📊 Kaynak: {signal.data_sources_count} veri
⏰ {signal.timestamp.strftime('%H:%M:%S')}"""
        
        return msg


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_generator: Optional[PremiumSignalGenerator] = None

def get_premium_generator() -> PremiumSignalGenerator:
    global _generator
    if _generator is None:
        _generator = PremiumSignalGenerator()
    return _generator


async def send_premium_signal(symbol: str = "BTCUSDT") -> bool:
    from src.utils.notifications import NotificationManager
    
    generator = get_premium_generator()
    notifier = NotificationManager()
    signal = await generator.generate(symbol)
    
    if signal:
        msg = generator.format_telegram_message(signal)
        await notifier.send_message_raw(msg)
        logger.info(f"📨 Premium: {symbol} → {signal.direction} (Conf:{signal.confluence_score}, Trend:{signal.trend_4h}/{signal.trend_1d})")
        return True
    return False


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        generator = get_premium_generator()
        signal = await generator.generate("BTCUSDT")
        if signal:
            print(generator.format_telegram_message(signal))
    
    asyncio.run(test())

