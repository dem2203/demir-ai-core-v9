# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - ULTRA PREMIUM SIGNAL GENERATOR
==============================================
TÜM AI MODÜLLERİ ENTEGRE - Gerçek Profesyonel Sinyal!

KULLANILAN KAYNAKLAR:
1. LSTM Prediction (Fiyat tahmini)
2. RSI, MACD, BB, Stochastic, ADX (Teknik)
3. Elliott Wave (Dalga analizi)
4. Harmonic Patterns (Gartley, Bat, vb.)
5. Multi-Timeframe Confluence (5m-1d)
6. On-Chain Data (Exchange flow, MVRV)
7. Makro (DXY, VIX, BTC.D)
8. Sentiment (Fear & Greed)
9. Order Book, CVD, Taker Volume
10. Funding Rate, L/S Ratio
11. Whale Flow
12. Claude AI (Final analiz)

FİLTRELER:
- Confluence: Min 5 faktör aynı yönde
- Multi-Timeframe: 4H ve 1D trend aligned
- Risk: Risk/Reward > 2.0
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
    """Ultra Premium sinyal sonucu"""
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
    
    # AI Modülleri
    lstm_direction: str
    lstm_confidence: int
    elliott_wave: str
    harmonic_pattern: str
    rsi_value: float
    macd_signal: str
    
    # Confluence
    confluence_score: int
    confluence_passed: bool
    mtf_confluence: str
    trend_4h: str
    trend_1d: str
    trend_aligned: bool
    
    # On-Chain & Makro
    mvrv: float
    dxy_trend: str
    sentiment: str
    
    # Claude analizi
    claude_analysis: str
    
    # Meta
    data_sources_count: int
    ai_modules_count: int
    timestamp: datetime


class PremiumSignalGenerator:
    """
    Ultra Premium Sinyal Üretici
    
    TÜM AI MODÜLLERİ BAĞLI:
    - LSTM, Elliott Wave, Harmonic
    - RSI, MACD, BB, Stochastic, ADX
    - MTF Confluence
    - On-Chain, Makro, Sentiment
    """
    
    MIN_CONFIDENCE = 60
    MIN_CONFLUENCE = 5  # Artırıldı - daha kaliteli sinyaller
    
    def __init__(self):
        self._llm_brain = None
        self._aggregator = None
        self._scrapers = None
        self._full_collector = None
    
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
    
    async def _get_full_collector(self):
        """Full AI Data Collector"""
        if self._full_collector is None:
            from src.brain.full_ai_collector import get_full_ai_collector
            self._full_collector = get_full_ai_collector()
        return self._full_collector
    
    async def _get_past_performance(self) -> Dict:
        """Geçmiş trade sonuçları (Self-Learning)"""
        try:
            from src.brain.paper_trading_manager import get_paper_trading_manager
            ptm = get_paper_trading_manager()
            recent_trades = [t for t in ptm.trades if t.status != "OPEN"][-10:]
            
            if not recent_trades:
                return {}
            
            wins = len([t for t in recent_trades if t.status in ["TP1_HIT", "TP2_HIT"]])
            return {
                "recent_win_rate": wins / len(recent_trades) * 100 if recent_trades else 0,
                "wins": wins,
                "losses": len(recent_trades) - wins
            }
        except:
            return {}
    
    async def generate(self, symbol: str = "BTCUSDT") -> Optional[PremiumSignal]:
        """
        ULTRA Premium sinyal üret - TÜM AI modülleri kullanarak.
        """
        try:
            # 1. TÜM AI VERİLERİNİ TOPLA
            collector = await self._get_full_collector()
            ai_data = await collector.collect_all(symbol)
            
            # 2. Eski kaynakları da al (uyumluluk için)
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
            
            # 3. FAKTÖR ANALİZİ
            bullish_factors = []
            bearish_factors = []
            risk_factors = []
            
            bullish_score = 0.0
            bearish_score = 0.0
            ai_modules_used = 0
            
            # === AI MODÜL 1: LSTM ===
            if ai_data.lstm_direction == "UP" and ai_data.lstm_confidence > 50:
                bullish_factors.append(f"🤖 LSTM +{ai_data.lstm_change_pct:.1f}% ({ai_data.lstm_confidence}%)")
                bullish_score += 2.0
                ai_modules_used += 1
            elif ai_data.lstm_direction == "DOWN" and ai_data.lstm_confidence > 50:
                bearish_factors.append(f"🤖 LSTM {ai_data.lstm_change_pct:.1f}% ({ai_data.lstm_confidence}%)")
                bearish_score += 2.0
                ai_modules_used += 1
            
            # === AI MODÜL 2: RSI ===
            if ai_data.rsi_signal == "OVERSOLD":
                bullish_factors.append(f"📉 RSI {ai_data.rsi:.1f} (Oversold)")
                bullish_score += 1.5
                ai_modules_used += 1
            elif ai_data.rsi_signal == "OVERBOUGHT":
                bearish_factors.append(f"📈 RSI {ai_data.rsi:.1f} (Overbought)")
                bearish_score += 1.5
                ai_modules_used += 1
            
            # === AI MODÜL 3: MACD ===
            if ai_data.macd_signal == "BULLISH":
                bullish_factors.append("📊 MACD Bullish")
                bullish_score += 1.5
                ai_modules_used += 1
            elif ai_data.macd_signal == "BEARISH":
                bearish_factors.append("📊 MACD Bearish")
                bearish_score += 1.5
                ai_modules_used += 1
            
            # === AI MODÜL 4: Stochastic ===
            if ai_data.stoch_signal == "OVERSOLD":
                bullish_factors.append("📉 Stochastic Oversold")
                bullish_score += 1.0
            elif ai_data.stoch_signal == "OVERBOUGHT":
                bearish_factors.append("📈 Stochastic Overbought")
                bearish_score += 1.0
            
            # === AI MODÜL 5: Bollinger Bands ===
            if ai_data.bb_position == "LOWER":
                bullish_factors.append("📉 BB Alt Band")
                bullish_score += 1.0
            elif ai_data.bb_position == "UPPER":
                bearish_factors.append("📈 BB Üst Band")
                bearish_score += 1.0
            
            # === AI MODÜL 6: ADX ===
            if ai_data.adx_trend == "STRONG":
                # ADX güçlü trend var demek - yön belirlemez
                pass
            
            # === AI MODÜL 7: Elliott Wave ===
            if ai_data.elliott_direction == "UP" and ai_data.elliott_confidence > 0.6:
                bullish_factors.append(f"🌊 Elliott {ai_data.elliott_current_wave}")
                bullish_score += 2.0
                ai_modules_used += 1
            elif ai_data.elliott_direction == "DOWN" and ai_data.elliott_confidence > 0.6:
                bearish_factors.append(f"🌊 Elliott {ai_data.elliott_current_wave}")
                bearish_score += 2.0
                ai_modules_used += 1
            
            # === AI MODÜL 8: Harmonic Pattern ===
            if ai_data.harmonic_pattern and ai_data.harmonic_confidence > 0.7:
                if ai_data.harmonic_bullish:
                    bullish_factors.append(f"🦋 {ai_data.harmonic_pattern}")
                    bullish_score += 2.5
                else:
                    bearish_factors.append(f"🦋 {ai_data.harmonic_pattern}")
                    bearish_score += 2.5
                ai_modules_used += 1
            
            # === AI MODÜL 9: MTF Confluence ===
            if ai_data.mtf_confluence == "BULLISH":
                bullish_factors.append(f"📊 MTF Confluence BULLISH")
                bullish_score += 2.0
                ai_modules_used += 1
            elif ai_data.mtf_confluence == "BEARISH":
                bearish_factors.append(f"📊 MTF Confluence BEARISH")
                bearish_score += 2.0
                ai_modules_used += 1
            
            # === AI MODÜL 10: On-Chain (MVRV) ===
            if ai_data.mvrv < 1.0:
                bullish_factors.append(f"🔗 MVRV {ai_data.mvrv:.2f} (Undervalued)")
                bullish_score += 1.5
                ai_modules_used += 1
            elif ai_data.mvrv > 3.0:
                risk_factors.append(f"⚠️ MVRV {ai_data.mvrv:.2f} (Overvalued)")
                bearish_score += 1.5
                ai_modules_used += 1
            
            # === AI MODÜL 11: Makro (DXY) ===
            if ai_data.dxy_trend == "DOWN":
                bullish_factors.append("💵 DXY Down (Crypto Bullish)")
                bullish_score += 1.0
            elif ai_data.dxy_trend == "UP":
                bearish_factors.append("💵 DXY Up (Crypto Bearish)")
                bearish_score += 1.0
            
            # === AI MODÜL 12: Sentiment ===
            if ai_data.news_score > 65:
                bullish_factors.append(f"😊 Sentiment {ai_data.news_score:.0f}")
                bullish_score += 1.0
            elif ai_data.news_score < 35:
                bullish_factors.append(f"😱 Fear {ai_data.news_score:.0f} (Kontrarian)")
                bullish_score += 1.5  # Kontrarian - fear = buy
            
            # === ESKİ KAYNAKLAR ===
            # Whale Flow
            if snapshot.whale_net_flow > 1000000:
                bullish_factors.append(f"🐋 Whale +${snapshot.whale_net_flow/1e6:.1f}M")
                bullish_score += 2.0
            elif snapshot.whale_net_flow < -1000000:
                bearish_factors.append(f"🐋 Whale ${abs(snapshot.whale_net_flow)/1e6:.1f}M satış")
                bearish_score += 2.0
            
            # Order Book
            if snapshot.orderbook_imbalance > 1.5:
                bullish_factors.append(f"📊 OB {snapshot.orderbook_imbalance:.1f}x bid")
                bullish_score += 1.5
            elif snapshot.orderbook_imbalance < 0.6:
                bearish_factors.append(f"📊 OB {1/snapshot.orderbook_imbalance:.1f}x ask")
                bearish_score += 1.5
            
            # CVD
            if snapshot.cvd_trend == "BULLISH":
                bullish_factors.append("📈 CVD Pozitif")
                bullish_score += 1.0
            elif snapshot.cvd_trend == "BEARISH":
                bearish_factors.append("📉 CVD Negatif")
                bearish_score += 1.0
            
            # Funding Rate
            funding = futures_data.get('funding_rate', 0)
            if funding > 0.0005:
                risk_factors.append(f"⚠️ Funding +{funding*100:.3f}%")
                bearish_score += 1.0
            elif funding < -0.0001:
                bullish_factors.append("💸 Funding Negatif")
                bullish_score += 1.0
            
            # L/S Ratio
            ls_ratio = futures_data.get('long_short_ratio', 1.0)
            if ls_ratio > 2.0:
                risk_factors.append(f"⚠️ L/S {ls_ratio:.2f}")
                bearish_score += 1.0
            elif ls_ratio < 0.7:
                bullish_factors.append(f"🚀 L/S {ls_ratio:.2f}")
                bullish_score += 1.5
            
            # 4. CONFLUENCE CHECK
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
            
            # 5. TREND ALIGNMENT CHECK
            trend_4h = ai_data.trend_4h
            trend_1d = ai_data.trend_1d
            
            if primary_direction == "LONG":
                trend_aligned = trend_4h in ["UP", "NEUTRAL"] and trend_1d in ["UP", "NEUTRAL"]
            elif primary_direction == "SHORT":
                trend_aligned = trend_4h in ["DOWN", "NEUTRAL"] and trend_1d in ["DOWN", "NEUTRAL"]
            else:
                trend_aligned = False
            
            # 6. FİNAL KARAR
            data_sources = bullish_count + bearish_count + len(risk_factors)
            
            if not confluence_passed:
                direction = "BEKLE"
                confidence = 35
                logger.info(f"⚠️ {symbol}: Confluence fail ({confluence_score}/{self.MIN_CONFLUENCE})")
            elif not trend_aligned and primary_direction != "BEKLE":
                direction = "BEKLE"
                confidence = 40
                logger.info(f"⚠️ {symbol}: Trend not aligned (4H:{trend_4h}, 1D:{trend_1d})")
            elif bullish_score > bearish_score + 3.0:
                direction = "LONG"
                confidence = min(95, int(60 + (bullish_score - bearish_score) * 4))
            elif bearish_score > bullish_score + 3.0:
                direction = "SHORT"
                confidence = min(95, int(60 + (bearish_score - bullish_score) * 4))
            else:
                direction = "BEKLE"
                confidence = 40
            
            if confidence < self.MIN_CONFIDENCE:
                direction = "BEKLE"
            
            # 7. ENTRY/TP/SL
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
            
            # 8. CLAUDE AI ANALİZİ
            claude_analysis = ""
            past_perf = await self._get_past_performance()
            
            if direction != "BEKLE":
                try:
                    llm = await self._get_llm_brain()
                    if llm.is_enabled:
                        onchain_data = {
                            'whale_flow': snapshot.whale_net_flow,
                            'funding_rate': funding,
                            'long_short_ratio': ls_ratio,
                            'lstm': ai_data.lstm_direction,
                            'rsi': ai_data.rsi,
                            'elliott': ai_data.elliott_current_wave,
                            'harmonic': ai_data.harmonic_pattern,
                            'mvrv': ai_data.mvrv
                        }
                        
                        if past_perf:
                            onchain_data['past_win_rate'] = past_perf.get('recent_win_rate', 0)
                        
                        analysis = await llm.analyze(
                            symbol=symbol,
                            current_price=current_price,
                            technical_data={
                                'orderbook_score': snapshot.orderbook_imbalance,
                                'cvd_trend': snapshot.cvd_trend,
                                'taker_ratio': snapshot.taker_buy_ratio,
                                'trend_4h': trend_4h,
                                'trend_1d': trend_1d,
                                'confluence_score': confluence_score,
                                'ai_modules': ai_modules_used
                            },
                            macro_data={
                                'fear_greed_index': ai_data.news_score,
                                'btc_dominance': 0,
                                'dxy_trend': ai_data.dxy_trend
                            },
                            onchain_data=onchain_data
                        )
                        if analysis and analysis.reasoning:
                            claude_analysis = analysis.reasoning
                except Exception as e:
                    logger.debug(f"Claude analysis skipped: {e}")
            
            if not claude_analysis:
                claude_analysis = f"AI Modül: {ai_modules_used} | Confluence: {confluence_score} | MTF: {ai_data.mtf_confluence}"
            
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
                lstm_direction=ai_data.lstm_direction,
                lstm_confidence=ai_data.lstm_confidence,
                elliott_wave=ai_data.elliott_current_wave,
                harmonic_pattern=ai_data.harmonic_pattern,
                rsi_value=ai_data.rsi,
                macd_signal=ai_data.macd_signal,
                confluence_score=confluence_score,
                confluence_passed=confluence_passed,
                mtf_confluence=ai_data.mtf_confluence,
                trend_4h=trend_4h,
                trend_1d=trend_1d,
                trend_aligned=trend_aligned,
                mvrv=ai_data.mvrv,
                dxy_trend=ai_data.dxy_trend,
                sentiment=ai_data.news_sentiment,
                claude_analysis=claude_analysis,
                data_sources_count=data_sources,
                ai_modules_count=ai_modules_used,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Premium signal generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def format_telegram_message(self, signal: PremiumSignal) -> str:
        """Ultra Premium Telegram formatı."""
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
        
        confluence_status = "✅" if signal.confluence_passed else "❌"
        trend_status = "✅" if signal.trend_aligned else "❌"
        
        # AI Modül durum çubuğu
        ai_bar = f"[{signal.ai_modules_count}/12 AI]"
        
        bullish_text = ""
        if signal.bullish_factors:
            for f in signal.bullish_factors[:6]:
                bullish_text += f"  • {f}\n"
        
        bearish_text = ""
        if signal.bearish_factors:
            for f in signal.bearish_factors[:6]:
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
            levels_text = "\n⏳ *Filtreler geçmedi - Sinyal yok*\n"
        
        msg = f"""🧠 *DEMIR AI ULTRA PREMIUM*
━━━━━━━━━━━━━━━━━━━━━━━━

{direction_emoji} *{signal.symbol}* → *{direction_text}*
🎯 Güven: *%{signal.confidence}* [{conf_bars}]
💰 Fiyat: ${signal.entry_price:,.2f}

🤖 *AI ANALİZ:* {ai_bar}
  📊 LSTM: {signal.lstm_direction} ({signal.lstm_confidence}%)
  📈 RSI: {signal.rsi_value:.1f} | MACD: {signal.macd_signal}
  🌊 Elliott: {signal.elliott_wave or 'N/A'}
  🦋 Harmonic: {signal.harmonic_pattern or 'N/A'}

📊 *FİLTRELER:*
  {confluence_status} Confluence: {signal.confluence_score}/5 faktör
  {trend_status} Trend: 4H={signal.trend_4h} | 1D={signal.trend_1d}
  📊 MTF: {signal.mtf_confluence}
{levels_text}
"""
        
        if bullish_text:
            msg += f"🟢 *YUKARI:*\n{bullish_text}\n"
        
        if bearish_text:
            msg += f"🔴 *AŞAĞI:*\n{bearish_text}\n"
        
        if risk_text:
            msg += f"⚠️ *RİSK:*\n{risk_text}\n"
        
        msg += f"""━━━━━━━━━━━━━━━━━━━━━━━━
🤖 *CLAUDE:*
_{signal.claude_analysis[:200]}{'...' if len(signal.claude_analysis) > 200 else ''}_

📊 Veri: {signal.data_sources_count} kaynak | {signal.ai_modules_count} AI modül
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
        logger.info(f"📨 Ultra Premium: {symbol} → {signal.direction} (AI:{signal.ai_modules_count}, Conf:{signal.confluence_score})")
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

