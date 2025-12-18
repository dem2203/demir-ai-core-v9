# -*- coding: utf-8 -*-
"""
DEMIR AI - Signal Orchestrator
Tüm veri kaynaklarını ve modelleri birleştirip güçlü sinyal üretir.

PHASE 49: Advanced Signal Fusion
- Tüm modüllerden sinyal toplama
- Dinamik ağırlık hesaplama
- Consensus mechanism
- Final güçlü sinyal üretimi
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Fix event loop conflicts
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

logger = logging.getLogger("SIGNAL_ORCHESTRATOR")


@dataclass
class ModuleSignal:
    """Bir modülden gelen sinyal"""
    module_name: str
    direction: str  # LONG / SHORT / NEUTRAL
    confidence: float  # 0-100
    weight: float  # 0-1 (ağırlık)
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FinalSignal:
    """Birleştirilmiş final sinyal"""
    direction: str  # LONG / SHORT / WAIT
    confidence: float  # 0-100
    strength: str  # STRONG / MODERATE / WEAK
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    contributing_modules: List[str]
    consensus_ratio: float  # Kaç modül aynı fikirde
    timestamp: datetime = field(default_factory=datetime.now)


class SignalOrchestrator:
    """
    Merkezi Sinyal Orkestratörü
    
    Tüm modülleri koordine eder:
    1. Markov Predictor
    2. LSTM Trend (trained model)
    3. Research Agent
    4. SMC Analyzer
    5. Whale Intelligence
    6. Liquidation Hunter
    7. Predictive Analyzer
    8. News Sentiment (YENİ)
    9. CME Gap Tracker (YENİ)
    10. Options Flow (YENİ)
    11. OnChain Intel (YENİ)
    
    Consensus kuralları:
    - Minimum 5/11 modül aynı yönde olmalı
    - Güven skoru > 65% olmalı
    - Risk/Reward > 2:1 olmalı
    """
    
    # Modül ağırlıkları (performansa göre ayarlanabilir)
    DEFAULT_WEIGHTS = {
        'MarkovPredictor': 0.08,
        'LSTMTrend': 0.10,
        'ResearchAgent': 0.10,
        'SMCAnalyzer': 0.08,
        'WhaleIntelligence': 0.08,
        'LiquidationHunter': 0.06,
        'PredictiveAnalyzer': 0.06,
        'NewsSentiment': 0.08,
        'CMEGapTracker': 0.08,
        'OptionsFlow': 0.06,
        'OnChainIntel': 0.06,
        'TradingViewTA': 0.16,  # NEW - High weight for proven data source
    }
    
    # Minimum sinyal gereksinimleri
    MIN_CONSENSUS_RATIO = 0.5  # En az %50 aynı fikirde (11 modül için)
    MIN_CONFIDENCE = 60  # Minimum güven
    MIN_RISK_REWARD = 2.0  # Minimum R:R
    
    def __init__(self):
        self.module_signals: List[ModuleSignal] = []
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.last_signal: Optional[FinalSignal] = None
        self.signal_history: List[FinalSignal] = []
    
    async def collect_all_signals(self, symbol: str = 'BTCUSDT', current_price: float = 0) -> List[ModuleSignal]:
        """Tüm modüllerden sinyal topla."""
        self.module_signals = []
        import requests
        
        # 1. Markov Predictor
        try:
            from src.brain.markov_predictor import MarkovPredictor
            markov = MarkovPredictor()
            
            resp = requests.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=2", timeout=5)
            if resp.status_code == 200:
                klines = resp.json()
                if len(klines) >= 2:
                    pct_change = ((float(klines[-1][4]) / float(klines[-2][4])) - 1) * 100
                    pred = markov.predict_1_2_hours(pct_change)
                    
                    self.module_signals.append(ModuleSignal(
                        module_name='MarkovPredictor',
                        direction=pred['combined_signal'].replace('WAIT', 'NEUTRAL'),
                        confidence=pred['signal_strength'],
                        weight=self.weights['MarkovPredictor'],
                        reasoning=f"1h: {pred['1_hour']['direction']}, 2h: {pred['2_hour']['direction']}"
                    ))
        except Exception as e:
            logger.warning(f"Markov signal failed: {e}")
        
        # 2. LSTM Trend (with trained model)
        try:
            from src.brain.models.lstm_trend import LSTMTrendPredictor
            import pandas as pd
            
            lstm = LSTMTrendPredictor()
            
            resp = requests.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=24", timeout=5)
            if resp.status_code == 200:
                klines = resp.json()
                df = pd.DataFrame({'close': [float(k[4]) for k in klines]})
                pred = lstm.predict(df)
                
                dir_map = {'UP': 'LONG', 'DOWN': 'SHORT', 'NEUTRAL': 'NEUTRAL'}
                self.module_signals.append(ModuleSignal(
                    module_name='LSTMTrend',
                    direction=dir_map.get(pred['direction'], 'NEUTRAL'),
                    confidence=pred['confidence'],
                    weight=self.weights['LSTMTrend'],
                    reasoning=f"Model: {pred.get('model', 'trained')}"
                ))
        except Exception as e:
            logger.warning(f"LSTM signal failed: {e}")
        
        # 3. Research Agent
        try:
            from src.brain.research_agent import ResearchAgent
            agent = ResearchAgent()
            
            research = await agent.research_coin(symbol)
            
            dir_map = {'BULLISH': 'LONG', 'BEARISH': 'SHORT', 'NEUTRAL': 'NEUTRAL'}
            self.module_signals.append(ModuleSignal(
                module_name='ResearchAgent',
                direction=dir_map.get(research.overall_bias, 'NEUTRAL'),
                confidence=research.overall_confidence,
                weight=self.weights['ResearchAgent'],
                reasoning=f"{len(research.findings)} bulgu analiz edildi"
            ))
        except Exception as e:
            logger.warning(f"Research signal failed: {e}")
        
        # 4. Whale Intelligence (YENİ - Binance API)
        try:
            from src.brain.whale_intelligence import WhaleIntelligence
            whale = WhaleIntelligence()
            whale_signal = whale.get_signal_for_orchestrator(symbol)
            
            if whale_signal.get('confidence', 0) > 0:
                self.module_signals.append(ModuleSignal(
                    module_name='WhaleIntelligence',
                    direction=whale_signal['direction'],
                    confidence=whale_signal.get('confidence', 50),
                    weight=self.weights['WhaleIntelligence'],
                    reasoning=whale_signal.get('reason', 'Whale analizi')
                ))
        except Exception as e:
            logger.warning(f"Whale signal failed: {e}")
        
        # 5. Predictive Analyzer
        try:
            from src.brain.predictive_analyzer import PredictiveAnalyzer
            predictor = PredictiveAnalyzer()
            
            async with asyncio.timeout(10):
                pred = await predictor.analyze_predictive_signals(symbol, current_price)
                
                if pred.get('has_signal'):
                    direction = 'LONG' if 'LONG' in pred.get('signal_type', '') else 'SHORT'
                    self.module_signals.append(ModuleSignal(
                        module_name='PredictiveAnalyzer',
                        direction=direction,
                        confidence=pred.get('confidence', 50),
                        weight=self.weights['PredictiveAnalyzer'],
                        reasoning=', '.join(pred.get('reasons', []))
                    ))
        except Exception as e:
            logger.warning(f"Predictive signal failed: {e}")
        
        # 6. Liquidation Hunter
        try:
            from src.brain.liquidation_hunter import LiquidationHunter
            hunter = LiquidationHunter()
            
            liq_data = await hunter.get_full_liquidation_analysis(symbol)
            
            if liq_data:
                long_liq = liq_data.get('long_liquidations', 0)
                short_liq = liq_data.get('short_liquidations', 0)
                
                if long_liq > short_liq * 1.5:
                    direction = 'SHORT'
                    confidence = 55
                elif short_liq > long_liq * 1.5:
                    direction = 'LONG'
                    confidence = 55
                else:
                    direction = 'NEUTRAL'
                    confidence = 30
                
                self.module_signals.append(ModuleSignal(
                    module_name='LiquidationHunter',
                    direction=direction,
                    confidence=confidence,
                    weight=self.weights['LiquidationHunter'],
                    reasoning=liq_data.get('interpretation', 'Likidasyon analizi')
                ))
            
            await hunter.close()
        except Exception as e:
            logger.warning(f"Liquidation signal failed: {e}")
        
        # 7. NEWS SENTIMENT - Always add signal
        try:
            from src.brain.news_scraper import CryptoNewsScraper
            scraper = CryptoNewsScraper()
            scraper.fetch_all_news(max_age_hours=4)
            sentiment = scraper.get_market_sentiment()
            
            dir_map = {'BULLISH': 'LONG', 'BEARISH': 'SHORT', 'NEUTRAL': 'NEUTRAL'}
            self.module_signals.append(ModuleSignal(
                module_name='NewsSentiment',
                direction=dir_map.get(sentiment['overall'], 'NEUTRAL'),
                confidence=sentiment.get('confidence', 40),
                weight=self.weights['NewsSentiment'],
                reasoning=f"{sentiment['bullish_count']} bullish, {sentiment['bearish_count']} bearish haber"
            ))
        except Exception as e:
            logger.warning(f"News signal failed: {e}")
        
        # 8. CME GAP TRACKER (YENİ)
        try:
            from src.brain.cme_gap_tracker import CMEGapTracker
            tracker = CMEGapTracker()
            gap_signal = tracker.get_signal_bias()
            
            if gap_signal.get('has_gap') and gap_signal.get('signal') != 'NEUTRAL':
                self.module_signals.append(ModuleSignal(
                    module_name='CMEGapTracker',
                    direction=gap_signal['signal'],
                    confidence=gap_signal.get('confidence', 50),
                    weight=self.weights['CMEGapTracker'],
                    reasoning=gap_signal.get('reason', 'CME Gap analizi')
                ))
        except Exception as e:
            logger.warning(f"CME Gap signal failed: {e}")
        
        # 9. OPTIONS FLOW - Always add signal
        try:
            from src.brain.options_flow import OptionsFlowAnalyzer
            analyzer = OptionsFlowAnalyzer()
            options = analyzer.get_signal_for_orchestrator()
            
            self.module_signals.append(ModuleSignal(
                module_name='OptionsFlow',
                direction=options.get('direction', 'NEUTRAL'),
                confidence=options.get('confidence', 40),
                weight=self.weights['OptionsFlow'],
                reasoning=options.get('reason', 'Options flow analizi')
            ))
        except Exception as e:
            logger.warning(f"Options signal failed: {e}")
        
        # 10. ON-CHAIN INTEL - Sync version (avoid async issues)
        try:
            from src.brain.whale_intelligence import WhaleIntelligence
            onchain = WhaleIntelligence()  # Use Binance API instead
            analysis = onchain.get_full_whale_analysis(symbol)
            
            if analysis.get('available'):
                self.module_signals.append(ModuleSignal(
                    module_name='OnChainIntel',
                    direction=analysis.get('whale_bias', 'NEUTRAL'),
                    confidence=analysis.get('confidence', 40),
                    weight=self.weights['OnChainIntel'],
                    reasoning=f"L/S: {analysis.get('long_short_ratio', 1):.2f}, Funding: {analysis.get('funding_rate_pct', 0):.3f}%"
                ))
        except Exception as e:
            logger.warning(f"OnChain signal failed: {e}")
        
        # 11. TRADINGVIEW TECHNICAL ANALYSIS (YENİ - En güvenilir kaynak)
        try:
            from src.brain.tv_playwright import TradingViewPlaywright
            tv = TradingViewPlaywright()
            tv_signal = tv.get_signal_for_orchestrator(symbol)
            
            if tv_signal.get('confidence', 0) > 0:
                self.module_signals.append(ModuleSignal(
                    module_name='TradingViewTA',
                    direction=tv_signal['direction'],
                    confidence=tv_signal.get('confidence', 50),
                    weight=self.weights['TradingViewTA'],
                    reasoning=tv_signal.get('reason', 'TradingView Technical Analysis')
                ))
        except Exception as e:
            logger.warning(f"TradingView signal failed: {e}")
        
        logger.info(f"Collected {len(self.module_signals)} signals from {len(self.weights)} modules")
        return self.module_signals
    
    def calculate_consensus(self) -> Tuple[str, float, float]:
        """Konsensüs hesapla."""
        if not self.module_signals:
            return 'WAIT', 0, 0
        
        long_score = 0
        short_score = 0
        neutral_score = 0
        total_weight = 0
        
        for sig in self.module_signals:
            total_weight += sig.weight
            weighted = sig.weight * (sig.confidence / 100)
            
            if sig.direction == 'LONG':
                long_score += weighted
            elif sig.direction == 'SHORT':
                short_score += weighted
            else:
                neutral_score += weighted
        
        if total_weight == 0:
            return 'WAIT', 0, 0
        
        # Normalize
        long_pct = long_score / total_weight
        short_pct = short_score / total_weight
        
        # Determine winner
        if long_pct > short_pct and long_pct > 0.4:
            consensus_dir = 'LONG'
            consensus_ratio = long_pct
        elif short_pct > long_pct and short_pct > 0.4:
            consensus_dir = 'SHORT'
            consensus_ratio = short_pct
        else:
            consensus_dir = 'WAIT'
            consensus_ratio = max(long_pct, short_pct)
        
        # Ortalama güven
        avg_confidence = sum(s.confidence * s.weight for s in self.module_signals) / total_weight
        
        return consensus_dir, consensus_ratio * 100, avg_confidence
    
    def generate_final_signal(self, current_price: float) -> Optional[FinalSignal]:
        """Final sinyal üret."""
        direction, consensus_ratio, avg_confidence = self.calculate_consensus()
        
        # Minimum gereksinimleri kontrol et
        if direction == 'WAIT':
            logger.info("No clear consensus - WAIT signal")
            return None
        
        if consensus_ratio < self.MIN_CONSENSUS_RATIO * 100:
            logger.info(f"Consensus too low: {consensus_ratio:.0f}% < {self.MIN_CONSENSUS_RATIO*100}%")
            return None
        
        if avg_confidence < self.MIN_CONFIDENCE:
            logger.info(f"Confidence too low: {avg_confidence:.0f}% < {self.MIN_CONFIDENCE}%")
            return None
        
        # Entry, SL, TP hesapla
        if direction == 'LONG':
            entry = current_price
            stop_loss = current_price * 0.98  # %2 SL
            take_profit = current_price * 1.04  # %4 TP
        else:  # SHORT
            entry = current_price
            stop_loss = current_price * 1.02  # %2 SL
            take_profit = current_price * 0.96  # %4 TP
        
        risk_reward = abs(take_profit - entry) / abs(entry - stop_loss) if entry != stop_loss else 0
        
        # R:R kontrolü
        if risk_reward < self.MIN_RISK_REWARD:
            logger.info(f"R:R too low: {risk_reward:.1f} < {self.MIN_RISK_REWARD}")
            return None
        
        # Sinyal gücü
        if avg_confidence > 80 and consensus_ratio > 80:
            strength = 'STRONG'
        elif avg_confidence > 65 and consensus_ratio > 65:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'
        
        # Katkıda bulunan modüller
        contributing = [s.module_name for s in self.module_signals if s.direction == direction]
        
        signal = FinalSignal(
            direction=direction,
            confidence=avg_confidence,
            strength=strength,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            contributing_modules=contributing,
            consensus_ratio=consensus_ratio
        )
        
        self.last_signal = signal
        self.signal_history.append(signal)
        
        logger.info(f"🎯 FINAL SIGNAL: {direction} | Confidence: {avg_confidence:.0f}% | R:R: {risk_reward:.1f}")
        
        return signal
    
    async def orchestrate(self, symbol: str = 'BTCUSDT') -> Optional[FinalSignal]:
        """
        Ana orkestrasyon fonksiyonu.
        Tüm süreci yönetir.
        """
        # Güncel fiyatı al
        try:
            import requests
            resp = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5)
            current_price = float(resp.json()['price']) if resp.status_code == 200 else 0
        except:
            current_price = 0
        
        if current_price == 0:
            logger.error("Cannot get current price")
            return None
        
        # Tüm sinyalleri topla
        await self.collect_all_signals(symbol, current_price)
        
        # Final sinyal üret
        return self.generate_final_signal(current_price)
    
    def get_signal_breakdown(self) -> Dict:
        """Sinyal detay raporu."""
        return {
            'signals_collected': len(self.module_signals),
            'modules': [
                {
                    'name': s.module_name,
                    'direction': s.direction,
                    'confidence': s.confidence,
                    'weight': s.weight,
                    'reasoning': s.reasoning
                }
                for s in self.module_signals
            ],
            'consensus': self.calculate_consensus(),
            'last_signal': {
                'direction': self.last_signal.direction if self.last_signal else None,
                'confidence': self.last_signal.confidence if self.last_signal else None,
                'strength': self.last_signal.strength if self.last_signal else None,
            } if self.last_signal else None
        }


# Convenience functions
def get_orchestrated_signal(symbol: str = 'BTCUSDT') -> Optional[Dict]:
    """Senkron orkestre sinyal."""
    orchestrator = SignalOrchestrator()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    signal = loop.run_until_complete(orchestrator.orchestrate(symbol))
    loop.close()
    
    if signal:
        return {
            'direction': signal.direction,
            'confidence': signal.confidence,
            'strength': signal.strength,
            'entry': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'risk_reward': signal.risk_reward,
            'modules': signal.contributing_modules,
            'consensus': signal.consensus_ratio
        }
    return None
