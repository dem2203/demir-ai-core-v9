import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

logger = logging.getLogger("MTF_TRANSFORMER")

class MultiTimeframeTransformer:
    """
    MULTI-TIMEFRAME TRANSFORMER ANALYZER (Çoklu Zaman Dilimi Transformatörü)
    
    Analyzes 15m, 1H, and 4H timeframes simultaneously with weighted attention mechanism.
    (15m, 1H ve 4H zaman dilimlerini ağırlıklı dikkat mekanizmasıyla eşzamanlı analiz eder)
    
    Features:
    - Cross-timeframe pattern coherence analysis
    - Divergence detection across timeframes
    - Regime-adaptive timeframe weighting
    - Multi-tf RSI, MACD, Trend confluence
    """
    
    # Timeframe weights based on market regime
    REGIME_WEIGHTS = {
        'TRENDING_BULL': {'15m': 0.2, '1h': 0.3, '4h': 0.5},  # Favor longer TF in trends
        'TRENDING_BEAR': {'15m': 0.2, '1h': 0.3, '4h': 0.5},
        'RANGING': {'15m': 0.4, '1h': 0.35, '4h': 0.25},  # Favor shorter TF in ranges
        'SCALPER_PARADISE': {'15m': 0.5, '1h': 0.3, '4h': 0.2},
        'ACCUMULATION': {'15m': 0.25, '1h': 0.35, '4h': 0.4},
        'DISTRIBUTION': {'15m': 0.25, '1h': 0.35, '4h': 0.4},
        'DEFAULT': {'15m': 0.3, '1h': 0.35, '4h': 0.35}  # Balanced
    }
    
    @staticmethod
    def analyze_multi_timeframe(
        data_map: Dict[str, pd.DataFrame], 
        regime: str = 'DEFAULT'
    ) -> Dict:
        """
        Comprehensive multi-timeframe analysis with Transformer-like attention
        
        Args:
            data_map: Dict with keys '15m', '1h', '4h' containing DataFrames
            regime: Current market regime for adaptive weighting
            
        Returns:
            Dict with MTF analysis results
        """
        if not all(tf in data_map for tf in ['15m', '1h', '4h']):
            logger.warning("Missing timeframe data for MTF analysis")
            return MultiTimeframeTransformer._empty_analysis()
        
        # Get adaptive weights
        weights = MultiTimeframeTransformer.REGIME_WEIGHTS.get(
            regime, 
            MultiTimeframeTransformer.REGIME_WEIGHTS['DEFAULT']
        )
        
        # Extract last candles
        tf_data = {
            '15m': data_map['15m'].iloc[-1],
            '1h': data_map['1h'].iloc[-1],
            '4h': data_map['4h'].iloc[-1]
        }
        
        # 1. Trend Alignment Analysis
        trend_analysis = MultiTimeframeTransformer._analyze_trend_alignment(tf_data)
        
        # 2. Momentum Confluence (RSI, MACD)
        momentum_analysis = MultiTimeframeTransformer._analyze_momentum_confluence(tf_data)
        
        # 3. Pattern Coherence
        pattern_analysis = MultiTimeframeTransformer._analyze_pattern_coherence(data_map)
        
        # 4. Cross-timeframe Divergences
        divergence_analysis = MultiTimeframeTransformer._detect_cross_tf_divergences(data_map)
        
        # 5. Weighted Decision Fusion
        final_decision = MultiTimeframeTransformer._fuse_decisions(
            trend_analysis,
            momentum_analysis,
            pattern_analysis,
            weights
        )
        
        return {
            'mtf_decision': final_decision['decision'],
            'mtf_confidence': final_decision['confidence'],
            'mtf_score': final_decision['score'],
            'trend_alignment': trend_analysis,
            'momentum_confluence': momentum_analysis,
            'pattern_coherence': pattern_analysis,
            'cross_tf_divergences': divergence_analysis,
            'timeframe_weights': weights,
            'regime': regime
        }
    
    @staticmethod
    def _analyze_trend_alignment(tf_data: Dict) -> Dict:
        """Analyze trend alignment across timeframes"""
        trends = {}
        
        for tf, candle in tf_data.items():
            # Multi-indicator trend detection
            ema_trend = 1 if candle['close'] > candle.get('vwap', candle['close']) else -1
            price_momentum = 1 if candle['close'] > candle['open'] else -1
            
            # Combine
            trend_score = (ema_trend + price_momentum) / 2  # -1 to 1
            
            if trend_score > 0.5:
                trends[tf] = 'BULLISH'
            elif trend_score < -0.5:
                trends[tf] = 'BEARISH'
            else:
                trends[tf] = 'NEUTRAL'
        
        # Check alignment
        if all(t == 'BULLISH' for t in trends.values()):
            alignment = 'STRONG_BULLISH'
            alignment_score = 100
        elif all(t == 'BEARISH' for t in trends.values()):
            alignment = 'STRONG_BEARISH'
            alignment_score = 100
        elif trends['4h'] == trends['1h'] and trends['4h'] != 'NEUTRAL':
            alignment = f"MODERATE_{trends['4h']}"
            alignment_score = 70
        elif trends['4h'] != 'NEUTRAL':
            alignment = f"WEAK_{trends['4h']}"
            alignment_score = 50
        else:
            alignment = 'CONFLICTING'
            alignment_score = 0
        
        return {
            'individual_trends': trends,
            'alignment': alignment,
            'alignment_score': alignment_score
        }
    
    @staticmethod
    def _analyze_momentum_confluence(tf_data: Dict) -> Dict:
        """Analyze RSI and MACD confluence across timeframes"""
        rsi_signals = {}
        macd_signals = {}
        
        for tf, candle in tf_data.items():
            # RSI analysis
            rsi = candle.get('rsi', 50)
            if rsi > 70:
                rsi_signals[tf] = 'OVERBOUGHT'
            elif rsi < 30:
                rsi_signals[tf] = 'OVERSOLD'
            elif rsi > 55:
                rsi_signals[tf] = 'BULLISH'
            elif rsi < 45:
                rsi_signals[tf] = 'BEARISH'
            else:
                rsi_signals[tf] = 'NEUTRAL'
            
            # MACD analysis
            macd_hist = candle.get('macd_hist', 0)
            if macd_hist > 0.001:
                macd_signals[tf] = 'BULLISH'
            elif macd_hist < -0.001:
                macd_signals[tf] = 'BEARISH'
            else:
                macd_signals[tf] = 'NEUTRAL'
        
        # Calculate confluence
        bullish_count = sum(1 for s in rsi_signals.values() if 'BULL' in s or 'OVERSOLD' in s)
        bearish_count = sum(1 for s in rsi_signals.values() if 'BEAR' in s or 'OVERBOUGHT' in s)
        
        if bullish_count == 3:
            confluence = 'STRONG_BULLISH'
            confluence_score = 90
        elif bearish_count == 3:
            confluence = 'STRONG_BEARISH'
            confluence_score = 90
        elif bullish_count >= 2:
            confluence = 'BULLISH'
            confluence_score = 65
        elif bearish_count >= 2:
            confluence = 'BEARISH'
            confluence_score = 65
        else:
            confluence = 'MIXED'
            confluence_score = 30
        
        return {
            'rsi_signals': rsi_signals,
            'macd_signals': macd_signals,
            'confluence': confluence,
            'confluence_score': confluence_score
        }
    
    @staticmethod
    def _analyze_pattern_coherence(data_map: Dict) -> Dict:
        """Analyze pattern coherence across timeframes"""
        # Simple pattern coherence based on recent candle patterns
        patterns = {}
        
        for tf, df in data_map.items():
            last_candles = df.tail(3)
            
            # Check for consistent directional movement
            consecutive_green = all(last_candles['close'] > last_candles['open'])
            consecutive_red = all(last_candles['close'] < last_candles['open'])
            
            if consecutive_green:
                patterns[tf] = 'BULLISH_CONTINUATION'
            elif consecutive_red:
                patterns[tf] = 'BEARISH_CONTINUATION'
            else:
                patterns[tf] = 'MIXED'
        
        # Check coherence
        bullish_count = sum(1 for p in patterns.values() if 'BULLISH' in p)
        bearish_count = sum(1 for p in patterns.values() if 'BEARISH' in p)
        
        if bullish_count >= 2:
            coherence = 'BULLISH_COHERENT'
            coherence_score = bullish_count * 30
        elif bearish_count >= 2:
            coherence = 'BEARISH_COHERENT'
            coherence_score = bearish_count * 30
        else:
            coherence = 'INCOHERENT'
            coherence_score = 0
        
        return {
            'tf_patterns': patterns,
            'coherence': coherence,
            'coherence_score': coherence_score
        }
    
    @staticmethod
    def _detect_cross_tf_divergences(data_map: Dict) -> Dict:
        """Detect divergences between price and indicators across timeframes"""
        divergences = []
        
        # Check RSI divergence between 1H and 4H
        if '1h' in data_map and '4h' in data_map:
            df_1h = data_map['1h'].tail(5)
            df_4h = data_map['4h'].tail(5)
            
            # Price trend
            price_1h_trend = df_1h['close'].iloc[-1] > df_1h['close'].iloc[0]
            price_4h_trend = df_4h['close'].iloc[-1] > df_4h['close'].iloc[0]
            
            # RSI trend
            rsi_1h_trend = df_1h['rsi'].iloc[-1] > df_1h['rsi'].iloc[0]
            rsi_4h_trend = df_4h['rsi'].iloc[-1] > df_4h['rsi'].iloc[0]
            
            # Detect divergence
            if price_1h_trend != rsi_1h_trend:
                div_type = 'BEARISH' if price_1h_trend else 'BULLISH'
                divergences.append(f"1H_RSI_{div_type}_DIVERGENCE")
            
            if price_4h_trend != rsi_4h_trend:
                div_type = 'BEARISH' if price_4h_trend else 'BULLISH'
                divergences.append(f"4H_RSI_{div_type}_DIVERGENCE")
        
        has_divergence = len(divergences) > 0
        divergence_warning = "⚠️ Cross-TF divergence detected" if has_divergence else None
        
        return {
            'detected_divergences': divergences,
            'has_divergence': has_divergence,
            'warning': divergence_warning
        }
    
    @staticmethod
    def _fuse_decisions(
        trend_analysis: Dict,
        momentum_analysis: Dict,
        pattern_analysis: Dict,
        weights: Dict
    ) -> Dict:
        """Fuse all analyses into final weighted decision"""
        
        # Convert qualitative signals to numeric scores
        def signal_to_score(signal: str) -> float:
            if 'STRONG_BULL' in signal or signal == 'BULLISH_COHERENT':
                return 1.0
            elif 'BULL' in signal or 'OVERSOLD' in signal:
                return 0.6
            elif 'STRONG_BEAR' in signal or signal == 'BEARISH_COHERENT':
                return -1.0
            elif 'BEAR' in signal or 'OVERBOUGHT' in signal:
                return -0.6
            else:
                return 0.0
        
        # Calculate weighted scores
        trend_score = signal_to_score(trend_analysis['alignment'])
        momentum_score = signal_to_score(momentum_analysis['confluence'])
        pattern_score = signal_to_score(pattern_analysis['coherence'])
        
        # Combine with component weights (trend has more weight)
        final_score = (
            trend_score * 0.45 +
            momentum_score * 0.35 +
            pattern_score * 0.20
        ) * 100  # Scale to 0-100
        
        # Confidence based on agreement
        agreements = [
            abs(trend_score) > 0.5,
            abs(momentum_score) > 0.5,
            abs(pattern_score) > 0.5
        ]
        confidence = (sum(agreements) / 3.0) * 100
        
        # Final decision
        if final_score > 40:
            decision = 'BUY'
        elif final_score < -40:
            decision = 'SELL'
        else:
            decision = 'NEUTRAL'
        
        return {
            'decision': decision,
            'score': final_score,
            'confidence': confidence,
            'component_scores': {
                'trend': trend_score * 100,
                'momentum': momentum_score * 100,
                'pattern': pattern_score * 100
            }
        }
    
    @staticmethod
    def _empty_analysis() -> Dict:
        """Return empty analysis structure"""
        return {
            'mtf_decision': 'NEUTRAL',
            'mtf_confidence': 0,
            'mtf_score': 0,
            'trend_alignment': {},
            'momentum_confluence': {},
            'pattern_coherence': {},
            'cross_tf_divergences': {'has_divergence': False},
            'timeframe_weights': MultiTimeframeTransformer.REGIME_WEIGHTS['DEFAULT'],
            'regime': 'UNKNOWN'
        }
