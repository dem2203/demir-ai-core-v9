import logging
from typing import Dict

logger = logging.getLogger("SIGNAL_FILTER")

class SignalFilter:
    """
    PRECISION FILTER - Signal Quality Control
    
    Calculates a 0-100 quality score for each signal.
    Only signals with score >= threshold are sent to Telegram.
    """
    
    def __init__(self, quality_threshold: int = 70):
        self.quality_threshold = quality_threshold
        logger.info(f"🎯 Precision Filter initialized (threshold={quality_threshold})")
    
    def calculate_signal_quality(self, signal_data: Dict, snapshot: Dict) -> int:
        """
        Calculate composite quality score (0-100).
        
        Scoring Breakdown:
        - HTF Trend Alignment: 0-20 points
        - Pattern Confirmation: 0-15 points
        - Whale Wall Support: 0-15 points
        - Confluence Quality: 0-20 points
        - RL Agent Confidence: 0-30 points
        """
        score = 0
        details = []
        
        # 1. HTF Trend Alignment (+20)
        brain_state = snapshot.get('brain_state', {})
        htf_attention = brain_state.get('htf_attention', 0)
        
        if signal_data['side'] == 'BUY' and htf_attention > 0.1:
            score += 20
            details.append("✅ HTF Bullish (+20)")
        elif signal_data['side'] == 'SELL' and htf_attention > 0.1:
            score += 20
            details.append("✅ HTF Bearish (+20)")
        else:
            details.append("⚠️ HTF Neutral (0)")
        
        # 2. Pattern Confirmation (+15)
        pattern = signal_data.get('pattern', 'None')
        if pattern and pattern != 'None':
            score += 15
            details.append(f"✅ Pattern: {pattern} (+15)")
        else:
            details.append("⚠️ No Pattern (0)")
        
        # 3. Whale Wall Support (+15)
        whale_support = snapshot.get('whale_support', 0)
        whale_resistance = snapshot.get('whale_resistance', 0)
        
        if (signal_data['side'] == 'BUY' and whale_support > 0) or \
           (signal_data['side'] == 'SELL' and whale_resistance > 0):
            score += 15
            details.append(f"✅ Whale Wall ({signal_data['side']}) (+15)")
        else:
            details.append("⚠️ No Whale Wall (0)")
        
        # 4. Confluence Quality (+20)
        quality = signal_data.get('quality', 'WEAK')
        if quality == 'STRONG':
            score += 20
            details.append("✅ Strong Confluence (+20)")
        elif quality == 'MODERATE':
            score += 10
            details.append("⚡ Moderate Confluence (+10)")
        else:
            details.append("⚠️ Weak/Conflicting (0)")
        
        # 5. RL Agent Confidence (+30)
        confidence = signal_data.get('confidence', 50)
        confidence_score = int(min(30, confidence * 0.3))
        score += confidence_score
        details.append(f"🤖 RL Confidence: {confidence:.0f}% (+{confidence_score})")
        
        logger.info(f"📊 Signal Quality Score: {score}/100 | {' | '.join(details)}")
        return score
    
    def should_send_signal(self, signal_data: Dict, snapshot: Dict) -> tuple[bool, int, str]:
        """
        Determine if signal meets quality threshold.
        
        Returns:
            (should_send, quality_score, reason)
        """
        score = self.calculate_signal_quality(signal_data, snapshot)
        
        if score >= self.quality_threshold:
            reason = f"✅ PREMIUM SIGNAL (Quality: {score}/100)"
            logger.info(f"🟢 SIGNAL APPROVED: {signal_data['symbol']} {signal_data['side']} | Score: {score}")
            return True, score, reason
        else:
            reason = f"⛔ LOW QUALITY (Score: {score}/100 < {self.quality_threshold})"
            logger.warning(f"🔴 SIGNAL REJECTED: {signal_data['symbol']} {signal_data['side']} | Score: {score}")
            return False, score, reason
