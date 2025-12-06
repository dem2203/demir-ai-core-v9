import logging
from typing import Dict, Tuple

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
        htf_direction = brain_state.get('htf_direction', 0)  # -1, 0, or 1
        
        # Check if signal aligns with HTF trend
        if signal_data['side'] == 'BUY' and htf_direction > 0:
            score += 20
            details.append("✅ HTF Bullish (+20)")
        elif signal_data['side'] == 'SELL' and htf_direction < 0:
            score += 20
            details.append("✅ HTF Bearish (+20)")
        else:
            details.append("⚠️ HTF Not Aligned (0)")
        
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
    
    def should_send_signal(self, signal_data: Dict, snapshot: Dict) -> Tuple[bool, int, str]:
        """
        Determine if signal meets quality threshold.
        
        STRICT FILTERING (Phase 21):
        - Confidence MUST be > 85%
        - Quality Score MUST be >= threshold
        """
        score = self.calculate_signal_quality(signal_data, snapshot)
        confidence = signal_data.get('confidence', 0)
        
        # 1. Strict Confidence Check
        if confidence <= 85:
            reason = f"⛔ LOW CONFIDENCE ({confidence:.1f}% <= 85%)"
            logger.info(f"🔇 SIGNAL MUTED: {signal_data['symbol']} | Conf: {confidence:.1f}%")
            return False, score, reason

        # 2. Quality Score Check
        if score >= self.quality_threshold:
            reason = f"✅ PREMIUM SIGNAL (Quality: {score}/100)"
            logger.info(f"🟢 SIGNAL APPROVED: {signal_data['symbol']} {signal_data['side']} | Score: {score}")
            return True, score, reason
        else:
            reason = f"⛔ LOW QUALITY (Score: {score}/100 < {self.quality_threshold})"
            logger.warning(f"🔴 SIGNAL REJECTED: {signal_data['symbol']} {signal_data['side']} | Score: {score}")
            return False, score, reason
