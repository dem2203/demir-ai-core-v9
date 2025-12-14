"""
Signal Quality Filter Module
Eliminates low-quality trading signals based on multiple criteria
"""
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("SIGNAL_FILTER")


class SignalQualityFilter:
    """
    Filters trading signals based on multiple quality criteria:
    1. Confidence threshold
    2. Multi-timeframe alignment
    3. SMC confirmation
    4. Volume confirmation
    5. Risk/Reward minimum
    """
    
    def __init__(
        self,
        min_confidence: float = 60.0,
        min_mtf_confluence: float = 50.0,
        min_risk_reward: float = 1.5,
        require_smc_alignment: bool = True,
        require_volume_confirm: bool = True
    ):
        self.min_confidence = min_confidence
        self.min_mtf_confluence = min_mtf_confluence
        self.min_risk_reward = min_risk_reward
        self.require_smc_alignment = require_smc_alignment
        self.require_volume_confirm = require_volume_confirm
        
        # Quality scoring weights
        self.weights = {
            'confidence': 0.25,
            'mtf_confluence': 0.20,
            'smc_alignment': 0.20,
            'risk_reward': 0.15,
            'volume': 0.10,
            'technical': 0.10
        }
    
    def filter_signal(
        self, 
        signal: Dict, 
        snapshot: Optional[Dict] = None
    ) -> Tuple[bool, float, str]:
        """
        Filter a trading signal based on quality criteria.
        
        Returns:
            (should_trade, quality_score, reason)
        """
        if not signal or not snapshot:
            return False, 0.0, "Missing signal or snapshot data"
        
        quality_score = 0.0
        reasons = []
        
        # 1. Confidence Check
        confidence = signal.get('ai_confidence', 0)
        if confidence >= self.min_confidence:
            quality_score += self.weights['confidence'] * min(confidence / 100, 1.0)
        else:
            reasons.append(f"Low confidence: {confidence:.1f}% < {self.min_confidence}%")
        
        # 2. MTF Confluence Check
        mtf = snapshot.get('mtf', {})
        mtf_score = mtf.get('confluence_score', 0)
        if mtf_score >= self.min_mtf_confluence:
            quality_score += self.weights['mtf_confluence'] * (mtf_score / 100)
        else:
            reasons.append(f"Low MTF confluence: {mtf_score}% < {self.min_mtf_confluence}%")
        
        # 3. SMC Alignment Check
        smc = snapshot.get('smc', {})
        smc_bias = smc.get('smc_bias', 'NEUTRAL')
        signal_direction = signal.get('ai_decision', 'NEUTRAL')
        
        smc_aligned = (
            (signal_direction == 'BUY' and smc_bias == 'BULLISH') or
            (signal_direction == 'SELL' and smc_bias == 'BEARISH') or
            not self.require_smc_alignment
        )
        
        if smc_aligned:
            quality_score += self.weights['smc_alignment']
        else:
            reasons.append(f"SMC misalignment: Signal={signal_direction}, SMC={smc_bias}")
        
        # 4. Risk/Reward Check
        sltp = snapshot.get('smart_sltp', {})
        rr1 = sltp.get('risk_reward_1', '0')
        
        # Parse R:R string like "1:2.5"
        try:
            if isinstance(rr1, str) and ':' in rr1:
                parts = rr1.split(':')
                rr_value = float(parts[1]) / float(parts[0])
            else:
                rr_value = float(rr1) if rr1 else 0
        except:
            rr_value = 0
        
        if rr_value >= self.min_risk_reward:
            quality_score += self.weights['risk_reward'] * min(rr_value / 3, 1.0)
        else:
            reasons.append(f"Low R:R: {rr_value:.2f} < {self.min_risk_reward}")
        
        # 5. Volume Confirmation
        vp = snapshot.get('volume_profile', {})
        price_position = vp.get('price_position', 'UNKNOWN')
        
        volume_confirms = (
            (signal_direction == 'BUY' and 'BELOW' in price_position) or
            (signal_direction == 'SELL' and 'ABOVE' in price_position) or
            not self.require_volume_confirm
        )
        
        if volume_confirms:
            quality_score += self.weights['volume']
        else:
            reasons.append(f"Volume not confirming: {price_position}")
        
        # 6. Technical Alignment
        tech_bias = snapshot.get('tech_bias', 'NEUTRAL')
        tech_aligned = (
            (signal_direction == 'BUY' and tech_bias == 'BULLISH') or
            (signal_direction == 'SELL' and tech_bias == 'BEARISH')
        )
        
        if tech_aligned:
            quality_score += self.weights['technical']
        
        # Final decision
        quality_score = min(quality_score * 100, 100)  # Convert to percentage
        
        # Minimum threshold for trading
        min_quality_threshold = 60.0
        should_trade = quality_score >= min_quality_threshold and len(reasons) <= 2
        
        # Generate summary reason
        if should_trade:
            reason = f"PASS: Quality {quality_score:.1f}%"
        else:
            reason = f"REJECT ({quality_score:.1f}%): " + "; ".join(reasons[:3])
        
        logger.info(f"🎯 Signal Filter: {signal_direction} => {reason}")
        
        return should_trade, quality_score, reason
    
    def get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
