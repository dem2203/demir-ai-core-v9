"""
Notification Priority System - Smart Alert Management
Phase 29.2

Calculates urgency level for signals based on:
- Confidence score
- MTF confluence
- SMC alignment
- R:R ratio
- Volume profile position
- Cross-exchange divergence

Urgency Levels:
- CRITICAL: Immediate action required (all factors aligned)
- HIGH: Strong setup (most factors aligned)
- MEDIUM: Good setup (standard quality)
- LOW: Informational only
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger("NOTIFICATION_PRIORITY")


@dataclass
class UrgencyConfig:
    """Configuration for each urgency level"""
    level: str
    emoji: str
    cooldown_minutes: int
    action_window: str  # How long user has to act
    sound_hint: str  # Telegram doesn't support actual sounds, but we can hint


# Urgency level configurations
URGENCY_CONFIGS = {
    'CRITICAL': UrgencyConfig(
        level='CRITICAL',
        emoji='🔥🔥🔥',
        cooldown_minutes=0,  # No cooldown for critical
        action_window='5-15 min',
        sound_hint='⚡ IMMEDIATE ACTION ⚡'
    ),
    'HIGH': UrgencyConfig(
        level='HIGH',
        emoji='🟠🟠',
        cooldown_minutes=15,
        action_window='15-30 min',
        sound_hint='⏰ Act Soon'
    ),
    'MEDIUM': UrgencyConfig(
        level='MEDIUM',
        emoji='🟡',
        cooldown_minutes=30,
        action_window='30-60 min',
        sound_hint='📋 Monitor'
    ),
    'LOW': UrgencyConfig(
        level='LOW',
        emoji='⚪',
        cooldown_minutes=60,
        action_window='1-4 hours',
        sound_hint='ℹ️ Informational'
    )
}


class NotificationPriority:
    """
    Calculates and manages notification urgency levels.
    
    Uses a scoring system based on multiple factors:
    - Base confidence: 0-40 points
    - MTF confluence: 0-20 points
    - SMC alignment: 0-15 points
    - R:R ratio: 0-15 points
    - Volume profile: 0-10 points
    
    Total Score -> Urgency:
    - 85-100: CRITICAL
    - 65-84: HIGH
    - 45-64: MEDIUM
    - 0-44: LOW
    """
    
    def __init__(self):
        self.urgency_cooldowns: Dict[str, datetime] = {}
    
    def calculate_urgency(self, signal: Dict, snapshot: Dict = None) -> Dict:
        """
        Calculate urgency level for a signal.
        
        Args:
            signal: Signal dict with confidence, side, etc.
            snapshot: Market snapshot with MTF, SMC, volume_profile, etc.
        
        Returns:
            {
                'urgency': 'CRITICAL'/'HIGH'/'MEDIUM'/'LOW',
                'score': 85,
                'breakdown': {...},
                'config': UrgencyConfig,
                'can_send': True/False (cooldown check)
            }
        """
        snapshot = snapshot or {}
        score = 0
        breakdown = {}
        
        # 1. Base Confidence Score (0-40 points)
        confidence = signal.get('confidence', 0)
        conf_score = min(40, int(confidence * 0.4))  # 100% conf = 40 points
        breakdown['confidence'] = conf_score
        score += conf_score
        
        # 2. MTF Confluence (0-20 points)
        mtf = snapshot.get('mtf', {})
        mtf_confluence = mtf.get('confluence_score', 0)
        mtf_score = min(20, int(mtf_confluence * 0.2))
        breakdown['mtf'] = mtf_score
        score += mtf_score
        
        # 3. SMC Alignment (0-15 points)
        smc = snapshot.get('smc', {})
        smc_bias = smc.get('smc_bias', 'NEUTRAL')
        signal_side = signal.get('side', '')
        
        smc_score = 0
        if smc_bias == 'BULLISH' and signal_side == 'BUY':
            smc_score = 15
        elif smc_bias == 'BEARISH' and signal_side == 'SELL':
            smc_score = 15
        elif smc_bias != 'NEUTRAL':
            smc_score = 5  # Partial alignment
        breakdown['smc'] = smc_score
        score += smc_score
        
        # 4. Risk:Reward Ratio (0-15 points)
        smart_sltp = snapshot.get('smart_sltp', {})
        rr1 = smart_sltp.get('risk_reward_1', 0)
        
        rr_score = 0
        if rr1 >= 3:
            rr_score = 15
        elif rr1 >= 2:
            rr_score = 10
        elif rr1 >= 1.5:
            rr_score = 5
        breakdown['risk_reward'] = rr_score
        score += rr_score
        
        # 5. Volume Profile Position (0-10 points)
        vp = snapshot.get('volume_profile', {})
        vp_position = vp.get('price_position', 'UNKNOWN')
        vp_signal = vp.get('volume_signal', {})
        
        vp_score = 0
        signal_is_long = signal_side == 'BUY'
        
        # Favorable positions
        if signal_is_long and vp_position == 'BELOW_VALUE_AREA':
            vp_score = 10  # Buying below value = good
        elif not signal_is_long and vp_position == 'ABOVE_VALUE_AREA':
            vp_score = 10  # Shorting above value = good
        elif vp_position == 'AT_VPOC':
            vp_score = 5  # At fair value, could go either way
        elif vp_signal.get('bias') == 'BULLISH' and signal_is_long:
            vp_score = 7
        elif vp_signal.get('bias') == 'BEARISH' and not signal_is_long:
            vp_score = 7
        breakdown['volume_profile'] = vp_score
        score += vp_score
        
        # 6. Cross-Exchange Bonus (0-5 bonus points)
        cross_exchange = snapshot.get('cross_exchange', {})
        divergence = cross_exchange.get('max_divergence', 0)
        if divergence > 0.1:  # Significant divergence
            bonus = min(5, int(divergence * 10))
            breakdown['cross_exchange_bonus'] = bonus
            score += bonus
        
        # Determine urgency level
        if score >= 85:
            urgency = 'CRITICAL'
        elif score >= 65:
            urgency = 'HIGH'
        elif score >= 45:
            urgency = 'MEDIUM'
        else:
            urgency = 'LOW'
        
        config = URGENCY_CONFIGS[urgency]
        
        # Check cooldown
        can_send = self._check_cooldown(signal.get('symbol', ''), urgency, config.cooldown_minutes)
        
        return {
            'urgency': urgency,
            'score': score,
            'breakdown': breakdown,
            'config': config,
            'can_send': can_send,
            'emoji': config.emoji,
            'action_window': config.action_window,
            'sound_hint': config.sound_hint
        }
    
    def _check_cooldown(self, symbol: str, urgency: str, cooldown_minutes: int) -> bool:
        """Check if we can send a notification based on cooldown"""
        if cooldown_minutes == 0:
            return True
        
        key = f"{symbol}_{urgency}"
        now = datetime.now()
        
        if key in self.urgency_cooldowns:
            last_sent = self.urgency_cooldowns[key]
            if now - last_sent < timedelta(minutes=cooldown_minutes):
                return False
        
        # Update cooldown
        self.urgency_cooldowns[key] = now
        return True
    
    def format_urgency_header(self, urgency_data: Dict) -> str:
        """
        Format urgency information for Telegram message header.
        """
        urgency = urgency_data['urgency']
        emoji = urgency_data['emoji']
        score = urgency_data['score']
        action_window = urgency_data['action_window']
        sound_hint = urgency_data['sound_hint']
        
        if urgency == 'CRITICAL':
            return (
                f"{emoji} **URGENCY: CRITICAL** {emoji}\n"
                f"📊 Score: {score}/100\n"
                f"{sound_hint}\n"
                f"⏱️ Act within: {action_window}"
            )
        elif urgency == 'HIGH':
            return (
                f"{emoji} **URGENCY: HIGH**\n"
                f"📊 Score: {score}/100 | {sound_hint}\n"
                f"⏱️ Window: {action_window}"
            )
        elif urgency == 'MEDIUM':
            return (
                f"{emoji} **URGENCY: MEDIUM** | Score: {score}/100\n"
                f"⏱️ Window: {action_window}"
            )
        else:
            return f"{emoji} **Urgency:** LOW | Score: {score}/100"
    
    def get_urgency_breakdown(self, urgency_data: Dict) -> str:
        """
        Get detailed breakdown of urgency score.
        """
        breakdown = urgency_data.get('breakdown', {})
        
        lines = ["📋 **Score Breakdown:**"]
        
        if 'confidence' in breakdown:
            lines.append(f"   • Confidence: {breakdown['confidence']}/40")
        if 'mtf' in breakdown:
            lines.append(f"   • MTF Confluence: {breakdown['mtf']}/20")
        if 'smc' in breakdown:
            lines.append(f"   • SMC Alignment: {breakdown['smc']}/15")
        if 'risk_reward' in breakdown:
            lines.append(f"   • Risk:Reward: {breakdown['risk_reward']}/15")
        if 'volume_profile' in breakdown:
            lines.append(f"   • Volume Profile: {breakdown['volume_profile']}/10")
        if 'cross_exchange_bonus' in breakdown:
            lines.append(f"   • Cross-Exchange Bonus: +{breakdown['cross_exchange_bonus']}")
        
        return "\n".join(lines)


# Singleton instance
_priority_instance: Optional[NotificationPriority] = None


def get_notification_priority() -> NotificationPriority:
    """Get or create the singleton NotificationPriority instance"""
    global _priority_instance
    if _priority_instance is None:
        _priority_instance = NotificationPriority()
    return _priority_instance


# Quick test
if __name__ == "__main__":
    priority = NotificationPriority()
    
    # Test signal
    test_signal = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'confidence': 85
    }
    
    # Test snapshot
    test_snapshot = {
        'mtf': {'confluence_score': 80, 'confluence_type': 'STRONG'},
        'smc': {'smc_bias': 'BULLISH'},
        'smart_sltp': {'risk_reward_1': 2.5},
        'volume_profile': {'price_position': 'BELOW_VALUE_AREA', 'volume_signal': {'bias': 'BULLISH'}}
    }
    
    result = priority.calculate_urgency(test_signal, test_snapshot)
    
    print(f"Urgency: {result['urgency']}")
    print(f"Score: {result['score']}")
    print(f"Breakdown: {result['breakdown']}")
    print(f"\nTelegram Header:")
    print(priority.format_urgency_header(result))
