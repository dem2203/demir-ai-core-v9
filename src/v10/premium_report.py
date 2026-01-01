# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - Premium Report Builder
=====================================
Tüm modüllerden veri toplayarak kapsamlı analiz raporu oluşturur.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class PremiumReport:
    """Premium Analiz Raporu Verisi"""
    symbol: str
    price: float
    action: str
    confidence: float
    
    # Seviyeler
    entry_min: float
    entry_max: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    
    # Breakout Hunter
    breakout_active: bool = False
    breakout_direction: str = ""  # BULLISH / BEARISH
    breakout_probability: int = 0
    breakout_imminent: bool = False
    
    # Teknik Göstergeler
    trend: str = ""
    rsi: float = 0
    macd_signal: str = ""
    bollinger_squeeze: bool = False
    
    # Whale & Orderbook
    orderbook_bias: int = 0
    whale_activity: str = ""
    ls_ratio: float = 0
    funding_rate: float = 0
    
    # Liquidation
    liq_magnet: float = 0
    liq_direction: str = ""
    
    # AI Council Votes
    claude_vote: str = ""
    claude_conf: int = 0
    claude_reason: str = ""
    
    gpt4_vote: str = ""
    gpt4_conf: int = 0
    gpt4_reason: str = ""
    
    deepseek_vote: str = ""
    deepseek_conf: int = 0
    deepseek_reason: str = ""
    
    # LSTM
    lstm_direction: str = ""
    lstm_change: float = 0
    lstm_target: float = 0
    
    # Risk Management
    leverage: int = 0
    margin_pct: float = 0
    position_size: float = 0
    
    def to_telegram_message(self) -> str:
        """Telegram için formatlanmış mesaj"""
        
        # Action emoji
        if self.action == "BUY":
            action_emoji = "🟢"
        elif self.action == "SELL":
            action_emoji = "🔴"
        else:
            action_emoji = "⏸️"
        
        # Breakout section
        breakout_section = ""
        if self.breakout_active:
            breakout_emoji = "📈" if self.breakout_direction == "BULLISH" else "📉"
            imminent = "⚡ YAKINDA!" if self.breakout_imminent else ""
            breakout_section = f"""
🚀 BREAKOUT HUNTER:
  ⚡ SQUEEZE TESPİT EDİLDİ!
  {breakout_emoji} Patlama Yönü: {self.breakout_direction} (%{self.breakout_probability})
  {imminent}
"""
        
        # Whale section
        orderbook_emoji = "📗" if self.orderbook_bias > 0 else "📕"
        long_pct = self.ls_ratio / (1 + self.ls_ratio) * 100 if self.ls_ratio > 0 else 50
        
        # Liq magnet direction
        liq_hint = ""
        if self.liq_magnet > self.price:
            liq_hint = "(yukarı çeker)"
        elif self.liq_magnet < self.price:
            liq_hint = "(aşağı çeker)"
        
        # AI Council section
        def vote_emoji(vote):
            if vote == "BUY":
                return "🟢"
            elif vote == "SELL":
                return "🔴"
            return "⏸️"
        
        # LSTM section
        lstm_section = ""
        if self.lstm_direction:
            lstm_emoji = "📈" if self.lstm_direction == "UP" else ("📉" if self.lstm_direction == "DOWN" else "➡️")
            lstm_section = f"""
🧠 LSTM TAHMİN:
  {lstm_emoji} Yön: {self.lstm_direction} ({self.lstm_change:+.2f}%)
  🎯 Hedef: ${self.lstm_target:,.0f}
"""
        
        # Risk section
        risk_section = ""
        if self.leverage > 0:
            risk_section = f"""
💰 RİSK YÖNETİMİ:
  ⚖️ Kaldıraç: {self.leverage}x
  💵 Marjin: %{self.margin_pct:.1f}
"""
        
        # Calculate TP percentage
        if self.action == "BUY":
            tp_pct = (self.take_profit - self.price) / self.price * 100
            sl_pct = (self.stop_loss - self.price) / self.price * 100
        else:
            tp_pct = (self.price - self.take_profit) / self.price * 100
            sl_pct = (self.price - self.stop_loss) / self.price * 100
        
        message = f"""🧠 DEMIR AI PRO ANALİZ - {self.symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 Fiyat: ${self.price:,.2f}
📍 Karar: {action_emoji} {self.action} | Güven: %{self.confidence:.0f}

🎯 SEVİYELER:
  🎟️ Entry: ${self.entry_min:,.0f} - ${self.entry_max:,.0f}
  🎯 TP: ${self.take_profit:,.0f} ({tp_pct:+.1f}%)
  🛑 SL: ${self.stop_loss:,.0f} ({sl_pct:+.1f}%)
  ⚖️ R/R: {self.risk_reward:.1f}x
{breakout_section}
📊 TEKNİK:
  📈 Trend: {self.trend}
  📊 RSI: {self.rsi:.0f}
  📉 MACD: {self.macd_signal}
  🔲 Bollinger: {"Squeeze" if self.bollinger_squeeze else "Normal"}

🐋 WHALE & LIQ:
  {orderbook_emoji} Orderbook: {self.orderbook_bias:+d}
  📊 L/S: {self.ls_ratio:.2f} ({long_pct:.0f}% Long)
  💵 Funding: {self.funding_rate:.4f}%
  🎯 Liq Magnet: ${self.liq_magnet:,.0f} {liq_hint}

🤖 AI COUNCIL:
  {vote_emoji(self.claude_vote)} Claude: {self.claude_vote} (%{self.claude_conf})
  {vote_emoji(self.gpt4_vote)} GPT-4: {self.gpt4_vote} (%{self.gpt4_conf})
  {vote_emoji(self.deepseek_vote)} DeepSeek: {self.deepseek_vote} (%{self.deepseek_conf})
{lstm_section}{risk_section}
━━━━━━━━━━━ DEMIR AI v10 ━━━━━━━━━━━"""
        
        return message


def build_premium_report(signal, breakout_signal=None, council_decision=None, liq_data=None) -> PremiumReport:
    """EarlySignal'dan Premium Report oluştur"""
    
    # Base data from signal
    report = PremiumReport(
        symbol=signal.symbol,
        price=signal.entry_zone[0],
        action=signal.action,
        confidence=signal.confidence,
        entry_min=signal.entry_zone[0],
        entry_max=signal.entry_zone[1],
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        risk_reward=signal.risk_reward
    )
    
    # Breakout Hunter data
    if breakout_signal:
        report.breakout_active = breakout_signal.is_squeeze
        report.breakout_direction = breakout_signal.direction
        report.breakout_probability = int(breakout_signal.breakout_probability)
        report.breakout_imminent = breakout_signal.is_imminent

    # Leading Signal data
    if signal.leading_signal:
        ls = signal.leading_signal
        # Fix: LeadingSignal has 'direction' (SignalDirection enum) not 'trend'
        if hasattr(ls.direction, 'value'):
            report.trend = ls.direction.value
        else:
             report.trend = str(ls.direction)
             
        report.rsi = ls.rsi_1h if hasattr(ls, 'rsi_1h') else 50
        report.bollinger_squeeze = "squeeze" in signal.reasoning.lower() if signal.reasoning else False
        
        # Whale & Orderbook from Leading Indicators
        report.orderbook_bias = getattr(ls, 'orderbook_score', 0)
        whale_score = getattr(ls, 'whale_score', 0)
        report.whale_activity = "Bullish" if whale_score > 0 else "Bearish" if whale_score < 0 else "Neutral"
    
    # AI Council data
    if council_decision:
        votes = council_decision.individual_votes if hasattr(council_decision, 'individual_votes') else {}
        
        if 'Claude' in votes:
            v = votes['Claude']
            report.claude_vote = v.get('vote', 'HOLD')
            report.claude_conf = int(v.get('confidence', 50))
            report.claude_reason = v.get('reason', '')[:50]
        
        if 'GPT-4' in votes:
            v = votes['GPT-4']
            report.gpt4_vote = v.get('vote', 'HOLD')
            report.gpt4_conf = int(v.get('confidence', 50))
            report.gpt4_reason = v.get('reason', '')[:50]
        
        if 'DeepSeek' in votes:
            v = votes['DeepSeek']
            report.deepseek_vote = v.get('vote', 'HOLD')
            report.deepseek_conf = int(v.get('confidence', 50))
            report.deepseek_reason = v.get('reason', '')[:50]
    
    # Liquidation data
    if liq_data:
        report.ls_ratio = liq_data.get('ls_ratio', 1.0)
        report.funding_rate = liq_data.get('funding_rate', 0)
        report.liq_magnet = liq_data.get('liquidation_magnet', 0)
    
    # LSTM data from ml_prediction
    if signal.ml_prediction:
        ml = signal.ml_prediction
        # Fix: PricePrediction is an object, not a dict
        if hasattr(ml, 'direction'):
            report.lstm_direction = ml.direction
            report.lstm_change = ml.predicted_change_pct
        elif isinstance(ml, dict):
            report.lstm_direction = ml.get('direction', '')
            report.lstm_change = ml.get('predicted_change_pct', 0)
            
        report.lstm_target = report.price * (1 + report.lstm_change / 100)
    
    # Risk data
    if signal.risk_profile:
        rp = signal.risk_profile
        # risk_profile is explicitly converted to dict in engine
        report.leverage = rp.get('leverage', 0)
        report.margin_pct = rp.get('position_size_pct', 0)
        report.position_size = rp.get('position_usd', 0)
    
    return report
