# -*- coding: utf-8 -*-
"""
DEMIR AI v11 - RISK MANAGER (Mental Stop)
=========================================
Bu modül, trading işlemlerinde "Ne kadar girmeliyim?" ve "Nerede çıkmalıyım?"
sorularını istatistiksel ve bilimsel yöntemlerle cevaplar.

Kurallar:
1. ASLA tüm kasayı basma (Max %5).
2. Kelly Criterion ile optimal büyüklüğü bul.
3. Her işlemde mutlaka Stop-Loss olsun.

Author: DEMIR AI Team
Date: 2026-01-04
"""
import numpy as np

class RiskManager:
    def __init__(self, check_circuit_breaker=True):
        self.max_position_size = 0.05  # Tek işlemde max %5
        self.daily_max_loss = 0.03     # Günlük max %3 kayıp (Circuit Breaker)
        self.check_circuit_breaker = check_circuit_breaker
        self.current_daily_pnl = 0.0

    def calculate_position_size(self, account_balance: float, win_rate: float, 
                              avg_win: float, avg_loss: float, confidence: float) -> float:
        """
        Kelly Criterion kullanarak pozisyon büyüklüğünü hesaplar.
        
        Formula: K = W - [(1 - W) / R]
        W: Win Rate
        R: Win/Loss Ratio
        """
        if avg_loss == 0:
            return self.max_position_size * account_balance

        win_loss_ratio = abs(avg_win / avg_loss)
        if win_loss_ratio == 0:
             return 0.0

        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Negatif Kelly = İşleme girme
        if kelly_pct <= 0:
            return 0.0
        
        # Risk Ayarı: Half-Kelly (Daha güvenli)
        safe_kelly = kelly_pct * 0.5
        
        # Güven ile çarp (Confidence Score)
        adjusted_size = safe_kelly * confidence
        
        # Asla max limiti geçme
        final_pct = min(adjusted_size, self.max_position_size)
        
        size = account_balance * final_pct
        return round(size, 2)

    def calculate_stop_loss(self, entry_price: float, side: str, atr: float, multiplier: float = 2.0) -> float:
        """
        ATR tabanlı dinamik Stop Loss seviyesi hesaplar.
        """
        stop_dist = atr * multiplier
        
        if side == "BUY":
            stop_price = entry_price - stop_dist
        else:  # SELL
            stop_price = entry_price + stop_dist
            
        return round(stop_price, 2)

    def calculate_take_profit(self, entry_price: float, side: str, stop_price: float, rr_ratio: float = 1.5) -> float:
        """
        Risk:Reward oranına göre Take Profit hesaplar.
        """
        risk = abs(entry_price - stop_price)
        reward = risk * rr_ratio
        
        if side == "BUY":
            tp_price = entry_price + reward
        else:  # SELL
            tp_price = entry_price - reward
            
        return round(tp_price, 2)
    
    def check_circuit_break(self, pnl_percent: float) -> bool:
        """
        Günlük kayıp limitini aştı mı kontrol et.
        """
        self.current_daily_pnl += pnl_percent
        
        if self.check_circuit_breaker and self.current_daily_pnl < -self.daily_max_loss:
            return True  # STOP TRADING
            
        return False
