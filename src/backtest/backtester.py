import pandas as pd
import numpy as np
import logging
import joblib
import asyncio
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from stable_baselines3 import PPO 

from src.brain.feature_engineering import FeatureEngineer
from src.brain.regime_classifier import RegimeClassifier
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("BACKTESTER")

from src.core.risk_manager import RiskManager

class Backtester:
    # ... (önceki kodlar aynı)
    
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto = BinanceConnector()
        self.macro = MacroConnector()
        self.regime_classifier = RegimeClassifier()
        self.risk_manager = RiskManager() # YENİ
        
        self.model_lstm = None
        self.scaler = None
        self.agent_rl = None
        self.trade_log = []

    # ... (load_brains ve diğer metodlar aynı)

    async def run_backtest(self, symbol="BTC/USDT", days=30, params=None):
        # ... (veri çekme ve hazırlık aynı)
        
        # Swing Points Hesapla (Backtest için)
        df['swing_high'] = df['high'].rolling(window=20).max()
        df['swing_low'] = df['low'].rolling(window=20).min()
        
        # ... (prediction döngüsü)
        
        for i, row in sim_df.iterrows():
            # ... (karar mantığı aynı)
            
            if position is None:
                if decision == "BUY":
                    position = 'LONG'
                    entry_price = current_price
                    
                    # SMART SL/TP HESAPLA
                    smart_levels = self.risk_manager.calculate_smart_levels(
                        entry_price=entry_price,
                        side="BUY",
                        swing_low=row.get('swing_low', entry_price*0.95),
                        swing_high=row.get('swing_high', entry_price*1.05),
                        whale_support=0, # Backtest'te yok
                        whale_resistance=0,
                        magnet_price=0,
                        atr=atr
                    )
                    
                    self.sl_price = smart_levels['sl']
                    self.tp_price = smart_levels['tp']
                    
                    self.trade_log.append({
                        "action": "BUY", "price": entry_price, "time": t_str, 
                        "score": f"L:{lstm_score:.2f}|R:{rl_action}", "balance": self.balance,
                        "setup": smart_levels['setup_type']
                    })
            
            elif position == 'LONG':
                # Smart Exit Kontrolü
                should_sell = False
                reason = ""
                
                if current_price < self.sl_price: should_sell = True; reason = "SL (Smart)"
                elif current_price > self.tp_price: should_sell = True; reason = "TP (Smart)"
                elif decision == "SELL": should_sell = True; reason = "Signal"
                
                if should_sell:
                    position = None
                    pnl_pct = (current_price - entry_price) / entry_price
                    pnl_amount = (self.balance * pnl_pct) - (self.balance * 0.001)
                    self.balance += pnl_amount
                    self.trade_log.append({
                        "action": "SELL", "price": current_price, "time": t_str, 
                        "score": f"L:{lstm_score:.2f}|R:{rl_action}", "pnl_pct": f"{pnl_pct*100:.2f}%", 
                        "reason": reason, "balance": self.balance
                    })
        
        # ... (sonuç hesaplama aynı)
