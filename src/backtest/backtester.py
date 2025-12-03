import pandas as pd
import numpy as np
import logging
import joblib
import asyncio
import os
from datetime import datetime
from tensorflow.keras.models import load_model

# Kendi modüllerimiz
from src.brain.feature_engineering import FeatureEngineer
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("BACKTESTER")

class Backtester:
    """
    DEMIR AI - TIME MACHINE (OPTIMIZABLE EDITION)
    
    Artık dışarıdan 'parametre' alabilir.
    Böylece Optimizer farklı Stop Loss/Take Profit oranlarını deneyebilir.
    """
    
    MODELS_DIR = "src/brain/models/storage"
    LOOKBACK = 60

    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto = BinanceConnector()
        self.macro = MacroConnector()
        self.model = None
        self.scaler = None
        self.trade_log = []

    def _get_paths(self, symbol):
        clean_sym = symbol.replace("/", "")
        model_path = os.path.join(self.MODELS_DIR, f"lstm_v11_{clean_sym}.h5")
        scaler_path = os.path.join(self.MODELS_DIR, f"scaler_{clean_sym}.pkl")
        return model_path, scaler_path

    def load_brain_for_symbol(self, symbol):
        m_path, s_path = self._get_paths(symbol)
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                self.model = load_model(m_path)
                self.scaler = joblib.load(s_path)
                return True
            except Exception as e:
                return False
        return False

    async def run_backtest(self, symbol="BTC/USDT", days=30, params=None):
        """
        params: { 'sl_mul': 1.5, 'tp_mul': 3.0, 'threshold': 0.50 }
        """
        # Varsayılan parametreler (Eğer Optimizer'dan gelmezse)
        if params is None:
            params = {'sl_mul': 1.5, 'tp_mul': 3.0, 'threshold': 0.51}

        logger.info(f"Starting Backtest for {symbol} with params: {params}")
        
        if not self.load_brain_for_symbol(symbol):
            return {"error": f"Brain not found for {symbol}."}

        # 1. Veri Çek
        limit = (days * 24) + 200 
        raw_crypto = await self.crypto.fetch_candles(symbol, limit=limit)
        await self.crypto.close()
        if not raw_crypto: return {"error": "No crypto data fetched."}
        
        crypto_df = FeatureEngineer.process_data(raw_crypto)
        macro_df = await self.macro.fetch_macro_data(period="2y", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        if df is None or len(df) < self.LOOKBACK:
            return {"error": "Not enough data."}

        # 2. Tahmin Hazırlığı
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume']]
        data_values = df[feature_cols].values
        
        try:
            scaled_data = self.scaler.transform(data_values)
        except: return {"error": "Scaler Mismatch."}
        
        X = []
        start_index = len(df) - (days * 24)
        if start_index < self.LOOKBACK: start_index = self.LOOKBACK
        indices = range(start_index, len(df))
        
        for i in indices:
            X.append(scaled_data[i-self.LOOKBACK:i])
            
        X = np.array(X)
        if len(X) == 0: return {"error": "No data for period."}
        
        predictions = self.model.predict(X, verbose=0) 
        
        # 3. Simülasyon (Parametrik)
        position = None 
        entry_price = 0
        
        sim_df = df.iloc[indices].reset_index(drop=True)
        
        for i, row in sim_df.iterrows():
            if i >= len(predictions): break
            
            current_price = row['close']
            atr = row['atr'] # Volatilite verisi
            ai_score = predictions[i][0]
            
            ts = row.get('timestamp', 0)
            try: readable_time = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')
            except: readable_time = str(ts)
            
            # --- DİNAMİK STRATEJİ ---
            
            # ALIM
            if position is None:
                if ai_score > params['threshold']: 
                    position = 'LONG'
                    entry_price = current_price
                    
                    self.trade_log.append({
                        "action": "BUY",
                        "price": entry_price,
                        "time": readable_time,
                        "score": f"{ai_score:.2f}",
                        "balance": self.balance
                    })

            # SATIM (ATR Bazlı Dinamik Stop/TP)
            elif position == 'LONG':
                # Dinamik Hedefler
                stop_price = entry_price - (atr * params['sl_mul'])
                target_price = entry_price + (atr * params['tp_mul'])
                
                should_sell = False
                reason = ""
                
                if current_price <= stop_price:
                    should_sell = True; reason = "Stop Loss (ATR)"
                elif current_price >= target_price:
                    should_sell = True; reason = "Take Profit (ATR)"
                elif ai_score < 0.40: # AI fikrini değiştirdi
                    should_sell = True; reason = "AI Exit Signal"
                
                if should_sell:
                    position = None
                    pnl_pct = (current_price - entry_price) / entry_price
                    pnl_amount = (self.balance * pnl_pct) - (self.balance * 0.001) 
                    self.balance += pnl_amount
                    
                    self.trade_log.append({
                        "action": "SELL",
                        "price": current_price,
                        "time": readable_time,
                        "score": f"{ai_score:.2f}",
                        "pnl_pct": f"{pnl_pct*100:.2f}%",
                        "reason": reason,
                        "balance": self.balance
                    })

        # Rapor
        sell_trades = [t for t in self.trade_log if t['action'] == 'SELL']
        total = len(sell_trades)
        wins = len([t for t in sell_trades if float(t['pnl_pct'].replace('%','')) > 0])
        win_rate = (wins / total * 100) if total > 0 else 0
        roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            "roi": roi,
            "total_trades": total,
            "win_rate": win_rate,
            "final_balance": self.balance,
            "trades": self.trade_log
        }
