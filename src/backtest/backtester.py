import pandas as pd
import numpy as np
import logging
import joblib
import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

try:
    from src.brain.feature_engineering import FeatureEngineer
    from src.brain.regime_classifier import RegimeClassifier
    from src.data_ingestion.connectors.binance_connector import BinanceConnector
    from src.data_ingestion.macro_connector import MacroConnector
    from src.core.risk_manager import RiskManager
except ImportError:
    FeatureEngineer = None
    RegimeClassifier = None
    BinanceConnector = None
    MacroConnector = None
    RiskManager = None

import requests

logger = logging.getLogger("BACKTESTER")


class Backtester:
    """
    DEMIR AI - BACKTEST ENGINE
    
    Geçmiş verilerle strateji testi.
    """
    
    RESULTS_FILE = "backtest_results.json"
    
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto = BinanceConnector() if BinanceConnector else None
        self.macro = MacroConnector() if MacroConnector else None
        self.regime_classifier = RegimeClassifier() if RegimeClassifier else None
        self.risk_manager = RiskManager() if RiskManager else None
        
        self.model_lstm = None
        self.scaler = None
        self.agent_rl = None
        self.trade_log = []
        self.results = {}
        
        logger.info("✅ Backtester initialized")

    async def fetch_historical_data(self, symbol: str = 'BTCUSDT', 
                                   interval: str = '1h',
                                   days: int = 30) -> pd.DataFrame:
        """Geçmiş verileri çek."""
        try:
            limit = min(1000, days * 24) if interval == '1h' else min(1000, days * 24 * 4)
            
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': interval, 'limit': limit},
                timeout=30
            )
            
            if resp.status_code != 200:
                logger.error(f"Failed to fetch historical data: {resp.status_code}")
                return pd.DataFrame()
            
            klines = resp.json()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Swing Points
            df['swing_high'] = df['high'].rolling(window=20).max()
            df['swing_low'] = df['low'].rolling(window=20).min()
            
            logger.info(f"Loaded {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Historical data fetch failed: {e}")
            return pd.DataFrame()

    async def run_backtest(self, symbol="BTCUSDT", days=30, params=None):
        """Backtest çalıştır."""
        self.trade_log = []
        self.balance = self.initial_balance
        
        # Veri çek
        df = await self.fetch_historical_data(symbol, '1h', days)
        if df.empty:
            return {'error': 'No data'}
        
        position = None
        entry_price = 0
        sl_price = 0
        tp_price = 0
        
        # Her mum için analiz
        for i in range(50, len(df) - 1):
            row = df.iloc[i]
            current_price = row['close']
            
            # Basit momentum signal
            signal = self._generate_signal(df.iloc[:i+1])
            decision = signal.get('direction', 'HOLD')
            
            if position is None:
                if decision == "LONG" and signal.get('confidence', 0) >= 60:
                    position = 'LONG'
                    entry_price = current_price
                    tp_price = current_price * 1.035
                    sl_price = current_price * 0.985
                    
                    self.trade_log.append({
                        "action": "BUY",
                        "price": entry_price,
                        "time": str(df.index[i]),
                        "confidence": signal.get('confidence', 0),
                        "balance": self.balance
                    })
                    
                elif decision == "SHORT" and signal.get('confidence', 0) >= 60:
                    position = 'SHORT'
                    entry_price = current_price
                    tp_price = current_price * 0.965
                    sl_price = current_price * 1.015
                    
                    self.trade_log.append({
                        "action": "SELL",
                        "price": entry_price,
                        "time": str(df.index[i]),
                        "confidence": signal.get('confidence', 0),
                        "balance": self.balance
                    })
            
            elif position == 'LONG':
                should_close = False
                reason = ""
                
                if current_price >= tp_price:
                    should_close = True
                    reason = "TP"
                elif current_price <= sl_price:
                    should_close = True
                    reason = "SL"
                
                if should_close:
                    pnl_pct = (current_price - entry_price) / entry_price
                    pnl_amount = self.balance * 0.1 * pnl_pct  # 10% position
                    self.balance += pnl_amount
                    
                    self.trade_log.append({
                        "action": "CLOSE_LONG",
                        "price": current_price,
                        "time": str(df.index[i]),
                        "reason": reason,
                        "pnl_pct": f"{pnl_pct*100:.2f}%",
                        "balance": self.balance
                    })
                    position = None
                    
            elif position == 'SHORT':
                should_close = False
                reason = ""
                
                if current_price <= tp_price:
                    should_close = True
                    reason = "TP"
                elif current_price >= sl_price:
                    should_close = True
                    reason = "SL"
                
                if should_close:
                    pnl_pct = (entry_price - current_price) / entry_price
                    pnl_amount = self.balance * 0.1 * pnl_pct
                    self.balance += pnl_amount
                    
                    self.trade_log.append({
                        "action": "CLOSE_SHORT",
                        "price": current_price,
                        "time": str(df.index[i]),
                        "reason": reason,
                        "pnl_pct": f"{pnl_pct*100:.2f}%",
                        "balance": self.balance
                    })
                    position = None
        
        # Sonuçları hesapla
        self.results = self._calculate_results(symbol)
        self._save_results()
        
        return self.results
    
    def _generate_signal(self, df: pd.DataFrame) -> Dict:
        """Test sinyali oluştur."""
        if len(df) < 24:
            return {'direction': 'HOLD', 'confidence': 0}
        
        # Momentum
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100
        
        # RSI approximation
        closes = df['close'].iloc[-15:]
        deltas = closes.diff().dropna()
        gains = deltas.where(deltas > 0, 0).mean()
        losses = (-deltas.where(deltas < 0, 0)).mean()
        rs = gains / losses if losses != 0 else 1
        rsi = 100 - (100 / (1 + rs))
        
        if price_change > 1.5 and rsi < 70:
            return {'direction': 'LONG', 'confidence': 50 + min(25, price_change * 5)}
        elif price_change < -1.5 and rsi > 30:
            return {'direction': 'SHORT', 'confidence': 50 + min(25, abs(price_change) * 5)}
        
        return {'direction': 'HOLD', 'confidence': 30}
    
    def _calculate_results(self, symbol: str) -> Dict:
        """Sonuçları hesapla."""
        if not self.trade_log:
            return {'symbol': symbol, 'total_trades': 0, 'win_rate': 0}
        
        # Win/Loss count
        wins = 0
        losses = 0
        
        for trade in self.trade_log:
            if 'reason' in trade:
                if trade['reason'] == 'TP':
                    wins += 1
                elif trade['reason'] == 'SL':
                    losses += 1
        
        total = wins + losses
        win_rate = (wins / total) * 100 if total > 0 else 0
        total_pnl_pct = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            'symbol': symbol,
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_results(self):
        """Sonuçları kaydet."""
        try:
            with open(self.RESULTS_FILE, 'w') as f:
                json.dump(self.results, f, indent=2)
        except Exception as e:
            logger.warning(f"Results save failed: {e}")
    
    def format_results_for_telegram(self) -> str:
        """Telegram formatında sonuç."""
        r = self.results
        
        if not r or 'error' in r:
            return "❌ Backtest hatası"
        
        win_emoji = "✅" if r['win_rate'] >= 55 else "⚠️" if r['win_rate'] >= 45 else "❌"
        pnl_emoji = "📈" if r['total_pnl_pct'] >= 0 else "📉"
        
        msg = f"""
📊 BACKTEST SONUCU
━━━━━━━━━━━━━━━━━━━━━━
{r['symbol']} - Son 30 Gün
━━━━━━━━━━━━━━━━━━━━━━
📈 Toplam İşlem: {r['total_trades']}
{win_emoji} Kazanma Oranı: %{r['win_rate']}

✅ Kazanç: {r['wins']} | ❌ Kayıp: {r['losses']}
━━━━━━━━━━━━━━━━━━━━━━
💰 Başlangıç: ${r['initial_balance']:,.2f}
💰 Son: ${r['final_balance']:,.2f}
{pnl_emoji} Toplam Kar: %{r['total_pnl_pct']}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_backtester = None

def get_backtester() -> Backtester:
    """Get or create backtester instance."""
    global _backtester
    if _backtester is None:
        _backtester = Backtester()
    return _backtester

