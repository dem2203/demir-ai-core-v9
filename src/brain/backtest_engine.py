# -*- coding: utf-8 -*-
"""
DEMIR AI - Backtest Engine
Geçmiş verilerle strateji testi.

PHASE 110: Backtest Engine
- Historical data loading
- Strategy simulation
- Win rate calculation
- Performance metrics
"""
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os

logger = logging.getLogger("BACKTEST")


@dataclass
class BacktestTrade:
    """Backtest işlemi."""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    pnl_pct: float
    result: str  # 'WIN', 'LOSS', 'OPEN'
    confidence: float


class BacktestEngine:
    """
    Backtest Motoru
    
    Geçmiş verilerle strateji testi yapar.
    """
    
    RESULTS_FILE = "backtest_results.json"
    
    def __init__(self):
        self.trades: List[BacktestTrade] = []
        self.results: Dict = {}
        logger.info("✅ Backtest Engine initialized")
    
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
            
            logger.info(f"Loaded {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Historical data fetch failed: {e}")
            return pd.DataFrame()
    
    async def run_backtest(self, symbol: str = 'BTCUSDT', 
                          days: int = 30,
                          tp_pct: float = 3.5,
                          sl_pct: float = 1.5,
                          min_confidence: float = 65) -> Dict:
        """
        Backtest çalıştır.
        
        Her mum için sinyal oluştur ve sonraki mumlarda TP/SL kontrol et.
        """
        self.trades = []
        
        # Veri çek
        df = await self.fetch_historical_data(symbol, '1h', days)
        if df.empty:
            return {'error': 'No data'}
        
        # Her mum için analiz
        for i in range(50, len(df) - 24):  # 24 mum ileriye bak
            current_row = df.iloc[i]
            price = current_row['close']
            
            # Basit sinyal simülasyonu (gerçek sistemde Living AI Brain kullanılır)
            signal = self._generate_test_signal(df.iloc[:i+1])
            
            if signal['confidence'] < min_confidence:
                continue
            
            if signal['direction'] == 'NEUTRAL':
                continue
            
            # TP/SL hesapla
            if signal['direction'] == 'LONG':
                tp = price * (1 + tp_pct / 100)
                sl = price * (1 - sl_pct / 100)
            else:
                tp = price * (1 - tp_pct / 100)
                sl = price * (1 + sl_pct / 100)
            
            # Sonraki 24 mumlarda kontrol et
            trade_result = self._check_trade_outcome(
                df.iloc[i+1:i+25],
                signal['direction'],
                price, tp, sl
            )
            
            trade = BacktestTrade(
                entry_time=df.index[i],
                exit_time=trade_result['exit_time'],
                direction=signal['direction'],
                entry_price=price,
                exit_price=trade_result['exit_price'],
                tp_price=tp,
                sl_price=sl,
                pnl_pct=trade_result['pnl_pct'],
                result=trade_result['result'],
                confidence=signal['confidence']
            )
            self.trades.append(trade)
        
        # Sonuçları hesapla
        self.results = self._calculate_results(symbol)
        self._save_results()
        
        return self.results
    
    def _generate_test_signal(self, df: pd.DataFrame) -> Dict:
        """Test sinyali oluştur (basit momentum)."""
        if len(df) < 24:
            return {'direction': 'NEUTRAL', 'confidence': 0}
        
        # Son 24 saatlik değişim
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100
        
        # Volume
        avg_volume = df['volume'].iloc[-24:].mean()
        recent_volume = df['volume'].iloc[-3:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # RSI approximation
        closes = df['close'].iloc[-15:]
        deltas = closes.diff().dropna()
        gains = deltas.where(deltas > 0, 0).mean()
        losses = (-deltas.where(deltas < 0, 0)).mean()
        rs = gains / losses if losses != 0 else 1
        rsi = 100 - (100 / (1 + rs))
        
        # Karar
        score = 0
        direction = 'NEUTRAL'
        
        if price_change > 2 and rsi < 70:
            score = 50 + min(25, price_change * 5)
            direction = 'LONG'
        elif price_change < -2 and rsi > 30:
            score = 50 + min(25, abs(price_change) * 5)
            direction = 'SHORT'
        
        if volume_ratio > 1.5:
            score += 15
        
        return {
            'direction': direction,
            'confidence': min(95, score)
        }
    
    def _check_trade_outcome(self, future_data: pd.DataFrame, 
                            direction: str, entry: float, tp: float, sl: float) -> Dict:
        """İşlem sonucunu kontrol et."""
        for idx, row in future_data.iterrows():
            high = row['high']
            low = row['low']
            
            if direction == 'LONG':
                # TP first
                if high >= tp:
                    return {
                        'exit_time': idx,
                        'exit_price': tp,
                        'pnl_pct': ((tp - entry) / entry) * 100,
                        'result': 'WIN'
                    }
                # SL
                if low <= sl:
                    return {
                        'exit_time': idx,
                        'exit_price': sl,
                        'pnl_pct': ((sl - entry) / entry) * 100,
                        'result': 'LOSS'
                    }
            else:  # SHORT
                # TP first
                if low <= tp:
                    return {
                        'exit_time': idx,
                        'exit_price': tp,
                        'pnl_pct': ((entry - tp) / entry) * 100,
                        'result': 'WIN'
                    }
                # SL
                if high >= sl:
                    return {
                        'exit_time': idx,
                        'exit_price': sl,
                        'pnl_pct': ((entry - sl) / entry) * 100,
                        'result': 'LOSS'
                    }
        
        # Timeout - use last price
        last_price = future_data['close'].iloc[-1] if len(future_data) > 0 else entry
        if direction == 'LONG':
            pnl = ((last_price - entry) / entry) * 100
        else:
            pnl = ((entry - last_price) / entry) * 100
        
        return {
            'exit_time': future_data.index[-1] if len(future_data) > 0 else None,
            'exit_price': last_price,
            'pnl_pct': pnl,
            'result': 'WIN' if pnl > 0 else 'LOSS'
        }
    
    def _calculate_results(self, symbol: str) -> Dict:
        """Sonuçları hesapla."""
        if not self.trades:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'win_rate': 0,
                'error': 'No trades'
            }
        
        wins = len([t for t in self.trades if t.result == 'WIN'])
        losses = len([t for t in self.trades if t.result == 'LOSS'])
        total = len(self.trades)
        
        total_pnl = sum(t.pnl_pct for t in self.trades)
        avg_win = np.mean([t.pnl_pct for t in self.trades if t.result == 'WIN']) if wins > 0 else 0
        avg_loss = np.mean([t.pnl_pct for t in self.trades if t.result == 'LOSS']) if losses > 0 else 0
        
        win_rate = (wins / total) * 100 if total > 0 else 0
        
        # Risk/Reward
        rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Max drawdown (basit)
        cumulative = np.cumsum([t.pnl_pct for t in self.trades])
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'symbol': symbol,
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'risk_reward': round(rr_ratio, 2),
            'max_drawdown': round(max_dd, 2),
            'profit_factor': round(abs(avg_win * wins) / abs(avg_loss * losses), 2) if losses > 0 and avg_loss != 0 else 0,
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
        
        if 'error' in r:
            return f"❌ Backtest hatası: {r['error']}"
        
        win_emoji = "✅" if r['win_rate'] >= 55 else "⚠️" if r['win_rate'] >= 45 else "❌"
        
        msg = f"""
📊 BACKTEST SONUCU
━━━━━━━━━━━━━━━━━━━━━━
{r['symbol']} - Son 30 Gün
━━━━━━━━━━━━━━━━━━━━━━
📈 Toplam İşlem: {r['total_trades']}
{win_emoji} Kazanma Oranı: %{r['win_rate']}

Kazanç: {r['wins']} | Kayıp: {r['losses']}
━━━━━━━━━━━━━━━━━━━━━━
💰 Toplam Kar: %{r['total_pnl']}
📈 Ortalama Kazanç: %{r['avg_win']}
📉 Ortalama Kayıp: %{r['avg_loss']}
━━━━━━━━━━━━━━━━━━━━━━
⚖️ Risk/Ödül: {r['risk_reward']}x
📉 Max Drawdown: %{r['max_drawdown']}
📊 Profit Factor: {r['profit_factor']}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_backtest = None

def get_backtest() -> BacktestEngine:
    """Get or create backtest instance."""
    global _backtest
    if _backtest is None:
        _backtest = BacktestEngine()
    return _backtest
