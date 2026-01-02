# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - PROFESSIONAL BACKTEST ENGINE
============================================
Historical strategy validation framework.

FEATURES:
1. Load historical klines (configurable timeframe)
2. Simulate signals with realistic slippage/fees
3. Calculate professional metrics:
   - Sharpe Ratio
   - Max Drawdown
   - Win Rate
   - Profit Factor
   - Sortino Ratio
4. Strategy comparison

Usage:
    engine = BacktestEngine()
    results = await engine.run_backtest("BTCUSDT", days=365)
    print(results.summary())
"""
import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path

logger = logging.getLogger("BACKTEST_ENGINE")


@dataclass
class BacktestTrade:
    """Single trade in backtest"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str  # LONG, SHORT
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    pnl_usd: float
    pnl_pct: float
    is_win: bool
    exit_reason: str  # TP_HIT, SL_HIT, TIME_EXIT, SIGNAL_EXIT


@dataclass
class BacktestMetrics:
    """Professional trading metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl_usd: float = 0.0
    total_pnl_pct: float = 0.0
    
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    avg_trade_duration_hours: float = 0.0
    longest_win_streak: int = 0
    longest_lose_streak: int = 0
    
    start_equity: float = 0.0
    end_equity: float = 0.0
    
    def summary(self) -> str:
        return f"""
📊 BACKTEST RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 PERFORMANCE
  • Total P&L: ${self.total_pnl_usd:+,.0f} ({self.total_pnl_pct:+.1f}%)
  • Start Equity: ${self.start_equity:,.0f}
  • End Equity: ${self.end_equity:,.0f}

📈 TRADE STATS
  • Total Trades: {self.total_trades}
  • Win Rate: {self.win_rate:.1f}%
  • Profit Factor: {self.profit_factor:.2f}
  • Avg Win: ${self.avg_win_usd:,.0f} ({self.avg_win_pct:+.1f}%)
  • Avg Loss: ${self.avg_loss_usd:,.0f} ({self.avg_loss_pct:.1f}%)

📉 RISK METRICS
  • Max Drawdown: {self.max_drawdown_pct:.1f}% (${self.max_drawdown_usd:,.0f})
  • Sharpe Ratio: {self.sharpe_ratio:.2f}
  • Sortino Ratio: {self.sortino_ratio:.2f}

⏱️ TIME STATS
  • Avg Trade Duration: {self.avg_trade_duration_hours:.1f} hours
  • Win Streak: {self.longest_win_streak}
  • Lose Streak: {self.longest_lose_streak}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    def is_profitable(self) -> bool:
        return self.total_pnl_usd > 0
    
    def is_acceptable(self) -> bool:
        """Check if strategy meets minimum requirements"""
        return (
            self.win_rate >= 45 and
            self.profit_factor >= 1.2 and
            self.max_drawdown_pct <= 25 and
            self.sharpe_ratio >= 0.5
        )


class BacktestEngine:
    """
    Professional Backtest Engine
    
    Simulates trading strategies on historical data with:
    - Realistic slippage and fees
    - Proper position sizing
    - Risk management integration
    """
    
    # === SIMULATION SETTINGS ===
    DEFAULT_SLIPPAGE_PCT = 0.05       # 0.05% slippage per trade
    DEFAULT_FEE_PCT = 0.04            # 0.04% fee (taker)
    DEFAULT_INITIAL_EQUITY = 10000    # $10k starting capital
    MAX_TRADES_PER_DAY = 5            # Limit overtrading
    
    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.trades: List[BacktestTrade] = []
        
        # Data directory
        self.data_dir = Path("data/backtest")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📊 Backtest Engine initialized")
    
    async def load_historical_data(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 365
    ) -> pd.DataFrame:
        """Load historical klines from Binance"""
        
        # Check cache first
        cache_key = f"{symbol}_{interval}_{days}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Check disk cache
        cache_file = self.data_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            self.data_cache[cache_key] = df
            logger.info(f"📂 Loaded cached data: {len(df)} candles")
            return df
        
        # Fetch from Binance
        logger.info(f"🌐 Fetching {days} days of {symbol} {interval} data...")
        
        all_klines = []
        end_time = int(datetime.now().timestamp() * 1000)
        batch_size = 1000  # Binance limit
        
        # Calculate how many candles we need
        interval_hours = {'1h': 1, '4h': 4, '1d': 24}.get(interval, 1)
        total_candles = (days * 24) // interval_hours
        
        async with aiohttp.ClientSession() as session:
            while len(all_klines) < total_candles:
                url = f"https://fapi.binance.com/fapi/v1/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': batch_size,
                    'endTime': end_time
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        klines = await resp.json()
                        if not klines:
                            break
                        all_klines = klines + all_klines
                        end_time = klines[0][0] - 1  # Before first candle
                    else:
                        logger.error(f"API error: {resp.status}")
                        break
                
                await asyncio.sleep(0.1)  # Rate limiting
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Clean and type convert
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df.set_index('timestamp')
        
        # Cache to disk
        df.to_parquet(cache_file)
        self.data_cache[cache_key] = df
        
        logger.info(f"✅ Loaded {len(df)} candles ({days} days)")
        return df
    
    async def run_backtest(
        self,
        symbol: str,
        signal_generator: Callable,
        days: int = 365,
        initial_equity: float = None,
        interval: str = "1h"
    ) -> BacktestMetrics:
        """
        Run backtest with given signal generator.
        
        signal_generator: async function(df, index) -> dict or None
            Returns: {'action': 'BUY'/'SELL', 'confidence': 80, 'sl': 100, 'tp': 110}
        """
        initial_equity = initial_equity or self.DEFAULT_INITIAL_EQUITY
        
        # Load data
        df = await self.load_historical_data(symbol, interval, days)
        
        # Initialize tracking
        equity = initial_equity
        peak_equity = initial_equity
        trades: List[BacktestTrade] = []
        equity_curve = [initial_equity]
        position = None
        daily_trade_count = {}
        
        logger.info(f"🚀 Starting backtest: {symbol} | {days} days | ${initial_equity:,.0f}")
        
        # Iterate through data
        for i in range(50, len(df)):  # Start after warmup period
            current_bar = df.iloc[i]
            current_time = df.index[i]
            current_date = current_time.date()
            
            # Check daily trade limit
            if daily_trade_count.get(current_date, 0) >= self.MAX_TRADES_PER_DAY:
                continue
            
            # If in position, check exit conditions
            if position:
                exit_price = None
                exit_reason = None
                
                # Check SL
                if position['direction'] == 'LONG':
                    if current_bar['low'] <= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'SL_HIT'
                    elif current_bar['high'] >= position['tp']:
                        exit_price = position['tp']
                        exit_reason = 'TP_HIT'
                else:  # SHORT
                    if current_bar['high'] >= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'SL_HIT'
                    elif current_bar['low'] <= position['tp']:
                        exit_price = position['tp']
                        exit_reason = 'TP_HIT'
                
                # Max hold time (48 candles)
                if not exit_price and i - position['entry_idx'] > 48:
                    exit_price = current_bar['close']
                    exit_reason = 'TIME_EXIT'
                
                # Exit position
                if exit_price:
                    # Apply slippage
                    if position['direction'] == 'LONG':
                        exit_price *= (1 - self.DEFAULT_SLIPPAGE_PCT / 100)
                    else:
                        exit_price *= (1 + self.DEFAULT_SLIPPAGE_PCT / 100)
                    
                    # Calculate P&L
                    if position['direction'] == 'LONG':
                        pnl_pct = (exit_price - position['entry']) / position['entry'] * 100
                    else:
                        pnl_pct = (position['entry'] - exit_price) / position['entry'] * 100
                    
                    pnl_pct -= self.DEFAULT_FEE_PCT  # Exit fee
                    pnl_usd = position['size'] * (pnl_pct / 100)
                    
                    trade = BacktestTrade(
                        entry_time=df.index[position['entry_idx']],
                        exit_time=current_time,
                        symbol=symbol,
                        direction=position['direction'],
                        entry_price=position['entry'],
                        exit_price=exit_price,
                        stop_loss=position['sl'],
                        take_profit=position['tp'],
                        position_size=position['size'],
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        is_win=pnl_usd > 0,
                        exit_reason=exit_reason
                    )
                    trades.append(trade)
                    
                    equity += pnl_usd
                    equity_curve.append(equity)
                    peak_equity = max(peak_equity, equity)
                    position = None
            
            # Generate signal if no position
            if not position:
                # Create slice for signal generator
                df_slice = df.iloc[:i+1]
                signal = await signal_generator(df_slice, i)
                
                if signal and signal.get('action') in ['BUY', 'SELL']:
                    entry_price = current_bar['close']
                    entry_price *= (1 + self.DEFAULT_SLIPPAGE_PCT / 100)  # Slippage
                    entry_price *= (1 + self.DEFAULT_FEE_PCT / 100)  # Entry fee
                    
                    # Position size (simple: 2% risk per trade)
                    risk_pct = 0.02
                    position_size = equity * risk_pct
                    
                    position = {
                        'direction': 'LONG' if signal['action'] == 'BUY' else 'SHORT',
                        'entry': entry_price,
                        'sl': signal.get('sl', entry_price * 0.98),
                        'tp': signal.get('tp', entry_price * 1.04),
                        'size': position_size,
                        'entry_idx': i
                    }
                    
                    daily_trade_count[current_date] = daily_trade_count.get(current_date, 0) + 1
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, initial_equity, equity, equity_curve)
        self.trades = trades
        
        logger.info(f"✅ Backtest complete: {len(trades)} trades | P&L: ${metrics.total_pnl_usd:+,.0f}")
        return metrics
    
    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        start_equity: float,
        end_equity: float,
        equity_curve: List[float]
    ) -> BacktestMetrics:
        """Calculate professional trading metrics"""
        
        metrics = BacktestMetrics()
        metrics.total_trades = len(trades)
        metrics.start_equity = start_equity
        metrics.end_equity = end_equity
        
        if not trades:
            return metrics
        
        # Basic stats
        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if not t.is_win]
        
        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = len(wins) / len(trades) * 100
        
        # P&L
        metrics.total_pnl_usd = sum(t.pnl_usd for t in trades)
        metrics.total_pnl_pct = (end_equity - start_equity) / start_equity * 100
        
        metrics.gross_profit = sum(t.pnl_usd for t in wins) if wins else 0
        metrics.gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 0
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf')
        
        # Averages
        if wins:
            metrics.avg_win_usd = sum(t.pnl_usd for t in wins) / len(wins)
            metrics.avg_win_pct = sum(t.pnl_pct for t in wins) / len(wins)
        if losses:
            metrics.avg_loss_usd = abs(sum(t.pnl_usd for t in losses) / len(losses))
            metrics.avg_loss_pct = abs(sum(t.pnl_pct for t in losses) / len(losses))
        
        # Drawdown
        peak = start_equity
        max_dd_usd = 0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = peak - eq
            max_dd_usd = max(max_dd_usd, dd)
        
        metrics.max_drawdown_usd = max_dd_usd
        metrics.max_drawdown_pct = max_dd_usd / peak * 100 if peak > 0 else 0
        
        # Sharpe & Sortino
        returns = [t.pnl_pct for t in trades]
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Annualized (assuming daily trades)
            metrics.sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
            
            # Sortino (downside deviation only)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                metrics.sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Streaks
        current_streak = 0
        max_win_streak = 0
        max_lose_streak = 0
        prev_win = None
        
        for t in trades:
            if prev_win is None or t.is_win == prev_win:
                current_streak += 1
            else:
                current_streak = 1
            
            if t.is_win:
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_lose_streak = max(max_lose_streak, current_streak)
            
            prev_win = t.is_win
        
        metrics.longest_win_streak = max_win_streak
        metrics.longest_lose_streak = max_lose_streak
        
        # Duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        metrics.avg_trade_duration_hours = np.mean(durations) if durations else 0
        
        return metrics
    
    async def compare_strategies(
        self,
        symbol: str,
        strategies: Dict[str, Callable],
        days: int = 365
    ) -> Dict[str, BacktestMetrics]:
        """Compare multiple strategies"""
        results = {}
        
        for name, signal_gen in strategies.items():
            logger.info(f"🧪 Testing strategy: {name}")
            metrics = await self.run_backtest(symbol, signal_gen, days)
            results[name] = metrics
        
        # Print comparison
        print("\n📊 STRATEGY COMPARISON")
        print("=" * 60)
        print(f"{'Strategy':<20} {'P&L':>10} {'Win Rate':>10} {'Sharpe':>10}")
        print("-" * 60)
        
        for name, m in sorted(results.items(), key=lambda x: x[1].total_pnl_usd, reverse=True):
            print(f"{name:<20} ${m.total_pnl_usd:>+9,.0f} {m.win_rate:>9.1f}% {m.sharpe_ratio:>9.2f}")
        
        return results


# Simple signal generators for testing
async def simple_ma_crossover(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """Simple Moving Average Crossover strategy"""
    if len(df) < 50:
        return None
    
    short_ma = df['close'].rolling(9).mean().iloc[-1]
    long_ma = df['close'].rolling(21).mean().iloc[-1]
    prev_short = df['close'].rolling(9).mean().iloc[-2]
    prev_long = df['close'].rolling(21).mean().iloc[-2]
    
    current_price = df['close'].iloc[-1]
    
    # Crossover
    if prev_short <= prev_long and short_ma > long_ma:
        return {
            'action': 'BUY',
            'confidence': 70,
            'sl': current_price * 0.98,
            'tp': current_price * 1.04
        }
    elif prev_short >= prev_long and short_ma < long_ma:
        return {
            'action': 'SELL',
            'confidence': 70,
            'sl': current_price * 1.02,
            'tp': current_price * 0.96
        }
    
    return None


async def rsi_mean_reversion(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """RSI Mean Reversion strategy"""
    if len(df) < 20:
        return None
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[-1]
    current_price = df['close'].iloc[-1]
    
    if current_rsi < 30:  # Oversold
        return {
            'action': 'BUY',
            'confidence': 75,
            'sl': current_price * 0.97,
            'tp': current_price * 1.03
        }
    elif current_rsi > 70:  # Overbought
        return {
            'action': 'SELL',
            'confidence': 75,
            'sl': current_price * 1.03,
            'tp': current_price * 0.97
        }
    
    return None


# Singleton
_backtest_engine: Optional[BacktestEngine] = None

def get_backtest_engine() -> BacktestEngine:
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine()
    return _backtest_engine
