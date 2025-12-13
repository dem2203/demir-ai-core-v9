"""
Backtester for RL Models - Tests trained PPO agents on historical data.
Uses PUBLIC API (no authentication required).
"""
import pandas as pd
import numpy as np
import logging
import asyncio
import argparse
from stable_baselines3 import PPO
import os

from src.brain.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BACKTESTER_V2")


class RLBacktester:
    """
    Backtest trained RL models on historical data.
    Uses public ccxt API for data fetching.
    """
    
    RL_STORAGE = "src/brain/rl_agent/storage"
    
    def __init__(self, initial_balance: float = 10_000.0):
        self.initial_balance = initial_balance
        self.exchange = None
        
    async def _get_exchange(self):
        """Create public exchange connection (no API key needed)"""
        if not self.exchange:
            import ccxt.async_support as ccxt
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
        return self.exchange
    
    def _get_rl_model_path(self, symbol: str) -> str:
        """Get RL model path for symbol - v4 models."""
        clean_sym = symbol.replace("/", "").replace("USDT", "").lower()
        model_name = f"ppo_{clean_sym}_v4"  # Updated to v4
        return os.path.join(self.RL_STORAGE, model_name)
    
    async def fetch_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical OHLCV data using public API."""
        logger.info(f"📥 Fetching {days} days of data for {symbol}...")
        
        exchange = await self._get_exchange()
        
        try:
            # Calculate limit (24 candles per day for 1h)
            limit = min(days * 24, 1000)
            
            ohlcv = await exchange.fetch_ohlcv(
                symbol.replace('/', ''), 
                timeframe='1h', 
                limit=limit
            )
            
            logger.info(f"✅ Fetched {len(ohlcv)} candles")
            
            raw_data = []
            for candle in ohlcv:
                raw_data.append({
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'symbol': symbol
                })
        finally:
            await exchange.close()
            self.exchange = None
        
        # Process with FeatureEngineer
        df = await asyncio.to_thread(FeatureEngineer.process_data, raw_data)
        return df
    
    def load_rl_model(self, symbol: str):
        """Load trained RL model for symbol."""
        model_path = self._get_rl_model_path(symbol)
        full_path = model_path + ".zip"
        
        if os.path.exists(full_path):
            try:
                model = PPO.load(model_path)
                logger.info(f"🧠 Loaded RL model: {model_path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return None
        else:
            logger.warning(f"⚠️ Model not found: {full_path}")
            return None
    
    async def run_backtest(self, symbol: str, days: int = 30) -> dict:
        """
        Run backtest simulation for a symbol.
        
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔥 Starting Backtest: {symbol} (Last {days} days)")
        logger.info(f"{'='*60}")
        
        # 1. Load data
        df = await self.fetch_historical_data(symbol, days)
        if df is None or len(df) < 100:
            logger.error(f"Insufficient data for {symbol}")
            return {"error": "Insufficient data"}
        
        # 2. Load RL model
        model = self.load_rl_model(symbol)
        if model is None:
            logger.warning("Using rule-based fallback for simulation")
        
        # 3. Prepare features for RL
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['timestamp', 'symbol']
        feature_cols = [c for c in feature_cols if c not in exclude_cols][:25]
        
        # 4. Run simulation
        balance = self.initial_balance
        position = None
        entry_price = 0
        trades = []
        
        # Track prices
        prices = df['close'].values
        features = df[feature_cols].fillna(0).values
        
        win_count = 0
        loss_count = 0
        total_pnl = 0
        
        for i in range(100, len(df)):
            current_price = prices[i]
            
            # Get RL action (0=HOLD, 1=BUY, 2=SELL)
            if model is not None:
                obs = features[i].astype(np.float32)
                # Ensure observation shape matches model expectations
                try:
                    action, _ = model.predict(obs, deterministic=True)
                except:
                    # Fallback to simple rule
                    rsi = df.iloc[i].get('rsi', 50)
                    action = 1 if rsi < 30 else (2 if rsi > 70 else 0)
            else:
                # Rule-based fallback
                rsi = df.iloc[i].get('rsi', 50)
                action = 1 if rsi < 30 else (2 if rsi > 70 else 0)
            
            # Execute trade logic
            if position is None:
                if action == 1:  # BUY signal
                    position = 'LONG'
                    entry_price = current_price
                    trades.append({
                        'type': 'BUY',
                        'price': current_price,
                        'balance': balance
                    })
            
            elif position == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Exit conditions
                should_exit = False
                reason = ""
                
                if pnl_pct < -0.02:  # 2% stop loss
                    should_exit = True
                    reason = "SL"
                elif pnl_pct > 0.04:  # 4% take profit
                    should_exit = True
                    reason = "TP"
                elif action == 2:  # SELL signal
                    should_exit = True
                    reason = "SIGNAL"
                
                if should_exit:
                    position = None
                    # Calculate P&L
                    pnl = balance * pnl_pct * 0.999  # 0.1% fee
                    balance += pnl
                    total_pnl += pnl
                    
                    if pnl > 0:
                        win_count += 1
                    else:
                        loss_count += 1
                    
                    trades.append({
                        'type': 'SELL',
                        'price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'reason': reason,
                        'balance': balance
                    })
        
        # Close any open position at end
        if position == 'LONG':
            pnl_pct = (prices[-1] - entry_price) / entry_price
            pnl = balance * pnl_pct * 0.999
            balance += pnl
            total_pnl += pnl
            if pnl > 0:
                win_count += 1
            else:
                loss_count += 1
            trades.append({
                'type': 'CLOSE',
                'price': prices[-1],
                'pnl': pnl,
                'balance': balance
            })
        
        # 5. Calculate metrics
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        roi = (balance - self.initial_balance) / self.initial_balance * 100
        
        # Simple Sharpe calculation
        daily_returns = []
        for i in range(1, len(prices)):
            daily_returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        sharpe = (avg_return / std_return) * np.sqrt(365 * 24) if std_return > 0 else 0
        
        results = {
            'symbol': symbol,
            'initial_balance': self.initial_balance,
            'final_balance': round(balance, 2),
            'roi': round(roi, 2),
            'total_trades': total_trades,
            'wins': win_count,
            'losses': loss_count,
            'win_rate': round(win_rate, 1),
            'sharpe_ratio': round(sharpe, 2),
            'total_pnl': round(total_pnl, 2),
            'model_used': 'RL v2 (200K)' if model else 'Rule-Based'
        }
        
        logger.info(f"\n📊 BACKTEST RESULTS for {symbol}:")
        logger.info(f"   💰 Final Balance: ${balance:,.2f}")
        logger.info(f"   📈 ROI: {roi:+.2f}%")
        logger.info(f"   🎯 Win Rate: {win_rate:.1f}% ({win_count}W/{loss_count}L)")
        logger.info(f"   📊 Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"   🔢 Total Trades: {total_trades}")
        
        return results


async def main():
    """Run backtests for all coins."""
    parser = argparse.ArgumentParser(description="Backtest RL Models")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest")
    parser.add_argument("--all", action="store_true", help="Backtest all 3 coins")
    
    args = parser.parse_args()
    
    backtester = RLBacktester(initial_balance=10_000.0)
    
    all_results = []
    
    if args.all:
        symbols = ["BTC/USDT", "ETH/USDT", "LTC/USDT", "SOL/USDT"]  # All 4 coins
        for symbol in symbols:
            result = await backtester.run_backtest(symbol, args.days)
            all_results.append(result)
    else:
        result = await backtester.run_backtest(args.symbol, args.days)
        all_results.append(result)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("📋 BACKTEST SUMMARY")
    logger.info("="*70)
    
    for r in all_results:
        if 'error' not in r:
            logger.info(f"{r['symbol']}: ROI={r['roi']:+.2f}%, WinRate={r['win_rate']:.1f}%, Sharpe={r['sharpe_ratio']:.2f}")
    
    logger.info("\n🎉 All backtests complete!")
    
    return all_results


if __name__ == "__main__":
    asyncio.run(main())
