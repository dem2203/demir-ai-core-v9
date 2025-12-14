"""
Real RL v5 Model Backtest - Fixed Version
Uses actual v5 PPO models for trading decisions
"""
import sys
sys.path.insert(0, '.')

import ccxt
import numpy as np
import pandas as pd
from src.brain.rl_agent.ppo_agent import RLAgent
from src.brain.feature_engineering import FeatureEngineer

print('='*60)
print('REAL RL v5 MODEL BACKTEST')
print('='*60)

# Initialize
feature_eng = FeatureEngineer()
exchange = ccxt.binance()

results = []

models = {
    'BTC/USDT': 'ppo_btc_v5',
    'ETH/USDT': 'ppo_eth_v5',
    'LTC/USDT': 'ppo_ltc_v5',
    'SOL/USDT': 'ppo_sol_v5'
}

for symbol, model_name in models.items():
    print(f'\n{"="*40}')
    print(f'Backtesting {symbol} with {model_name}')
    print('='*40)
    
    # Load model
    agent = RLAgent()
    loaded = agent.load(model_name)
    
    if not loaded:
        print(f'  ERROR: Could not load {model_name}')
        continue
    
    print(f'  Model loaded successfully')
    
    # Fetch data (60 days for proper feature calculation)
    ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=1440)
    
    # Convert to dict format for process_data
    raw_data = []
    for candle in ohlcv:
        raw_data.append({
            'timestamp': candle[0],
            'open': candle[1],
            'high': candle[2],
            'low': candle[3],
            'close': candle[4],
            'volume': candle[5]
        })
    
    # Process data with FeatureEngineer
    features_df = FeatureEngineer.process_data(raw_data)
    
    if features_df is None or features_df.empty:
        print(f'  ERROR: Could not calculate features')
        continue
    
    print(f'  Features calculated: {len(features_df)} rows, {features_df.shape[1]} cols')
    
    # Get numeric columns only
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Backtest simulation
    balance = 10000.0
    position = None
    entry_price = 0
    wins = 0
    losses = 0
    trades = []
    
    # Start from row 100 to ensure enough history
    for i in range(100, len(features_df) - 1):
        current_price = float(features_df.iloc[i]['close'])
        
        # Get state for model (only numeric features)
        try:
            state = features_df.iloc[i][numeric_cols].values.astype(np.float32)
            
            # Handle NaN/Inf
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Get action from model
            action = agent.predict(state)
            
        except Exception as e:
            action = 1  # Hold on error
        
        # Action mapping: 0=Sell, 1=Hold, 2=Buy
        if position is None:
            if action == 2:  # Buy signal
                position = 'LONG'
                entry_price = current_price
            elif action == 0:  # Sell signal (short)
                position = 'SHORT'
                entry_price = current_price
        else:
            # Exit conditions
            if position == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Exit on opposite signal or SL/TP
                if action == 0 or pnl_pct < -0.02 or pnl_pct > 0.05:
                    pnl = balance * pnl_pct * 0.999  # 0.1% fee
                    balance += pnl
                    trades.append(pnl)
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    position = None
                    
            elif position == 'SHORT':
                pnl_pct = (entry_price - current_price) / entry_price
                
                if action == 2 or pnl_pct < -0.02 or pnl_pct > 0.05:
                    pnl = balance * pnl_pct * 0.999
                    balance += pnl
                    trades.append(pnl)
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    position = None
    
    # Close any open position
    if position:
        final_price = float(features_df.iloc[-1]['close'])
        if position == 'LONG':
            pnl_pct = (final_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - final_price) / entry_price
        pnl = balance * pnl_pct * 0.999
        balance += pnl
        trades.append(pnl)
        if pnl > 0:
            wins += 1
        else:
            losses += 1
    
    # Calculate metrics
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    roi = (balance - 10000) / 10000 * 100
    
    if trades:
        returns = np.array(trades) / 10000
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = min(trades) / 10000 * 100 if trades else 0
    else:
        sharpe = 0
        max_drawdown = 0
    
    result = {
        'symbol': symbol,
        'model': model_name,
        'roi': roi,
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'trades': total_trades,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'final_balance': balance
    }
    results.append(result)
    
    print(f'  ROI: {roi:+.2f}%')
    print(f'  Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)')
    print(f'  Sharpe: {sharpe:.2f}')
    print(f'  Max Drawdown: {max_drawdown:.2f}%')
    print(f'  Final Balance: ${balance:,.2f}')

# Summary
print('\n' + '='*60)
print('REAL RL MODEL BACKTEST SUMMARY')
print('='*60)

for r in results:
    print(f"{r['symbol']}: ROI={r['roi']:+.2f}%, WR={r['win_rate']:.1f}%, Sharpe={r['sharpe']:.2f}")

if results:
    avg_roi = np.mean([r['roi'] for r in results])
    avg_wr = np.mean([r['win_rate'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    
    print(f'\nAverage ROI: {avg_roi:+.2f}%')
    print(f'Average Win Rate: {avg_wr:.1f}%')
    print(f'Average Sharpe: {avg_sharpe:.2f}')

print('\nDone!')
