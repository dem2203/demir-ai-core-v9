"""v5 Model Backtest Report Generator"""
import sys
sys.path.insert(0, '.')

import ccxt
import numpy as np
import json

print('='*60)
print('v5 MODEL BACKTEST REPORT')
print('='*60)

results = []

for symbol in ['BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'SOL/USDT']:
    print(f'\nBacktesting {symbol}...')
    
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=720)  # 30 days
    
    prices = [c[4] for c in ohlcv]
    
    balance = 10000
    position = None
    entry_price = 0
    wins = 0
    losses = 0
    trades = []
    
    for i in range(50, len(prices)):
        current = prices[i]
        avg_recent = np.mean(prices[i-20:i])
        
        if position is None:
            if current < avg_recent * 0.98:
                position = 'LONG'
                entry_price = current
        else:
            pnl_pct = (current - entry_price) / entry_price
            
            if pnl_pct < -0.02 or pnl_pct > 0.04 or current > avg_recent * 1.02:
                pnl = balance * pnl_pct * 0.999
                balance += pnl
                trades.append(pnl)
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = None
    
    if position:
        pnl_pct = (prices[-1] - entry_price) / entry_price
        pnl = balance * pnl_pct * 0.999
        balance += pnl
        trades.append(pnl)
        if pnl > 0:
            wins += 1
        else:
            losses += 1
    
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    roi = (balance - 10000) / 10000 * 100
    
    # Sharpe calculation
    if trades:
        returns = np.array(trades) / 10000
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0
    
    result = {
        'symbol': symbol,
        'roi': roi,
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'trades': total_trades,
        'sharpe': sharpe,
        'final_balance': balance
    }
    results.append(result)
    
    print(f'  ROI: {roi:+.2f}%')
    print(f'  Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)')
    print(f'  Sharpe: {sharpe:.2f}')
    print(f'  Final: ${balance:,.2f}')

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
for r in results:
    print(f"{r['symbol']}: ROI={r['roi']:+.2f}%, WR={r['win_rate']:.1f}%, Sharpe={r['sharpe']:.2f}")

avg_roi = np.mean([r['roi'] for r in results])
avg_sharpe = np.mean([r['sharpe'] for r in results])
print(f'\nAverage ROI: {avg_roi:+.2f}%')
print(f'Average Sharpe: {avg_sharpe:.2f}')

# Save results
with open('backtest_report.json', 'w') as f:
    json.dump({'results': results, 'avg_roi': avg_roi, 'avg_sharpe': avg_sharpe}, f, indent=2)

print('\n✅ Report saved to backtest_report.json')
