"""Test Telegram Signal - v5 Backtest Report"""
import sys
sys.path.insert(0, '.')
import asyncio
import os

# Check if TELEGRAM_TOKEN exists
if not os.environ.get('TELEGRAM_TOKEN'):
    print("ERROR: TELEGRAM_TOKEN not set in environment")
    print("Please set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID")
    exit(1)

from src.utils.notifications import Notifier

async def send_test():
    notifier = Notifier()
    
    # Test signal - v5 Backtest Report
    test_signal = {
        'symbol': 'BTC/USDT',
        'signal': 'BUY',
        'ai_decision': 'BUY',
        'ai_confidence': 85.0,
        'reason': 'v5 Backtest: +9.24% ROI, 54.5% Win Rate, Sharpe 4.63'
    }
    
    # Snapshot with advanced data
    test_snapshot = {
        'price': 101500,
        'smart_sltp': {
            'valid': True,
            'direction': 'LONG',
            'stop_loss': 98000,
            'take_profit_1': 105000,
            'take_profit_2': 110000,
            'take_profit_3': 115000,
            'risk_pct': 3.4,
            'risk_reward_1': '1:1',
            'risk_reward_2': '1:2.5',
            'risk_reward_3': '1:4',
            'quality': 'GOOD'
        },
        'mtf': {
            'confluence_score': 75,
            'trends': {
                '1h': {'trend': 'BULLISH'},
                '4h': {'trend': 'BULLISH'},
                '1d': {'trend': 'NEUTRAL'}
            }
        },
        'smc': {
            'smc_bias': 'BULLISH'
        }
    }
    
    print("Sending test signal to Telegram...")
    
    try:
        result = await notifier.send_signal(test_signal, test_snapshot)
        if result:
            print("SUCCESS: Test signal sent!")
        else:
            print("FAILED: Signal returned False")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(send_test())
