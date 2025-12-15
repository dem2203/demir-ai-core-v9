import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.data_ingestion.money_flow_analyzer import MoneyFlowAnalyzer

async def test():
    analyzer = MoneyFlowAnalyzer()
    print("Fetching multi-timeframe data...")
    data = await analyzer.get_market_money_flow()
    
    print("\n" + "="*50)
    print("RAW DATA:")
    print("="*50)
    print(f"Market Buyer %: {data.get('market_buyer_pct')}")
    print(f"Buying Power: {data.get('buying_power')}")
    print(f"Timeframe Flows: {data.get('timeframe_flows')}")
    print(f"Coin Count: {len(data.get('coin_details', []))}")
    
    if data.get('coin_details'):
        print("\nTop 3 Coins:")
        for coin in data['coin_details'][:3]:
            print(f"  {coin['symbol']}: Flow {coin['flow_pct']}% | 15m {coin['buyer_15m']}% | Mts {coin['mts']} | {coin['arrows']}")

asyncio.run(test())
