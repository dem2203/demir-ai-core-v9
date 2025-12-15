import asyncio
from src.data_ingestion.money_flow_analyzer import MoneyFlowAnalyzer

async def test():
    analyzer = MoneyFlowAnalyzer()
    data = await analyzer.get_market_money_flow()
    
    print("=" * 50)
    print("MONEY FLOW TEST")
    print("=" * 50)
    print(f"Market Buyer %: {data.get('market_buyer_pct')}")
    print(f"Buying Power: {data.get('buying_power')}")
    print(f"Market Flow: {data.get('market_flow')}")
    print(f"Top Inflow Count: {len(data.get('top_inflow', []))}")
    print(f"Top Inflow: {data.get('top_inflow', [])[:5]}")
    print(f"Coin Flows Keys: {list(data.get('coin_flows', {}).keys())[:5]}")
    
    # Show formatted telegram message
    print("\n" + "=" * 50)
    print("TELEGRAM FORMAT:")
    print("=" * 50)
    print(analyzer.format_for_telegram(data))

asyncio.run(test())
