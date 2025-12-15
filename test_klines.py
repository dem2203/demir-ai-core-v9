import asyncio
import aiohttp

async def test_klines_api():
    """Binance Klines API - index 10 is taker buy quote volume"""
    # Get last 24 candles of 1h = 24 hours of data
    url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=24"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                klines = await response.json()
                
                print("=" * 60)
                print("BTCUSDT KLINES DATA (Last 24 hours)")
                print("=" * 60)
                print("Kline format:")
                print("  [0] Open time")
                print("  [5] Volume (base)")
                print("  [7] Quote volume (total USDT traded)")
                print("  [9] Number of trades")
                print("  [10] Taker buy base volume")
                print("  [11] Taker buy quote volume <-- WE NEED THIS!")
                
                print("\n" + "=" * 60)
                print("LAST 3 CANDLES:")
                print("=" * 60)
                
                for kline in klines[-3:]:
                    quote_vol = float(kline[7])
                    taker_buy_quote = float(kline[10])  # Index 10 is taker buy BASE volume
                    taker_buy_quote_vol = float(kline[11]) if len(kline) > 11 else 0
                    
                    print(f"  Quote Volume: ${quote_vol:,.0f}")
                    print(f"  Taker Buy Base Vol: {taker_buy_quote:,.4f}")
                    if taker_buy_quote_vol > 0:
                        buyer_pct = (taker_buy_quote_vol / quote_vol) * 100
                        print(f"  Taker Buy Quote Vol: ${taker_buy_quote_vol:,.0f}")
                        print(f"  >>> BUYER %: {buyer_pct:.1f}%")
                    print()
                
                # Calculate 24h aggregate
                total_quote = sum(float(k[7]) for k in klines)
                total_taker_buy = sum(float(k[11]) if len(k) > 11 else 0 for k in klines)
                
                if total_quote > 0 and total_taker_buy > 0:
                    buyer_24h = (total_taker_buy / total_quote) * 100
                    print("=" * 60)
                    print(f"24H AGGREGATE BUYER %: {buyer_24h:.1f}%")
                    print("=" * 60)
            else:
                print(f"API Error: {response.status}")

asyncio.run(test_klines_api())
