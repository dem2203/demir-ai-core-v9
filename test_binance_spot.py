import asyncio
import aiohttp

async def test_spot_api():
    """Check Binance SPOT API - has taker buy volume"""
    url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                
                print("=" * 60)
                print("BTCUSDT SPOT API TICKER DATA")
                print("=" * 60)
                for key, value in data.items():
                    print(f"  {key}: {value}")
                
                print("\n" + "=" * 60)
                print("KEY FIELDS FOR MONEY FLOW:")
                print("=" * 60)
                print(f"  quoteVolume: {data.get('quoteVolume', 'NOT FOUND')}")
                print(f"  volume: {data.get('volume', 'NOT FOUND')}")
                
                # Calculate buyer percentage
                total_vol = float(data.get('quoteVolume', 0))
                taker_buy_vol = float(data.get('takerBuyQuoteAssetVol', 0) or data.get('takerBuyQuoteVolume', 0))
                
                if total_vol > 0 and taker_buy_vol > 0:
                    buyer_pct = (taker_buy_vol / total_vol) * 100
                    print(f"\n  BUYER % = {buyer_pct:.1f}%")
                else:
                    print(f"\n  takerBuyQuoteAssetVol: NOT FOUND")
            else:
                print(f"API Error: {response.status}")

asyncio.run(test_spot_api())
