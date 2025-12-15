import asyncio
import aiohttp

async def test_binance_api():
    """Check actual Binance Futures API response fields"""
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                
                # Find BTCUSDT
                btc_data = None
                for item in data:
                    if item.get('symbol') == 'BTCUSDT':
                        btc_data = item
                        break
                
                if btc_data:
                    print("=" * 60)
                    print("BTCUSDT TICKER DATA FROM BINANCE FUTURES API")
                    print("=" * 60)
                    for key, value in btc_data.items():
                        print(f"  {key}: {value}")
                    
                    print("\n" + "=" * 60)
                    print("KEY FIELDS FOR MONEY FLOW:")
                    print("=" * 60)
                    print(f"  quoteVolume: {btc_data.get('quoteVolume', 'NOT FOUND')}")
                    print(f"  volume: {btc_data.get('volume', 'NOT FOUND')}")
                    print(f"  takerBuyQuoteAssetVol: {btc_data.get('takerBuyQuoteAssetVol', 'NOT FOUND')}")
                    print(f"  takerBuyBaseAssetVolume: {btc_data.get('takerBuyBaseAssetVolume', 'NOT FOUND')}")
            else:
                print(f"API Error: {response.status}")

asyncio.run(test_binance_api())
