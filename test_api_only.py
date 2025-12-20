# -*- coding: utf-8 -*-
"""
DEMIR AI - SIMPLIFIED DATA SOURCE TEST (API ONLY)
==================================================
Playwright olmadan, sadece API'lerle veri kaynaklarını test eder.
"""
import asyncio
import aiohttp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

async def test_api_sources():
    print("=" * 60)
    print("DEMIR AI - API DATA SOURCE TEST")
    print("Time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 60)
    
    results = {}
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        
        # 1. BINANCE PRICE
        print("\n[1] Binance Price...")
        try:
            async with session.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT") as resp:
                data = await resp.json()
                price = float(data.get('price', 0))
                print(f"    OK - BTC Price: ${price:,.2f}")
                results['binance_price'] = True
        except Exception as e:
            print(f"    FAIL - {e}")
            results['binance_price'] = False
        
        # 2. ORDER BOOK
        print("\n[2] Order Book Depth...")
        try:
            async with session.get("https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=20") as resp:
                data = await resp.json()
                bids = sum(float(b[1]) for b in data.get('bids', []))
                asks = sum(float(a[1]) for a in data.get('asks', []))
                imb = bids / asks if asks > 0 else 1.0
                print(f"    OK - Bid: {bids:.2f}, Ask: {asks:.2f}, Imbalance: {imb:.2f}x")
                results['orderbook'] = True
        except Exception as e:
            print(f"    FAIL - {e}")
            results['orderbook'] = False
        
        # 3. FUNDING RATE
        print("\n[3] Funding Rate...")
        try:
            async with session.get("https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1") as resp:
                data = await resp.json()
                rate = float(data[0]['fundingRate']) * 100 if data else 0
                print(f"    OK - Funding: {rate:.4f}%")
                results['funding'] = True
        except Exception as e:
            print(f"    FAIL - {e}")
            results['funding'] = False
        
        # 4. OPEN INTEREST
        print("\n[4] Open Interest...")
        try:
            async with session.get("https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT") as resp:
                data = await resp.json()
                oi = float(data.get('openInterest', 0))
                print(f"    OK - OI: {oi:,.0f} BTC")
                results['oi'] = True
        except Exception as e:
            print(f"    FAIL - {e}")
            results['oi'] = False
        
        # 5. LONG/SHORT RATIO
        print("\n[5] Long/Short Ratio...")
        try:
            async with session.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=1") as resp:
                data = await resp.json()
                if data:
                    ratio = float(data[0].get('longShortRatio', 1))
                    print(f"    OK - L/S Ratio: {ratio:.2f}")
                    results['ls_ratio'] = True
                else:
                    print(f"    EMPTY - No data")
                    results['ls_ratio'] = False
        except Exception as e:
            print(f"    FAIL - {e}")
            results['ls_ratio'] = False
        
        # 6. TAKER BUY/SELL
        print("\n[6] Taker Buy/Sell Ratio...")
        try:
            async with session.get("https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=5m&limit=1") as resp:
                data = await resp.json()
                if data:
                    buy_vol = float(data[0].get('buyVol', 0))
                    sell_vol = float(data[0].get('sellVol', 0))
                    total = buy_vol + sell_vol
                    buy_ratio = buy_vol / total if total > 0 else 0.5
                    print(f"    OK - Buy Ratio: {buy_ratio:.2f} ({buy_ratio*100:.0f}%)")
                    results['taker'] = True
                else:
                    print(f"    EMPTY - No data")
                    results['taker'] = False
        except Exception as e:
            print(f"    FAIL - {e}")
            results['taker'] = False
        
        # 7. FEAR & GREED
        print("\n[7] Fear & Greed Index...")
        try:
            async with session.get("https://api.alternative.me/fng/") as resp:
                data = await resp.json()
                if data.get('data'):
                    val = int(data['data'][0].get('value', 50))
                    label = data['data'][0].get('value_classification', 'Neutral')
                    print(f"    OK - Index: {val} ({label})")
                    results['fear_greed'] = True
                else:
                    results['fear_greed'] = False
        except Exception as e:
            print(f"    FAIL - {e}")
            results['fear_greed'] = False
        
        # 8. DEFI TVL (DefiLlama)
        print("\n[8] DeFi TVL...")
        try:
            async with session.get("https://api.llama.fi/v2/historicalChainTvl") as resp:
                data = await resp.json()
                if data:
                    latest = data[-1] if isinstance(data, list) else {}
                    tvl = latest.get('tvl', 0)
                    print(f"    OK - Total TVL: ${tvl/1e9:.2f}B")
                    results['defi_tvl'] = True
                else:
                    results['defi_tvl'] = False
        except Exception as e:
            print(f"    FAIL - {e}")
            results['defi_tvl'] = False
        
        # 9. BYBIT CROSS-EXCHANGE
        print("\n[9] Cross-Exchange (Bybit)...")
        try:
            async with session.get("https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT") as resp:
                data = await resp.json()
                if data.get('result', {}).get('list'):
                    bybit_price = float(data['result']['list'][0]['lastPrice'])
                    print(f"    OK - Bybit BTC: ${bybit_price:,.2f}")
                    results['bybit'] = True
                else:
                    results['bybit'] = False
        except Exception as e:
            print(f"    FAIL - {e}")
            results['bybit'] = False
        
        # 10. WHALE TRACKER (WebSocket status)
        print("\n[10] Whale Tracker (WebSocket)...")
        try:
            from src.brain.whale_tracker import get_whale_tracker
            tracker = get_whale_tracker()
            summary = tracker.get_whale_summary()
            print(f"    OK - Net Flow: ${summary.get('net_flow_usd', 0):,.0f}")
            results['whale'] = True
        except Exception as e:
            print(f"    FAIL - {e}")
            results['whale'] = False
    
    # SUMMARY
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    live_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\nLIVE DATA: {live_count}/{total_count} sources")
    print("\nDetails:")
    for name, is_live in results.items():
        status = "OK" if is_live else "FAIL"
        print(f"  [{status}] {name}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    asyncio.run(test_api_sources())
