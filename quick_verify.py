# Quick data verification
import requests
from datetime import datetime

print("=" * 60)
print("  DEMIR AI - GERCEK VERI DOGRULAMA")
print("=" * 60)

# 1. BTC Fiyat
print()
print("1. BINANCE FIYAT")
resp = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"}, timeout=10)
btc = float(resp.json()["price"])
print(f"   BTC: ${btc:,.2f}")
print("   Kaynak: api.binance.com")

# 2. Fear & Greed
print()
print("2. FEAR & GREED")
resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
fg = resp.json()["data"][0]
print(f"   Index: {fg['value']} ({fg['value_classification']})")
print("   Kaynak: api.alternative.me")

# 3. Funding
print()
print("3. FUNDING RATE")
resp = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params={"symbol": "BTCUSDT", "limit": 1}, timeout=10)
fr = float(resp.json()[0]["fundingRate"]) * 100
print(f"   Funding: {fr:.4f}%")
print("   Kaynak: fapi.binance.com")

# 4. L/S Ratio
print()
print("4. LONG/SHORT RATIO")
resp = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio", params={"symbol": "BTCUSDT", "period": "1h", "limit": 1}, timeout=10)
ls = float(resp.json()[0]["longShortRatio"])
print(f"   L/S: {ls:.2f}")
print("   Kaynak: fapi.binance.com")

# 5. Order Book
print()
print("5. ORDER BOOK (WHALE)")
resp = requests.get("https://api.binance.com/api/v3/depth", params={"symbol": "BTCUSDT", "limit": 100}, timeout=10)
data = resp.json()
bid = sum(float(b[1]) for b in data["bids"])
ask = sum(float(a[1]) for a in data["asks"])
if bid/ask > 1.5:
    whale = "ALICI AGIRLIKLI"
elif bid/ask < 0.67:
    whale = "SATICI AGIRLIKLI"
else:
    whale = "DENGELI"
print(f"   Bid: {bid:.1f} BTC, Ask: {ask:.1f} BTC")
print(f"   Oran: {bid/ask:.2f} ({whale})")
print("   Kaynak: api.binance.com")

print()
print("=" * 60)
print("  TUM VERILER GERCEK API KAYNAKLARINDAN!")
print("  MOCK DATA YOK!")
print("=" * 60)
