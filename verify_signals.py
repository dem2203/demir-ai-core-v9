# -*- coding: utf-8 -*-
"""
Signal Verification Script
===========================
Teknik analiz sinyallerini dogrular:
1. Binance'den gercek veri ceker
2. RSI, EMA, ATR hesaplar
3. Sistem ciktisiyla karsilastirir
"""
import asyncio
import aiohttp
import json
from datetime import datetime


async def fetch_klines(symbol: str, interval: str = "1h", limit: int = 100):
    """Binance'den mum verileri cek."""
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
    return None


async def fetch_orderbook(symbol: str):
    """Order book cek."""
    url = f"https://fapi.binance.com/fapi/v1/depth"
    params = {"symbol": symbol, "limit": 20}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
    return None


async def fetch_funding(symbol: str):
    """Funding rate cek."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": 1}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data:
                    return float(data[-1]['fundingRate']) * 100
    return None


def calculate_rsi(closes, period=14):
    """RSI hesapla."""
    if len(closes) < period + 1:
        return None
    
    gains = []
    losses = []
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if len(gains) < period:
        return None
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(closes, period):
    """EMA hesapla."""
    if len(closes) < period:
        return None
    
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    
    for close in closes[period:]:
        ema = (close - ema) * multiplier + ema
    
    return ema


async def verify_symbol(symbol: str):
    """Tek coin icin dogrulama."""
    print(f"\n{'='*60}")
    print(f"  {symbol} DOGRULAMA")
    print(f"{'='*60}")
    
    # 1. Kline verileri
    klines = await fetch_klines(symbol)
    if not klines:
        print("[HATA] Kline verisi alinamadi")
        return False
    
    closes = [float(k[4]) for k in klines]
    current_price = closes[-1]
    
    print(f"\n[FIYAT]")
    print(f"  Guncel: ${current_price:,.2f}")
    print(f"  24h Onceki: ${closes[-24]:,.2f}")
    print(f"  Degisim: {((current_price - closes[-24]) / closes[-24] * 100):+.2f}%")
    
    # 2. RSI
    rsi = calculate_rsi(closes)
    print(f"\n[RSI (14)]")
    print(f"  Hesaplanan: {rsi:.1f}")
    if rsi > 70:
        print(f"  Durum: OVERBOUGHT (Satis baskisi olabilir)")
    elif rsi < 30:
        print(f"  Durum: OVERSOLD (Alim firsati olabilir)")
    else:
        print(f"  Durum: NOTR")
    
    # 3. EMA'lar
    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    ema50 = calculate_ema(closes, 50)
    
    print(f"\n[EMA]")
    print(f"  EMA9: ${ema9:,.2f}")
    print(f"  EMA21: ${ema21:,.2f}")
    print(f"  EMA50: ${ema50:,.2f}")
    
    if current_price > ema9 > ema21:
        print(f"  Trend: YUKARI (Fiyat EMA'larin uzerinde)")
    elif current_price < ema9 < ema21:
        print(f"  Trend: ASAGI (Fiyat EMA'larin altinda)")
    else:
        print(f"  Trend: KARISIK")
    
    # 4. Order Book
    orderbook = await fetch_orderbook(symbol)
    if orderbook:
        bid_vol = sum(float(b[1]) for b in orderbook['bids'])
        ask_vol = sum(float(a[1]) for a in orderbook['asks'])
        total = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total * 100
        
        print(f"\n[ORDER BOOK]")
        print(f"  Bid Volume: {bid_vol:.2f}")
        print(f"  Ask Volume: {ask_vol:.2f}")
        print(f"  Imbalance: {imbalance:+.1f}%")
        
        if imbalance > 10:
            print(f"  Durum: ALIM AGIR")
        elif imbalance < -10:
            print(f"  Durum: SATIM AGIR")
        else:
            print(f"  Durum: DENGELI")
    
    # 5. Funding Rate
    funding = await fetch_funding(symbol)
    if funding is not None:
        print(f"\n[FUNDING RATE]")
        print(f"  Oran: {funding:.4f}%")
        
        if funding > 0.03:
            print(f"  Durum: COK LONG (Short squeeze riski)")
        elif funding < -0.03:
            print(f"  Durum: COK SHORT (Long squeeze riski)")
        else:
            print(f"  Durum: NORMAL")
    
    # 6. Sonuc
    print(f"\n[SONUC]")
    score = 0
    reasons = []
    
    if rsi < 40:
        score += 1
        reasons.append("RSI dusuk (alim firsati)")
    elif rsi > 60:
        score -= 1
        reasons.append("RSI yuksek (satis baskisi)")
    
    if current_price > ema21:
        score += 1
        reasons.append("Fiyat EMA21 uzerinde")
    else:
        score -= 1
        reasons.append("Fiyat EMA21 altinda")
    
    if orderbook and imbalance > 10:
        score += 1
        reasons.append("Order book alim agir")
    elif orderbook and imbalance < -10:
        score -= 1
        reasons.append("Order book satim agir")
    
    if score >= 2:
        print(f"  >>> BULLISH ({score} puan)")
    elif score <= -2:
        print(f"  >>> BEARISH ({score} puan)")
    else:
        print(f"  >>> NOTR ({score} puan)")
    
    for reason in reasons:
        print(f"      - {reason}")
    
    return True


async def compare_with_system(symbol: str):
    """Sistemin ciktisiyla karsilastir."""
    print(f"\n{'='*60}")
    print(f"  SISTEM CIKTISI KARSILASTIRMASI")
    print(f"{'='*60}")
    
    try:
        # Dashboard data
        with open("dashboard_data.json", "r") as f:
            data = json.load(f)
        
        if symbol in data:
            snap = data[symbol]
            print(f"\n[SISTEM RAPORU: {symbol}]")
            print(f"  AI Karar: {snap.get('ai_decision', 'N/A')}")
            print(f"  AI Guven: {snap.get('ai_confidence', 0):.0f}%")
            print(f"  Sinyal Tipi: {snap.get('signal_type', 'N/A')}")
            
            if 'leading_indicators' in snap:
                print(f"\n[LEADING INDICATORS]")
                for key, val in snap['leading_indicators'].items():
                    print(f"    {key}: {val}")
        else:
            print(f"  {symbol} icin sistem verisi bulunamadi")
            
    except Exception as e:
        print(f"  Dashboard verisi okunamadi: {e}")


async def main():
    print("\n" + "="*60)
    print("  DEMIR AI - SINYAL DOGRULAMA")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    for symbol in symbols:
        await verify_symbol(symbol)
    
    # Sistem karsilastirmasi
    await compare_with_system("BTCUSDT")
    
    print("\n" + "="*60)
    print("  DOGRULAMA TAMAMLANDI")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
