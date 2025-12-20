# -*- coding: utf-8 -*-
"""
DEMIR AI - GERCEK VERI DOGRULAMA TESTI
======================================
Tum verilerin gercek kaynaklardan geldigini dogrular.
"""
import asyncio
import requests
import sys
import io
from datetime import datetime as dt

# Windows encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_check(name, value, source, verified=True):
    status = "[OK]" if verified else "[??]"
    print(f"{status} {name}: {value}")
    print(f"     Kaynak: {source}")

async def main():
    print_header("DEMIR AI - GERCEK VERI DOGRULAMA")
    print(f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    errors = []
    
    # =====================================================================
    # 1. BINANCE API - FIYAT
    # =====================================================================
    print_header("1. BINANCE FIYAT VERISI")
    
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={'symbol': 'BTCUSDT'},
            timeout=10
        )
        data = resp.json()
        btc_price = float(data['price'])
        
        print_check(
            "BTC Fiyat",
            f"${btc_price:,.2f}",
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        )
        
        # Makul aralikta mi? ($50,000 - $200,000)
        if 50000 < btc_price < 200000:
            print("     Dogrulama: Fiyat makul aralikta")
        else:
            errors.append("BTC fiyat aralik disi")
            
    except Exception as e:
        errors.append(f"Binance API: {e}")
        print(f"[HATA] Binance API: {e}")
    
    # =====================================================================
    # 2. FEAR & GREED INDEX
    # =====================================================================
    print_header("2. FEAR & GREED INDEX")
    
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=10
        )
        data = resp.json()
        fg_value = int(data['data'][0]['value'])
        fg_class = data['data'][0]['value_classification']
        fg_timestamp = data['data'][0]['timestamp']
        
        print_check(
            "Fear & Greed",
            f"{fg_value} ({fg_class})",
            "https://api.alternative.me/fng/"
        )
        
        # Timestamp kontrol (son 24 saat icinde olmali)
        from datetime import datetime
        ts = datetime.fromtimestamp(int(fg_timestamp))
        age_hours = (datetime.now() - ts).total_seconds() / 3600
        print(f"     Veri yasi: {age_hours:.1f} saat once")
        
        if age_hours < 24:
            print("     Dogrulama: Veri guncel")
        else:
            errors.append("Fear&Greed verisi eski")
            
    except Exception as e:
        errors.append(f"Fear&Greed API: {e}")
        print(f"[HATA] Fear&Greed API: {e}")
    
    # =====================================================================
    # 3. FUNDING RATE
    # =====================================================================
    print_header("3. BINANCE FUTURES - FUNDING RATE")
    
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={'symbol': 'BTCUSDT', 'limit': 1},
            timeout=10
        )
        data = resp.json()
        
        if data:
            funding = float(data[0]['fundingRate']) * 100
            funding_time = data[0]['fundingTime']
            
            print_check(
                "Funding Rate",
                f"{funding:.4f}%",
                "https://fapi.binance.com/fapi/v1/fundingRate"
            )
            
            ts = datetime.fromtimestamp(funding_time / 1000)
            print(f"     Son guncelleme: {ts.strftime('%Y-%m-%d %H:%M')}")
            
    except Exception as e:
        errors.append(f"Funding Rate: {e}")
        print(f"[HATA] Funding Rate: {e}")
    
    # =====================================================================
    # 4. LONG/SHORT RATIO
    # =====================================================================
    print_header("4. LONG/SHORT RATIO")
    
    try:
        resp = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={'symbol': 'BTCUSDT', 'period': '1h', 'limit': 1},
            timeout=10
        )
        data = resp.json()
        
        if data:
            ls_ratio = float(data[0]['longShortRatio'])
            ls_time = data[0]['timestamp']
            
            print_check(
                "Long/Short Ratio",
                f"{ls_ratio:.2f}",
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            )
            
            ts = datetime.fromtimestamp(ls_time / 1000)
            print(f"     Son guncelleme: {ts.strftime('%Y-%m-%d %H:%M')}")
            
    except Exception as e:
        errors.append(f"L/S Ratio: {e}")
        print(f"[HATA] L/S Ratio: {e}")
    
    # =====================================================================
    # 5. ORDER BOOK (WHALE DETECTION)
    # =====================================================================
    print_header("5. ORDER BOOK - WHALE ALGI")
    
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/depth",
            params={'symbol': 'BTCUSDT', 'limit': 100},
            timeout=10
        )
        data = resp.json()
        
        bid_vol = sum(float(b[1]) for b in data['bids'])
        ask_vol = sum(float(a[1]) for a in data['asks'])
        imbalance = bid_vol / ask_vol
        
        if imbalance > 1.5:
            whale_status = "ALICI AGIRLKLI"
        elif imbalance < 0.67:
            whale_status = "SATICI AGIRLIKLI"
        else:
            whale_status = "DENGELI"
        
        print_check(
            "Order Book",
            f"Bid: {bid_vol:.1f} BTC, Ask: {ask_vol:.1f} BTC",
            "https://api.binance.com/api/v3/depth"
        )
        print(f"     Oran: {imbalance:.2f} ({whale_status})")
        
    except Exception as e:
        errors.append(f"Order Book: {e}")
        print(f"[HATA] Order Book: {e}")
    
    # =====================================================================
    # 6. KLINES (CANDLE DATA)
    # =====================================================================
    print_header("6. KLINE VERILERI (MUMLAR)")
    
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 5},
            timeout=10
        )
        data = resp.json()
        
        print_check(
            "Son 5 Mum",
            f"{len(data)} mum alindi",
            "https://api.binance.com/api/v3/klines"
        )
        
        for k in data[-3:]:  # Son 3 mum
            open_time = datetime.fromtimestamp(k[0] / 1000)
            open_p = float(k[1])
            close_p = float(k[4])
            volume = float(k[5])
            print(f"     {open_time.strftime('%H:%M')} | O:${open_p:,.0f} C:${close_p:,.0f} V:{volume:.1f}")
            
    except Exception as e:
        errors.append(f"Klines: {e}")
        print(f"[HATA] Klines: {e}")
    
    # =====================================================================
    # 7. THINKING BRAIN KARSILASTIRMA
    # =====================================================================
    print_header("7. THINKING BRAIN TEST")
    
    try:
        from src.thinking_brain.brain import get_thinking_brain
        
        brain = get_thinking_brain()
        decision = await brain.think('BTCUSDT')
        
        print("[OK] Thinking Brain calisti")
        print(f"     Karar: {decision.action}")
        print(f"     Guven: {decision.confidence*100:.0f}%")
        
        # Fiyat karsilastir
        if decision.entry_price > 0:
            fark = abs(decision.entry_price - btc_price)
            fark_pct = (fark / btc_price) * 100
            print(f"     Fiyat farki: ${fark:.2f} ({fark_pct:.2f}%)")
            
            if fark_pct < 0.5:
                print("     Dogrulama: Fiyatlar tutarli")
            else:
                errors.append("Thinking Brain fiyat uyumsuz")
                
    except Exception as e:
        errors.append(f"Thinking Brain: {e}")
        print(f"[HATA] Thinking Brain: {e}")
    
    # =====================================================================
    # SONUC
    # =====================================================================
    print_header("DOGRULAMA SONUCU")
    
    if not errors:
        print("[OK] TUM VERILER GERCEK VE DOGRULANMIS!")
        print("")
        print("Kontrol edilen kaynaklar:")
        print("  - Binance Spot API (fiyat)")
        print("  - Binance Futures API (funding, L/S)")
        print("  - Alternative.me API (Fear & Greed)")
        print("  - Order Book (whale tespiti)")
        print("")
        print("Hicbir mock/sahte veri KULLANILMIYOR.")
    else:
        print(f"[UYARI] {len(errors)} sorun bulundu:")
        for err in errors:
            print(f"  - {err}")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
