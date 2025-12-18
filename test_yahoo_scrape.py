# -*- coding: utf-8 -*-
"""Test Yahoo Finance scraping for Gold - with symbol-specific extraction"""
import requests
import re
import json

symbols = {
    'gold': 'GC=F',
    'nasdaq': '^IXIC',
    'vix': '^VIX',
}

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

for name, symbol in symbols.items():
    url = f"https://finance.yahoo.com/quote/{symbol}/"
    r = requests.get(url, headers=headers, timeout=15)
    
    # Look for the specific symbol's price in the JSON data
    # Find the quote data for THIS symbol specifically
    # Pattern: "GC=F".*?"regularMarketPrice":{"raw":
    escaped_symbol = re.escape(symbol)
    pattern = rf'"{escaped_symbol}".*?"regularMarketPrice":\{{"raw":([\d.]+)'
    m = re.search(pattern, r.text, re.DOTALL)
    
    if m:
        price = float(m.group(1))
        print(f"{name.upper()} ({symbol}): ${price:,.2f}")
    else:
        # Try alternative: look for quote summary
        alt_pattern = rf'"symbol":"{escaped_symbol}".*?"regularMarketPrice":\{{"raw":([\d.]+)'
        m2 = re.search(alt_pattern, r.text, re.DOTALL)
        if m2:
            price = float(m2.group(1))
            print(f"{name.upper()} ({symbol}): ${price:,.2f} (alt)")
        else:
            print(f"{name.upper()} ({symbol}): NOT FOUND")
