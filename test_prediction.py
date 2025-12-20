# Quick test for predictive engine
import asyncio
import sys
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.brain.predictive_engine import get_predictive_engine

async def test():
    engine = get_predictive_engine()
    
    # Veri topla
    data = await engine._collect_all_data('BTCUSDT')
    print(f"Current price: ${data['current_price']:,.2f}")
    print(f"Funding: {data['funding_rate']:.4f}%")
    print(f"L/S Ratio: {data['ls_ratio']:.2f}")
    print(f"Fear/Greed: {data['fear_greed']}")
    print(f"Orderbook Imbalance: {data['orderbook_imbalance']:.2f}")
    print()
    
    # Faktorleri analiz et
    factors = await engine._analyze_all_factors('BTCUSDT', data)
    
    print('FACTOR ANALYSIS:')
    for f in factors:
        print(f"  {f['name']}: {f['signal']} ({f['value']:.0f}) - {f['description']}")
    
    # Confluence
    direction, score, count = engine._calculate_confluence(factors)
    print(f"\nCONFLUENCE: {direction} {score:.0f}% ({count} factors agree)")
    print(f"Min required: {engine.MIN_CONFLUENCE_SCORE}% and {engine.MIN_FACTORS_AGREE} factors")

asyncio.run(test())
