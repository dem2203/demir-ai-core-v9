from src.brain.advanced_scrapers import AdvancedMarketScrapers

s = AdvancedMarketScrapers()
r = s.get_liquidation_levels('BTC')

print(f"OI: {r.get('oi_formatted', 'N/A')}")
print(f"Funding: {r.get('funding_rate_formatted', 'N/A')}")
print(f"L/S Ratio: {r.get('long_short_ratio', 'N/A')}")
print(f"Taker Buy Ratio: {r.get('taker_buy_ratio', 'N/A')}")
print(f"Direction: {r.get('direction', 'N/A')}")
print(f"Action: {r.get('action', 'N/A')}")
