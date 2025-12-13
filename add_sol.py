# Add SOL to rl_model_map
with open('src/brain/market_analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace LTC line to add SOL
old = "'LTC/USDT': 'ppo_ltc_v3'   # 2 years data, 500K steps, Sharpe 0.09"
new = """'LTC/USDT': 'ppo_ltc_v3',  # 2 years data
            'SOL/USDT': 'ppo_sol_v3'   # NEW - v5 pending"""

content = content.replace(old, new)

with open('src/brain/market_analyzer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("SOL added to rl_model_map!")
