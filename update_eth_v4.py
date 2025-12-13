# Update ETH to v4 in rl_model_map
with open('src/brain/market_analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace ETH v3 with v4
old = "'ETH/USDT': 'ppo_eth_v3',  # 2 years data, 500K steps, Sharpe 0.10"
new = "'ETH/USDT': 'ppo_eth_v4',  # 5 years data, v4 complete"

content = content.replace(old, new)

with open('src/brain/market_analyzer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("ETH v4 updated in rl_model_map!")
