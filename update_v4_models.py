# Update LTC and SOL to v4 in rl_model_map
with open('src/brain/market_analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Update LTC v3 -> v4
content = content.replace(
    "'LTC/USDT': 'ppo_ltc_v3',  # 2 years data",
    "'LTC/USDT': 'ppo_ltc_v4',  # 5 years data, Sharpe 0.07"
)

# Update SOL v3 -> v4
content = content.replace(
    "'SOL/USDT': 'ppo_sol_v3'   # NEW - v5 pending",
    "'SOL/USDT': 'ppo_sol_v4'   # 5 years data, v4 complete"
)

with open('src/brain/market_analyzer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("LTC v4 + SOL v4 updated in rl_model_map!")
