---
description: Resume Demir AI project with full context
---
// turbo-all

# Resume Demir AI Project

## 1. Read Project Summary
Read the PROJECT_SUMMARY.md file to understand the project:
```
view_file C:\Users\idemi\.gemini\antigravity\scratch\demir-ai-core-v9\PROJECT_SUMMARY.md
```

## 2. Set Working Directory
The project is at: `C:\Users\idemi\.gemini\antigravity\scratch\demir-ai-core-v9`

## 3. Key Rules to Follow

### Code Rules
- **ZERO MOCK DATA**: Dashboard must show only real data, never fake/placeholder
- **Per-coin collapsibles**: Each coin (BTC, ETH, LTC, SOL) has its own expandable section
- **Signal Quality Filter**: All signals must pass SignalQualityFilter (min 60% confidence)

### Model Rules  
- Current RL models: v5 (ppo_btc_v5, ppo_eth_v5, ppo_ltc_v5, ppo_sol_v5)
- Current LSTM models: v11
- Model map is in `market_analyzer.py` lines 119-123
- TradingEnv expects 28 features (not 32!)

### Deployment Rules
- Railway auto-deploys on git push to main
- Always `git push origin main` after changes
- Check Railway logs if crash happens

### Import Rules
- Use `SignalQualityFilter` NOT `SignalFilter`
- Class in `src/core/signal_filter.py`

## 4. Check Recent Changes
```powershell
cd C:\Users\idemi\.gemini\antigravity\scratch\demir-ai-core-v9
git log --oneline -10
```

## 5. Check Current Task Status
Read task.md to see what was in progress:
```
view_file C:\Users\idemi\.gemini\antigravity\brain\edf024f3-f460-4f6c-a297-76b60a4538ef\task.md
```

## 6. Key Entry Points

| Purpose | File |
|---------|------|
| Dashboard | `dashboard.py` |
| Main Engine | `src/core/engine.py` |
| Market Analysis | `src/brain/market_analyzer.py` |
| RL Training | `src/brain/rl_agent/trainer.py` |
| Notifications | `src/utils/notifications.py` |

## 7. Completed Phases

- Phase 23: SMC, MTF, Volume Profile, Smart SL/TP
- Phase 24: v4 model training (5 years)
- Phase 25: v5 model training (all 4 coins)
- Phase 26: Dashboard cleanup + backtest
- Phase 27: v6 macro features + signal filter
- Phase 28: Ensemble + Kelly + Auto-Retrain

## 8. Environment Variables (Railway)

| Key | Purpose |
|-----|---------|
| BINANCE_API_KEY | Market data |
| TELEGRAM_TOKEN | Signal notifications |
| TELEGRAM_CHAT_ID | Chat ID |
| FRED_API_KEY | Macro data |
| GOOGLE_API_KEY | Gemini Vision |
| OPENAI_API_KEY | GPT-4o Vision |
