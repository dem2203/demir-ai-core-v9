# DEMIR AI CORE v9 - Project Summary
**Last Updated:** 2025-12-14

## 🎯 Project Overview

Demir AI is an advanced crypto trading AI system deployed on Railway with:
- **Dashboard:** Streamlit at Railway public URL
- **Repo:** github.com/dem2203/demir-ai-core-v9
- **4 Target Coins:** BTC, ETH, LTC, SOL

---

## 📁 Key File Structure

```
demir-ai-core-v9/
├── main.py                    # Entry point
├── dashboard.py              # Streamlit dashboard
├── src/
│   ├── brain/
│   │   ├── market_analyzer.py       # MAIN: All analysis orchestration
│   │   ├── ensemble_model.py        # Phase 28: RL+LSTM voting
│   │   ├── feature_engineering.py   # Feature calculation
│   │   ├── smc_analyzer.py          # Smart Money Concepts
│   │   ├── mtf_analyzer.py          # Multi-Timeframe
│   │   ├── volume_profile.py        # Volume Profile
│   │   ├── smart_sltp.py            # Smart SL/TP
│   │   ├── rl_agent/
│   │   │   ├── ppo_agent.py         # RL Agent
│   │   │   ├── trading_env.py       # Gym environment
│   │   │   ├── trainer.py           # RL Trainer
│   │   │   ├── auto_retrain.py      # Phase 28: Weekly retrain
│   │   │   └── storage/             # Model files (.zip)
│   │   └── models/lstm/             # LSTM models (.keras)
│   ├── core/
│   │   ├── engine.py                # BotEngine
│   │   ├── signal_filter.py         # Phase 27: Quality filter
│   │   └── position_sizer.py        # Phase 28: Kelly sizing
│   ├── data_ingestion/
│   │   ├── macro_connector.py       # MacroData + v6 ratios
│   │   └── connectors/binance_connector.py
│   └── utils/
│       └── notifications.py         # Telegram signals
└── data/dashboard_data.json         # Dashboard live data
```

---

## 🧠 Model Versions

| Type | Version | Files |
|------|---------|-------|
| RL (PPO) | v5 | `ppo_btc_v5.zip`, `ppo_eth_v5.zip`, `ppo_ltc_v5.zip`, `ppo_sol_v5.zip` |
| LSTM | v11 | `lstm_btc_v11.keras`, etc. |

**Model Map in `market_analyzer.py` lines 119-123:**
```python
self.rl_model_map = {
    'BTC/USDT': 'ppo_btc_v5',
    'ETH/USDT': 'ppo_eth_v5',
    'LTC/USDT': 'ppo_ltc_v5',
    'SOL/USDT': 'ppo_sol_v5'
}
```

---

## 📊 Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 23 | Advanced Signals (SMC, MTF, VP, SL/TP) | ✅ |
| 24 | v4 Model Training (5 years data) | ✅ |
| 25 | v5 Model Training (all 4 coins) | ✅ |
| 26 | Dashboard cleanup + backtest | ✅ |
| 27 | v6 Macro Features + Signal Filter | ✅ |
| 28 | Ensemble + Kelly + Auto-Retrain | ✅ |
| 29 | Signal Enhancement + Multi-Exchange + Portfolio | ✅ |

---

## 🆕 Phase 27-28 New Features

### Phase 27 (Short-Term)
1. **v6 Macro Features** (`macro_connector.py`):
   - `eth_btc_ratio`, `gold_btc_ratio`, `sp500_btc_ratio`
2. **Signal Quality Filter** (`signal_filter.py`):
   - Filters low-quality signals based on confidence, MTF, SMC, R:R

### Phase 28 (Medium-Term)
1. **Ensemble Model** (`ensemble_model.py`, 265 lines):
   - RL+LSTM weighted voting, dynamic weights
2. **Position Sizer** (`position_sizer.py`, 275 lines):
   - Kelly criterion + volatility + drawdown protection
3. **Auto-Retrain Pipeline** (`auto_retrain.py`, 365 lines):
   - Weekly scheduled retraining with rollback

---

## 🔧 Important Fixes Applied

| Issue | Fix | Commit |
|-------|-----|--------|
| Duplicate dashboard sections | Removed, now per-coin collapsibles | `c543dbb` |
| v5 model map update | Updated in market_analyzer.py | `5e10427` |
| SignalFilter import error | Changed to SignalQualityFilter | `866a04a` |

---

## 🌐 Environment Variables (Railway)

| Key | Purpose |
|-----|---------|
| `BINANCE_API_KEY` | Market data |
| `TELEGRAM_TOKEN` | Signal notifications |
| `TELEGRAM_CHAT_ID` | Chat ID for signals |
| `FRED_API_KEY` | Macro data |
| `GOOGLE_API_KEY` | Gemini Vision |
| `OPENAI_API_KEY` | GPT-4o Vision |

---

## 📋 Next Steps (Future)

1. **Multi-Exchange** - Bybit, OKX support
2. **Options Trading** - BTC/ETH options hedging
3. **Portfolio Optimization** - Multi-coin allocation
4. **Live Trading** - Actual trade execution

---

## 🚨 Common Issues & Solutions

| Error | Solution |
|-------|----------|
| `NameError: SignalFilter` | Use `SignalQualityFilter` in notifications.py |
| Model not loading | Check `rl_model_map` in market_analyzer.py |
| Dashboard not updating | Check `dashboard_data.json` permissions |
| Feature mismatch in backtest | TradingEnv expects 28 features |

---

## 📝 How to Resume Work

1. Clone repo: `git clone github.com/dem2203/demir-ai-core-v9`
2. Check task.md in `.gemini/antigravity/brain/*/task.md`
3. Look at recent commits: `git log --oneline -10`
4. Main entry points:
   - Dashboard: `dashboard.py`
   - Engine: `src/core/engine.py`
   - Analysis: `src/brain/market_analyzer.py`
