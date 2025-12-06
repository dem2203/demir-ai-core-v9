# 🦅 DEMIR AI - Institutional Trading Terminal
**Version 21.0 | Phase 21: Telegram Intelligence**

![Status](https://img.shields.io/badge/Status-OPERATIONAL-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![AI](https://img.shields.io/badge/AI-Multi--Model-orange)
![Railway](https://img.shields.io/badge/Deployed%20on-Railway-violet)

## 🏗️ System Overview
Demir AI is an autonomous institutional-grade trading system powered by a **Hybrid Intelligence** architecture. It combines quantitative finance (LSTM, RL) with generative AI (GPT-4o, Gemini) to dominate the market.

### 🧠 AI Superpowers
1.  **Dual Vision Cortex (Active):**
    *   Uses **Google Gemini 1.5** + **OpenAI GPT-4o** to "see" charts like a human trader.
    *   Cross-validates technical patterns (Head & Shoulders, Flags) between models.
    *   Generates a consensus "Visual Score" (0-100).
2.  **Liquidation Hunter:**
    *   Maps "Magnet Levels" where high leverage traders are likely to get liquidated.
    *   Analyzes Open Interest (OI) flows and Funding Rate extremes to bet against the crowd.
3.  **Neural Brain:**
    *   **LSTM:** Predicts next-candle price close with high probability.
    *   **RL Agent (PPO):** Learns optimal portfolio allocation through millions of simulated steps.
4.  **Macro Engine:**
    *   Ingests real-time **DXY** (Dollar Strength) and **VIX** (Fear Index) to adjust risk exposure.

## 🚀 Deployment
This project is automatically deployed to **Railway** on every push to `main`.

### Dashboard Features
The system exposes a state-of-the-art Streamlit dashboard:
*   **📡 Live Market Intelligence:**
    *   Real-time price & AI signals.
    *   "Whale Wall" detection (Order Book Imbalance).
    *   Kelly Criterion risk sizing.
*   **🧠 Neural Brain Monitor:**
    *   Visualizes the RL Agent's "Attention Map" (what it's focusing on).
    *   Shows Dual Vision analysis results & charts.
*   **💼 Advisory Portfolio:**
    *   Paper trading simulation with PnL tracking.
*   **🧪 Backtest Lab:**
    *   Run historical simulations on any asset.

## 📂 Project Structure
```
src/
├── brain/          # The Cortex
│   ├── vision_analyst.py   # GPT-4o + Gemini Vision
│   ├── liquidation_hunter.py # Anti-Crowd Logic
│   └── rl_trainer.py       # PPO Reinforcement Learning
├── core/           # The Engine
│   ├── engine.py           # Main Orchestrator
│   └── risk_shield.py      # Capital Protection
├── data_ingestion/ # The Senses (Binance, Macro)
└── ui/             # The Face (Streamlit Dashboard)
```

## 📜 Recent Updates
- **Phase 21 (Current):**
    - **Smart Telegram:** Anti-Spam deduplication & Strict Filters (>85%).
    - **Heartbeat:** Hourly system integrity checks.
    - **Optimization:** Deduplicated signal broadcasting.

---
*Created by Demir AI Team. Proprietary Software.*
