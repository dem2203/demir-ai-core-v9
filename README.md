# 🦅 DEMIR AI - Institutional Trading Terminal
**Version 23.0 | Phase 23: Smart Money Concepts**

![Status](https://img.shields.io/badge/Status-OPERATIONAL-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![AI](https://img.shields.io/badge/AI-Multi--Model-orange)
![Railway](https://img.shields.io/badge/Deployed%20on-Railway-violet)

## 🏗️ System Overview
Demir AI is an autonomous institutional-grade trading system powered by a **Hybrid Intelligence** architecture. It combines quantitative finance (LSTM, RL) with generative AI (GPT-4o, Gemini) and **Smart Money Concepts (SMC)** to dominate the market.

### 🧠 AI Superpowers (v23)
1.  **Smart Money Concepts (NEW!):**
    *   Detects **Order Blocks** - institutional entry zones
    *   Identifies **Fair Value Gaps (FVG)** - price imbalances
    *   Maps **Liquidity Zones** - where stops accumulate
2.  **Multi-Timeframe Confluence:**
    *   Analyzes 1H, 4H, 1D trends simultaneously
    *   Confluence scoring (0-100%)
    *   Entry quality assessment
3.  **Volume Profile:**
    *   VPOC (Volume Point of Control)
    *   HVN/LVN zones (High/Low Volume Nodes)
    *   Price magnet detection
4.  **Smart SL/TP:**
    *   SMC-based stop losses (below Order Blocks)
    *   Multi-target take profits (TP1, TP2, TP3)
    *   R:R ratios calculated automatically
5.  **RL Agent v4 (5 Years, 500K Steps):**
    *   BTC: Sharpe 0.13, +10.7% ROI (30d backtest)
    *   ETH, LTC, SOL: All trained on 5 years of data
6.  **Dual Vision Cortex:**
    *   Uses **Google Gemini 1.5** + **OpenAI GPT-4o** to "see" charts.
7.  **Liquidation Hunter:**
    *   Maps "Magnet Levels" where high leverage traders get liquidated.

## 🚀 Deployment
This project is automatically deployed to **Railway** on every push to `main`.

### Dashboard Features
*   **📡 Live Market Intelligence:**
    *   Real-time SMC analysis (Order Blocks, FVG)
    *   MTF Confluence display (1H/4H/1D)
    *   Volume Profile (VPOC, VAH, VAL)
    *   Smart Entry/Exit levels with R:R
*   **🧠 Neural Brain Monitor:**
    *   RL Agent v4 decision visualization
    *   Dual Vision analysis results
*   **💼 Advisory Portfolio:**
    *   Paper trading with advanced risk sizing
*   **🧪 Backtest Lab:**
    *   Test v4 models on historical data

## 📂 Project Structure
```
src/
├── brain/              # The Cortex
│   ├── smc_analyzer.py     # Smart Money Concepts (NEW)
│   ├── mtf_analyzer.py     # Multi-Timeframe Confluence (NEW)
│   ├── volume_profile.py   # Volume Profile Analysis (NEW)
│   ├── smart_sltp.py       # Intelligent SL/TP (NEW)
│   ├── vision_analyst.py   # GPT-4o + Gemini Vision
│   └── rl_agent/           # PPO v4 Models
├── core/               # The Engine
│   ├── engine.py           # Main Orchestrator
│   └── risk_shield.py      # Capital Protection
└── data_ingestion/     # The Senses (Binance, Macro)
```

## 📜 Recent Updates
- **Phase 23 (Current):**
    - **SMC Analyzer:** Order Blocks, Fair Value Gaps, Liquidity Zones
    - **MTF Confluence:** 1H/4H/1D trend alignment scoring
    - **Volume Profile:** VPOC, VAH, VAL, HVN/LVN detection
    - **Smart SL/TP:** Multi-target TP with R:R ratios
    - **v4 Models:** All 4 coins (BTC/ETH/LTC/SOL) trained on 5 years
    - **Enhanced Telegram:** R:R ratios, MTF confluence in signals

---
*Created by Demir AI Team. Proprietary Software.*
