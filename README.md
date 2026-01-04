# 🤖 DEMIR AI v11 - Quantitative Trading System

![Status](https://img.shields.io/badge/Status-LIVE-green)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![ML](https://img.shields.io/badge/ML-LightGBM-orange)
![Risk](https://img.shields.io/badge/Risk-Kelly%20Criterion-red)
![Railway](https://img.shields.io/badge/Deploy-Railway-violet)

---

## 🧠 Ne Yapıyor?

**DEMIR AI v11**, kripto para piyasalarında (BTC, ETH) yapay zeka destekli **otomatik sinyal üretimi** yapan profesyonel bir quant trading sistemidir.

### ⚙️ Nasıl Çalışıyor?

```
📊 Binance Veri → 🔬 74 Teknik İndikatör → 🤖 LightGBM ML Modeli → 📈 Sinyal Üretimi → 📱 Telegram Bildirimi
```

1. **Veri Toplama:** Binance Futures'dan dakikalık OHLCV verisi çeker
2. **Feature Engineering:** RSI, MACD, Bollinger, ATR, OBV ve 70+ teknik indikatör hesaplar
3. **ML Prediction:** LightGBM modeli ile yön tahmini yapar (BUY/SELL)
4. **Risk Yönetimi:** Kelly Criterion ile pozisyon büyüklüğü, ATR-bazlı Stop Loss/Take Profit hesaplar
5. **Sinyal Gönderimi:** %80+ güvenli sinyalleri Telegram'dan bildirir

---

## 📊 Backtest Sonuçları

| Coin | İşlem Sayısı (Yıllık) | Win Rate | Max Drawdown |
|:--|:--|:--|:--|
| **BTCUSDT** | ~7,200 | %44 | %8 |
| **ETHUSDT** | ~7,200 | %49 | %3 |

> **Not:** Eski sistemde %99 batış yaşanırken, yeni risk yönetimi ile maksimum kayıp %8'e düşürüldü.

---

## 🏗️ Proje Yapısı

```
demir-ai-core-v9/
├── src/
│   ├── data_pipeline/
│   │   └── collector.py      # Binance veri indirme (async)
│   ├── features/
│   │   └── technical.py      # 74 teknik indikatör
│   ├── models/
│   │   └── trainer.py        # LightGBM eğitimi
│   ├── risk/
│   │   └── position_sizer.py # Kelly Criterion
│   ├── execution/
│   │   ├── backtester.py     # Profesyonel backtest
│   │   ├── signal_generator.py # Canlı sinyal üretimi
│   │   └── notifier.py       # Telegram bildirimi
│   └── ...
├── data/
│   ├── raw/                  # Ham OHLCV verileri (parquet)
│   └── models/               # Eğitilmiş LightGBM modelleri
├── run_backtest.py           # Backtest çalıştırma
├── run_live_trading.py       # Canlı sistem çalıştırma
└── Dockerfile                # Railway deployment
```

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Backtest
```bash
python run_backtest.py --download --train --backtest --symbols BTCUSDT ETHUSDT
```

### Canlı Mod (Railway)
```bash
python run_live_trading.py
```

### Ortam Değişkenleri (.env)
```
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 🛡️ Risk Yönetimi

| Mekanizma | Açıklama |
|:--|:--|
| **Kelly Criterion** | Optimal pozisyon büyüklüğü hesaplama |
| **ATR Stop Loss** | Volatiliteye göre dinamik stop loss |
| **Risk:Reward 1:1.5** | Her işlemde minimum 1.5x kar hedefi |
| **%80 Güven Eşiği** | Sadece yüksek güvenli sinyallere giriş |

---

## 📱 Telegram Sinyalleri

Sistem sinyal bulduğunda şu formatta mesaj atar:

```
🟢 SİNYAL ALARMI 🟢
━━━━━━━━━━━━━━━━━━
🪙 ETHUSDT
↕️ Yön: BUY
💵 Fiyat: 2450.50
🎯 Güven: %82.3

🛑 SL: 2420.00
💰 TP: 2495.75
🎲 Büyüklük: $150.00
━━━━━━━━━━━━━━━━━━
🤖 Demir AI v11 Live
```

---

## 📈 Teknoloji Stack

- **Python 3.11**
- **LightGBM** - Gradient Boosting ML
- **Pandas/NumPy** - Veri işleme
- **aiohttp** - Async HTTP (Binance API)
- **PyArrow** - Parquet dosya formatı
- **python-telegram-bot** - Telegram entegrasyonu

---

## ⚠️ Sorumluluk Reddi

Bu yazılım sadece eğitim amaçlıdır. Gerçek para ile işlem yapmadan önce kendi araştırmanızı yapın. Yatırım kararlarınızdan tamamen siz sorumlusunuz.

---

*Created by Demir AI Team | v11.0 | January 2026*
