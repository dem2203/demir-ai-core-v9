# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - PACKAGE INIT
============================
Yeni prediktif trading sistemi.

Modüller:
- data_hub: Merkezi veri toplama
- predictor: Tahmin motoru
- smart_notifier: Telegram bildirimleri
- engine: Ana döngü
"""
from src.v10.data_hub import get_data_hub, DataHub, MarketSnapshot
from src.v10.predictor import get_predictor, PredictorEngine, TradingSignal, SignalType
from src.v10.smart_notifier import get_notifier, SmartNotifier
from src.v10.engine import get_v10_engine, V10Engine, run_v10

__all__ = [
    'get_data_hub', 'DataHub', 'MarketSnapshot',
    'get_predictor', 'PredictorEngine', 'TradingSignal', 'SignalType',
    'get_notifier', 'SmartNotifier',
    'get_v10_engine', 'V10Engine', 'run_v10'
]

__version__ = "10.0.0"
