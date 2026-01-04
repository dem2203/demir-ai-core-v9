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

Note: Imports are lazy to avoid circular dependencies and missing modules.
"""

# Lazy imports - will fail silently if module is missing
def get_data_hub():
    from src.v10.data_hub import get_data_hub as _get
    return _get()

def get_predictor():
    from src.v10.predictor import get_predictor as _get
    return _get()

def get_notifier():
    from src.v10.smart_notifier import get_notifier as _get
    return _get()

def get_v10_engine():
    from src.v10.engine import get_v10_engine as _get
    return _get()

__version__ = "10.0.0"
