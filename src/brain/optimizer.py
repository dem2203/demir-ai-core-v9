import logging
import asyncio
import pandas as pd
from src.backtest.backtester import Backtester

logger = logging.getLogger("STRATEGY_OPTIMIZER")

class StrategyOptimizer:
    """
    GRID SEARCH OPTIMIZER
    Farklı strateji kombinasyonlarını deneyerek en iyi ROI'yi bulur.
    """
    
    def __init__(self):
        self.backtester = Backtester()

    async def optimize(self, symbol="BTC/USDT", days=30):
        logger.info(f"Starting Optimization for {symbol}...")
        
        # DENECEK KOMBİNASYONLAR
        # sl_mul: Stop Loss (ATR katı)
        # tp_mul: Take Profit (ATR katı)
        # threshold: LSTM Güven Eşiği
        param_grid = [
            {'sl_mul': 1.5, 'tp_mul': 3.0, 'threshold': 0.55}, # Dengeli
            {'sl_mul': 1.0, 'tp_mul': 2.0, 'threshold': 0.51}, # Scalper (Dar Stop, Düşük Eşik)
            {'sl_mul': 2.0, 'tp_mul': 5.0, 'threshold': 0.60}, # Swing (Geniş Stop, Yüksek Eşik)
            {'sl_mul': 1.5, 'tp_mul': 3.0, 'threshold': 0.45}, # Agresif (Düşük Eşik)
            {'sl_mul': 3.0, 'tp_mul': 6.0, 'threshold': 0.55}  # Trend Follower
        ]
        
        results = []
        
        # Her bir ayar için simülasyon yap
        for params in param_grid:
            # Yeni instance oluştur (Temiz hafıza)
            bt = Backtester(initial_balance=10000)
            res = await bt.run_backtest(symbol, days, params)
            
            if "error" not in res:
                results.append({
                    "params": params,
                    "roi": res['roi'],
                    "win_rate": res['win_rate'],
                    "trades": res['total_trades'],
                    "balance": res['final_balance']
                })
        
        # En çok kazandıranı bul
        if not results: return {"best_config": None, "all_results": []}
            
        results.sort(key=lambda x: x['roi'], reverse=True)
        best_result = results[0]
        
        return {
            "best_config": best_result,
            "all_results": results
        }
