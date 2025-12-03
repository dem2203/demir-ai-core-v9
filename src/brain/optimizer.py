import logging
import asyncio
import pandas as pd
from src.backtest.backtester import Backtester

logger = logging.getLogger("STRATEGY_OPTIMIZER")

class StrategyOptimizer:
    """
    HİPERPARAMETRE OPTİMİZASYON MOTORU
    
    Farklı strateji ayarlarını (Stop Loss, Take Profit, Giriş Eşiği)
    geçmiş veride simüle eder ve en yüksek karı getiren ayarı bulur.
    """
    
    def __init__(self):
        self.backtester = Backtester()

    async def optimize(self, symbol="BTC/USDT", days=30):
        """
        Grid Search yöntemiyle en iyi ayarları arar.
        """
        logger.info(f"Starting Optimization for {symbol}...")
        
        # DENENECEK AYARLAR (GRID)
        # Çok fazla kombinasyon yaparsak Railway yavaşlar, o yüzden kritik olanları seçtik.
        param_grid = [
            {'sl_mul': 1.5, 'tp_mul': 3.0, 'threshold': 0.51}, # Dengeli
            {'sl_mul': 1.0, 'tp_mul': 2.0, 'threshold': 0.51}, # Dar Stop (Scalp)
            {'sl_mul': 2.0, 'tp_mul': 4.0, 'threshold': 0.51}, # Geniş Stop (Swing)
            {'sl_mul': 1.5, 'tp_mul': 3.0, 'threshold': 0.60}, # Sadece çok eminse gir
            {'sl_mul': 1.5, 'tp_mul': 3.0, 'threshold': 0.45}, # Agresif giriş
        ]
        
        results = []
        
        for params in param_grid:
            # Her ayar için backtest çalıştır (Yeni bir Backtester instance'ı ile)
            bt = Backtester(initial_balance=10000)
            res = await bt.run_backtest(symbol, days, params)
            
            if "error" not in res:
                results.append({
                    "params": params,
                    "roi": res['roi'],
                    "win_rate": res['win_rate'],
                    "trades": res['total_trades']
                })
        
        # Sonuçları ROI'ye göre sırala (En yüksek kar en üstte)
        results.sort(key=lambda x: x['roi'], reverse=True)
        
        best_result = results[0] if results else None
        
        return {
            "best_config": best_result,
            "all_results": results
        }
