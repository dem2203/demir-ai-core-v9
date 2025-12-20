# -*- coding: utf-8 -*-
"""
DEMIR AI - LIVE TEST SYSTEM
============================
Tum modulleri test eder ve detayli rapor uretir.

Usage:
    python live_test.py
"""
import asyncio
import logging
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LIVE_TEST")


class LiveTestSystem:
    """Canli test sistemi."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
    
    async def run_all_tests(self):
        """Tum testleri calistir."""
        print("\n" + "="*60)
        print("   DEMIR AI - LIVE TEST SYSTEM")
        print("   " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*60 + "\n")
        
        tests = [
            ("1. Imports", self.test_imports),
            ("2. Binance API", self.test_binance_api),
            ("3. Derivatives API", self.test_derivatives_api),
            ("4. CoinGecko API", self.test_coingecko_api),
            ("5. Fear & Greed API", self.test_fear_greed_api),
            ("6. Unified Brain", self.test_unified_brain),
            ("7. Early Warning", self.test_early_warning),
            ("8. Performance Tracker", self.test_performance_tracker),
            ("9. Auto Trainer", self.test_auto_trainer),
            ("10. LSTM Model", self.test_lstm_model),
            ("11. WebSocket", self.test_websocket),
        ]
        
        for name, test_func in tests:
            print(f"\n{'-'*60}")
            print(f"Testing: {name}")
            print(f"{'-'*60}")
            
            try:
                result = await test_func()
                self.results[name] = result
                
                if result['status'] == 'PASS':
                    print(f"Result: PASS")
                elif result['status'] == 'WARN':
                    print(f"Result: WARN - {result.get('message', '')}")
                    self.warnings.append(f"{name}: {result.get('message', '')}")
                else:
                    print(f"Result: FAIL - {result.get('error', '')}")
                    self.errors.append(f"{name}: {result.get('error', '')}")
                    
                if result.get('data'):
                    for key, value in result['data'].items():
                        print(f"  {key}: {value}")
                        
            except Exception as e:
                print(f"Result: ERROR - {e}")
                self.results[name] = {'status': 'FAIL', 'error': str(e)}
                self.errors.append(f"{name}: {e}")
        
        # Final Report
        self._print_report()
        
        return self.results
    
    async def test_imports(self) -> Dict:
        """Test all critical imports."""
        imports = {
            'unified_brain': False,
            'early_warning': False,
            'auto_trainer': False,
            'signal_performance': False,
            'macro_connector': False,
            'correlation_connector': False,
            'trading_env': False,
        }
        
        try:
            from src.brain.unified_brain import get_unified_brain
            imports['unified_brain'] = True
        except Exception as e:
            return {'status': 'FAIL', 'error': f'unified_brain: {e}'}
        
        try:
            from src.brain.early_warning import get_warning_system
            imports['early_warning'] = True
        except Exception as e:
            return {'status': 'FAIL', 'error': f'early_warning: {e}'}
        
        try:
            from src.brain.auto_trainer import get_trainer
            imports['auto_trainer'] = True
        except Exception as e:
            return {'status': 'FAIL', 'error': f'auto_trainer: {e}'}
        
        try:
            from src.brain.signal_performance import get_performance_tracker
            imports['signal_performance'] = True
        except Exception as e:
            return {'status': 'FAIL', 'error': f'signal_performance: {e}'}
        
        try:
            from src.data_ingestion.macro_connector import MacroConnector
            imports['macro_connector'] = True
        except Exception as e:
            return {'status': 'FAIL', 'error': f'macro_connector: {e}'}
        
        try:
            from src.data_ingestion.correlation_connector import CorrelationConnector
            imports['correlation_connector'] = True
        except Exception as e:
            return {'status': 'FAIL', 'error': f'correlation_connector: {e}'}
        
        try:
            from src.brain.rl_agent.trading_env import TradingEnv
            imports['trading_env'] = True
            obs_dim = TradingEnv.OBSERVATION_DIM
            imports['observation_dim'] = obs_dim
        except Exception as e:
            return {'status': 'FAIL', 'error': f'trading_env: {e}'}
        
        all_passed = all(v for k, v in imports.items() if k != 'observation_dim')
        
        return {
            'status': 'PASS' if all_passed else 'FAIL',
            'data': imports
        }
    
    async def test_binance_api(self) -> Dict:
        """Test Binance API connection."""
        import requests
        
        try:
            # Price endpoint
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={'symbol': 'BTCUSDT'},
                timeout=10
            )
            
            if resp.status_code != 200:
                return {'status': 'FAIL', 'error': f'HTTP {resp.status_code}'}
            
            data = resp.json()
            btc_price = float(data['price'])
            
            # Klines endpoint
            kline_resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 5},
                timeout=10
            )
            
            if kline_resp.status_code != 200:
                return {'status': 'WARN', 'message': 'Klines failed', 'data': {'btc_price': btc_price}}
            
            klines = kline_resp.json()
            
            return {
                'status': 'PASS',
                'data': {
                    'btc_price': f"${btc_price:,.2f}",
                    'klines_count': len(klines),
                    'latest_close': f"${float(klines[-1][4]):,.2f}"
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_derivatives_api(self) -> Dict:
        """Test Binance Futures API."""
        import requests
        
        try:
            # Funding rate
            fr_resp = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={'symbol': 'BTCUSDT', 'limit': 1},
                timeout=10
            )
            
            if fr_resp.status_code != 200:
                return {'status': 'FAIL', 'error': f'Funding rate HTTP {fr_resp.status_code}'}
            
            fr_data = fr_resp.json()
            funding_rate = float(fr_data[0]['fundingRate']) * 100 if fr_data else 0
            
            # Long/Short ratio
            ls_resp = requests.get(
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                params={'symbol': 'BTCUSDT', 'period': '1h', 'limit': 1},
                timeout=10
            )
            
            ls_ratio = 1.0
            if ls_resp.status_code == 200:
                ls_data = ls_resp.json()
                if ls_data:
                    ls_ratio = float(ls_data[0]['longShortRatio'])
            
            return {
                'status': 'PASS',
                'data': {
                    'funding_rate': f"{funding_rate:.4f}%",
                    'long_short_ratio': f"{ls_ratio:.2f}"
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_coingecko_api(self) -> Dict:
        """Test CoinGecko API."""
        import requests
        
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=10
            )
            
            if resp.status_code != 200:
                return {'status': 'FAIL', 'error': f'HTTP {resp.status_code}'}
            
            data = resp.json()['data']
            btc_dominance = data['market_cap_percentage']['btc']
            total_market_cap = data['total_market_cap']['usd']
            
            return {
                'status': 'PASS',
                'data': {
                    'btc_dominance': f"{btc_dominance:.1f}%",
                    'total_market_cap': f"${total_market_cap/1e12:.2f}T"
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_fear_greed_api(self) -> Dict:
        """Test Fear & Greed Index API."""
        import requests
        
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=10
            )
            
            if resp.status_code != 200:
                return {'status': 'FAIL', 'error': f'HTTP {resp.status_code}'}
            
            data = resp.json()['data'][0]
            value = int(data['value'])
            classification = data['value_classification']
            
            return {
                'status': 'PASS',
                'data': {
                    'fear_greed_index': value,
                    'classification': classification
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_unified_brain(self) -> Dict:
        """Test Unified Brain analysis."""
        try:
            from src.brain.unified_brain import get_unified_brain
            
            brain = get_unified_brain()
            
            # Test analyze (should return Signal or None)
            signal = await brain.analyze('BTCUSDT')
            
            if signal:
                return {
                    'status': 'PASS',
                    'data': {
                        'symbol': signal.symbol,
                        'direction': signal.direction,
                        'confidence': f"{signal.confidence:.0f}%",
                        'entry': f"${signal.entry_price:,.2f}",
                        'tp': f"${signal.take_profit:,.2f}",
                        'sl': f"${signal.stop_loss:,.2f}",
                        'rr': f"{signal.risk_reward:.1f}"
                    }
                }
            else:
                return {
                    'status': 'WARN',
                    'message': 'No signal generated (confidence below threshold)',
                    'data': {'min_confidence': brain.MIN_CONFIDENCE}
                }
                
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_early_warning(self) -> Dict:
        """Test Early Warning system."""
        try:
            from src.brain.early_warning import get_warning_system
            
            ws = get_warning_system()
            warnings = await ws.scan_all()
            
            warning_summary = {}
            for w in warnings:
                warning_summary[f"{w.symbol}_{w.warning_type}"] = w.severity
            
            return {
                'status': 'PASS',
                'data': {
                    'total_warnings': len(warnings),
                    'critical': len([w for w in warnings if w.severity == 'CRITICAL']),
                    'high': len([w for w in warnings if w.severity == 'HIGH']),
                    'medium': len([w for w in warnings if w.severity == 'MEDIUM']),
                    'samples': list(warning_summary.items())[:3]
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_performance_tracker(self) -> Dict:
        """Test Performance Tracker."""
        try:
            from src.brain.signal_performance import get_performance_tracker
            
            tracker = get_performance_tracker()
            stats = tracker.get_stats()
            
            return {
                'status': 'PASS',
                'data': {
                    'total_signals': stats.get('total_signals', 0),
                    'active_signals': stats.get('active_signals', 0),
                    'win_rate': f"{stats.get('win_rate', 0):.1f}%"
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_auto_trainer(self) -> Dict:
        """Test Auto Trainer."""
        try:
            from src.brain.auto_trainer import get_trainer
            
            trainer = get_trainer()
            status = trainer.get_training_status()
            
            # Test data fetch
            df = await trainer._fetch_training_data('BTCUSDT', limit=50)
            
            return {
                'status': 'PASS',
                'data': {
                    'is_training': status.get('is_training', False),
                    'last_train': status.get('last_train', {}),
                    'data_fetch_ok': df is not None,
                    'data_rows': len(df) if df is not None else 0
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_lstm_model(self) -> Dict:
        """Test LSTM model loading."""
        try:
            from src.brain.models.lstm_trend import LSTMTrendPredictor
            
            model = LSTMTrendPredictor(symbol='BTCUSDT')
            
            return {
                'status': 'PASS' if model.trained else 'WARN',
                'message': 'Model not trained yet' if not model.trained else None,
                'data': {
                    'model_loaded': model.trained,
                    'symbol': 'BTCUSDT'
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_websocket(self) -> Dict:
        """Test WebSocket connection (quick test)."""
        try:
            import websockets
            
            url = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
            
            async with websockets.connect(url, ping_interval=5, ping_timeout=5) as ws:
                # Receive one message
                message = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(message)
                
                return {
                    'status': 'PASS',
                    'data': {
                        'connected': True,
                        'trade_price': f"${float(data.get('p', 0)):,.2f}",
                        'trade_qty': f"{float(data.get('q', 0)):.4f} BTC"
                    }
                }
                
        except asyncio.TimeoutError:
            return {'status': 'WARN', 'message': 'Timeout waiting for trade data'}
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def _print_report(self):
        """Print final report."""
        print("\n" + "="*60)
        print("   FINAL REPORT")
        print("="*60)
        
        passed = len([r for r in self.results.values() if r['status'] == 'PASS'])
        warned = len([r for r in self.results.values() if r['status'] == 'WARN'])
        failed = len([r for r in self.results.values() if r['status'] == 'FAIL'])
        total = len(self.results)
        
        print(f"\nResults: {passed} PASS / {warned} WARN / {failed} FAIL (Total: {total})")
        
        if self.warnings:
            print(f"\nWarnings:")
            for w in self.warnings:
                print(f"  - {w}")
        
        if self.errors:
            print(f"\nErrors:")
            for e in self.errors:
                print(f"  - {e}")
        
        print("\n" + "="*60)
        
        if failed == 0:
            print("STATUS: ALL SYSTEMS OPERATIONAL")
        else:
            print(f"STATUS: {failed} SYSTEM(S) NEED ATTENTION")
        
        print("="*60 + "\n")


async def main():
    """Main entry point."""
    tester = LiveTestSystem()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
