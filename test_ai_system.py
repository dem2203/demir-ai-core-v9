# Test AI Signal System
import asyncio
import sys
sys.path.insert(0, '.')

async def test_signal_system():
    print('='*60)
    print('DEMIR AI SIGNAL SYSTEM TEST')
    print('='*60)
    
    # 1. Test SignalOrchestrator
    print('\n1. SIGNAL ORCHESTRATOR TEST')
    print('-'*40)
    try:
        from src.brain.signal_orchestrator import SignalOrchestrator
        orch = SignalOrchestrator()
        
        print(f'Registered modules: {len(orch.weights)}')
        for mod, weight in orch.weights.items():
            print(f'  - {mod}: {weight*100:.0f}%')
        
        # Collect signals
        print('\nCollecting signals from all modules...')
        signals = await orch.collect_all_signals('BTCUSDT', 100000)
        print(f'Signals collected: {len(signals)}/{len(orch.weights)}')
        
        for sig in signals:
            print(f'  OK {sig.module_name}: {sig.direction} ({sig.confidence:.0f}%)')
    except Exception as e:
        print(f'  ERROR: {e}')
    
    # 2. Test Signal Tracker
    print('\n2. SIGNAL TRACKER TEST')
    print('-'*40)
    try:
        from src.notifications.signal_tracker import SignalTracker
        tracker = SignalTracker()
        
        stats = tracker.get_statistics()
        print(f'Total signals: {stats.get("total_signals", 0)}')
        print(f'Win rate: {stats.get("win_rate", 0):.1f}%')
        print(f'Active: {len(tracker.get_active_signals())}')
    except Exception as e:
        print(f'  ERROR: {e}')
    
    # 3. Test News Scraper
    print('\n3. NEWS SCRAPER TEST')
    print('-'*40)
    try:
        from src.brain.news_scraper import CryptoNewsScraper
        scraper = CryptoNewsScraper()
        scraper.fetch_all_news(max_age_hours=4)
        sentiment = scraper.get_market_sentiment()
        print(f'Sentiment: {sentiment["overall"]}')
        print(f'Bullish: {sentiment["bullish_count"]} | Bearish: {sentiment["bearish_count"]}')
    except Exception as e:
        print(f'  ERROR: {e}')
    
    # 4. Test CME Gap
    print('\n4. CME GAP TRACKER TEST')
    print('-'*40)
    try:
        from src.brain.cme_gap_tracker import CMEGapTracker
        tracker = CMEGapTracker()
        status = tracker.get_gap_status()
        print(f'Status: {status["status"]}')
        if status.get('size_usd'):
            print(f'Gap: ${status["size_usd"]:,.0f}')
    except Exception as e:
        print(f'  ERROR: {e}')
    
    # 5. Test Options Flow
    print('\n5. OPTIONS FLOW TEST')
    print('-'*40)
    try:
        from src.brain.options_flow import OptionsFlowAnalyzer
        analyzer = OptionsFlowAnalyzer()
        options = analyzer.analyze()
        if options.get('available'):
            print(f'C/P Ratio: {options["call_put_ratio"]:.2f}')
            print(f'Max Pain: ${options["max_pain"]:,.0f}')
            print(f'IV Rank: {options["iv_rank"]:.0f}%')
        else:
            print('Options data not available')
    except Exception as e:
        print(f'  ERROR: {e}')
    
    # 6. Test Live Price
    print('\n6. LIVE PRICE VERIFICATION')
    print('-'*40)
    try:
        import requests
        resp = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', timeout=5)
        if resp.status_code == 200:
            price = float(resp.json()['price'])
            print(f'BTC/USDT: ${price:,.2f} - REAL DATA')
    except Exception as e:
        print(f'  ERROR: {e}')
    
    print('\n' + '='*60)
    print('TEST COMPLETE')
    print('='*60)

if __name__ == '__main__':
    asyncio.run(test_signal_system())
