# -*- coding: utf-8 -*-
"""
Early Signal Engine Test Script
================================
Early Signal Engine'i yerelde test eder.
"""
import asyncio
import logging
import sys
import os

# Windows encoding fix
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TEST_EARLY_SIGNAL")


async def test_early_signal_engine():
    """
    Early Signal Engine'i test et.
    """
    print("\n" + "="*60)
    print("[TEST] EARLY SIGNAL ENGINE TEST")
    print("="*60 + "\n")
    
    try:
        # Import
        from src.v10.early_signal_engine import get_early_signal_engine
        from src.v10.leading_indicators import get_leading_indicators
        
        print("[OK] Import basarili\n")
        
        # Leading Indicators test
        print("[INFO] Leading Indicators Test...")
        print("-" * 40)
        
        li = await get_leading_indicators()
        
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            print(f"\n[ANALYZE] {symbol} analizi...")
            signal = await li.calculate_all(symbol)
            
            print(f"   Direction: {signal.direction.value}")
            print(f"   Strength: {signal.strength:.1f}")
            print(f"   Confidence: {signal.confidence:.1f}")
            print(f"   Reasoning: {signal.reasoning}")
            
            print("   Indicators:")
            for ind in signal.indicators:
                status = "[+]" if ind.value > 10 else "[-]" if ind.value < -10 else "[=]"
                print(f"     {status} {ind.name}: {ind.value:+.1f} ({ind.direction.value})")
        
        await li.close()
        
        # Early Signal Engine test
        print("\n" + "-" * 40)
        print("[INFO] Early Signal Engine Test...")
        print("-" * 40)
        
        engine = await get_early_signal_engine()
        
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            print(f"\n[SIGNAL] {symbol} Early Signal...")
            early_signal = await engine.analyze(symbol)
            
            print(f"   Action: {early_signal.action}")
            print(f"   Confidence: {early_signal.confidence:.1f}%")
            print(f"   Entry Zone: ${early_signal.entry_zone[0]:,.2f} - ${early_signal.entry_zone[1]:,.2f}")
            print(f"   Stop Loss: ${early_signal.stop_loss:,.2f}")
            print(f"   Take Profit: ${early_signal.take_profit:,.2f}")
            print(f"   R/R: {early_signal.risk_reward:.2f}")
            print(f"   Reasoning: {early_signal.reasoning}")
        
        await engine.close()
        
        print("\n" + "="*60)
        print("[SUCCESS] TUM TESTLER BASARILI!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] TEST HATASI: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_early_signal_engine())
    sys.exit(0 if success else 1)
