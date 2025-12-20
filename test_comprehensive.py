# -*- coding: utf-8 -*-
"""
DEMIR AI - COMPREHENSIVE DATA SOURCE TEST
==========================================
17 veri kaynağı + 13 tetikleyiciyi test eder.
Hangi kaynak canlı data, hangisi boş - detaylı rapor.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

async def test_all_data_sources():
    print("=" * 60)
    print("🧪 DEMIR AI - VERİ KAYNAĞI TESTİ")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {}
    
    # =========================================
    # 17 VERİ KAYNAĞI TESTİ
    # =========================================
    
    print("\n" + "=" * 60)
    print("📊 17 VERİ KAYNAĞI ANALİZİ")
    print("=" * 60)
    
    try:
        from src.brain.institutional_aggregator import get_aggregator
        agg = get_aggregator()
        
        # BTC için snapshot al
        print("\n🔄 Veri toplanıyor (BTCUSDT)...")
        snapshot = await agg.get_live_snapshot("BTCUSDT")
        
        # Her kaynağı analiz et
        sources = [
            ("1. Whale Activity", snapshot.whale_net_flow != 0 or snapshot.whale_trade_count != 0, 
             f"Net Flow: ${snapshot.whale_net_flow:,.0f}, Trades: {snapshot.whale_trade_count}"),
            
            ("2. Order Book", snapshot.orderbook_imbalance != 1.0, 
             f"Imbalance: {snapshot.orderbook_imbalance:.2f}, Bid: {snapshot.orderbook_bid_volume:.2f}, Ask: {snapshot.orderbook_ask_volume:.2f}"),
            
            ("3. Liquidation", snapshot.liq_nearest_level != 0, 
             f"Long: ${snapshot.liq_long_total:,.0f}, Short: ${snapshot.liq_short_total:,.0f}, Nearest: ${snapshot.liq_nearest_level:,.0f}"),
            
            ("4. Funding Rate", snapshot.funding_rate != 0, 
             f"Rate: {snapshot.funding_rate:.4f}%, Predicted: {snapshot.funding_predicted:.4f}%"),
            
            ("5. Open Interest", snapshot.open_interest != 0, 
             f"OI: {snapshot.open_interest:,.0f}, 1h: {snapshot.oi_change_1h:.2f}%, 24h: {snapshot.oi_change_24h:.2f}%"),
            
            ("6. Long/Short Ratio", snapshot.long_short_ratio != 1.0, 
             f"Ratio: {snapshot.long_short_ratio:.2f}, Long: {snapshot.long_account_pct:.1f}%, Short: {snapshot.short_account_pct:.1f}%"),
            
            ("7. CVD", snapshot.cvd_value != 0 or snapshot.cvd_trend != "NEUTRAL", 
             f"Value: {snapshot.cvd_value:,.0f}, Trend: {snapshot.cvd_trend}"),
            
            ("8. Exchange Flow", snapshot.exchange_netflow != 0, 
             f"Inflow: ${snapshot.exchange_inflow:,.0f}, Outflow: ${snapshot.exchange_outflow:,.0f}, Net: ${snapshot.exchange_netflow:,.0f}"),
            
            ("9. Stablecoin", snapshot.usdt_supply_change != 0 or snapshot.usdc_supply_change != 0, 
             f"USDT: {snapshot.usdt_supply_change:.2f}%, USDC: {snapshot.usdc_supply_change:.2f}%"),
            
            ("10. DeFi TVL", snapshot.defi_tvl != 0, 
             f"TVL: ${snapshot.defi_tvl/1e9:.2f}B, Change: {snapshot.defi_tvl_change_24h:.2f}%"),
            
            ("11. Options", snapshot.put_call_ratio != 1.0 or snapshot.max_pain_price != 0, 
             f"Put/Call: {snapshot.put_call_ratio:.2f}, Max Pain: ${snapshot.max_pain_price:,.0f}"),
            
            ("12. CME Gap", snapshot.cme_gap_price != 0, 
             f"Gap: ${snapshot.cme_gap_price:,.0f}, Filled: {snapshot.cme_gap_filled}, Dir: {snapshot.cme_gap_direction}"),
            
            ("13. Cross-Exchange", snapshot.binance_price != 0, 
             f"Binance: ${snapshot.binance_price:,.2f}, CB Premium: {snapshot.coinbase_premium:.3f}%, Bybit: {snapshot.bybit_premium:.3f}%"),
            
            ("14. ETF/Grayscale", snapshot.etf_flow_daily != 0 or snapshot.grayscale_premium != 0, 
             f"Daily Flow: ${snapshot.etf_flow_daily:.0f}M, GBTC Premium: {snapshot.grayscale_premium:.2f}%"),
            
            ("15. Fear & Greed", snapshot.fear_greed_index != 50, 
             f"Index: {snapshot.fear_greed_index}, Label: {snapshot.fear_greed_label}"),
            
            ("16. Network", snapshot.hash_rate != 0, 
             f"Hash: {snapshot.hash_rate:.2f} EH/s, Change: {snapshot.hash_rate_change:.2f}%"),
            
            ("17. Taker Volume", snapshot.taker_buy_ratio != 0.5, 
             f"Buy Ratio: {snapshot.taker_buy_ratio:.2f}, Buy: {snapshot.taker_buy_volume:,.0f}, Sell: {snapshot.taker_sell_volume:,.0f}"),
        ]
        
        live_count = 0
        for name, is_live, details in sources:
            status = "✅ CANLI" if is_live else "❌ BOŞ"
            print(f"\n{name}: {status}")
            print(f"   {details}")
            results[name] = {"live": is_live, "details": details}
            if is_live:
                live_count += 1
        
        print(f"\n{'=' * 60}")
        print(f"📊 ÖZET: {live_count}/17 kaynak CANLI data döndürüyor")
        print(f"{'=' * 60}")
        
    except Exception as e:
        print(f"❌ Aggregator hatası: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================
    # 13 TETİKLEYİCİ TESTİ
    # =========================================
    
    print("\n" + "=" * 60)
    print("⚡ 13 TETİKLEYİCİ ANALİZİ")
    print("=" * 60)
    
    try:
        alert_snapshot = await agg.check_sudden_triggers("BTCUSDT")
        
        print(f"\n🔔 Aktif Tetikleyici Sayısı: {alert_snapshot.active_trigger_count}/13")
        print(f"📊 Dominant Yön: {alert_snapshot.dominant_direction}")
        print(f"⚠️ Genel Seviye: {alert_snapshot.overall_severity}")
        print(f"📢 Uyarı Gönderilmeli: {'EVET' if alert_snapshot.should_alert else 'HAYIR'}")
        
        if alert_snapshot.triggers:
            print("\n🔥 Aktif Tetikleyiciler:")
            for t in alert_snapshot.triggers:
                print(f"   • {t.name}: {t.value} ({t.severity}) - {t.direction}")
                print(f"     {t.message}")
        else:
            print("\n💤 Şu an aktif tetikleyici yok.")
            
    except Exception as e:
        print(f"❌ Trigger test hatası: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================
    # TELEGRAM TEST
    # =========================================
    
    print("\n" + "=" * 60)
    print("📱 TELEGRAM BAĞLANTI TESTİ")
    print("=" * 60)
    
    try:
        from src.config.settings import Config
        
        token_ok = bool(Config.TELEGRAM_TOKEN)
        chat_ok = bool(Config.TELEGRAM_CHAT_ID)
        
        print(f"\n🔑 Token: {'✅ MEVCUT' if token_ok else '❌ EKSİK'}")
        print(f"💬 Chat ID: {'✅ MEVCUT' if chat_ok else '❌ EKSİK'}")
        
        if token_ok and chat_ok:
            from src.utils.notifications import NotificationManager
            notifier = NotificationManager()
            
            test_msg = f"""🧪 *TEST MESAJI*
━━━━━━━━━━━━━━━━
✅ Sistem çalışıyor!
📊 17 kaynak bağlı
⚡ 13 tetikleyici aktif
━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}"""
            
            await notifier.send_message_raw(test_msg)
            print("📤 Test mesajı gönderildi!")
        
    except Exception as e:
        print(f"❌ Telegram test hatası: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 TEST TAMAMLANDI")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    asyncio.run(test_all_data_sources())
