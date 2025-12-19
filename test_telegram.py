# -*- coding: utf-8 -*-
"""
Telegram Test Messages - Send examples of all notification types
"""
import asyncio
import sys
sys.path.insert(0, '.')

async def send_test_messages():
    """Send test messages for all notification types."""
    from src.utils.notifications import NotificationManager
    from datetime import datetime
    
    notifier = NotificationManager()
    
    if not notifier.telegram_token or not notifier.telegram_chat_id:
        print("❌ Telegram not configured!")
        return
    
    print("✅ Telegram configured, sending test messages...")
    
    # 1. AI OBSERVATION (from ai_observation.py format)
    observation_msg = """
🧠 AI TRADER ANALİZİ - BTCUSDT
━━━━━━━━━━━━━━━━━━━━━━
🌍 Avrupa Session | ORTA volatilite
💰 Fiyat: $97,250.00
📊 15dk: +0.38% | Hacim: 4.6x
💹 Order Flow: 🟢 ALICI BASKISI

📋 5 MODÜL ANALİZİ:
  📈 Fiyat: +0.38%
  🐋 Whale: LONG (%65)
  😱 Fear&Greed: 25 (Korku = ALIM FIRSATI!)
  📐 Fib: $96,500 desteğinde
  🎯 Liq Magnet: Yukarı ($98,500)

🟢 KOMBİNE SİNYAL: LONG
🟢 Güven: %72 (YÜKSEK)
━━━━━━━━━━━━━━━━━━━━━━
🎯 İŞLEM DETAYLARI:
▸ Giriş: $97,250
▸ Hedef: $99,400 (+2.2%)
▸ Stop: $96,150 (-1.1%)
▸ R:R = 1:2.0
━━━━━━━━━━━━━━━━━━━━━━
📝 TRADER YORUMU:
🌍 Avrupa session'ındayız. Orta hacim.
📈 BTC yukarı yönlü sinyal veriyor.
💹 Order flow: Taker alımları güçlü.
━━━━━━━━━━━━━━━━━━━━━━
⏰ """ + datetime.now().strftime('%d.%m.%Y %H:%M') + """
🔬 TEST MESAJI - Phase 126
"""
    await notifier.send_message_raw(observation_msg.strip())
    print("✅ 1/3 AI Observation sent")
    await asyncio.sleep(2)
    
    # 2. FULL SIGNAL
    signal_msg = """
🚀 DEMIR AI SİNYAL - ETHUSDT
━━━━━━━━━━━━━━━━━━━━━━
🟢 YÖN: LONG ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━
📍 GİRİŞ: $3,420.00
🛡️ STOP: $3,350.00 (2.0% risk)
━━━ HEDEFLER ━━━
🎯 TP1: $3,490.00 (R:R 2.0)
🎯 TP2: $3,560.00 (R:R 4.0)
🎯 TP3: $3,650.00 (R:R 6.5)
━━━━━━━━━━━━━━━━━━━━━━
🧠 MODÜLLER:
• LSTM: UP (%68)
• Whale: LONG
• SMC: Bullish OB
• MTF: 3/3 BULLISH
━━━━━━━━━━━━━━━━━━━━━━
📊 Güven: %78
💰 Size: 3.5%
⏰ """ + datetime.now().strftime('%d.%m.%Y %H:%M') + """
🔬 TEST MESAJI - Phase 126
"""
    await notifier.send_message_raw(signal_msg.strip())
    print("✅ 2/3 Full Signal sent")
    await asyncio.sleep(2)
    
    # 3. WHALE ALERT
    alert_msg = """
🐋 WHALE ALERT - BTCUSDT
━━━━━━━━━━━━━━━━━━━━━━
💰 $85M Long pozisyon açıldı!

📊 Detaylar:
• Exchange: Binance
• Leverage: 10x tahmini
• Fiyat: $97,200

🎯 Olası etki: Yükseliş baskısı
⏰ """ + datetime.now().strftime('%d.%m.%Y %H:%M') + """
🔬 TEST MESAJI - Phase 126
"""
    await notifier.send_message_raw(alert_msg.strip())
    print("✅ 3/3 Whale Alert sent")
    
    print("\n✅ TÜM TEST MESAJLARI GÖNDERİLDİ!")
    print("Telegram'ı kontrol et!")

if __name__ == "__main__":
    asyncio.run(send_test_messages())
