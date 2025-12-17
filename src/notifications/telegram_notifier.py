# -*- coding: utf-8 -*-
"""
DEMIR AI - Advanced Telegram Notifier
7/24 akıllı bildirim sistemi.

PHASE 50: Comprehensive Notification System
- Sinyal bildirimi (Entry/SL/TP)
- TP/SL sonuç bildirimi
- Risk/Fırsat uyarısı
- Haber bildirimi
- Heartbeat (1 saat sessizlik)
- 4 coin takibi
"""
import logging
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

# Import Config for token and chat_id
try:
    from src.config.settings import Config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

logger = logging.getLogger("TELEGRAM_NOTIFIER")


class TelegramNotifier:
    """
    Gelişmiş Telegram Bildirim Sistemi
    
    Bildirim Tipleri:
    1. SIGNAL - Yeni sinyal (Entry/SL/TP)
    2. TP_HIT - Take Profit vuruldu
    3. SL_HIT - Stop Loss vuruldu
    4. WARNING - Risk/Fırsat uyarısı
    5. NEWS - Önemli haber
    6. HEARTBEAT - Sistem aktif (1 saat sessizlikten sonra)
    """
    
    # Takip edilen coinler
    COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT']
    
    def __init__(self, token: str = None, chat_id: str = None):
        # Use Config if available (existing system integration)
        if HAS_CONFIG:
            self.token = token or Config.TELEGRAM_TOKEN or ''
            self.chat_id = chat_id or Config.TELEGRAM_CHAT_ID or ''
        else:
            self.token = token or os.getenv('TELEGRAM_BOT_TOKEN') or ''
            self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID') or ''
        
        self.last_heartbeat = datetime.now()
        self.last_signal_time: Dict[str, datetime] = {}
        self.enabled = bool(self.token and self.chat_id)
        
        if self.enabled:
            logger.info("✅ Telegram Notifier initialized with Config settings")
        else:
            logger.warning("⚠️ Telegram not configured (missing token or chat_id)")
    
    def send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """Telegram'a mesaj gönder."""
        if not self.enabled:
            logger.info(f"[TELEGRAM DISABLED] {text[:100]}...")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            resp = requests.post(url, json=payload, timeout=10)
            
            if resp.status_code == 200:
                logger.info(f"✅ Telegram message sent")
                return True
            else:
                logger.error(f"Telegram error: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    # ==========================================
    # 1. SIGNAL NOTIFICATION
    # ==========================================
    def send_signal(self, symbol: str, direction: str, entry: float,
                   stop_loss: float, tp1: float, tp2: float,
                   confidence: float, modules: List[str],
                   signal_id: str) -> bool:
        """
        Yeni sinyal bildirimi.
        """
        dir_emoji = "📈" if direction == "LONG" else "📉"
        
        sl_pct = ((stop_loss / entry) - 1) * 100 if direction == 'LONG' else ((entry / stop_loss) - 1) * 100
        tp1_pct = ((tp1 / entry) - 1) * 100 if direction == 'LONG' else ((entry / tp1) - 1) * 100
        tp2_pct = ((tp2 / entry) - 1) * 100 if direction == 'LONG' else ((entry / tp2) - 1) * 100
        
        message = f"""
🎯 <b>DEMIR AI - YENİ SİNYAL</b>

{dir_emoji} <b>{direction} {symbol}</b>

💰 Entry: <code>${entry:,.2f}</code>
🛑 Stop Loss: <code>${stop_loss:,.2f}</code> ({sl_pct:+.1f}%)
🎯 TP1: <code>${tp1:,.2f}</code> ({tp1_pct:+.1f}%)
🎯 TP2: <code>${tp2:,.2f}</code> ({tp2_pct:+.1f}%)

📊 Güven: <b>{confidence:.0f}%</b>
🤝 {len(modules)}/7 modül aynı fikirde

📝 <i>Katkıda bulunanlar:</i>
• {', '.join(modules)}

🔔 <b>ID:</b> <code>{signal_id}</code>
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
        """.strip()
        
        result = self.send_message(message)
        
        if result:
            self.last_signal_time[symbol] = datetime.now()
        
        return result
    
    # ==========================================
    # 2. TP/SL RESULT NOTIFICATION
    # ==========================================
    def send_result(self, symbol: str, direction: str, result_type: str,
                   entry: float, exit_price: float, profit_pct: float,
                   duration_hours: float, signal_id: str) -> bool:
        """
        TP veya SL sonuç bildirimi.
        
        result_type: TP1_HIT, TP2_HIT, SL_HIT
        """
        if 'TP' in result_type:
            emoji = "✅"
            result_text = "TP VURULDU! 🎉"
            profit_emoji = "📊"
        else:
            emoji = "❌"
            result_text = "SL VURULDU 😔"
            profit_emoji = "📉"
        
        dir_emoji = "📈" if direction == "LONG" else "📉"
        
        message = f"""
{emoji} <b>DEMIR AI - {result_text}</b>

{dir_emoji} <b>{symbol} {direction}</b>

💰 Entry: <code>${entry:,.2f}</code>
🏁 Çıkış: <code>${exit_price:,.2f}</code>
{profit_emoji} Sonuç: <b>{profit_pct:+.2f}%</b>

⏱️ Süre: {duration_hours:.1f} saat
🔔 ID: <code>{signal_id}</code>

⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
        """.strip()
        
        return self.send_message(message)
    
    # ==========================================
    # 3. WARNING/OPPORTUNITY NOTIFICATION
    # ==========================================
    def send_warning(self, symbol: str, warning_type: str,
                    description: str, potential_signal: str = None,
                    timeframe: str = None) -> bool:
        """
        Risk veya fırsat uyarısı.
        
        warning_type: WHALE_ALERT, HIGH_VOLATILITY, LIQUIDATION_RISK, OPPORTUNITY
        """
        type_emojis = {
            'WHALE_ALERT': '🐋',
            'HIGH_VOLATILITY': '⚡',
            'LIQUIDATION_RISK': '⚠️',
            'OPPORTUNITY': '💡',
            'NEWS': '📰'
        }
        
        emoji = type_emojis.get(warning_type, '⚠️')
        
        message = f"""
{emoji} <b>DEMIR AI - UYARI</b>

<b>{warning_type.replace('_', ' ')}</b>
📍 {symbol}

{description}
        """.strip()
        
        if potential_signal:
            message += f"\n\n📊 <i>Potansiyel sinyal:</i> {potential_signal}"
        
        if timeframe:
            message += f"\n⏰ <i>Tahmini:</i> {timeframe}"
        
        message += f"\n\n🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        
        return self.send_message(message)
    
    # ==========================================
    # 4. NEWS NOTIFICATION
    # ==========================================
    def send_news(self, headline: str, source: str,
                 market_impact: str, sentiment: str) -> bool:
        """
        Önemli haber bildirimi.
        
        sentiment: BULLISH, BEARISH, NEUTRAL
        """
        sentiment_emoji = {
            'BULLISH': '🟢 YÜKSELİŞ',
            'BEARISH': '🔴 DÜŞÜŞ',
            'NEUTRAL': '⚪ NÖTR'
        }
        
        message = f"""
📰 <b>DEMIR AI - ÖNEMLİ HABER</b>

📢 <b>{headline}</b>

📊 Piyasa Etkisi: <b>{market_impact}</b>
🎯 Tahmin: <b>{sentiment_emoji.get(sentiment, sentiment)}</b>

🔗 Kaynak: {source}
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
        """.strip()
        
        return self.send_message(message)
    
    # ==========================================
    # 5. HEARTBEAT NOTIFICATION
    # ==========================================
    def send_heartbeat(self, prices: Dict[str, float] = None) -> bool:
        """
        1 saat sinyal yoksa sistem aktif bildirimi.
        """
        now = datetime.now()
        
        message = f"""
💚 <b>DEMIR AI - SİSTEM AKTİF</b>

⏰ Son 1 saat: <i>Güçlü sinyal yok</i>
📊 Piyasa: <b>NÖTR/BEKLE</b>
        """.strip()
        
        if prices:
            message += "\n\n📉 <b>Güncel Fiyatlar:</b>"
            for symbol, price in prices.items():
                short_name = symbol.replace('USDT', '')
                message += f"\n• {short_name}: ${price:,.2f}"
        
        message += f"\n\n🔄 <i>Takipte...</i>"
        message += f"\n⏰ {now.strftime('%d.%m.%Y %H:%M')}"
        
        result = self.send_message(message)
        
        if result:
            self.last_heartbeat = now
        
        return result
    
    # ==========================================
    # 6. STATUS/STATS NOTIFICATION
    # ==========================================
    def send_stats(self, stats: Dict) -> bool:
        """
        Performans istatistikleri.
        """
        message = f"""
📊 <b>DEMIR AI - PERFORMANS</b>

📈 Toplam Sinyal: {stats.get('total_signals', 0)}
✅ Kazanç: {stats.get('wins', 0)}
❌ Kayıp: {stats.get('losses', 0)}
📊 Win Rate: <b>{stats.get('win_rate', 0):.1f}%</b>

💰 Net Kar: <b>{stats.get('net_profit_pct', 0):+.2f}%</b>
📍 Aktif Sinyal: {stats.get('active_count', 0)}

⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
        """.strip()
        
        return self.send_message(message)
    
    # ==========================================
    # HEARTBEAT CHECK
    # ==========================================
    def should_send_heartbeat(self) -> bool:
        """1 saatten fazla sinyal yoksa True."""
        # Son sinyal zamanını kontrol et
        if not self.last_signal_time:
            # Hiç sinyal gönderilmemiş, son heartbeat'e bak
            time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
            return time_since_heartbeat >= 3600  # 1 saat
        
        # En son sinyal ne zaman gönderildi
        latest_signal = max(self.last_signal_time.values())
        time_since_signal = (datetime.now() - latest_signal).total_seconds()
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        
        # 1 saatten fazla sinyal yok VE 1 saatten fazla heartbeat yok
        return time_since_signal >= 3600 and time_since_heartbeat >= 3600
    
    def get_current_prices(self) -> Dict[str, float]:
        """4 coin için güncel fiyatları al."""
        prices = {}
        for symbol in self.COINS:
            try:
                resp = requests.get(
                    f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
                    timeout=5
                )
                if resp.status_code == 200:
                    prices[symbol] = float(resp.json()['price'])
            except:
                pass
        return prices
    
    def check_and_send_heartbeat(self) -> bool:
        """Heartbeat gerekiyorsa gönder."""
        if self.should_send_heartbeat():
            prices = self.get_current_prices()
            return self.send_heartbeat(prices)
        return False
    
    # ==========================================
    # 7. MONEY FLOW REPORT (INOUT)
    # ==========================================
    def send_money_flow_report(self, money_flow_data: Dict) -> bool:
        """
        Mikabot tarzı para akışı raporu.
        /inout komutuyla tetiklenir.
        """
        try:
            # Header
            msg = "📊 <b>Marketteki Tüm Coinlere Olan Nakit Girişi Raporu.</b>\n"
            msg += f"<i>Kısa Vadeli Market Alım Gücü:</i> <b>{money_flow_data.get('buying_power', '+0.0X')}</b>\n"
            msg += f"<i>Marketteki Hacim Payı:</i> <b>%{money_flow_data.get('market_buyer_pct', 50):.1f}</b>\n"
            msg += "━━━━━━━━━━━━━━━━━━━━\n"
            
            # Timeframe analysis
            tf_flows = money_flow_data.get('timeframe_flows', {})
            for tf in ['15m', '1h', '4h', '12h', '1d']:
                pct = tf_flows.get(tf, 50)
                arrow = "🟢▲" if pct >= 50 else "🔴▼"
                msg += f"{tf}=> <b>%{pct:.1f}</b> {arrow}\n"
            
            msg += "━━━━━━━━━━━━━━━━━━━━\n"
            
            # Coin details
            msg += "<b>En çok nakit girişi olanlar.</b>\n"
            msg += "<i>(Sonunda 🔺 olanlar sağlıklıdır)</i>\n"
            msg += "<i>Nakitin nereye aktığını gösterir.</i>\n\n"
            
            coin_details = money_flow_data.get('coin_details', [])
            
            if coin_details:
                for coin in coin_details[:5]:  # Top 5
                    symbol = coin.get('symbol', '???')
                    flow_pct = coin.get('flow_pct', 0)
                    buyer_15m = coin.get('buyer_15m', 50)
                    mts = coin.get('mts', 0)
                    arrows = coin.get('arrows', '➖➖➖➖➖➖')
                    
                    msg += f"<b>{symbol}</b> Nakit: %{flow_pct:.1f} 15m:%{buyer_15m:.0f} Mts:{mts} {arrows}\n"
            else:
                # Fallback format
                for symbol, data in money_flow_data.get('top_inflow', {}).items():
                    if isinstance(data, dict):
                        pct = data.get('flow_pct', 0)
                    else:
                        pct = data
                    clean_symbol = symbol.replace('USDT', '')
                    arrows = "🔺🔺🔺" if pct >= 55 else "🔺" if pct >= 50 else "🔻"
                    msg += f"🔹 <b>{clean_symbol}</b> Nakit: %{pct:.1f} {arrows}\n"
            
            msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"<i>Güncelleme: {datetime.now().strftime('%H:%M')}</i>"
            
            return self.send_message(msg)
            
        except Exception as e:
            logger.error(f"Money flow report failed: {e}")
            return False
    
    # ==========================================
    # 8. TELEGRAM COMMAND HANDLER
    # ==========================================
    def check_telegram_commands(self, money_flow_analyzer=None) -> None:
        """
        Telegram komutlarını kontrol et.
        
        Desteklenen komutlar:
        - inout, /inout : Para akışı raporu
        - status, /status : Sistem durumu
        - help, /help : Yardım
        """
        if not self.enabled:
            return
        
        try:
            # Initialize last_update_id if not exists
            if not hasattr(self, 'last_update_id'):
                self.last_update_id = 0
            
            # Get updates from Telegram
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 1,
                'allowed_updates': ['message']
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('ok') and data.get('result'):
                    for update in data['result']:
                        self.last_update_id = update['update_id']
                        
                        message = update.get('message', {})
                        text = message.get('text', '').lower().strip()
                        chat_id = str(message.get('chat', {}).get('id', ''))
                        
                        # Only respond to authorized chat
                        if chat_id != str(self.chat_id):
                            continue
                        
                        # Handle commands
                        if text in ['inout', '/inout', 'moneyflow', '/moneyflow']:
                            logger.info(f"📩 Telegram command received: {text}")
                            self._handle_inout_command(money_flow_analyzer)
                        
                        elif text in ['status', '/status', 'durum', '/durum']:
                            logger.info(f"📩 Telegram command received: {text}")
                            self._handle_status_command()
                        
                        elif text in ['help', '/help', 'yardim', '/yardim']:
                            self._handle_help_command()
                            
        except requests.exceptions.Timeout:
            pass  # Normal, non-blocking check
        except Exception as e:
            logger.debug(f"Telegram command check error: {e}")
    
    def _handle_inout_command(self, money_flow_analyzer=None):
        """inout komutu - Para akışı raporu."""
        if money_flow_analyzer is None:
            # Try to import and create analyzer
            try:
                from src.data_ingestion.money_flow_analyzer import MoneyFlowAnalyzer
                money_flow_analyzer = MoneyFlowAnalyzer()
            except Exception as e:
                self.send_message("⚠️ Money Flow Analyzer kullanılamıyor")
                return
        
        try:
            self.send_message("⏳ Nakit akışı hesaplanıyor...")
            
            # Get fresh money flow data
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            money_flow_data = loop.run_until_complete(money_flow_analyzer.get_market_money_flow())
            loop.close()
            
            self.send_money_flow_report(money_flow_data)
            
        except Exception as e:
            self.send_message(f"❌ Hata: {e}")
            logger.error(f"Money flow command error: {e}")
    
    def _handle_status_command(self):
        """status komutu - Sistem durumu."""
        try:
            from src.notifications.signal_tracker import SignalTracker
            tracker = SignalTracker()
            stats = tracker.get_statistics()
            active = len(tracker.get_active_signals())
        except:
            stats = {}
            active = 0
        
        msg = "📊 <b>DEMIR AI DURUM</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"⏰ Zaman: {datetime.now().strftime('%H:%M:%S')}\n"
        msg += f"🟢 Sistem: AKTIF\n"
        msg += f"📈 Aktif Pozisyon: {active}\n"
        msg += f"📊 Win Rate: {stats.get('win_rate', 0):.0f}%\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"<i>/inout - Para akışı raporu</i>\n"
        msg += f"<i>/help - Komut listesi</i>"
        
        self.send_message(msg)
    
    def _handle_help_command(self):
        """help komutu - Yardım."""
        msg = "📋 <b>KOMUT LİSTESİ</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"📊 <code>/inout</code> - Nakit akışı raporu\n"
        msg += f"📈 <code>/status</code> - Sistem durumu\n"
        msg += f"❓ <code>/help</code> - Bu yardım mesajı\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"<i>Mikabot-tarzı para akışı analizi</i>"
        
        self.send_message(msg)


# Convenience functions
def get_notifier() -> TelegramNotifier:
    """Singleton notifier."""
    return TelegramNotifier()


def quick_signal(symbol: str, direction: str, entry: float,
                sl: float, tp1: float, tp2: float,
                confidence: float) -> bool:
    """Hızlı sinyal gönder."""
    notifier = TelegramNotifier()
    signal_id = f"{symbol[:3]}-{datetime.now().strftime('%Y%m%d%H%M')}"
    return notifier.send_signal(symbol, direction, entry, sl, tp1, tp2, confidence, ['Manual'], signal_id)

