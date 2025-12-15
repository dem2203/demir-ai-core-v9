import logging
import requests
import asyncio
import json
import os
from datetime import datetime, timedelta
from src.config.settings import Config
from src.core.signal_filter import SignalQualityFilter
from src.utils.notification_priority import get_notification_priority, NotificationPriority

logger = logging.getLogger("NOTIFICATION_MANAGER")


class DedupCache:
    """
    Prevents spamming the same signal repeatedly.
    Now Price-Aware (Phase 21 Fix).
    """
    def __init__(self, cooldown_minutes: int = 60, price_threshold: float = 0.01):
        self.cache = {}
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.price_threshold = price_threshold # 1% default

    def is_duplicate(self, symbol: str, side: str, current_price: float) -> bool:
        key = f"{symbol}_{side}"
        now = datetime.now()
        
        if key in self.cache:
            last_time, last_price = self.cache[key]
            
            # 1. Time Check
            if now - last_time < self.cooldown:
                # 2. Price Check (If moved > 1%, it's NEW)
                price_diff = abs(current_price - last_price) / last_price
                if price_diff < self.price_threshold:
                    return True # Duplicate (Close in time AND price)
        
        # Update cache
        self.cache[key] = (now, current_price)
        return False

class NotificationManager:
    """
    DEMIR AI V20.0 - TELEGRAM ALERT SYSTEM
    
    Sends critical signals through Telegram.
    """
    
    def __init__(self):
        self.telegram_token = Config.TELEGRAM_TOKEN
        self.telegram_chat_id = Config.TELEGRAM_CHAT_ID
        self.telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage" if self.telegram_token else None
        self.signal_filter = SignalQualityFilter(min_confidence=70.0)
        self.rejected_log_path = "rejected_signals.json"
        self.dedup_cache = DedupCache(cooldown_minutes=60) # Phase 21: Dedup
        
        # Phase 30 Fix: Early warning deduplication (4 hour cooldown for same symbol+direction)
        self.early_warning_cache = {}  # {symbol_direction: last_sent_time}
        
        # Phase 31: Active positions tracking - prevents repeat signals until TP/SL hit
        # Format: {symbol: {'side': 'BUY', 'entry': price, 'sl': price, 'tp1': price, 'timestamp': datetime}}
        self.active_positions = {}
        
        self.priority_calculator = get_notification_priority()  # Phase 29: Smart urgency

    async def send_signal(self, signal: dict, snapshot: dict = None):
        """
        Sends ENHANCED signal to Telegram (Phase 29).
        
        New features:
        - Urgency level (CRITICAL/HIGH/MEDIUM/LOW)
        - Volume Profile zones (POC, VAH, VAL)
        - Cross-exchange price comparison
        """
        if not self.telegram_token or not self.telegram_chat_id:
            return
            
        # 1. STRICT FILTER: Only STRONG signals
        if signal.get('quality') != 'STRONG':
            return
        
        # 2. DEDUP CHECK
        price = signal.get('entry_price', signal.get('price', 0))
        if self.dedup_cache.is_duplicate(signal['symbol'], signal['side'], price):
            return
        
        # 2.5 POSITION TRACKING CHECK (Phase 31)
        # If we already have an active position for this symbol, check if TP/SL was hit
        symbol = signal['symbol']
        if symbol in self.active_positions:
            pos = self.active_positions[symbol]
            
            # Check if position expired (24 hour max)
            hours_since = (datetime.now() - pos['timestamp']).total_seconds() / 3600
            if hours_since >= 24:
                logger.info(f"📊 Position expired after 24h: {symbol}")
                del self.active_positions[symbol]
            else:
                # Check if TP1 or SL was hit
                current_price = price
                if pos['side'] == 'BUY':  # LONG position
                    if pos['tp1'] > 0 and current_price >= pos['tp1']:
                        logger.info(f"🎯 LONG TP1 HIT: {symbol} at ${current_price:,.2f}")
                        del self.active_positions[symbol]
                    elif pos['sl'] > 0 and current_price <= pos['sl']:
                        logger.info(f"🛡️ LONG SL HIT: {symbol} at ${current_price:,.2f}")
                        del self.active_positions[symbol]
                    else:
                        logger.debug(f"⏳ Active LONG position exists for {symbol}, skipping signal")
                        return
                else:  # SHORT position
                    if pos['tp1'] > 0 and current_price <= pos['tp1']:
                        logger.info(f"🎯 SHORT TP1 HIT: {symbol} at ${current_price:,.2f}")
                        del self.active_positions[symbol]
                    elif pos['sl'] > 0 and current_price >= pos['sl']:
                        logger.info(f"🛡️ SHORT SL HIT: {symbol} at ${current_price:,.2f}")
                        del self.active_positions[symbol]
                    else:
                        logger.debug(f"⏳ Active SHORT position exists for {symbol}, skipping signal")
                        return
        
        # 3. CALCULATE URGENCY (Phase 29)
        urgency_data = self.priority_calculator.calculate_urgency(signal, snapshot)
        
        # 4. Check urgency-based cooldown
        if not urgency_data.get('can_send', True):
            logger.debug(f"Signal blocked by urgency cooldown: {signal['symbol']}")
            return
        
        try:
            side_icon = "🟢 LONG 🚀" if signal['side'] == "BUY" else "🔴 SHORT 🔻"
            conf = signal['confidence']
            
            # Get enhanced data from snapshot
            snapshot = snapshot or {}
            smart_sltp = snapshot.get('smart_sltp', {})
            mtf = snapshot.get('mtf', {})
            smc = snapshot.get('smc', {})
            volume_profile = snapshot.get('volume_profile', {})
            cross_exchange = snapshot.get('cross_exchange', {})
            
            # Entry and levels
            entry = signal['entry_price']
            sl = smart_sltp.get('stop_loss', signal.get('sl_price', 0))
            tp1 = smart_sltp.get('take_profit_1', signal.get('tp_price', 0))
            tp2 = smart_sltp.get('take_profit_2', 0)
            tp3 = smart_sltp.get('take_profit_3', 0)
            
            # Risk/Reward ratios
            rr1 = smart_sltp.get('risk_reward_1', 0)
            rr2 = smart_sltp.get('risk_reward_2', 0)
            rr3 = smart_sltp.get('risk_reward_3', 0)
            risk_pct = smart_sltp.get('risk_pct', 0)
            
            # Setup quality
            quality = smart_sltp.get('quality', signal.get('quality', 'UNKNOWN'))
            quality_emoji = "✅" if quality == "EXCELLENT" else "👍" if quality == "GOOD" else "⚠️"
            
            # === BUILD ENHANCED MESSAGE ===
            
            # Header with Urgency
            urgency = urgency_data['urgency']
            urgency_emoji = urgency_data['emoji']
            urgency_score = urgency_data['score']
            action_window = urgency_data['action_window']
            
            message = f"{side_icon} **{signal['symbol']}**\n"
            message += f"━━━━━━━━━━━━━━━━━━\n"
            
            # Urgency Banner (Phase 29)
            if urgency == 'CRITICAL':
                message += f"{urgency_emoji} **URGENCY: CRITICAL** {urgency_emoji}\n"
                message += f"⚡ ACT NOW | Score: {urgency_score}/100\n"
                message += f"⏱️ Window: {action_window}\n"
            elif urgency == 'HIGH':
                message += f"{urgency_emoji} **URGENCY: HIGH** | Score: {urgency_score}/100\n"
                message += f"⏱️ Window: {action_window}\n"
            else:
                message += f"{urgency_emoji} Urgency: {urgency} | Score: {urgency_score}/100\n"
            
            message += f"━━━━━━━━━━━━━━━━━━\n"
            
            # Quality and Confidence
            message += f"{quality_emoji} **Setup:** {quality}\n"
            message += f"🧠 **Confidence:** {conf:.1f}%\n"
            message += f"📉 **Reason:** {signal.get('reason', 'AI Model')[:50]}\n"
            
            # MTF Confluence
            mtf_score = mtf.get('confluence_score', 0)
            if mtf_score > 0:
                mtf_type = mtf.get('confluence_type', 'N/A')
                trends = mtf.get('trends', {})
                trend_1h = trends.get('1h', {}).get('trend', '?')
                trend_4h = trends.get('4h', {}).get('trend', '?')
                trend_1d = trends.get('1d', {}).get('trend', '?')
                t1_e = "🟢" if trend_1h == "BULLISH" else "🔴" if trend_1h == "BEARISH" else "⚪"
                t4_e = "🟢" if trend_4h == "BULLISH" else "🔴" if trend_4h == "BEARISH" else "⚪"
                td_e = "🟢" if trend_1d == "BULLISH" else "🔴" if trend_1d == "BEARISH" else "⚪"
                message += f"📊 **MTF:** {mtf_score}% ({mtf_type})\n"
                message += f"   {t1_e}1H {t4_e}4H {td_e}1D\n"
            
            # SMC Bias
            smc_bias = smc.get('smc_bias', 'N/A')
            if smc_bias != 'N/A':
                bias_e = "🟢" if smc_bias == "BULLISH" else "🔴" if smc_bias == "BEARISH" else "⚪"
                message += f"🎯 **SMC:** {bias_e} {smc_bias}\n"
            
            # === VOLUME PROFILE SECTION (Phase 29.1) ===
            vpoc = volume_profile.get('vpoc', 0)
            vah = volume_profile.get('vah', 0)
            val = volume_profile.get('val', 0)
            vp_position = volume_profile.get('price_position', '')
            
            if vpoc > 0:
                message += f"━━━ VOLUME PROFILE ━━━\n"
                message += f"📊 **POC:** ${vpoc:,.0f}\n"
                message += f"   VAH: ${vah:,.0f} | VAL: ${val:,.0f}\n"
                
                if vp_position:
                    pos_emoji = "🔼" if "ABOVE" in vp_position else "🔽" if "BELOW" in vp_position else "↔️"
                    message += f"   {pos_emoji} Price: {vp_position.replace('_', ' ')}\n"
            
            # === CROSS-EXCHANGE SECTION (Phase 29.3) ===
            if cross_exchange and cross_exchange.get('exchanges_online', 0) > 1:
                message += f"━━━ CROSS-EXCHANGE ━━━\n"
                for ex in ['binance', 'bybit', 'coinbase']:
                    if ex in cross_exchange and cross_exchange[ex].get('price', 0) > 0:
                        ex_data = cross_exchange[ex]
                        ex_price = ex_data['price']
                        deviation = ex_data.get('deviation', 0)
                        dev_str = f"+{deviation:.2f}%" if deviation > 0 else f"{deviation:.2f}%"
                        dev_emoji = "🟢" if deviation > 0 else "🔴" if deviation < 0 else "⚪"
                        message += f"   {dev_emoji} {ex.capitalize()}: ${ex_price:,.0f} ({dev_str})\n"
                
                max_div = cross_exchange.get('max_divergence', 0)
                if max_div >= 0.1:
                    message += f"   ⚡ **Divergence: {max_div:.2f}%**\n"
            
            message += f"━━━━━━━━━━━━━━━━━━\n"
            
            # Entry/SL/TP Section
            message += f"🚪 **ENTRY:** ${entry:,.2f}\n"
            if sl > 0:
                message += f"🛡️ **STOP:** ${sl:,.2f} ({risk_pct:.1f}% risk)\n"
            message += f"━━━ TARGETS ━━━\n"
            if tp1 > 0:
                message += f"🎯 **TP1:** ${tp1:,.2f} (R:R {rr1})\n"
            if tp2 > 0:
                message += f"🎯 **TP2:** ${tp2:,.2f} (R:R {rr2})\n"
            if tp3 > 0:
                message += f"🎯 **TP3:** ${tp3:,.2f} (R:R {rr3})\n"
            
            message += f"━━━━━━━━━━━━━━━━━━\n"
            message += f"💰 **Size:** {signal.get('kelly_size', 'N/A')}%\n"
            message += f"⚠️ _DYOR - AI Advisory Only v29_"
            
            await self.send_message_raw(message)
            logger.info(f"✅ Phase 29 Enhanced Signal sent: {signal['symbol']} [Urgency: {urgency}]")
            
            # Phase 31: Register position for tracking
            self.active_positions[signal['symbol']] = {
                'side': signal['side'],
                'entry': entry,
                'sl': sl,
                'tp1': tp1,
                'timestamp': datetime.now()
            }
            logger.info(f"📊 Position registered: {signal['symbol']} {signal['side']} Entry=${entry:,.2f} SL=${sl:,.2f} TP1=${tp1:,.2f}")
            
        except Exception as e:
            logger.error(f"Telegram Error: {e}")

    async def send_heartbeat(self, price_info: dict):
        """Sends hourly heartbeat if no signals were sent."""
        if not self.telegram_token: return
        
        try:
            msg = (
                f"💓 **System Heartbeat**\n"
                f"━━━━━━━━━━━━━━\n"
                f"🤖 AI Engine: **ONLINE**\n"
                f"📅 Time: {datetime.now().strftime('%H:%M')}\n"
                f"━━━━━━━━━━━━━━\n"
            )
            
            for sym, price in price_info.items():
                msg += f"🔹 **{sym}:** ${price:,.2f}\n"
                
            msg += f"━━━━━━━━━━━━━━\n"
            msg += f"Scanning for opportunities..."
            
            await self.send_message_raw(msg)
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
    
    async def send_money_flow_report(self, money_flow_data: dict):
        """
        Mikabot tarzı para akışı raporu gönderir.
        (Sends Mikabot-style money flow report to Telegram)
        """
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        try:
            # Header
            msg = "📊 **Marketteki Nakit Akışı Raporu**\n"
            msg += f"Kısa Vadeli Alım Gücü: **{money_flow_data.get('buying_power', '0X')}**\n"
            msg += f"Marketteki Hacim Payı: %{money_flow_data.get('market_buyer_pct', 50):.1f}\n"
            msg += "━━━━━━━━━━━━━━━━━━━━\n"
            
            # Timeframe analysis
            market_flow = money_flow_data.get('market_flow', {})
            for tf in ['15m', '1h', '4h', '12h', '1d']:
                pct = market_flow.get(tf, 50)
                arrow = "🔺" if pct >= 50 else "🔻"
                msg += f"{tf}=> %{pct:.1f} {arrow}\n"
            
            msg += "━━━━━━━━━━━━━━━━━━━━\n"
            
            # Top inflow coins
            msg += "**En Çok Nakit Girişi Olanlar:**\n"
            msg += "_(🔺: alım baskısı, 🔻: satış baskısı)_\n\n"
            
            for symbol, pct in money_flow_data.get('top_inflow', [])[:5]:
                clean_symbol = symbol.replace('USDT', '')
                
                # Momentum arrows
                if pct >= 60:
                    arrows = "🔺🔺🔺"
                elif pct >= 55:
                    arrows = "🔺🔺"
                elif pct >= 50:
                    arrows = "🔺"
                elif pct >= 45:
                    arrows = "🔻"
                elif pct >= 40:
                    arrows = "🔻🔻"
                else:
                    arrows = "🔻🔻🔻"
                
                msg += f"🔹 **{clean_symbol}** Nakit: %{pct:.1f} {arrows}\n"
            
            msg += "━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"_Güncelleme: {datetime.now().strftime('%H:%M')}_"
            
            await self.send_message_raw(msg)
            logger.info("📊 Money Flow report sent to Telegram")
            
        except Exception as e:
            logger.error(f"Money flow report failed: {e}")
    
    
    async def send_early_warning(self, symbol: str, warnings: list, visual_prediction: dict = None):
        """
        PROAKTIF ERKEN UYARI MESAJI (Proactive Early Warning)
        
        İndikatörlerden FARKLI olarak:
        - Hareket OLMADAN ÖNCE uyarır
        - Ne olacağını TAHMİN eder
        - ERKEN GİRİŞ noktası verir
        """
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        if not warnings:
            return
        
        # Phase 30 Fix: Check if we already sent this warning recently (4 hour cooldown)
        # Create a key from symbol + first warning title/type
        first_warning = warnings[0] if warnings else {}
        warning_key = f"{symbol}_{first_warning.get('title', 'unknown')}"
        
        now = datetime.now()
        if warning_key in self.early_warning_cache:
            last_sent = self.early_warning_cache[warning_key]
            hours_since = (now - last_sent).total_seconds() / 3600
            if hours_since < 4:  # 4 hour cooldown
                logger.debug(f"Early warning skipped (sent {hours_since:.1f}h ago): {warning_key}")
                return
        
        # Update cache
        self.early_warning_cache[warning_key] = now
        
        try:
            # Build message
            msg = f"⚡ **ERKEN UYARI - {symbol}**\n"
            msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
            
            # Add each warning
            for w in warnings[:3]:  # Max 3 warnings
                priority_emoji = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '⚪'
                }.get(w.get('priority'), '⚪')
                
                msg += f"{priority_emoji} **{w.get('title', 'Uyarı')}**\n"
                msg += f"   {w.get('message', '')}\n"
                msg += f"   ➡️ _{w.get('action', '')}_\n\n"
            
            # Add AI prediction if available
            if visual_prediction and visual_prediction.get('probability', 0) >= 60:
                msg += f"━━━━━━━━━━━━━━━━━━━━\n"
                msg += f"🧠 **AI TAHMİNİ**\n"
                msg += f"📊 Trend: **{visual_prediction.get('trend', 'N/A')}**\n"
                msg += f"🎯 Olasılık: **%{visual_prediction.get('probability', 0)}**\n"
                
                prediction = visual_prediction.get('prediction', '')
                if prediction:
                    msg += f"📈 Tahmin: _{prediction}_\n"
                
                early_entry = visual_prediction.get('early_entry_price')
                target = visual_prediction.get('target_price')
                stop = visual_prediction.get('stop_loss')
                
                if early_entry:
                    msg += f"\n💰 **ERKEN GİRİŞ:**\n"
                    msg += f"   📍 Giriş: ${early_entry:,.0f}\n" if isinstance(early_entry, (int, float)) else f"   📍 Giriş: {early_entry}\n"
                    if target:
                        msg += f"   🎯 Hedef: ${target:,.0f}\n" if isinstance(target, (int, float)) else f"   🎯 Hedef: {target}\n"
                    if stop:
                        msg += f"   🛡️ Stop: ${stop:,.0f}\n" if isinstance(stop, (int, float)) else f"   🛡️ Stop: {stop}\n"
                
                time_horizon = visual_prediction.get('time_horizon')
                if time_horizon:
                    msg += f"   ⏰ Süre: {time_horizon}\n"
                
                risk_warning = visual_prediction.get('risk_warning')
                if risk_warning:
                    msg += f"\n⚠️ Risk: _{risk_warning}_\n"
            
            msg += f"\n━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"_Bu bir ERKEN UYARIDIR. Hareket henüz başlamadı._"
            
            await self.send_message_raw(msg)
            logger.info(f"📢 Early Warning sent for {symbol}: {len(warnings)} warnings")
            
        except Exception as e:
            logger.error(f"Early Warning send failed: {e}")

    async def send_message_raw(self, text: str):
        """Send raw text to Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        try:
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: requests.post(self.telegram_url, data=payload))
        except Exception as e:
            logger.error(f"Telegram Error: {e}")
    
    def _log_rejected_signal(self, signal: dict, quality_score: int, reason: str):
        """Log rejected signals to JSON file for dashboard review"""
        try:
            rejected_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": signal['symbol'],
                "side": signal['side'],
                "quality_score": quality_score,
                "reason": reason,
                "pattern": signal.get('pattern', 'None'),
                "confidence": signal.get('confidence', 0)
            }
            
            # Append to log file
            if os.path.exists(self.rejected_log_path):
                with open(self.rejected_log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(rejected_data)
            
            # Keep only last 50 rejected signals
            if len(logs) > 50:
                logs = logs[-50:]
            
            with open(self.rejected_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"📝 Rejected signal logged: {signal['symbol']} (Score: {quality_score})")
        except Exception as e:
            logger.error(f"Error logging rejected signal: {e}")
    
    # ============================================
    # TELEGRAM COMMAND HANDLER (Phase 31)
    # ============================================
    
    async def check_telegram_commands(self, money_flow_analyzer=None):
        """
        Check for incoming Telegram commands.
        Supported commands:
        - inout / /inout : Trigger Money Flow report
        - status / /status : System status
        """
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        try:
            # Initialize last_update_id if not exists
            if not hasattr(self, 'last_update_id'):
                self.last_update_id = 0
            
            # Get updates from Telegram
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 1,  # Short timeout for non-blocking
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
                        if chat_id != str(self.telegram_chat_id):
                            continue
                        
                        # Handle commands
                        if text in ['inout', '/inout', 'moneyflow', '/moneyflow']:
                            logger.info(f"📩 Telegram command received: {text}")
                            await self._handle_inout_command(money_flow_analyzer)
                        
                        elif text in ['status', '/status', 'durum', '/durum']:
                            logger.info(f"📩 Telegram command received: {text}")
                            await self._handle_status_command()
                        
                        elif text in ['help', '/help', 'yardim', '/yardim']:
                            await self._handle_help_command()
                            
        except requests.exceptions.Timeout:
            pass  # Normal, non-blocking check
        except Exception as e:
            logger.debug(f"Telegram command check error: {e}")
    
    async def _handle_inout_command(self, money_flow_analyzer):
        """Handle inout command - send Money Flow report"""
        if money_flow_analyzer is None:
            await self.send_message_raw("⚠️ Money Flow Analyzer not available")
            return
        
        try:
            await self.send_message_raw("⏳ Nakit akışı hesaplanıyor...")
            
            # Get fresh money flow data
            money_flow_data = await money_flow_analyzer.get_market_money_flow()
            await self.send_money_flow_report(money_flow_data)
            
        except Exception as e:
            await self.send_message_raw(f"❌ Hata: {e}")
            logger.error(f"Money flow command error: {e}")
    
    async def _handle_status_command(self):
        """Handle status command"""
        from datetime import datetime
        
        msg = "📊 **DEMIR AI DURUM**\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"⏰ Zaman: {datetime.now().strftime('%H:%M:%S')}\n"
        msg += f"🟢 Sistem: AKTIF\n"
        msg += f"📈 Aktif Pozisyon: {len(self.active_positions)}\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"_/inout - Para akışı raporu_\n"
        msg += f"_/help - Komut listesi_"
        
        await self.send_message_raw(msg)
    
    async def _handle_help_command(self):
        """Handle help command"""
        msg = "📋 **KOMUT LİSTESİ**\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"📊 `/inout` - Nakit akışı raporu\n"
        msg += f"📈 `/status` - Sistem durumu\n"
        msg += f"❓ `/help` - Bu yardım mesajı\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"_Mikabot-tarzı para akışı analizi_"
        
        await self.send_message_raw(msg)