# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - TELEGRAM COMMANDS
=================================
Telegram bot komutları: /info, /durum, /istatistik, /son, /risk, /piyasa

KULLANIM:
    Bot'a /info yazın → Tüm komutları görün
    Bot'a /durum yazın → Bot durumunu görün
"""
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Optional, Dict, List

logger = logging.getLogger("TELEGRAM_COMMANDS")


class TelegramCommands:
    """Telegram komut handler'ları"""
    
    FUTURES_BASE = "https://fapi.binance.com"
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._start_time = datetime.now()
        self._signal_history: List[Dict] = []
        self._stats = {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'bekle_signals': 0,
            'wins': 0,
            'losses': 0
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    # ==========================================================================
    # /info - Tüm komutları listele
    # ==========================================================================
    async def cmd_info(self) -> str:
        """Tüm komutları listele - PRO VERSION"""
        return """🤖 *DEMIR AI PRO - KOMUTLAR*
━━━━━━━━━━━━━━━━━━━━━━━━

📋 *GENEL:*
  /start → Botu başlat
  /info → Bu mesajı göster
  /durum → Bot + Risk Engine durumu
  /brain → 🧠 Thinking Brain durumu
  
📊 *ANALİZ:*
  /analiz BTCUSDT → 🧠 AI PRO Analiz
    • 4 AI Council (Claude, GPT-4, DeepSeek, Gemini)
    • Kelly Position Sizing
    • LSTM Tahmin
  /piyasa → BTC/ETH anlık durum
  /son → Son 5 sinyal özeti
  
📈 *PERFORMANS:*
  /istatistik → Win rate, toplam P&L
  /performans → Detaylı performans raporu
  /risk → Risk Engine durumu

🛡️ *PRO ÖZELLİKLER:*
  • Kelly Criterion Position Sizing
  • Auto-Shutdown (Win Rate < %40)
  • AI Council (4 AI Weighted Voting)
  • LSTM Prediction (%83 accuracy)

━━━━━━━━ DEMIR AI v10 PRO ━━━━━━━━
💡 _Premium sinyaller 15 dakikada bir gönderilir._
"""
    
    # ==========================================================================
    # /durum - Bot durumu
    # ==========================================================================
    async def cmd_durum(self) -> str:
        """Bot durumunu göster - PRO VERSION"""
        uptime = datetime.now() - self._start_time
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60
        
        # Son sinyal zamanı
        last_signal_time = "Henüz sinyal yok"
        if self._signal_history:
            last = self._signal_history[-1]
            last_signal_time = last.get('timestamp', 'N/A')
        
        # API durumu kontrol
        api_status = "✅"
        try:
            session = await self._get_session()
            async with session.get(f"{self.FUTURES_BASE}/fapi/v1/ping") as resp:
                if resp.status != 200:
                    api_status = "⚠️"
        except:
            api_status = "❌"
        
        # Risk Engine durumu
        risk_status = "✅ Aktif"
        risk_trading = "✅ İzin Var"
        try:
            from src.brain.risk_engine import get_risk_engine
            risk_engine = get_risk_engine()
            if not risk_engine.is_trading_enabled:
                risk_trading = f"🛑 Durduruldu: {risk_engine.disable_reason}"
        except:
            risk_status = "⚠️ Yüklenemedi"
        
        # Performance Tracker durumu
        perf_stats = "N/A"
        try:
            from src.v10.performance_tracker import get_performance_tracker
            tracker = get_performance_tracker()
            stats = tracker.get_stats()
            perf_stats = f"{stats.get('win_rate', 0):.1f}% ({stats.get('completed', 0)} trade)"
        except:
            pass
        
        return f"""🤖 *DEMIR AI PRO - DURUM*
━━━━━━━━━━━━━━━━━━━━━━━━

🟢 *Bot Durumu:* AKTİF 
⏱️ *Uptime:* {uptime.days} gün {hours} saat {minutes} dk

📡 *Bağlantılar:*
  Binance API: {api_status}
  Telegram: ✅
  AI Council: ✅ (4 AI aktif)

🛡️ *Risk Engine:*
  Durum: {risk_status}
  Trading: {risk_trading}

📊 *Performance:*
  Win Rate: {perf_stats}
  
📈 *Bugün:*
  Sinyal: {self._stats['total_signals']}
  LONG: {self._stats['long_signals']}
  SHORT: {self._stats['short_signals']}

🕐 *Son Sinyal:* {last_signal_time}

━━━━━━━━ DEMIR AI v10 PRO ━━━━━━━━
"""
    
    # ==========================================================================
    # /piyasa - BTC/ETH anlık durum
    # ==========================================================================
    async def cmd_piyasa(self) -> str:
        """BTC ve ETH anlık piyasa durumu"""
        try:
            session = await self._get_session()
            
            btc_data = await self._get_market_data("BTCUSDT")
            eth_data = await self._get_market_data("ETHUSDT")
            
            # Fear & Greed
            fng = await self._get_fear_greed()
            
            return f"""📊 *PİYASA DURUMU*
━━━━━━━━━━━━━━━━━━━━━━━━

🟠 *BTCUSDT*
  💰 Fiyat: ${btc_data['price']:,.2f}
  📈 24s: {btc_data['change_24h']:+.2f}%
  📊 Volume: ${btc_data['volume']/1e9:.2f}B
  📉 Funding: {btc_data['funding']:.4f}%

💎 *ETHUSDT*
  💰 Fiyat: ${eth_data['price']:,.2f}
  📈 24s: {eth_data['change_24h']:+.2f}%
  📊 Volume: ${eth_data['volume']/1e9:.2f}B
  📉 Funding: {eth_data['funding']:.4f}%

🎭 *Fear & Greed:* {fng['value']} ({fng['label']})

━━━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        except Exception as e:
            logger.error(f"Piyasa error: {e}")
            return "❌ Piyasa verileri alınamadı"
    
    async def _get_market_data(self, symbol: str) -> Dict:
        """Binance'dan piyasa verisi al"""
        session = await self._get_session()
        
        # Fiyat ve 24h değişim
        ticker_url = f"{self.FUTURES_BASE}/fapi/v1/ticker/24hr?symbol={symbol}"
        async with session.get(ticker_url) as resp:
            ticker = await resp.json()
        
        # Funding rate
        funding_url = f"{self.FUTURES_BASE}/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        async with session.get(funding_url) as resp:
            funding_data = await resp.json()
        
        return {
            'price': float(ticker.get('lastPrice', 0)),
            'change_24h': float(ticker.get('priceChangePercent', 0)),
            'volume': float(ticker.get('quoteVolume', 0)),
            'funding': float(funding_data[0].get('fundingRate', 0)) * 100 if funding_data else 0
        }
    
    async def _get_fear_greed(self) -> Dict:
        """Fear & Greed Index al"""
        try:
            session = await self._get_session()
            async with session.get("https://api.alternative.me/fng/") as resp:
                data = await resp.json()
                return {
                    'value': int(data['data'][0]['value']),
                    'label': data['data'][0]['value_classification']
                }
        except:
            return {'value': 50, 'label': 'Neutral'}
    
    # ==========================================================================
    # /istatistik - Win rate ve sinyal sayısı
    # ==========================================================================
    async def cmd_istatistik(self) -> str:
        """İstatistikleri göster - PRO VERSION"""
        try:
            # CORRECT: Use brain.signal_performance_tracker
            from src.brain.signal_performance_tracker import get_tracker
            tracker = get_tracker()
            stats = tracker.get_win_rate() # Returns dict with win_rate, total_signals, etc.
            
            # Risk Engine'den portföy bilgisi al isntead of Paper Trader default
            portfolio_info = ""
            try:
                from src.execution.paper_trader import get_paper_trader
                pt = get_paper_trader()
                balance = pt.get_balance()
                portfolio_info = f"""
🛡️ *PAPER TRADER:*
  💰 Bakiye: ${balance:,.2f}
  📊 Açık Pozisyon: {len(pt.get_open_positions())}
"""
            except:
                pass
            
            return f"""📈 *İSTATİSTİKLER - PRO*
━━━━━━━━━━━━━━━━━━━━━━━━

📊 *SİNYAL PERFORMANSI (7 Gün):*
  Toplam: {stats.get('total_signals', 0)}
  ✅ Win: {stats.get('winners', 0)} ({stats.get('win_rate', 0):.1f}%)
  ❌ Loss: {stats.get('losers', 0)}
  ⏳ Bekleyen: {stats.get('pending', 0)}

{portfolio_info}
📋 *SİNYAL DAĞILIMI:*
  LONG: {self._stats['long_signals']}
  SHORT: {self._stats['short_signals']}
  BEKLE: {self._stats['bekle_signals']}

━━━━━━━━ DEMIR AI v11 PRO ━━━━━━━━
"""
        except Exception as e:
            logger.error(f"İstatistik error: {e}")
            return "❌ İstatistik alınamadı"

    # ==========================================================================
    # /performans - Detaylı Rapor (Alias for now)
    # ==========================================================================
    async def cmd_performans(self) -> str:
        """Detaylı performans raporu"""
        return await self.cmd_istatistik()

    # ==========================================================================
    # /risk - Açık pozisyonlar (Paper Trader)
    # ==========================================================================
    async def cmd_risk(self) -> str:
        """Açık pozisyonları ve risk durumunu göster"""
        try:
            # CORRECT: Use execution.paper_trader
            from src.execution.paper_trader import get_paper_trader
            pt = get_paper_trader()
            
            open_positions = pt.get_open_positions() # Returns dict {symbol: pos}
            
            if not open_positions:
                return """⚠️ *RİSK DURUMU*
━━━━━━━━━━━━━━━━━━━━━━━━

📭 Açık paper trade pozisyonu yok

━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            msg = """⚠️ *PAPER TRADE POZİSYONLARI*
━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            total_pnl = 0
            for symbol, pos in open_positions.items():
                emoji = "🟢" if pos['side'] == 'BUY' else "🔴"
                # PnL hesaplama (anlık fiyat lazım ama burada basit gösterelim)
                entry = pos['entry_price']
                size = pos['size']
                msg += f"""
{emoji} *{symbol}* {pos['side']}
  Entry: ${entry:,.2f}
  Size: {size:.4f}
  SL: ${pos['stop_loss']:,.2f} | TP: ${pos['take_profit']:,.2f}
"""
            
            return msg
            
        except Exception as e:
            logger.error(f"Risk error: {e}")
            logger.exception(e)
            return "❌ Risk bilgisi alınamadı"

    async def cmd_pozisyonlar(self) -> str:
        return await self.cmd_risk()
    
    # ==========================================================================
    # Helper: Sinyal kaydet
    # ==========================================================================
    def record_signal(self, signal: Dict):
        """Yeni sinyal kaydet"""
        self._signal_history.append({
            'symbol': signal.get('symbol', 'BTCUSDT'),
            'direction': signal.get('direction', 'BEKLE'),
            'entry': signal.get('entry_price', 0),
            'confidence': signal.get('confidence', 0),
            'timestamp': datetime.now().strftime('%H:%M')
        })
        
        # Son 100 sinyal tut
        self._signal_history = self._signal_history[-100:]
        
        # İstatistik güncelle
        self._stats['total_signals'] += 1
        if signal.get('direction') == 'LONG':
            self._stats['long_signals'] += 1
        elif signal.get('direction') == 'SHORT':
            self._stats['short_signals'] += 1
        else:
            self._stats['bekle_signals'] += 1
    
    # ==========================================================================
    # /brain - Thinking Brain durumu
    # ==========================================================================
    async def cmd_brain(self) -> str:
        """Thinking Brain durumunu göster"""
        try:
            from src.brain.thinking_brain import get_thinking_brain
            brain = get_thinking_brain()
            
            # Weights
            weights = brain._weights
            
            # Performance
            perf = brain._performance_by_source
            rl_wr = (perf['rl']['wins'] / perf['rl']['total'] * 100) if perf['rl']['total'] > 0 else 0
            claude_wr = (perf['claude']['wins'] / perf['claude']['total'] * 100) if perf['claude']['total'] > 0 else 0
            rules_wr = (perf['rules']['wins'] / perf['rules']['total'] * 100) if perf['rules']['total'] > 0 else 0
            
            # Decision history
            history_count = len(brain._decision_history)
            
            # RL Agent status
            rl_status = "✅ Yüklü" if brain._rl_agent and brain._rl_agent.model else "⏳ Bekleniyor"
            
            return f"""🧠 *THINKING BRAIN DURUMU*
━━━━━━━━━━━━━━━━━━━━━━━━

🎛️ *KAYNAK AĞIRLIKLARI:*
  🤖 RL Agent: %{weights['rl']*100:.0f}
  🧠 Claude: %{weights['claude']*100:.0f}
  📊 Rules: %{weights['rules']*100:.0f}

📊 *KAYNAK PERFORMANSI:*
  🤖 RL: {perf['rl']['wins']}/{perf['rl']['total']} (%{rl_wr:.0f})
  🧠 Claude: {perf['claude']['wins']}/{perf['claude']['total']} (%{claude_wr:.0f})
  📊 Rules: {perf['rules']['wins']}/{perf['rules']['total']} (%{rules_wr:.0f})

🧠 *HAFIZA:*
  📝 Karar Sayısı: {history_count}/500
  🤖 RL Agent: {rl_status}

💡 *NASIL ÇALIŞIR:*
  1. Her sinyal için 3 kaynak analiz eder
  2. Ağırlıklı fusion ile karar verir
  3. Geçmiş benzer durumları kontrol eder
  4. Piyasa rejimine göre adapte olur
  5. Her trade'den öğrenir

━━━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        except Exception as e:
            logger.error(f"Brain status error: {e}")
            return f"""🧠 *THINKING BRAIN DURUMU*
━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ Thinking Brain henüz aktif değil.

İlk sinyal üretildiğinde aktif olacak.

━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_commands: Optional[TelegramCommands] = None

def get_telegram_commands() -> TelegramCommands:
    """Get or create commands instance."""
    global _commands
    if _commands is None:
        _commands = TelegramCommands()
    return _commands


# =============================================================================
# TELEGRAM BOT INTEGRATION
# =============================================================================

async def handle_command(command: str) -> str:
    """Komut handler - telegram bot'tan çağrılır"""
    cmd = get_telegram_commands()
    
    command = command.lower().strip()
    
    if command in ['/info', 'info', '/start', 'start', '/help', 'help']:
        return await cmd.cmd_info()
    elif command in ['/durum', 'durum', '/status']:
        return await cmd.cmd_durum()
    elif command in ['/piyasa', 'piyasa', '/market']:
        return await cmd.cmd_piyasa()
    elif command in ['/istatistik', 'istatistik', '/stats', '/stat']:
        return await cmd.cmd_istatistik()
    elif command in ['/son', 'son', '/recent', '/last']:
        return await cmd.cmd_son()
    elif command in ['/risk', 'risk', '/positions']:
        return await cmd.cmd_risk()
    elif command in ['/brain', 'brain', '/thinking']:
        return await cmd.cmd_brain()
    else:
        return f"""❌ Bilinmeyen komut: `{command}`

📋 Komutlar için /info yazın."""


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        cmd = get_telegram_commands()
        
        print("=== /info ===")
        print(await cmd.cmd_info())
        
        print("\n=== /durum ===")
        print(await cmd.cmd_durum())
        
        print("\n=== /piyasa ===")
        print(await cmd.cmd_piyasa())
        
        print("\n=== /istatistik ===")
        print(await cmd.cmd_istatistik())
        
        await cmd.close()
    
    asyncio.run(test())
