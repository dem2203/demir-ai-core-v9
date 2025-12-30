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
        """Tüm komutları listele"""
        return """🤖 *DEMIR AI - KOMUTLAR*
━━━━━━━━━━━━━━━━━━━━━━━━

📋 *GENEL:*
  /info → Bu mesajı göster
  /durum → Bot durumu ve uptime
  
📊 *ANALİZ:*
  /piyasa → BTC/ETH anlık durum
  /son → Son 5 sinyal özeti
  
📈 *İSTATİSTİK:*
  /istatistik → Win rate, sinyal sayısı
  /risk → Açık pozisyonlar

━━━━━━━━━━━━━━━━━━━━━━━━
💡 _Premium sinyaller 15 dakikada bir otomatik gönderilir._
"""
    
    # ==========================================================================
    # /durum - Bot durumu
    # ==========================================================================
    async def cmd_durum(self) -> str:
        """Bot durumunu göster"""
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
        
        return f"""🤖 *DEMIR AI - DURUM*
━━━━━━━━━━━━━━━━━━━━━━━━

🟢 *Bot Durumu:* AKTİF 
⏱️ *Uptime:* {uptime.days} gün {hours} saat {minutes} dk

📡 *Bağlantılar:*
  Binance API: {api_status}
  Telegram: ✅
  Claude AI: ✅

📊 *Bugün:*
  Sinyal: {self._stats['total_signals']}
  LONG: {self._stats['long_signals']}
  SHORT: {self._stats['short_signals']}

🕐 *Son Sinyal:* {last_signal_time}

━━━━━━━━━━━━━━━━━━━━━━━━
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
        """İstatistikleri göster"""
        try:
            # Paper trading manager'dan istatistik al
            from src.brain.paper_trading_manager import get_paper_trading_manager
            ptm = get_paper_trading_manager()
            
            stats = ptm.get_stats()
            total = stats.get('total_trades', 0)
            wins = stats.get('wins', 0)
            losses = stats.get('losses', 0)
            win_rate = (wins / total * 100) if total > 0 else 0
            total_pnl = stats.get('total_pnl', 0)
            
            return f"""📈 *İSTATİSTİKLER*
━━━━━━━━━━━━━━━━━━━━━━━━

📊 *GENEL:*
  Toplam Trade: {total}
  Kazanan: {wins} ✅
  Kaybeden: {losses} ❌
  Win Rate: %{win_rate:.1f}

💰 *PERFORMANS:*
  Toplam PnL: {total_pnl:+.2f}%
  Bugün: {stats.get('today_pnl', 0):+.2f}%
  
📋 *SİNYAL DAĞILIMI:*
  LONG: {self._stats['long_signals']}
  SHORT: {self._stats['short_signals']}
  BEKLE: {self._stats['bekle_signals']}

━━━━━━━━━━━━━━━━━━━━━━━━
"""
        except Exception as e:
            logger.error(f"İstatistik error: {e}")
            return f"""📈 *İSTATİSTİKLER*
━━━━━━━━━━━━━━━━━━━━━━━━

📊 Sinyal Sayısı: {self._stats['total_signals']}
  LONG: {self._stats['long_signals']}
  SHORT: {self._stats['short_signals']}
  BEKLE: {self._stats['bekle_signals']}

━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    # ==========================================================================
    # /son - Son 5 sinyal
    # ==========================================================================
    async def cmd_son(self) -> str:
        """Son 5 sinyali göster"""
        if not self._signal_history:
            return "📭 Henüz sinyal geçmişi yok"
        
        msg = """📋 *SON 5 SİNYAL*
━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        recent = self._signal_history[-5:][::-1]  # Son 5, ters sıra
        
        for i, sig in enumerate(recent, 1):
            emoji = "🟢" if sig['direction'] == 'LONG' else "🔴" if sig['direction'] == 'SHORT' else "⏸️"
            msg += f"""
{i}. {emoji} *{sig['symbol']}* → {sig['direction']}
   💰 Entry: ${sig['entry']:,.2f}
   🎯 Conf: %{sig['confidence']}
   ⏰ {sig['timestamp']}
"""
        
        msg += "\n━━━━━━━━━━━━━━━━━━━━━━━━"
        return msg
    
    # ==========================================================================
    # /risk - Açık pozisyonlar
    # ==========================================================================
    async def cmd_risk(self) -> str:
        """Açık pozisyonları ve risk durumunu göster"""
        try:
            from src.brain.paper_trading_manager import get_paper_trading_manager
            ptm = get_paper_trading_manager()
            
            open_trades = ptm.get_open_trades()
            
            if not open_trades:
                return """⚠️ *RİSK DURUMU*
━━━━━━━━━━━━━━━━━━━━━━━━

📭 Açık pozisyon yok

━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            msg = """⚠️ *RİSK DURUMU*
━━━━━━━━━━━━━━━━━━━━━━━━

📊 *AÇIK POZİSYONLAR:*
"""
            
            total_risk = 0
            for trade in open_trades:
                emoji = "🟢" if trade['direction'] == 'LONG' else "🔴"
                pnl = trade.get('unrealized_pnl', 0)
                pnl_emoji = "📈" if pnl > 0 else "📉"
                
                msg += f"""
{emoji} *{trade['symbol']}* {trade['direction']}
  Entry: ${trade['entry']:,.2f}
  TP: ${trade['tp1']:,.2f}
  SL: ${trade['sl']:,.2f}
  {pnl_emoji} PnL: {pnl:+.2f}%
"""
                total_risk += abs(trade.get('risk_pct', 0))
            
            msg += f"""
━━━━━━━━━━━━━━━━━━━━━━━━
📊 Toplam Risk: %{total_risk:.1f}
"""
            return msg
            
        except Exception as e:
            logger.error(f"Risk error: {e}")
            return "❌ Risk bilgisi alınamadı"
    
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
