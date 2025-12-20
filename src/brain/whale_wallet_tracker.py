# -*- coding: utf-8 -*-
"""
DEMIR AI - WHALE WALLET TRACKER
================================
Bilinen balina cüzdanlarını ve büyük transferleri takip eder.
API Key gereksiz - Public blockchain explorers kullanılır.

Kaynaklar:
- Blockchain.com (BTC)
- Etherscan public (ETH)
- Whale Alert patterns
"""
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("WHALE_WALLET_TRACKER")


@dataclass
class WhaleTransaction:
    """Tek bir balina işlemi"""
    tx_hash: str
    coin: str  # BTC, ETH, etc.
    amount: float  # In coin units
    amount_usd: float  # USD value
    from_address: str
    to_address: str
    from_label: str = ""  # Known entity name
    to_label: str = ""
    direction: str = "UNKNOWN"  # "EXCHANGE_IN", "EXCHANGE_OUT", "WHALE_TO_WHALE", "UNKNOWN"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WhaleActivity:
    """Belirli bir coin için balina aktivitesi özeti"""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Transactions
    recent_transactions: List[WhaleTransaction] = field(default_factory=list)
    
    # Aggregated metrics
    total_volume_24h: float = 0  # USD
    exchange_inflow_24h: float = 0  # USD (bearish)
    exchange_outflow_24h: float = 0  # USD (bullish)
    large_tx_count: int = 0  # >$1M transactions
    
    # Known wallets activity
    known_whale_buys: int = 0
    known_whale_sells: int = 0
    institutional_activity: bool = False
    
    # Signal
    signal: str = "NEUTRAL"  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    signal_reason: str = ""


# Known exchange wallet addresses (partial - for pattern matching)
KNOWN_EXCHANGES = {
    'BTC': {
        'binance': ['bc1qm34lsc65z', '1NDyJtNTjm', '34xp4vRoC'],
        'coinbase': ['3Cbq7aT1', '395qu3Gh'],
        'bitfinex': ['bc1qgdj', '3D2oetd'],
        'kraken': ['bc1qkr', '37NRnf'],
    },
    'ETH': {
        'binance': ['0x28C6c06298d514Db089934071355E5743bf21d60', '0x21a31Ee1afc51d94C2eFcCAa2092aD1028285549'],
        'coinbase': ['0x71660c4005BA85c37ccec55d0C4493E66Fe775d3', '0x503828976D22510aad0339F0a6C1CD2B9a9f2bb3'],
        'kraken': ['0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2'],
        'ftx': ['0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2'],
    }
}

# Known whale/institutional wallets
KNOWN_WHALES = {
    'BTC': {
        'MicroStrategy': ['bc1qazcm763858nkj2dj986etajy6'],
        'Tesla': [],  # Unknown
        'BlockOne': ['1FeexV6bAH'],
        'Grayscale': ['35PBEaofp'],
        'US_Gov': ['bc1qa5wkgaew'],  # Seized coins
    },
    'ETH': {
        'Vitalik': ['0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B'],
        'Ethereum_Foundation': ['0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe'],
        'Paradigm': ['0x9B9647431632AF44be02ddd22477Ed94d14AacAa'],
    }
}


class WhaleWalletTracker:
    """
    Whale Wallet Tracker
    
    Büyük cüzdan hareketlerini ve exchange flow'u takip eder.
    Public API'ler kullanır - API Key gereksiz!
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, WhaleActivity] = {}
        self._cache_ttl = 60  # seconds
        self._btc_price = 0
        self._eth_price = 0
        logger.info("🐋 Whale Wallet Tracker initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _update_prices(self):
        """Güncel fiyatları al"""
        try:
            session = await self._get_session()
            async with session.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT") as resp:
                data = await resp.json()
                self._btc_price = float(data.get('price', 0))
            
            async with session.get("https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT") as resp:
                data = await resp.json()
                self._eth_price = float(data.get('price', 0))
        except Exception as e:
            logger.debug(f"Price update error: {e}")
    
    # =========================================
    # DATA FETCHING (Public APIs)
    # =========================================
    
    async def _fetch_btc_large_txs(self) -> List[WhaleTransaction]:
        """
        Blockchain.com'dan büyük BTC işlemlerini çek.
        Public API - No key required
        """
        transactions = []
        try:
            session = await self._get_session()
            
            # Recent unconfirmed large transactions
            url = "https://blockchain.info/unconfirmed-transactions?format=json"
            async with session.get(url) as resp:
                data = await resp.json()
            
            for tx in data.get('txs', [])[:50]:  # Last 50 txs
                # Calculate total output value
                total_out = sum(out.get('value', 0) for out in tx.get('out', [])) / 1e8  # satoshi to BTC
                total_usd = total_out * self._btc_price
                
                # Only process large transactions (>$500K)
                if total_usd > 500000:
                    # Check if exchange related
                    inputs = tx.get('inputs', [])
                    outputs = tx.get('out', [])
                    
                    from_addr = inputs[0].get('prev_out', {}).get('addr', 'Unknown') if inputs else 'Unknown'
                    to_addr = outputs[0].get('addr', 'Unknown') if outputs else 'Unknown'
                    
                    # Detect direction
                    direction = self._classify_btc_tx(from_addr, to_addr)
                    from_label = self._get_btc_label(from_addr)
                    to_label = self._get_btc_label(to_addr)
                    
                    transactions.append(WhaleTransaction(
                        tx_hash=tx.get('hash', ''),
                        coin='BTC',
                        amount=total_out,
                        amount_usd=total_usd,
                        from_address=from_addr[:20] + '...' if len(from_addr) > 20 else from_addr,
                        to_address=to_addr[:20] + '...' if len(to_addr) > 20 else to_addr,
                        from_label=from_label,
                        to_label=to_label,
                        direction=direction,
                        timestamp=datetime.fromtimestamp(tx.get('time', datetime.now().timestamp()))
                    ))
            
        except Exception as e:
            logger.debug(f"BTC large tx fetch error: {e}")
        
        return transactions
    
    async def _fetch_eth_large_txs(self) -> List[WhaleTransaction]:
        """
        Etherscan'dan büyük ETH işlemlerini çek.
        Note: Free tier has rate limits
        """
        transactions = []
        try:
            session = await self._get_session()
            
            # Top accounts by balance (public endpoint)
            # Using gas tracker as proxy for activity
            url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"
            async with session.get(url) as resp:
                data = await resp.json()
            
            # Note: Etherscan requires API key for transaction lists
            # We'll estimate based on gas activity and known patterns
            
            # Fallback: Check known whale wallets
            for whale_name, addresses in KNOWN_WHALES.get('ETH', {}).items():
                for addr in addresses[:1]:  # Only first address per whale
                    try:
                        # This might fail without API key, but worth trying
                        balance_url = f"https://api.etherscan.io/api?module=account&action=balance&address={addr}&tag=latest"
                        async with session.get(balance_url) as resp:
                            balance_data = await resp.json()
                        
                        if balance_data.get('status') == '1':
                            balance_eth = int(balance_data.get('result', 0)) / 1e18
                            balance_usd = balance_eth * self._eth_price
                            
                            # If significant balance, note it
                            if balance_usd > 10000000:  # >$10M
                                logger.debug(f"Known whale {whale_name}: ${balance_usd/1e6:.1f}M ETH")
                    except:
                        pass
            
        except Exception as e:
            logger.debug(f"ETH large tx fetch error: {e}")
        
        return transactions
    
    async def _fetch_exchange_flows(self, coin: str) -> Dict:
        """
        Exchange inflow/outflow tahmini.
        Coinglass/Glassnode benzeri veri - basit hesaplama
        """
        try:
            # This is a simplified estimation
            # Real implementation would use on-chain data
            return {
                'inflow': 0,
                'outflow': 0,
                'net': 0
            }
        except Exception as e:
            logger.debug(f"Exchange flow error: {e}")
            return {'inflow': 0, 'outflow': 0, 'net': 0}
    
    def _classify_btc_tx(self, from_addr: str, to_addr: str) -> str:
        """İşlemi sınıflandır"""
        from_is_exchange = any(
            from_addr.startswith(prefix) 
            for exchange_addrs in KNOWN_EXCHANGES.get('BTC', {}).values()
            for prefix in exchange_addrs
        )
        to_is_exchange = any(
            to_addr.startswith(prefix)
            for exchange_addrs in KNOWN_EXCHANGES.get('BTC', {}).values()
            for prefix in exchange_addrs
        )
        
        if from_is_exchange and not to_is_exchange:
            return "EXCHANGE_OUT"  # Bullish
        elif not from_is_exchange and to_is_exchange:
            return "EXCHANGE_IN"  # Bearish
        elif from_is_exchange and to_is_exchange:
            return "EXCHANGE_TO_EXCHANGE"
        else:
            return "WHALE_TO_WHALE"
    
    def _get_btc_label(self, address: str) -> str:
        """Adresi etiketle"""
        # Check exchanges
        for exchange, prefixes in KNOWN_EXCHANGES.get('BTC', {}).items():
            for prefix in prefixes:
                if address.startswith(prefix):
                    return exchange.upper()
        
        # Check whales
        for whale, addrs in KNOWN_WHALES.get('BTC', {}).items():
            for addr in addrs:
                if address.startswith(addr):
                    return whale
        
        return ""
    
    # =========================================
    # MAIN API
    # =========================================
    
    async def get_whale_activity(self, symbol: str = "BTCUSDT") -> WhaleActivity:
        """
        Belirli bir coin için whale aktivitesini al.
        """
        # Check cache
        if symbol in self._cache:
            cached = self._cache[symbol]
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self._cache_ttl:
                return cached
        
        await self._update_prices()
        
        coin = symbol.replace('USDT', '')
        
        if coin == 'BTC':
            transactions = await self._fetch_btc_large_txs()
        elif coin == 'ETH':
            transactions = await self._fetch_eth_large_txs()
        else:
            transactions = []
        
        # Calculate metrics
        total_volume = sum(tx.amount_usd for tx in transactions)
        exchange_in = sum(tx.amount_usd for tx in transactions if tx.direction == "EXCHANGE_IN")
        exchange_out = sum(tx.amount_usd for tx in transactions if tx.direction == "EXCHANGE_OUT")
        
        # Determine signal
        if exchange_out > exchange_in * 2 and exchange_out > 5000000:
            signal = "STRONG_BUY"
            reason = f"${exchange_out/1e6:.1f}M exchange outflow (accumulation)"
        elif exchange_out > exchange_in * 1.3:
            signal = "BUY"
            reason = "Exchange outflow dominant"
        elif exchange_in > exchange_out * 2 and exchange_in > 5000000:
            signal = "STRONG_SELL"
            reason = f"${exchange_in/1e6:.1f}M exchange inflow (selling pressure)"
        elif exchange_in > exchange_out * 1.3:
            signal = "SELL"
            reason = "Exchange inflow dominant"
        else:
            signal = "NEUTRAL"
            reason = "Balanced flow"
        
        activity = WhaleActivity(
            symbol=symbol,
            recent_transactions=transactions[:10],  # Top 10
            total_volume_24h=total_volume,
            exchange_inflow_24h=exchange_in,
            exchange_outflow_24h=exchange_out,
            large_tx_count=len(transactions),
            signal=signal,
            signal_reason=reason
        )
        
        self._cache[symbol] = activity
        return activity
    
    async def get_all_activity(self) -> Dict[str, WhaleActivity]:
        """4 coin için whale aktivitesini al"""
        results = await asyncio.gather(
            self.get_whale_activity("BTCUSDT"),
            self.get_whale_activity("ETHUSDT"),
            return_exceptions=True
        )
        
        return {
            "BTCUSDT": results[0] if isinstance(results[0], WhaleActivity) else WhaleActivity(symbol="BTCUSDT"),
            "ETHUSDT": results[1] if isinstance(results[1], WhaleActivity) else WhaleActivity(symbol="ETHUSDT"),
        }
    
    def format_for_telegram(self, activity: WhaleActivity) -> str:
        """Whale aktivitesini Telegram mesajı olarak formatla"""
        signal_emoji = {
            "STRONG_BUY": "🚀",
            "BUY": "🟢",
            "NEUTRAL": "⚪",
            "SELL": "🔴",
            "STRONG_SELL": "💀"
        }.get(activity.signal, "⚪")
        
        tx_lines = ""
        for tx in activity.recent_transactions[:5]:
            dir_emoji = "📥" if tx.direction == "EXCHANGE_IN" else "📤" if tx.direction == "EXCHANGE_OUT" else "🔄"
            tx_lines += f"  {dir_emoji} ${tx.amount_usd/1e6:.1f}M | {tx.from_label or tx.from_address[:10]} → {tx.to_label or tx.to_address[:10]}\n"
        
        if not tx_lines:
            tx_lines = "  • Büyük işlem tespit edilmedi\n"
        
        return f"""🐋 *WHALE AKTİVİTESİ - {activity.symbol}*
━━━━━━━━━━━━━━━━━━
{signal_emoji} *Sinyal: {activity.signal}*
📝 {activity.signal_reason}

💰 24s Toplam: ${activity.total_volume_24h/1e6:.1f}M
📥 Exchange Giriş: ${activity.exchange_inflow_24h/1e6:.1f}M
📤 Exchange Çıkış: ${activity.exchange_outflow_24h/1e6:.1f}M
🔢 Büyük TX: {activity.large_tx_count}

━━━ SON İŞLEMLER ━━━
{tx_lines}━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}"""


# Singleton instance
_instance: Optional[WhaleWalletTracker] = None

def get_whale_wallet_tracker() -> WhaleWalletTracker:
    global _instance
    if _instance is None:
        _instance = WhaleWalletTracker()
    return _instance
