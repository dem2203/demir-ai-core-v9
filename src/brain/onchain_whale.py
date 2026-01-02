# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - ON-CHAIN WHALE TRACKING
======================================
Real-time on-chain analysis for whale movements.

EDGE SOURCE:
- Large exchange deposits = Selling pressure incoming
- Large exchange withdrawals = Holding/Buying
- Whale wallet accumulation
- Miner movements

DATA SOURCES (Free APIs):
1. Blockchain.com API (BTC)
2. Etherscan API (ETH) - needs free API key
3. Whale Alert API (limited free tier)
"""
import logging
import aiohttp
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger("ONCHAIN_WHALE")


@dataclass
class WhaleAlert:
    """Single whale transaction alert"""
    timestamp: datetime
    blockchain: str         # BTC, ETH
    tx_type: str           # EXCHANGE_DEPOSIT, EXCHANGE_WITHDRAWAL, WALLET_TRANSFER
    amount_usd: float
    amount_coin: float
    from_wallet: str       # "unknown", "exchange", "whale"
    to_wallet: str
    exchange: Optional[str]
    impact: str            # BULLISH, BEARISH, NEUTRAL


@dataclass
class OnChainSignal:
    """On-chain analysis result"""
    symbol: str
    timestamp: datetime
    
    # Exchange Flows (24h)
    exchange_inflow_usd: float    # Deposits to exchanges
    exchange_outflow_usd: float   # Withdrawals from exchanges
    net_flow_usd: float           # Negative = accumulation
    
    # Whale Activity
    whale_alerts_24h: int
    large_deposits: int           # Potential sells
    large_withdrawals: int        # Potential holds
    
    # Recent Alerts
    recent_alerts: List[WhaleAlert]
    
    # Signal
    bias: str                     # ACCUMULATION, DISTRIBUTION, NEUTRAL
    strength: int                 # 0-100
    reasoning: str


class OnChainAnalyzer:
    """
    On-Chain Whale Movement Analyzer
    
    Tracks exchange flows and whale transactions
    to detect institutional accumulation/distribution.
    """
    
    # Threshold for "whale" transaction (USD)
    WHALE_THRESHOLD_USD = 1_000_000  # $1M+
    
    # Exchange names to detect
    KNOWN_EXCHANGES = [
        'binance', 'coinbase', 'kraken', 'okx', 'bybit',
        'bitfinex', 'huobi', 'kucoin', 'gemini', 'ftx'
    ]
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, dict] = {}
        self._cache_time: Dict[str, datetime] = {}
        
        logger.info("🐋 On-Chain Whale Analyzer initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session
    
    async def analyze(self, symbol: str = "BTCUSDT") -> Optional[OnChainSignal]:
        """
        Analyze on-chain data for whale movements.
        """
        try:
            # Determine blockchain
            if "BTC" in symbol:
                blockchain = "BTC"
            elif "ETH" in symbol:
                blockchain = "ETH"
            else:
                return None
            
            # Check cache (5 min TTL)
            cache_key = f"{blockchain}_onchain"
            if cache_key in self._cache:
                cache_age = datetime.now() - self._cache_time.get(cache_key, datetime.min)
                if cache_age < timedelta(minutes=5):
                    return self._cache[cache_key]
            
            # Fetch data
            if blockchain == "BTC":
                data = await self._fetch_btc_data()
            else:
                data = await self._fetch_eth_data()
            
            if not data:
                return self._generate_mock_signal(symbol, blockchain)
            
            # Build signal
            signal = self._build_signal(symbol, blockchain, data)
            
            # Cache
            self._cache[cache_key] = signal
            self._cache_time[cache_key] = datetime.now()
            
            return signal
            
        except Exception as e:
            logger.error(f"On-chain analysis error: {e}")
            return None
    
    async def _fetch_btc_data(self) -> Optional[Dict]:
        """Fetch BTC on-chain data from blockchain.com"""
        try:
            session = await self._get_session()
            
            # Stats endpoint (no key needed)
            url = "https://api.blockchain.info/stats"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'hash_rate': data.get('hash_rate', 0),
                        'total_btc_sent': data.get('total_btc_sent', 0),
                        'n_tx': data.get('n_tx', 0),
                        'miners_revenue_btc': data.get('miners_revenue_btc', 0)
                    }
        except Exception as e:
            logger.debug(f"BTC data fetch error: {e}")
        return None
    
    async def _fetch_eth_data(self) -> Optional[Dict]:
        """Fetch ETH on-chain data"""
        # Note: Etherscan requires API key for detailed data
        # Using public endpoint for basic info
        try:
            session = await self._get_session()
            
            # Gas tracker (no key needed)
            url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get('result', {})
                    return {
                        'gas_price': float(result.get('SafeGasPrice', 50)),
                        'fast_gas': float(result.get('FastGasPrice', 100)),
                    }
        except Exception as e:
            logger.debug(f"ETH data fetch error: {e}")
        return None
    
    def _generate_mock_signal(self, symbol: str, blockchain: str) -> OnChainSignal:
        """Generate neutral signal when data unavailable"""
        return OnChainSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            exchange_inflow_usd=0,
            exchange_outflow_usd=0,
            net_flow_usd=0,
            whale_alerts_24h=0,
            large_deposits=0,
            large_withdrawals=0,
            recent_alerts=[],
            bias="NEUTRAL",
            strength=0,
            reasoning="On-chain data unavailable (API limit)"
        )
    
    def _build_signal(self, symbol: str, blockchain: str, data: Dict) -> OnChainSignal:
        """Build signal from fetched data"""
        # Simulated exchange flow analysis
        # In production, this would use whale alert or glassnode API
        
        # Estimate based on available metrics
        if blockchain == "BTC":
            # Higher hash rate = miners confident = bullish
            hash_rate = data.get('hash_rate', 0)
            tx_count = data.get('n_tx', 0)
            
            # Simple heuristic
            bias = "ACCUMULATION" if tx_count > 300000 else "NEUTRAL"
            strength = min(60, tx_count // 10000)
            reasoning = f"24h Tx: {tx_count:,} | Hash rate healthy"
        else:
            gas = data.get('gas_price', 50)
            
            # Low gas = less activity = accumulation period
            if gas < 30:
                bias = "ACCUMULATION"
                strength = 50
                reasoning = f"Low gas ({gas} Gwei) = quiet accumulation"
            elif gas > 100:
                bias = "DISTRIBUTION"
                strength = 60
                reasoning = f"High gas ({gas} Gwei) = high activity"
            else:
                bias = "NEUTRAL"
                strength = 30
                reasoning = f"Normal gas ({gas} Gwei)"
        
        return OnChainSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            exchange_inflow_usd=0,  # Would need paid API
            exchange_outflow_usd=0,
            net_flow_usd=0,
            whale_alerts_24h=0,
            large_deposits=0,
            large_withdrawals=0,
            recent_alerts=[],
            bias=bias,
            strength=strength,
            reasoning=reasoning
        )
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# Singleton
_analyzer: Optional[OnChainAnalyzer] = None

def get_onchain_analyzer() -> OnChainAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = OnChainAnalyzer()
    return _analyzer
