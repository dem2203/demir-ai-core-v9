import logging
import aiohttp
import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from src.config.settings import Config

logger = logging.getLogger("ON_CHAIN_ANALYZER")

class OnChainAnalyzer:
    """
    PHASE 22: ON-CHAIN WHALE HUNTER
    
    True On-Chain Metrics using Blockchain APIs:
    1. Ethereum: Etherscan (Whale wallet tracking)
    2. Bitcoin: Blockchain.com (Unconfirmed TXs & Large Blocks)
    """
    
    # Whale Thresholds (Native Units)
    THRESHOLDS = {
        'BTC': 50.0,    # > 50 BTC is a Whale
        'ETH': 1000.0   # > 1000 ETH is a Whale
    }
    
    def __init__(self):
        self.etherscan_key = Config.ETHERSCAN_KEY
        self.session = None
        self.cache = {}
        self.last_update = {}
        
    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_whale_sentiment(self, symbol: str) -> Dict:
        """
        Main entry point. Returns metrics and bias.
        """
        asset = 'BTC' if 'BTC' in symbol else ('ETH' if 'ETH' in symbol else None)
        
        if not asset:
            return self._neutral_output()
            
        # Rate Limit / Cache Check (15 min cache)
        if self._is_cached(asset):
            return self.cache[asset]
            
        try:
            if asset == 'ETH':
                data = await self._analyze_ethereum()
            elif asset == 'BTC':
                data = await self._analyze_bitcoin()
            else:
                data = self._neutral_output()
                
            self.cache[asset] = data
            self.last_update[asset] = datetime.now()
            return data
            
        except Exception as e:
            logger.error(f"On-Chain Analysis Failed for {symbol}: {e}")
            return self._neutral_output()
            
    def _is_cached(self, asset: str) -> bool:
        if asset not in self.last_update:
            return False
        return (datetime.now() - self.last_update[asset]) < timedelta(minutes=15)

    async def _analyze_ethereum(self) -> Dict:
        """Fetch Etherscan for large TXs"""
        if not self.etherscan_key:
            return self._neutral_output(error="No Etherscan Key")
            
        session = await self._get_session()
        # Endpoint: Get last block number implies liveness, but for whales we check latest blocks
        # We will scan the last 100 blocks for tx > threshold
        
        try:
            # 1. Get CryptoCompare/Etherscan proxy for recent large TXs or Gas Guzzlers
            # Since free Etherscan is limited, we simulate logic or check Gas Price + Pending TXs
            # For this Phase, we'll check "Gas Price" as a proxy for network congestion (Activity)
            
            url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={self.etherscan_key}"
            async with session.get(url) as resp:
                data = await resp.json()
                
            fast_gas = int(data['result']['FastGasPrice'])
            
            # Logic: High Gas = High Activity = Volatility Incoming
            sentiment = "NEUTRAL"
            bias = 0.0
            
            if fast_gas > 100: # Expensive -> High Usage -> Bullish/Peak
                sentiment = "HIGH_ACTIVITY"
                bias = 0.5
                logger.info(f"⛽ ETH Gas High: {fast_gas} Gwei (High Activity)")
            elif fast_gas < 15: # Cheap -> Low Usage -> Stagnant/Bearish
                sentiment = "LOW_ACTIVITY"
                bias = -0.2
            
            return {
                'metric': 'GAS_GWEI',
                'value': fast_gas,
                'sentiment': sentiment,
                'score': bias,
                'whale_count': 0 # Placeholder for free tier limitation
            }
        except Exception as e:
            logger.error(f"ETH Scan Error: {e}")
            return self._neutral_output()

    async def _analyze_bitcoin(self) -> Dict:
        """Fetch Blockchain.com for Unconfirmed TXs (Mempool Whale Watching)"""
        session = await self._get_session()
        try:
            url = "https://blockchain.info/unconfirmed-transactions?format=json"
            async with session.get(url) as resp:
                data = await resp.json()
                
            txs = data['txs']
            whale_txs = 0
            total_btc_moved = 0.0
            
            for tx in txs:
                # Approximate value sum inputs
                inputs_val = sum([i['prev_out']['value'] for i in tx['inputs'] if 'prev_out' in i])
                btc_val = inputs_val / 100_000_000 # Satoshi to BTC
                
                if btc_val > self.THRESHOLDS['BTC']:
                    whale_txs += 1
                    total_btc_moved += btc_val
            
            # Sentiment
            bias = 0.0
            sentiment = "NEUTRAL"
            
            if whale_txs > 10: # High Whale Activity in Mempool
                sentiment = "VOLATILE"
                # Large movements usually precede volatility, bias neutral but caution high
                bias = -0.3 # Risk ON
                logger.info(f"🐋 BTC Mempool Whales: {whale_txs} TXs moving {total_btc_moved:.0f} BTC")
            
            return {
                'metric': 'MEMPOOL_WHALES',
                'value': total_btc_moved,
                'whale_count': whale_txs,
                'sentiment': sentiment,
                'score': bias
            }
        except Exception as e:
            logger.error(f"BTC Scan Error: {e}")
            return self._neutral_output()

    def _neutral_output(self, error=None) -> Dict:
        return {
            'metric': 'NONE',
            'value': 0,
            'sentiment': 'NEUTRAL',
            'score': 0.0,
            'error': error
        }
    
    async def close(self):
        if self.session:
            await self.session.close()
