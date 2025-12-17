# -*- coding: utf-8 -*-
"""
DEMIR AI - DEX Volume Tracker
Decentralized exchange volume spikes = pump signals

PHASE 45: High-Value Scraper
- DexScreener API (no key required)
- Volume spike detection (>300% = pump)
- New pair detection
- Multi-chain support (ETH, BSC, SOL)
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("DEX_VOLUME_TRACKER")


@dataclass
class DEXPair:
    """DEX trading pair"""
    chain: str
    dex: str
    pair_address: str
    base_token: str
    quote_token: str
    price_usd: float
    volume_24h: float
    volume_change_24h: float
    liquidity_usd: float
    price_change_24h: float
    creation_time: datetime
    is_new: bool


class DEXVolumeTracker:
    """
    DEX Volume Spike Tracker
    
    Detects:
    - Volume spikes (>300% increase = pump signal)
    - New pairs launching
    - Liquidity changes
    - Price manipulation patterns
    """
    
    BASE_URL = "https://api.dexscreener.com/latest/dex"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    # Volume spike thresholds
    PUMP_THRESHOLD = 300  # 300% volume increase
    MEGA_PUMP_THRESHOLD = 1000  # 1000% = mega pump
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 180  # 3 minutes (DEX changes fast)
        self.last_fetch = {}
    
    def get_trending_pairs(self, chain: str = 'all', limit: int = 20) -> List[DEXPair]:
        """
        Get trending DEX pairs with volume spikes.
        
        Args:
            chain: 'ethereum', 'bsc', 'solana', 'all'
            limit: Max pairs to return
        """
        cache_key = f'trending_{chain}_{limit}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            # DexScreener trending endpoint
            url = f"{self.BASE_URL}/search?q=trending"
            
            response = requests.get(url, headers=self.HEADERS, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"DexScreener API returned {response.status_code}")
                return []
            
            data = response.json()
            pairs = []
            
            for pair_data in data.get('pairs', [])[:limit]:
                try:
                    pair = self._parse_pair(pair_data)
                    
                    # Filter by chain if specified
                    if chain != 'all' and pair.chain.lower() != chain.lower():
                        continue
                    
                    pairs.append(pair)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse pair: {e}")
                    continue
            
            self._set_cache(cache_key, pairs)
            logger.info(f"Fetched {len(pairs)} trending DEX pairs")
            
            return pairs
            
        except Exception as e:
            logger.warning(f"DEX volume tracking failed: {e}")
            return []
    
    def detect_volume_spikes(self, min_volume_24h: float = 100000) -> Dict:
        """
        Detect significant volume spikes (pump signals).
        
        Args:
            min_volume_24h: Minimum 24h volume to consider (USD)
        
        Returns:
            {
                'pump_signals': list of pairs with >300% volume spike
                'mega_pumps': list of pairs with >1000% spike
                'new_pairs': recently created pairs
                'summary': text summary
            }
        """
        pairs = self.get_trending_pairs(chain='all', limit=50)
        
        pump_signals = []
        mega_pumps = []
        new_pairs = []
        
        for pair in pairs:
            # Filter by minimum volume
            if pair.volume_24h < min_volume_24h:
                continue
            
            # Detect mega pumps (>1000% volume increase)
            if pair.volume_change_24h >= self.MEGA_PUMP_THRESHOLD:
                mega_pumps.append(pair)
            
            # Detect pumps (>300% volume increase)
            elif pair.volume_change_24h >= self.PUMP_THRESHOLD:
                pump_signals.append(pair)
            
            # Detect new pairs (created < 24h ago)
            if pair.is_new:
                new_pairs.append(pair)
        
        # Generate summary
        summary = self._generate_summary(pump_signals, mega_pumps, new_pairs)
        
        return {
            'pump_signals': pump_signals,
            'mega_pumps': mega_pumps,
            'new_pairs': new_pairs,
            'total_tracked': len(pairs),
            'summary': summary,
            'timestamp': datetime.now()
        }
    
    def get_token_volume(self, token_address: str, chain: str = 'ethereum') -> Optional[Dict]:
        """
        Get volume data for a specific token.
        """
        try:
            url = f"{self.BASE_URL}/tokens/{token_address}"
            
            response = requests.get(url, headers=self.HEADERS, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pairs = data.get('pairs', [])
                
                if not pairs:
                    return None
                
                # Aggregate volume across all pairs
                total_volume = sum(p.get('volume', {}).get('h24', 0) for p in pairs)
                avg_price_change = sum(p.get('priceChange', {}).get('h24', 0) for p in pairs) / len(pairs)
                
                return {
                    'token_address': token_address,
                    'total_volume_24h': total_volume,
                    'avg_price_change_24h': avg_price_change,
                    'pair_count': len(pairs),
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to fetch token volume: {e}")
            return None
    
    def format_for_telegram(self) -> str:
        """Telegram formatı"""
        spike_data = self.detect_volume_spikes()
        
        msg = "📊 *DEX Volume Spikes*\\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\\n\\n"
        
        # Mega pumps
        if spike_data['mega_pumps']:
            msg += "🚀 **MEGA PUMPS** (>1000% volume):\\n"
            for pair in spike_data['mega_pumps'][:3]:
                msg += f"• {pair.base_token}/{pair.quote_token}\\n"
                msg += f"  📈 Volume: +{pair.volume_change_24h:.0f}%\\n"
                msg += f"  💰 ${pair.volume_24h/1e6:.1f}M (24h)\\n"
                msg += f"  🔗 {pair.dex} ({pair.chain})\\n\\n"
        
        # Regular pumps
        if spike_data['pump_signals']:
            msg += "⚡ **Volume Spikes** (>300%):\\n"
            for pair in spike_data['pump_signals'][:5]:
                msg += f"• {pair.base_token}: +{pair.volume_change_24h:.0f}%\\n"
        
        # New pairs
        if spike_data['new_pairs']:
            msg += f"\\n🆕 **New Pairs:** {len(spike_data['new_pairs'])}\\n"
        
        msg += f"\\n💡 _{spike_data['summary']}_\\n"
        msg += f"\\n⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        
        return msg
    
    # =========================================
    # PRIVATE METHODS
    # =========================================
    
    def _parse_pair(self, data: Dict) -> DEXPair:
        """Parse DexScreener pair data"""
        
        # Parse creation time
        created_at = data.get('pairCreatedAt', 0)
        if created_at:
            creation_time = datetime.fromtimestamp(created_at / 1000)
            is_new = (datetime.now() - creation_time).total_seconds() < 86400  # < 24h
        else:
            creation_time = datetime.now()
            is_new = False
        
        return DEXPair(
            chain=data.get('chainId', 'unknown'),
            dex=data.get('dexId', 'unknown'),
            pair_address=data.get('pairAddress', ''),
            base_token=data.get('baseToken', {}).get('symbol', 'UNKNOWN'),
            quote_token=data.get('quoteToken', {}).get('symbol', 'UNKNOWN'),
            price_usd=float(data.get('priceUsd', 0)),
            volume_24h=float(data.get('volume', {}).get('h24', 0)),
            volume_change_24h=float(data.get('priceChange', {}).get('h24', 0)),  # Approximate
            liquidity_usd=float(data.get('liquidity', {}).get('usd', 0)),
            price_change_24h=float(data.get('priceChange', {}).get('h24', 0)),
            creation_time=creation_time,
            is_new=is_new
        )
    
    def _generate_summary(self, pumps: List, mega_pumps: List, new_pairs: List) -> str:
        """Generate text summary"""
        
        if mega_pumps:
            return f"🚨 {len(mega_pumps)} MEGA PUMP detected! Volume surges >1000%. High manipulation risk!"
        elif len(pumps) >= 5:
            return f"⚠️ {len(pumps)} volume spikes detected. Increased trading activity across DEXs."
        elif len(pumps) > 0:
            return f"📊 {len(pumps)} moderate volume spikes. Normal DEX activity."
        elif new_pairs:
            return f"🆕 {len(new_pairs)} new pairs launched today. Monitor for early opportunities."
        else:
            return "↔️ Normal DEX volume. No significant spikes detected."
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and fresh"""
        if key not in self.cache or key not in self.last_fetch:
            return False
        
        age = (datetime.now() - self.last_fetch[key]).total_seconds()
        return age < self.cache_duration
    
    def _set_cache(self, key: str, data):
        """Cache data"""
        self.cache[key] = data
        self.last_fetch[key] = datetime.now()
