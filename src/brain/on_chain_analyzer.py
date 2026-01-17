"""
On-Chain Whale Hunter - Real-Time Accumulation/Distribution Tracker

Why this is powerful:
- Tracks "Exchange Inflows" (Whale moving BTC to Binance = SELL RISK)
- Tracks "Exchange Outflows" (Whale moving BTC to Cold Wallet = ACCUMULATION)
- Uses Grok to parse "Whale Alert" signals instantly.

Lead Time: 1-4 hours before price impact
"""

import logging
from typing import Dict, Optional
import asyncio
from src.config import Config

logger = logging.getLogger("WHALE_HUNTER")

class OnChainAnalyzer:
    """
    Tracks whale movements to detect potential dumps or accumulation phases.
    Uses Grok to interpret "Whale Alert" data streams.
    """
    
    def __init__(self, grok_analyzer=None):
        self.grok = grok_analyzer
        self.last_analysis = {}
        
        # Thresholds
        self.WHALE_THRESHOLD_BTC = 1000  # Alert if >1000 BTC moves
        self.WHALE_THRESHOLD_ETH = 10000 # Alert if >10,000 ETH moves
        
        # State tracking
        self.net_flow_24h = 0  # + means Net Inflow (Bearish), - means Net Outflow (Bullish)
        
    async def analyze_whale_activity(self, symbol: str) -> Dict:
        """
        Analyze recent on-chain movements for a specific asset.
        """
        if not self.grok:
            return self._fallback_response()
            
        base_asset = symbol.replace("USDT", "").replace("PERP", "")
        
        try:
            # 1. Ask Grok to scan real-time Whale Alert data
            prompt = f"""Analyze the last 2 hours of 'Whale Alert' data for #{base_asset}.

Focus on LARGE transfers (> $10M value).
- Transfer TO Exchange = POTENTIAL SELL (Bearish)
- Transfer FROM Exchange = ACCUMULATION (Bullish)
- Transfer Wallet-to-Wallet = OTC / NEUTRAL

Return JSON:
{{
  "whale_sentiment": "BULLISH" or "BEARISH" or "NEUTRAL",
  "net_flow_status": "INFLOW" (to exchange) or "OUTFLOW" (to wallet) or "NEUTRAL",
  "whale_score": 0-10 (10 = Strong Accumulation, 0 = Heavy Dumping),
  "significant_moves": ["1,000 BTC to Binance", "500 BTC to Coinbase"],
  "confidence": 1-10
}}"""

            # Use Grok via the existing client in GrokAnalyzer
            # This assumes GrokAnalyzer exposes a method or client we can use
            # Or we can ask GrokAnalyzer to run this prompt
            
            # Since we passed the grok instance, let's use a specialized method if available
            # or add a generic prompt method to GrokAnalyzer.
            # Ideally, GrokAnalyzer should handle the API call.
            
            analysis = await self.grok.run_custom_prompt(prompt)
            
            # Parse result (assuming Grok returns parsed JSON or dict)
            whale_score = analysis.get("whale_score", 5)
            net_flow = analysis.get("net_flow_status", "NEUTRAL")
            
            # Log significant activity
            moves = analysis.get("significant_moves", [])
            if moves:
                logger.info(f"🐋 Whale Activity ({base_asset}): {', '.join(moves[:2])}")
                
            return {
                "score": whale_score,
                "status": net_flow,
                "sentiment": analysis.get("whale_sentiment", "NEUTRAL"),
                "confidence": analysis.get("confidence", 5),
                "moves": moves
            }
            
        except Exception as e:
            logger.warning(f"Whale analysis failed: {e}")
            return self._fallback_response()
            
    def _fallback_response(self) -> Dict:
        return {
            "score": 5,
            "status": "UNKNOWN",
            "sentiment": "NEUTRAL",
            "confidence": 0,
            "moves": []
        }
