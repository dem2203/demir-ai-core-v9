import logging
import asyncio
from typing import Dict, Optional
from src.brain.macro import MacroBrain
from src.infrastructure.binance_api import BinanceAPI

logger = logging.getLogger("GLOBAL_CORTEX")

class StrategicDirective:
    def __init__(self, mode: str, allowed_direction: str, risk_level: str, reasoning: str):
        self.mode = mode  # AGGRESSIVE, CONSERVATIVE, DEFENSIVE, SLEEP
        self.allowed_direction = allowed_direction # LONG, SHORT, BOTH, NONE
        self.risk_level = risk_level # HIGH, MEDIUM, LOW
        self.reasoning = reasoning

class GlobalCortex:
    """
    The Director.
    Synthesizes Macro + Crypto Structure to issue Trading Directives.
    """
    def __init__(self, binance: BinanceAPI):
        self.macro = MacroBrain()
        self.binance = binance
        
    async def think(self) -> Dict[str, StrategicDirective]:
        """
        Main decision loop. 
        Returns directives for BTCUSDT and ETHUSDT.
        """
        # 1. Gather Intelligence
        world_state = await self.macro.analyze_world()
        
        # 2. Analyze Specific Charts (The "Eyes")
        btc_analysis = await self._analyze_structure("BTCUSDT")
        eth_analysis = await self._analyze_structure("ETHUSDT")
        
        # 3. Formulate Strategy
        directives = {}
        
        # --- BTC Strategy ---
        directives["BTCUSDT"] = self._formulate_plan(world_state, btc_analysis, "BTC")
        
        # --- ETH Strategy ---
        directives["ETHUSDT"] = self._formulate_plan(world_state, eth_analysis, "ETH", btc_context=btc_analysis)
        
        return directives

    async def _analyze_structure(self, symbol: str) -> dict:
        """
        Deep AI analysis of market structure (Highs/Lows, Volume, Trend)
        """
        df = await self.binance.fetch_candles(symbol, timeframe="1h", limit=100)
        if df.empty:
            return {"trend": "UNKNOWN", "volatility": 0}
            
        # Basic Price Action Logic (Placeholder for Deep AI)
        last_close = df['close'].iloc[-1]
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        sma200 = df['close'].rolling(200).mean().iloc[-1]
        
        # Trend Detection
        trend = "NEUTRAL"
        if last_close > sma50 > sma200: trend = "BULLISH"
        elif last_close < sma50 < sma200: trend = "BEARISH"
        
        # Volatility
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        
        return {
            "price": last_close,
            "trend": trend,
            "sma50": sma50,
            "sma200": sma200,
            "atr": atr,
            "volume_trend": "RISING" if df['volume'].iloc[-1] > df['volume'].mean() else "FALLING"
        }

    def _formulate_plan(self, macro: dict, technical: dict, asset: str, btc_context: dict = None) -> StrategicDirective:
        """
        Synthesize Macro + Technicals into a Directive.
        """
        reasons = []
        mode = "CONSERVATIVE"
        direction = "NONE"
        risk = "LOW"
        
        # Macro Filter
        if macro['regime'] == "RISK_OFF":
            reasons.append("⚠️ Macro is Risk-Off (High VIX/DXY). Defensive Mode.")
            mode = "DEFENSIVE"
            if technical['trend'] == "BEARISH":
                direction = "SHORT"
                reasons.append(f"{asset} is in Bear Trend. Shorts Approved.")
            else:
                reasons.append(f"{asset} is holding up, but Macro is bad. Stay cash.")
                
        elif macro['regime'] == "RISK_ON":
            reasons.append("🌍 Macro is Risk-On. Aggressive Mode.")
            mode = "AGGRESSIVE"
            if technical['trend'] == "BULLISH":
                direction = "LONG"
                reasons.append(f"{asset} confirming Bull Trend. Longs Approved.")
            elif technical['trend'] == "BEARISH":
                direction = "SHORT" # Counter-trend or correction
                mode = "CONSERVATIVE"
                reasons.append(f"{asset} lagging macro. Cautious Short or Wait.")
                
        else: # NEUTRAL
            reasons.append("⚖️ Macro is Neutral. Trading Range.")
            direction = "BOTH"
            risk = "MEDIUM"
            
        # ETH Specific check
        if asset == "ETH" and btc_context:
            if btc_context['trend'] == "BEARISH" and direction == "LONG":
                direction = "NONE"
                reasons.append("⛔ BTC is Bearish. Cancelling ETH Longs.")
        
        return StrategicDirective(mode, direction, risk, " | ".join(reasons))
