# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - CHART VISION ANALYZER
====================================
Grafik görsellerini oluşturup Vision AI modellerine gönderir.

Özellikler:
1. Candlestick chart oluşturma
2. Technical indicators overlay
3. Support/Resistance çizgileri
4. Vision AI entegrasyonu (GPT-4V, Gemini Vision, Claude Vision)
"""
import os
import io
import base64
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger("CHART_VISION")


@dataclass
class ChartAnalysis:
    """Vision AI'ın grafik analizi sonucu"""
    model_name: str
    patterns_detected: List[str]
    trend_assessment: str  # BULLISH, BEARISH, NEUTRAL
    key_levels: List[float]
    confidence: int
    reasoning: str
    error: Optional[str] = None


class ChartGenerator:
    """
    Grafik Oluşturucu
    
    Matplotlib/Plotly kullanarak chart oluşturur ve PNG olarak kaydeder.
    """
    
    def __init__(self):
        self._chart_cache = {}
        self._cache_duration = timedelta(minutes=5)
        logger.info("📊 Chart Generator initialized")
    
    def generate_candlestick_chart(
        self,
        symbol: str,
        klines: List[Dict],
        indicators: Dict = None,
        support_levels: List[float] = None,
        resistance_levels: List[float] = None
    ) -> bytes:
        """
        Candlestick chart oluştur ve PNG bytes döndür
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-GUI backend
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import Rectangle
            import numpy as np
            from datetime import datetime
            
            # Kline verilerini hazırla
            if not klines or len(klines) < 10:
                logger.warning(f"Insufficient kline data for {symbol}")
                return None
            
            dates = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            for k in klines[-100:]:  # Son 100 mum
                if isinstance(k, dict):
                    dates.append(datetime.fromtimestamp(k.get('timestamp', 0) / 1000))
                    opens.append(float(k.get('open', 0)))
                    highs.append(float(k.get('high', 0)))
                    lows.append(float(k.get('low', 0)))
                    closes.append(float(k.get('close', 0)))
                    volumes.append(float(k.get('volume', 0)))
                elif isinstance(k, (list, tuple)) and len(k) >= 6:
                    dates.append(datetime.fromtimestamp(k[0] / 1000))
                    opens.append(float(k[1]))
                    highs.append(float(k[2]))
                    lows.append(float(k[3]))
                    closes.append(float(k[4]))
                    volumes.append(float(k[5]))
            
            if not dates:
                return None
            
            # Create figure with 2 subplots (price + volume)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                           gridspec_kw={'height_ratios': [3, 1]},
                                           facecolor='#1a1a2e')
            
            fig.suptitle(f'{symbol} - Technical Analysis', fontsize=14, color='white')
            
            # Style
            ax1.set_facecolor('#16213e')
            ax2.set_facecolor('#16213e')
            
            # Candlesticks
            width = 0.6
            for i, (d, o, h, l, c) in enumerate(zip(dates, opens, highs, lows, closes)):
                color = '#26a69a' if c >= o else '#ef5350'  # Green/Red
                
                # Wick
                ax1.plot([i, i], [l, h], color=color, linewidth=1)
                
                # Body
                body_height = abs(c - o)
                body_bottom = min(o, c)
                rect = Rectangle((i - width/2, body_bottom), width, body_height,
                                facecolor=color, edgecolor=color)
                ax1.add_patch(rect)
            
            # Moving averages if available
            if indicators:
                if 'ema21' in indicators and indicators['ema21']:
                    ema = indicators['ema21'][-100:]
                    ax1.plot(range(len(ema)), ema, color='#ffd700', linewidth=1.5, 
                            label='EMA21', alpha=0.8)
                
                if 'ema50' in indicators and indicators['ema50']:
                    ema = indicators['ema50'][-100:]
                    ax1.plot(range(len(ema)), ema, color='#ff6b6b', linewidth=1.5,
                            label='EMA50', alpha=0.8)
            
            # Support/Resistance levels
            if support_levels:
                for level in support_levels[:3]:
                    ax1.axhline(y=level, color='#4ecdc4', linestyle='--', 
                               linewidth=1, alpha=0.7, label=f'S: ${level:,.0f}')
            
            if resistance_levels:
                for level in resistance_levels[:3]:
                    ax1.axhline(y=level, color='#ff6b6b', linestyle='--',
                               linewidth=1, alpha=0.7, label=f'R: ${level:,.0f}')
            
            # Volume bars
            colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' 
                     for i in range(len(closes))]
            ax2.bar(range(len(volumes)), volumes, color=colors, alpha=0.7)
            
            # Styling
            ax1.set_ylabel('Price', color='white')
            ax2.set_ylabel('Volume', color='white')
            ax1.tick_params(colors='white')
            ax2.tick_params(colors='white')
            ax1.grid(True, alpha=0.2)
            ax2.grid(True, alpha=0.2)
            
            if ax1.get_legend_handles_labels()[0]:
                ax1.legend(loc='upper left', facecolor='#16213e', labelcolor='white')
            
            # Add timestamp
            ax2.set_xlabel(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
                          color='gray', fontsize=8)
            
            plt.tight_layout()
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, facecolor='#1a1a2e',
                       bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            logger.info(f"📈 Generated chart for {symbol}")
            return buf.getvalue()
            
        except ImportError as e:
            logger.warning(f"Matplotlib not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return None
    
    def image_to_base64(self, image_bytes: bytes) -> str:
        """PNG bytes'ı base64'e çevir"""
        if not image_bytes:
            return ""
        return base64.b64encode(image_bytes).decode('utf-8')


class VisionAnalyzer:
    """
    Vision AI Analizörü
    
    Grafik görsellerini GPT-4V, Gemini Vision, Claude Vision'a gönderir.
    """
    
    VISION_PROMPT = """Analyze this cryptocurrency price chart. 

You are an expert technical analyst. Look at the candlestick patterns, trend direction, 
support/resistance levels, and any notable formations.

Identify:
1. Current trend (BULLISH, BEARISH, or NEUTRAL)
2. Key chart patterns (head and shoulders, double top/bottom, triangles, flags, etc.)
3. Important price levels visible on the chart
4. Volume analysis if visible
5. Potential next movement

Return your analysis in JSON format:
```json
{
    "trend": "BULLISH" | "BEARISH" | "NEUTRAL",
    "patterns": ["Pattern 1", "Pattern 2"],
    "key_levels": [price1, price2, price3],
    "confidence": 0-100,
    "reasoning": "Brief explanation",
    "next_move": "Expected direction and target"
}
```

ONLY return JSON, nothing else."""

    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        self.gpt_vision_enabled = bool(self.openai_key)
        self.gemini_vision_enabled = bool(self.google_key)
        self.claude_vision_enabled = bool(self.anthropic_key)
        
        enabled = []
        if self.gpt_vision_enabled:
            enabled.append("GPT-4V")
        if self.gemini_vision_enabled:
            enabled.append("Gemini")
        if self.claude_vision_enabled:
            enabled.append("Claude")
        
        logger.info(f"👁️ Vision Analyzers enabled: {', '.join(enabled) if enabled else 'None'}")
    
    async def analyze_with_gpt4v(self, image_base64: str, symbol: str) -> ChartAnalysis:
        """GPT-4 Vision ile analiz"""
        if not self.gpt_vision_enabled:
            return ChartAnalysis(
                model_name="GPT-4V",
                patterns_detected=[],
                trend_assessment="UNKNOWN",
                key_levels=[],
                confidence=0,
                reasoning="GPT-4V not available",
                error="API key not configured"
            )
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o",  # GPT-4o has vision
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.VISION_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
            )
            
            return self._parse_vision_response(response.choices[0].message.content, "GPT-4V")
            
        except Exception as e:
            logger.error(f"GPT-4V analysis error: {e}")
            return ChartAnalysis(
                model_name="GPT-4V",
                patterns_detected=[],
                trend_assessment="UNKNOWN",
                key_levels=[],
                confidence=0,
                reasoning=str(e)[:200],
                error=str(e)
            )
    
    async def analyze_with_gemini(self, image_base64: str, symbol: str) -> ChartAnalysis:
        """Gemini Vision ile analiz"""
        if not self.gemini_vision_enabled:
            return ChartAnalysis(
                model_name="Gemini",
                patterns_detected=[],
                trend_assessment="UNKNOWN",
                key_levels=[],
                confidence=0,
                reasoning="Gemini not available",
                error="API key not configured"
            )
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.google_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Decode base64 to bytes for Gemini
            import base64
            image_bytes = base64.b64decode(image_base64)
            
            # Use PIL to create image
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_bytes))
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content([self.VISION_PROMPT, image])
            )
            
            return self._parse_vision_response(response.text, "Gemini")
            
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return ChartAnalysis(
                model_name="Gemini",
                patterns_detected=[],
                trend_assessment="UNKNOWN",
                key_levels=[],
                confidence=0,
                reasoning=str(e)[:200],
                error=str(e)
            )
    
    async def analyze_with_claude(self, image_base64: str, symbol: str) -> ChartAnalysis:
        """Claude Vision ile analiz"""
        if not self.claude_vision_enabled:
            return ChartAnalysis(
                model_name="Claude",
                patterns_detected=[],
                trend_assessment="UNKNOWN",
                key_levels=[],
                confidence=0,
                reasoning="Claude not available",
                error="API key not configured"
            )
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=500,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_base64
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": self.VISION_PROMPT
                                }
                            ]
                        }
                    ]
                )
            )
            
            return self._parse_vision_response(response.content[0].text, "Claude")
            
        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
            return ChartAnalysis(
                model_name="Claude",
                patterns_detected=[],
                trend_assessment="UNKNOWN",
                key_levels=[],
                confidence=0,
                reasoning=str(e)[:200],
                error=str(e)
            )
    
    def _parse_vision_response(self, response_text: str, model_name: str) -> ChartAnalysis:
        """Vision AI yanıtını parse et"""
        import json
        
        try:
            # JSON bloğunu bul
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            data = json.loads(json_str)
            
            return ChartAnalysis(
                model_name=model_name,
                patterns_detected=data.get("patterns", []),
                trend_assessment=data.get("trend", "NEUTRAL").upper(),
                key_levels=[float(l) for l in data.get("key_levels", [])],
                confidence=min(100, max(0, int(data.get("confidence", 50)))),
                reasoning=data.get("reasoning", "") + " | " + data.get("next_move", "")
            )
            
        except Exception as e:
            logger.warning(f"{model_name} vision parse error: {e}")
            return ChartAnalysis(
                model_name=model_name,
                patterns_detected=[],
                trend_assessment="UNKNOWN",
                key_levels=[],
                confidence=30,
                reasoning=response_text[:300] if response_text else str(e),
                error=str(e)
            )
    
    async def analyze_chart(self, image_base64: str, symbol: str) -> List[ChartAnalysis]:
        """Tüm Vision AI'larla paralel analiz"""
        tasks = []
        
        if self.gpt_vision_enabled:
            tasks.append(self.analyze_with_gpt4v(image_base64, symbol))
        
        if self.gemini_vision_enabled:
            tasks.append(self.analyze_with_gemini(image_base64, symbol))
        
        if self.claude_vision_enabled:
            tasks.append(self.analyze_with_claude(image_base64, symbol))
        
        if not tasks:
            logger.warning("No Vision AI models available")
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        analyses = []
        for result in results:
            if isinstance(result, ChartAnalysis):
                analyses.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Vision analysis failed: {result}")
        
        return analyses


# Singleton
_chart_generator: Optional[ChartGenerator] = None
_vision_analyzer: Optional[VisionAnalyzer] = None


def get_chart_generator() -> ChartGenerator:
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = ChartGenerator()
    return _chart_generator


def get_vision_analyzer() -> VisionAnalyzer:
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer()
    return _vision_analyzer


async def analyze_chart_with_vision(
    symbol: str,
    klines: List[Dict],
    indicators: Dict = None,
    support_levels: List[float] = None,
    resistance_levels: List[float] = None
) -> List[ChartAnalysis]:
    """Quick access function - chart oluştur ve analiz et"""
    
    generator = get_chart_generator()
    analyzer = get_vision_analyzer()
    
    # Generate chart
    chart_bytes = generator.generate_candlestick_chart(
        symbol, klines, indicators, support_levels, resistance_levels
    )
    
    if not chart_bytes:
        return []
    
    # Convert to base64
    chart_base64 = generator.image_to_base64(chart_bytes)
    
    # Analyze with Vision AI
    return await analyzer.analyze_chart(chart_base64, symbol)
