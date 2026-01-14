import logging
import base64
import aiohttp
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from src.config import Config

logger = logging.getLogger("GEMINI_VISION")

class GeminiVisionAnalyzer:
    """
    Chart analysis using Gemini Vision API
    Analyzes candlestick charts visually for patterns, breakouts, volume spikes
    """
    
    def __init__(self):
        self.api_key = Config.GOOGLE_API_KEY
        self.model = "gemini-2.0-flash-exp"  # Latest Gemini with vision
        
    async def analyze_chart_visual(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Create candlestick chart image and analyze with Gemini Vision
        
        Returns:
            {
                'verdict': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'confidence': 1-10,
                'reasoning': str,
                'patterns': [list of patterns detected]
            }
        """
        try:
            # Generate chart image
            chart_image = self._create_candlestick_chart(df, symbol)
            
            # Convert to base64
            buffered = BytesIO()
            chart_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Send to Gemini Vision
            prompt = self._build_analysis_prompt()
            result = await self._call_gemini_vision(img_base64, prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini Vision analysis failed: {e}")
            return {
                'verdict': 'NEUTRAL',
                'confidence': 5,
                'reasoning': f'Analysis error: {str(e)}',
                'patterns': []
            }
    
    def _create_candlestick_chart(self, df: pd.DataFrame, symbol: str) -> Image:
        """
        Create a simple candlestick chart image from DataFrame
        """
        # Image dimensions
        width, height = 800, 400
        img = Image.new('RGB', (width, height), color='#1e1e1e')
        draw = ImageDraw.Draw(img)
        
        # Calculate scaling
        df_recent = df.tail(50)  # Last 50 candles
        price_min = df_recent['low'].min()
        price_max = df_recent['high'].max()
        price_range = price_max - price_min
        
        candle_width = width / len(df_recent) * 0.7
        
        # Draw candlesticks
        for i, (idx, row) in enumerate(df_recent.iterrows()):
            x = i * (width / len(df_recent))
            
            # Scale prices to canvas
            open_y = height - ((row['open'] - price_min) / price_range * height * 0.9)
            close_y = height - ((row['close'] - price_min) / price_range * height * 0.9)
            high_y = height - ((row['high'] - price_min) / price_range * height * 0.9)
            low_y = height - ((row['low'] - price_min) / price_range * height * 0.9)
            
            # Color
            color = '#26a69a' if row['close'] > row['open'] else '#ef5350'
            
            # Draw wick
            draw.line([(x, high_y), (x, low_y)], fill=color, width=1)
            
            # Draw body
            body_top = min(open_y, close_y)
            body_bottom = max(open_y, close_y)
            draw.rectangle([x - candle_width/2, body_top, x + candle_width/2, body_bottom], 
                          fill=color, outline=color)
        
        # Add volume bars at bottom (simplified)
        volume_max = df_recent['volume'].max()
        for i, (idx, row) in enumerate(df_recent.iterrows()):
            x = i * (width / len(df_recent))
            vol_height = (row['volume'] / volume_max) * 80
            color = '#26a69a' if row['close'] > row['open'] else '#ef5350'
            draw.rectangle([x - candle_width/2, height - vol_height, x + candle_width/2, height],
                          fill=color + '80')  # Semi-transparent
        
        return img
    
    def _build_analysis_prompt(self) -> str:
        """Build the prompt for Gemini Vision analysis"""
        return """You are a professional crypto trader analyzing this BTC/ETH chart.

Analyze this candlestick chart and provide:

1. **Pattern Detection:**
   - Is there a breakout (price breaking consolidation)?
   - Volume spike visible? (tall volume bars)
   - Trend direction (up, down, sideways)?
   - Support/Resistance breaks?

2. **Trading Signal:**
   - BULLISH: Strong upward momentum, breakout, volume confirmation
   - BEARISH: Downward pressure, breakdown, selling volume
   - NEUTRAL: Ranging, unclear, low volume

3. **Confidence:** Rate 1-10 based on:
   - Pattern clarity (10 = obvious breakout, 1 = unclear)
   - Volume confirmation (high volume = higher confidence)
   - Trend strength

Respond in JSON format:
{
  "verdict": "BULLISH" or "BEARISH" or "NEUTRAL",
  "confidence": 1-10,
  "reasoning": "Brief explanation of what you see",
  "patterns": ["breakout", "volume_spike", etc]
}

Be decisive. If you see a clear move, say BULLISH/BEARISH. Only say NEUTRAL if truly unclear."""
    
    async def _call_gemini_vision(self, image_base64: str, prompt: str) -> dict:
        """Call Gemini Vision API"""
        if not self.api_key:
            raise Exception("GOOGLE_API_KEY not configured")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 500
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=15) as response:
                if response.status != 200:
                    raise Exception(f"Gemini API error: {response.status}")
                
                data = await response.json()
                text_response = data['candidates'][0]['content']['parts'][0]['text']
                
                # Parse JSON from response
                import json
                # Extract JSON from markdown code blocks if present
                if "```json" in text_response:
                    text_response = text_response.split("```json")[1].split("```")[0]
                elif "```" in text_response:
                    text_response = text_response.split("```")[1].split("```")[0]
                
                result = json.loads(text_response.strip())
                
                logger.info(f"👁️ Gemini Vision: {result['verdict']} ({result['confidence']}/10)")
                logger.info(f"   Reasoning: {result['reasoning'][:100]}")
                
                return result
