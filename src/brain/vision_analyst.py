import logging
import os
import io
import json
import base64
from typing import Dict, Optional, List
import google.generativeai as genai
from openai import OpenAI
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

from src.config.settings import Config

logger = logging.getLogger("VISION_ANALYST")

class VisionAnalyst:
    """
    DUAL VISION CORTEX - Multi-Model AI Vision Analysis
    
    Capabilities:
    1. Generate technical charts from raw OHLCV data
    2. Analyze chart images using DUAL Multimodal AIs:
       - Google Gemini (Fast, wide context)
       - OpenAI GPT-4o (Precise, technical)
    3. Cross-validate patterns and trends between models
    4. Identify visual patterns that mathematical indicators might miss
    """
    
    def __init__(self):
        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        self.gemini_active = False
        self.openai_active = False
        
        # Initialize Gemini
        if self.google_key:
            try:
                genai.configure(api_key=self.google_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                self.gemini_active = True
                logger.info("👁️ Gemini Vision: ACTIVE")
            except Exception as e:
                logger.error(f"Gemini initialization failed: {e}")
        else:
            logger.warning("⚠️ GOOGLE_API_KEY not found. Gemini disabled.")
            
        # Initialize OpenAI
        if self.openai_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_key)
                self.openai_active = True
                logger.info("👁️ OpenAI (GPT-4o) Vision: ACTIVE")
            except Exception as e:
                logger.error(f"OpenAI initialization failed: {e}")
        else:
            logger.warning("⚠️ OPENAI_API_KEY not found. GPT-4o disabled.")
            
        if not self.gemini_active and not self.openai_active:
            logger.warning("⚠️ NO VISION APIs AVAILABLE. Visual Cortex disabled.")
            
    def analyze_chart(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Generate chart image and analyze it using DUAL AI Vision (Gemini + GPT-4o).
        Returns combined consensus analysis.
        """
        if (not self.gemini_active and not self.openai_active) or df.empty:
            return self._empty_analysis()
            
        try:
            # 1. Generate Chart Image
            img_bytes = self._generate_chart_image(symbol, df)
            
            if not img_bytes:
                return self._empty_analysis()
            
            # 2. Query BOTH AIs (if available)
            analyses = []
            
            if self.gemini_active:
                gemini_result = self._query_gemini(img_bytes)
                if gemini_result:
                    analyses.append(("Gemini", gemini_result))
                    
            if self.openai_active:
                openai_result = self._query_openai(img_bytes)
                if openai_result:
                    analyses.append(("GPT-4o", openai_result))
            
            # 3. Combine Results
            if len(analyses) == 0:
                return self._empty_analysis()
            elif len(analyses) == 1:
                # Single model available
                _, result = analyses[0]
                result['dual_vision'] = False
                result['agreement'] = 'N/A'
                return result
            else:
                # Dual Vision: Compare and Combine
                return self._combine_analyses(analyses)
            
        except Exception as e:
            logger.error(f"Visual Analysis failed for {symbol}: {e}")
            return self._empty_analysis()
            
    def _query_gemini(self, img_bytes: bytes) -> Optional[Dict]:
        """Query Google Gemini Vision API"""
        try:
            prompt = self._get_analysis_prompt()
            
            image_part = {
                "mime_type": "image/png",
                "data": img_bytes
            }
            
            response = self.gemini_model.generate_content([prompt, image_part])
            return self._parse_response(response.text, "Gemini")
            
        except Exception as e:
            logger.error(f"Gemini query failed: {e}")
            return None
            
    def _query_openai(self, img_bytes: bytes) -> Optional[Dict]:
        """Query OpenAI GPT-4o Vision API"""
        try:
            prompt = self._get_analysis_prompt()
            
            # Convert to base64
            import base64
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return self._parse_response(response.choices[0].message.content, "GPT-4o")
            
        except Exception as e:
            logger.error(f"OpenAI query failed: {e}")
            return None
            
    def _combine_analyses(self, analyses: List[tuple]) -> Dict:
        """
        Combine analyses from multiple models.
        Returns consensus analysis with agreement flag.
        """
        gemini_data = next((data for name, data in analyses if name == "Gemini"), None)
        gpt_data = next((data for name, data in analyses if name == "GPT-4o"), None)
        
        # Average scores
        avg_score = (gemini_data['visual_score'] + gpt_data['visual_score']) // 2
        
        # Check trend agreement
        gemini_trend = gemini_data['trend']
        gpt_trend = gpt_data['trend']
        
        if gemini_trend == gpt_trend:
            consensus_trend = gemini_trend
            agreement = "STRONG"
        elif (gemini_trend == "NEUTRAL" or gpt_trend == "NEUTRAL"):
            consensus_trend = gemini_trend if gemini_trend != "NEUTRAL" else gpt_trend
            agreement = "MODERATE"
        else:
            # Disagreement
            consensus_trend = "NEUTRAL"
            agreement = "CONFLICT"
        
        # Combine reasoning
        combined_reasoning = f"🟢 Gemini: {gemini_data['reasoning']}\n\n🔵 GPT-4o: {gpt_data['reasoning']}"
        
        return {
            "trend": consensus_trend,
            "pattern": gemini_data.get('pattern', 'None'),  # Prefer first model's pattern
            "visual_score": avg_score,
            "reasoning": combined_reasoning,
            "dual_vision": True,
            "agreement": agreement,
            "gemini_score": gemini_data['visual_score'],
            "gpt_score": gpt_data['visual_score'],
            "timestamp": datetime.now().isoformat()
        }
            
    def _generate_chart_image(self, symbol: str, df: pd.DataFrame) -> Optional[bytes]:
        """Convert DataFrame to candlestick chart image bytes"""
        try:
            # Create simple candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            )])
            
            # Clean layout for AI (remove noise)
            fig.update_layout(
                title=f"{symbol} Price Action",
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                width=800,
                height=500
            )
            
            # Convert to image bytes
            img_bytes = fig.to_image(format="png", engine="kaleido")
            return img_bytes
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None
            
    def _get_analysis_prompt(self) -> str:
        """Standard prompt for both models"""
        return """
You are an expert technical analyst. I will provide a cryptocurrency price chart.
Analyze the visual patterns and price action.

Focus on:
1. The immediate trend (Bullish/Bearish/Neutral)
2. Any visible chart patterns (Bull Flag, Head & Shoulders, Double Bottom, etc.)
3. Key mental levels (Support/Resistance)

Respond ONLY with a JSON object in this exact format:
{
    "trend": "BULLISH/BEARISH/NEUTRAL",
    "pattern": "Pattern Name or None",
    "visual_score": 0-100 (Where 0 is extremely bearish, 100 is extremely bullish, 50 neutral),
    "reasoning": "Brief explanation of what you see"
}
Do not include markdown formatting or extra text. Just the JSON.
"""
            
    def _parse_response(self, text: str, model_name: str) -> Dict:
        """Parse JSON response from AI"""
        try:
            # Clean up potential markdown code blocks
            clean_text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            
            # Validate fields
            return {
                "trend": data.get("trend", "NEUTRAL").upper(),
                "pattern": data.get("pattern", "None"),
                "visual_score": int(data.get("visual_score", 50)),
                "reasoning": data.get("reasoning", "Analysis unavailable"),
                "timestamp": datetime.now().isoformat(),
                "model": model_name
            }
        except Exception as e:
            logger.error(f"Failed to parse {model_name} response: {text[:100]}... Error: {e}")
            return {
                "trend": "NEUTRAL",
                "pattern": "None",
                "visual_score": 50,
                "reasoning": f"{model_name} parsing failed",
                "timestamp": datetime.now().isoformat(),
                "model": model_name
            }
            
    def _empty_analysis(self) -> Dict:
        return {
            "trend": "NEUTRAL",
            "pattern": "None",
            "visual_score": 50,
            "reasoning": "Visual analysis disabled or failed",
            "dual_vision": False,
            "agreement": "N/A",
            "timestamp": datetime.now().isoformat()
        }
