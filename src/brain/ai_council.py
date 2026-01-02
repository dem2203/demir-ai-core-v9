# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - AI COUNCIL (Yapay Zeka Konseyi)
==============================================
4 Güçlü AI Modeli Birlikte Karar Veriyor:
- Claude Haiku (Anthropic)
- GPT-4o-mini (OpenAI)
- Gemini Flash (Google)
- DeepSeek V3 (DeepSeek)

Her AI aynı market verisini analiz eder, sonra oylama yapılır.
Çoğunluk kazanır. Eşitlik durumunda Claude'un kararı geçerli.
"""
import os
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger("AI_COUNCIL")


# ============================================================
# DATA CLASSES
# ============================================================

# === AI SPECIALIZATION ROLES ===
AI_ROLES = {
    "Claude": {
        "role": "Risk Analyst",
        "weight": 1.5,  # Higher weight for risk assessment
        "has_veto": True,  # Can veto high-risk trades
        "focus": "risk, drawdown, position sizing, capital protection",
        "prompt_addon": """SEN BİR RİSK ANALİSTİSİN. Ana görevin:
1. Bu trade'in risklerini değerlendir
2. Potansiyel kayıp senaryolarını analiz et
3. Position sizing öner
4. VETO gerekiyorsa açıkça belirt

Eğer risk çok yüksekse, direction'ı HOLD yap ve risk_factors'da VETO sebebini yaz."""
    },
    "GPT-4": {
        "role": "Macro Strategist",
        "weight": 1.2,
        "has_veto": False,
        "focus": "macro trends, market regime, big picture analysis",
        "prompt_addon": """SEN BİR MAKRO STRATEJİSTSİN. Ana görevin:
1. Genel piyasa trendini değerlendir
2. Alt coin/BTC korelasyonunu analiz et
3. Risk-on/Risk-off durumunu belirle
4. Uzun vadeli trend yönünü öner"""
    },
    "DeepSeek": {
        "role": "Technical Analyst",
        "weight": 1.0,
        "has_veto": False,
        "focus": "entry/exit timing, price patterns, indicators",
        "prompt_addon": """SEN BİR TEKNİK ANALİSTSİN. Ana görevin:
1. Optimal giriş zamanlamasını belirle
2. Destek/direnç seviyelerini analiz et
3. Pattern tamamlanma durumunu değerlendir
4. Kesin TP ve SL seviyeleri öner"""
    },
    "Gemini": {
        "role": "Sentiment Analyst",
        "weight": 0.8,
        "has_veto": False,
        "focus": "news sentiment, social sentiment, fear/greed",
        "prompt_addon": """SEN BİR SENTİMENT ANALİSTİSİN. Ana görevin:
1. Haber ve sosyal medya sentimentini değerlendir
2. Fear & Greed durumunu analiz et
3. Crowd positioning'i değerlendir
4. Contrarian sinyalleri tespit et"""
    }
}

@dataclass
class AIAnalysis:
    """Tek bir AI'ın analiz sonucu"""
    model_name: str
    direction: str  # BUY, SELL, HOLD
    confidence: int  # 0-100
    reasoning: str
    key_factors: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    response_time_ms: int = 0
    error: Optional[str] = None


@dataclass
class CouncilDecision:
    """AI Council'ın nihai kararı"""
    final_direction: str  # BUY, SELL, HOLD
    final_confidence: int  # Ortalama güven
    vote_count: Dict[str, int]  # {'BUY': 3, 'SELL': 1, 'HOLD': 0}
    unanimous: bool  # Oybirliği var mı?
    individual_analyses: List[AIAnalysis]
    combined_reasoning: str
    veto_active: bool = False
    veto_reason: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_telegram_format(self) -> str:
        """Telegram mesajı formatı"""
        votes = " | ".join([f"{k}: {v}" for k, v in self.vote_count.items() if v > 0])
        
        models_summary = []
        for analysis in self.individual_analyses:
            emoji = "🟢" if analysis.direction == "BUY" else "🔴" if analysis.direction == "SELL" else "⚪"
            models_summary.append(f"{emoji} {analysis.model_name}: {analysis.direction} ({analysis.confidence}%)")
        
        msg = f"""🤖 *AI COUNCIL KARARI*
━━━━━━━━━━━━━━━━━━━━

📊 *OYLAMA:* {votes}
{'✅ OYBİRLİĞİ!' if self.unanimous else ''}

*Model Kararları:*
{chr(10).join(models_summary)}

📍 *FİNAL:* {self.final_direction} ({self.final_confidence}%)

💭 *ÖZET:*
{self.combined_reasoning[:300]}...
"""
        return msg


# ============================================================
# BASE ANALYZER CLASS
# ============================================================

class AIAnalyzer(ABC):
    """AI Analizör Base Class"""
    
    MODEL_NAME = "BaseAI"
    
    def __init__(self):
        self.enabled = False
        self.api_key = None
        self.call_count = 0
        self.total_cost = 0.0
    
    @abstractmethod
    async def analyze(
        self,
        symbol: str,
        current_price: float,
        market_data: Dict
    ) -> AIAnalysis:
        """Market verisini analiz et"""
        pass
    
    def _build_prompt(self, symbol: str, current_price: float, market_data: Dict) -> str:
        """Tüm AI'lar için ENRICHED prompt - çok daha detaylı"""
        
        # Extract enriched context if available
        enriched = market_data.get('enriched_context', '')
        data_quality = market_data.get('data_quality_score', 100)
        data_warnings = market_data.get('data_warnings', [])
        
        # Vision analysis results if available
        vision_analysis = market_data.get('vision_analysis', '')
        
        return f"""Sen dünya çapında tanınan bir kripto hedge fund yöneticisisin. 
Milyarlarca dolarlık portföy yönetiyorsun. Aşağıdaki kapsamlı verileri analiz et.

## 📊 MARKET VERİLERİ

**Sembol:** {symbol}
**Güncel Fiyat:** ${current_price:,.2f}
**Veri Kalitesi:** {data_quality:.0f}/100 {'' if data_quality > 80 else '⚠️ Düşük kalite!'}
{chr(10).join(['⚠️ ' + w for w in data_warnings[:3]]) if data_warnings else ''}

### 📈 Teknik Göstergeler (Zenginleştirilmiş)
- **RSI (1h):** {market_data.get('rsi', 'N/A')}
  - Percentile (30d): {market_data.get('rsi_percentile', 'N/A')}%
  - Trend: {market_data.get('rsi_trend', 'N/A')}
  - {market_data.get('rsi_context', '')}

- **LSTM Tahmin:** {market_data.get('lstm_direction', 'N/A')} ({market_data.get('lstm_change', 0):+.2f}%)
  - Güven: {market_data.get('lstm_confidence', 'N/A')}%

- **Trend:** {market_data.get('trend', 'N/A')}
- **Volatilite:** {market_data.get('volatility_state', 'NORMAL')}
- **Wyckoff Fazı:** {market_data.get('wyckoff_phase', 'N/A')}

### 📗 Order Flow & Whale Activity
- **Orderbook İmbalance:** {market_data.get('orderbook_score', 0):+.0f}%
  - Context: {market_data.get('orderbook_context', 'N/A')}
  
- **Whale Aktivite:** {market_data.get('whale_score', 0):+.0f}
  - Son 24 saat: {market_data.get('whale_flow_24h', 'N/A')}
  - Context: {market_data.get('whale_context', 'N/A')}

### 🌍 Makro Veriler (Zenginleştirilmiş)
- **Fear & Greed Index:** {market_data.get('fear_greed', 50)}
  - Percentile (30d): {market_data.get('fear_greed_percentile', 50)}%
  - Historical context: {market_data.get('fear_greed_context', 'N/A')}

- **BTC Dominance:** {market_data.get('btc_dominance', 0):.1f}%
  - 24h değişim: {market_data.get('btc_dominance_change', 0):+.2f}%
  
- **Market Regime:** {market_data.get('regime', 'UNKNOWN')}
  - Süre: {market_data.get('regime_duration', 'N/A')} saat

- **Haber Sentimenti:** {market_data.get('news_sentiment', 'NEUTRAL')}
- **Opsiyon Sentimenti:** {market_data.get('options_sentiment', 'NEUTRAL')}

### 💧 Likidasyon & Funding
- **Funding Rate:** {market_data.get('funding_rate', 0):.4f}%
  - Context: {market_data.get('funding_context', 'N/A')}
  
- **Long/Short Ratio:** {market_data.get('ls_ratio', 1.0):.2f}
  - Context: {market_data.get('ls_context', 'N/A')}
  
- **OI Değişim (24h):** {market_data.get('oi_change', 0):+.1f}%

### 📍 Kritik Seviyeler
- **Daily R1 (Direnç):** ${market_data.get('daily_r1', 0):,.0f}
- **Daily S1 (Destek):** ${market_data.get('daily_s1', 0):,.0f}
- **Distance to R1:** {market_data.get('distance_to_r1', 0):+.2f}%
- **Distance to S1:** {market_data.get('distance_to_s1', 0):+.2f}%

### 📰 Son Haberler
{market_data.get('recent_news_summary', 'Önemli haber yok')}

### 🐋 Son Whale Hareketleri
{market_data.get('whale_movements_summary', 'Büyük hareket yok')}

{f'''### 👁️ GRAFİK ANALİZİ (Vision AI)
{vision_analysis}
''' if vision_analysis else ''}

{f'''### 📊 ZENGİNLEŞTİRİLMİŞ BAĞLAM
{enriched}
''' if enriched else ''}

## 🎯 GÖREV

Yukarıdaki TÜM verileri dikkatle analiz et. Sadece basit indikatörlere değil, 
BAĞLAMA (context), TARİHSEL pozisyona (percentile) ve TREND'lere dikkat et.

Şu formatta JSON döndür:

```json
{{
    "direction": "BUY" | "SELL" | "HOLD",
    "confidence": 0-100,
    "reasoning": "Detaylı açıklama (max 300 karakter) - hangi faktörler kritik?",
    "key_factors": ["En önemli 3-5 faktör"],
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "risk_factors": ["Dikkat edilmesi gereken riskler"],
    "price_target": hedef_fiyat_veya_null,
    "stop_loss": stop_fiyat_veya_null,
    "timeframe": "1h" | "4h" | "1d"
}}
```

⚠️ SADECE JSON döndür, başka bir şey yazma.
⚠️ Bear market'ta LONG, Bull market'ta SHORT sinyali verirken ÇOK DİKKATLİ OL."""
    
    def _parse_response(self, response_text: str, model_name: str) -> AIAnalysis:
        """AI yanıtını parse et"""
        try:
            # JSON bloğunu bul
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            data = json.loads(json_str)
            
            return AIAnalysis(
                model_name=model_name,
                direction=data.get("direction", "HOLD").upper(),
                confidence=min(100, max(0, int(data.get("confidence", 50)))),
                reasoning=data.get("reasoning", "")[:500],
                key_factors=data.get("key_factors", [])[:5],
                risk_level=data.get("risk_level", "MEDIUM").upper(),
                price_target=data.get("price_target"),
                stop_loss=data.get("stop_loss")
            )
        except Exception as e:
            logger.error(f"{model_name} parse error: {e}")
            return AIAnalysis(
                model_name=model_name,
                direction="HOLD",
                confidence=30,
                reasoning=f"Parse error: {str(e)[:100]}",
                key_factors=[],
                risk_level="HIGH",
                error=str(e)
            )


# ============================================================
# CLAUDE ANALYZER
# ============================================================

class ClaudeAnalyzer(AIAnalyzer):
    """Claude Haiku Analizör"""
    
    MODEL_NAME = "Claude"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.enabled = bool(self.api_key)
        self.model = "claude-3-haiku-20240307"
        
        if self.enabled:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"✅ {self.MODEL_NAME} initialized")
            except Exception as e:
                logger.error(f"❌ Claude init error: {e}")
                self.enabled = False
    
    async def analyze(self, symbol: str, current_price: float, market_data: Dict) -> AIAnalysis:
        if not self.enabled:
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning="Claude API not available",
                key_factors=[],
                risk_level="HIGH",
                error="API disabled"
            )
        
        import time
        start = time.time()
        
        try:
            prompt = self._build_prompt(symbol, current_price, market_data)
            
            # Sync call in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            
            response_text = response.content[0].text
            self.call_count += 1
            
            analysis = self._parse_response(response_text, self.MODEL_NAME)
            analysis.response_time_ms = int((time.time() - start) * 1000)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning=str(e)[:200],
                key_factors=[],
                risk_level="HIGH",
                error=str(e)
            )


# ============================================================
# GPT-4 ANALYZER
# ============================================================

class GPTAnalyzer(AIAnalyzer):
    """GPT-4o-mini Analizör"""
    
    MODEL_NAME = "GPT-4"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = bool(self.api_key)
        self.model = "gpt-4o-mini"
        
        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"✅ {self.MODEL_NAME} initialized")
            except Exception as e:
                logger.error(f"❌ GPT init error: {e}")
                self.enabled = False
    
    async def analyze(self, symbol: str, current_price: float, market_data: Dict) -> AIAnalysis:
        if not self.enabled:
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning="OpenAI API not available",
                key_factors=[],
                risk_level="HIGH",
                error="API disabled"
            )
        
        import time
        start = time.time()
        
        try:
            prompt = self._build_prompt(symbol, current_price, market_data)
            
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
            )
            
            response_text = response.choices[0].message.content
            self.call_count += 1
            
            analysis = self._parse_response(response_text, self.MODEL_NAME)
            analysis.response_time_ms = int((time.time() - start) * 1000)
            
            return analysis
            
        except Exception as e:
            logger.error(f"GPT analysis error: {e}")
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning=str(e)[:200],
                key_factors=[],
                risk_level="HIGH",
                error=str(e)
            )


# ============================================================
# GEMINI ANALYZER
# ============================================================

class GeminiAnalyzer(AIAnalyzer):
    """Google Gemini Flash Analizör"""
    
    MODEL_NAME = "Gemini"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.enabled = False  # TEMPORARILY DISABLED - Quota exceeded, use 3 AI models
        self.model = "gemini-2.0-flash-exp"
        
        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
                logger.info(f"✅ {self.MODEL_NAME} initialized")
            except Exception as e:
                logger.error(f"❌ Gemini init error: {e}")
                self.enabled = False
    
    async def analyze(self, symbol: str, current_price: float, market_data: Dict) -> AIAnalysis:
        if not self.enabled:
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning="Google API not available",
                key_factors=[],
                risk_level="HIGH",
                error="API disabled"
            )
        
        import time
        start = time.time()
        
        try:
            prompt = self._build_prompt(symbol, current_price, market_data)
            
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(prompt)
            )
            
            response_text = response.text
            self.call_count += 1
            
            analysis = self._parse_response(response_text, self.MODEL_NAME)
            analysis.response_time_ms = int((time.time() - start) * 1000)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning=str(e)[:200],
                key_factors=[],
                risk_level="HIGH",
                error=str(e)
            )


# ============================================================
# DEEPSEEK ANALYZER
# ============================================================

class DeepSeekAnalyzer(AIAnalyzer):
    """DeepSeek V3 Analizör"""
    
    MODEL_NAME = "DeepSeek"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.enabled = bool(self.api_key)
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com"
        
        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                logger.info(f"✅ {self.MODEL_NAME} initialized")
            except Exception as e:
                logger.error(f"❌ DeepSeek init error: {e}")
                self.enabled = False
    
    async def analyze(self, symbol: str, current_price: float, market_data: Dict) -> AIAnalysis:
        if not self.enabled:
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning="DeepSeek API not available",
                key_factors=[],
                risk_level="HIGH",
                error="API disabled"
            )
        
        import time
        start = time.time()
        
        try:
            prompt = self._build_prompt(symbol, current_price, market_data)
            
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
            )
            
            response_text = response.choices[0].message.content
            self.call_count += 1
            
            analysis = self._parse_response(response_text, self.MODEL_NAME)
            analysis.response_time_ms = int((time.time() - start) * 1000)
            
            return analysis
            
        except Exception as e:
            logger.error(f"DeepSeek analysis error: {e}")
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning=str(e)[:200],
                key_factors=[],
                risk_level="HIGH",
                error=str(e)
            )


# ============================================================
# AI COUNCIL - MAIN ORCHESTRATOR
# ============================================================

class AICouncil:
    """
    AI Council - 4 AI Modeli Birlikte Karar Veriyor
    
    Oylama Kuralları:
    1. Çoğunluk kazanır (3/4 veya 2/4)
    2. Eşitlik durumunda Claude'un kararı geçerli
    3. Tüm AI'lar HOLD derse → Sinyal yok
    4. Oybirliği varsa güven %10 artar
    """
    
    def __init__(self):
        self.analyzers: List[AIAnalyzer] = []
        self.enabled = False
        self._initialize_analyzers()
        
        # Budget tracking
        self.daily_budget = float(os.getenv("AI_DAILY_BUDGET", "5.0"))
        self.daily_cost = 0.0
        self.last_reset = datetime.now().date()
        
        logger.info(f"🤖 AI Council initialized with {len([a for a in self.analyzers if a.enabled])} active models")
    
    def _initialize_analyzers(self):
        """Tüm AI analizörleri başlat"""
        # DEBUG: Log all AI API keys status
        logger.info("=" * 50)
        logger.info("🔍 AI COUNCIL API KEY CHECK:")
        logger.info(f"  ANTHROPIC_API_KEY: {'✅ SET' if os.getenv('ANTHROPIC_API_KEY') else '❌ MISSING'}")
        logger.info(f"  OPENAI_API_KEY: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ MISSING'}")
        logger.info(f"  GOOGLE_API_KEY: {'✅ SET' if os.getenv('GOOGLE_API_KEY') else '❌ MISSING'}")
        logger.info(f"  DEEPSEEK_API_KEY: {'✅ SET' if os.getenv('DEEPSEEK_API_KEY') else '❌ MISSING'}")
        logger.info("=" * 50)
        
        self.analyzers = [
            ClaudeAnalyzer(),
            GPTAnalyzer(),
            GeminiAnalyzer(),
            DeepSeekAnalyzer()
        ]
        
        active = [a.MODEL_NAME for a in self.analyzers if a.enabled]
        self.enabled = len(active) >= 1
        
        if active:
            logger.info(f"🟢 Active AI Models: {', '.join(active)}")
        else:
            logger.warning("⚠️ No AI models available! Check API keys in Railway.")
    
    async def analyze(
        self,
        symbol: str,
        current_price: float,
        market_data: Dict
    ) -> CouncilDecision:
        """
        Tüm AI'lardan analiz al ve oylama yap
        """
        if not self.enabled:
            return self._empty_decision("No AI models available")
        
        # Budget check
        self._check_budget()
        
        # Paralel olarak tüm AI'ları çağır
        active_analyzers = [a for a in self.analyzers if a.enabled]
        
        logger.info(f"🗳️ AI Council voting for {symbol}...")
        
        tasks = [
            analyzer.analyze(symbol, current_price, market_data)
            for analyzer in active_analyzers
        ]
        
        analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Hataları filtrele
        valid_analyses = []
        for i, result in enumerate(analyses):
            if isinstance(result, Exception):
                logger.error(f"AI {active_analyzers[i].MODEL_NAME} failed: {result}")
                valid_analyses.append(AIAnalysis(
                    model_name=active_analyzers[i].MODEL_NAME,
                    direction="HOLD",
                    confidence=0,
                    reasoning=str(result)[:100],
                    key_factors=[],
                    risk_level="HIGH",
                    error=str(result)
                ))
            else:
                valid_analyses.append(result)
        
        # Oylama
        return self._vote(valid_analyses)
    
    def _vote(self, analyses: List[AIAnalysis]) -> CouncilDecision:
        """Weighted voting with specialization"""
        # === WEIGHTED VOTE COUNTING ===
        weighted_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        vote_count = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        for analysis in analyses:
            if analysis.direction in vote_count:
                # Get role weight (default 1.0 if not found)
                role_info = AI_ROLES.get(analysis.model_name, {"weight": 1.0})
                weight = role_info.get("weight", 1.0)
                
                # Weight by confidence too (0.5 to 1.5 multiplier)
                confidence_mult = 0.5 + (analysis.confidence / 100)
                final_weight = weight * confidence_mult
                
                weighted_scores[analysis.direction] += final_weight
                vote_count[analysis.direction] += 1
        
        # === VETO CHECK (Risk Analyst) ===
        veto_active = False
        veto_reason = ""
        
        # Check if Risk Analyst (Claude) wants to veto
        claude_analysis = next((a for a in analyses if a.model_name == "Claude"), None)
        if claude_analysis:
            role_info = AI_ROLES.get("Claude", {})
            if role_info.get("has_veto") and claude_analysis.risk_level == "HIGH":
                # Check for explicit veto signals
                veto_keywords = ["veto", "tehlikeli", "riskli", "danger", "avoid"]
                reasoning_lower = claude_analysis.reasoning.lower()
                if any(kw in reasoning_lower for kw in veto_keywords) or claude_analysis.confidence < 30:
                    veto_active = True
                    veto_reason = f"🛡️ Risk Analyst VETO: {claude_analysis.reasoning[:100]}"
                    logger.warning(f"🛡️ VETO ACTIVATED by Risk Analyst (Claude)")
        
        # === DETERMINE WINNER ===
        if veto_active:
            final_direction = "HOLD"
            avg_confidence = 30
        else:
            # Weighted winner
            max_score = max(weighted_scores.values())
            winners = [d for d, s in weighted_scores.items() if s == max_score]
            
            if len(winners) == 1:
                final_direction = winners[0]
            else:
                # Tie-breaker: Claude's vote
                if claude_analysis:
                    final_direction = claude_analysis.direction
                else:
                    final_direction = "HOLD"
            
            # Calculate average confidence for winning direction
            direction_analyses = [a for a in analyses if a.direction == final_direction]
            if direction_analyses:
                # Weighted average confidence
                total_weight = 0
                weighted_conf = 0
                for a in direction_analyses:
                    w = AI_ROLES.get(a.model_name, {}).get("weight", 1.0)
                    weighted_conf += a.confidence * w
                    total_weight += w
                avg_confidence = int(weighted_conf / total_weight) if total_weight > 0 else 50
            else:
                avg_confidence = 50
        
        # Calculate max_votes for unanimity check
        max_votes = max(vote_count.values()) if vote_count else 0
        
        # Oybirliği bonusu
        unanimous = max_votes == len(analyses) and len(analyses) > 1
        if unanimous:
            avg_confidence = min(100, avg_confidence + 10)
        
        # VETO kontrolü - Tüm AI'lar HOLD derse (daha az agresif)
        # Eski: Çoğunluk HOLD = VETO (çok agresif, paper trade engelliyor)
        # Yeni: Oybirliğiyle HOLD = VETO (daha fazla trade, daha fazla öğrenme)
        veto_active = False
        veto_reason = ""
        total_models = len(analyses)
        if vote_count["HOLD"] == total_models and total_models > 0:
            veto_active = True
            veto_reason = f"AI Council: {vote_count['HOLD']}/{total_models} HOLD oyu (oybirliği)"
        elif vote_count["HOLD"] >= total_models - 1 and total_models >= 3:
            # 3 AI'dan 2'si HOLD derse uyarı ver ama VETO yapma
            logger.warning(f"⚠️ High HOLD votes ({vote_count['HOLD']}/{total_models}) - proceeding with caution")
        
        # Combined reasoning
        key_reasons = []
        for a in analyses:
            if a.reasoning and a.direction == final_direction:
                key_reasons.append(f"[{a.model_name}] {a.reasoning[:100]}")
        combined = " | ".join(key_reasons[:3])
        
        decision = CouncilDecision(
            final_direction=final_direction,
            final_confidence=avg_confidence,
            vote_count=vote_count,
            unanimous=unanimous,
            individual_analyses=analyses,
            combined_reasoning=combined,
            veto_active=veto_active,
            veto_reason=veto_reason
        )
        
        logger.info(
            f"🗳️ Council Decision: {final_direction} ({avg_confidence}%) | "
            f"Votes: BUY={vote_count['BUY']}, SELL={vote_count['SELL']}, HOLD={vote_count['HOLD']} | "
            f"{'UNANIMOUS!' if unanimous else ''}"
        )
        
        return decision
    
    def _empty_decision(self, reason: str) -> CouncilDecision:
        """Boş karar döndür"""
        return CouncilDecision(
            final_direction="HOLD",
            final_confidence=0,
            vote_count={"BUY": 0, "SELL": 0, "HOLD": 0},
            unanimous=False,
            individual_analyses=[],
            combined_reasoning=reason,
            veto_active=True,
            veto_reason=reason
        )
    
    def _check_budget(self):
        """Günlük bütçe kontrolü"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_cost = 0.0
            self.last_reset = today
        
        if self.daily_cost >= self.daily_budget:
            logger.warning(f"⚠️ Daily AI budget exceeded: ${self.daily_cost:.2f} >= ${self.daily_budget:.2f}")
    
    def get_stats(self) -> Dict:
        """İstatistikler"""
        return {
            "active_models": [a.MODEL_NAME for a in self.analyzers if a.enabled],
            "total_calls": sum(a.call_count for a in self.analyzers),
            "daily_cost": self.daily_cost,
            "daily_budget": self.daily_budget
        }


# ============================================================
# SINGLETON
# ============================================================

_ai_council: Optional[AICouncil] = None


def get_ai_council() -> AICouncil:
    """Get or create singleton AICouncil"""
    global _ai_council
    if _ai_council is None:
        _ai_council = AICouncil()
    return _ai_council


async def council_analyze(symbol: str, current_price: float, market_data: Dict) -> CouncilDecision:
    """Quick access function"""
    council = get_ai_council()
    return await council.analyze(symbol, current_price, market_data)
