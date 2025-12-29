# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - FRACTAL ANALYZER (Fractal Memory)
================================================
"Tarih tekerrürden ibarettir."

Bu modül, şu anki fiyat hareketini (son 50 mum) geçmişteki 
binlerce mumluk veriyle kıyaslar. Benzer bir şekil (Pattern) 
bulursa, o tarihten sonra ne olduğunu analiz eder.

Yöntem:
1. Son 50 mumu al (Query Pattern).
2. Geçmişteki tüm 50 mumluk pencereleri al (Sliding Window).
3. Cosine Similarity ile benzerliği ölç.
4. Benzerlik > %90 ise "MATCH" kabul et.
5. Match sonrası fiyatın ne yaptığına bak (Profit Factor).
"""
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("FRACTAL_MEMORY")

@dataclass
class FractalMatch:
    match_index: int
    similarity: float  # 0.0 to 1.0
    future_change_pct: float  # What happened next?
    timestamp: str = ""

class FractalAnalyzer:
    """Geçmiş grafiklerde benzer desenleri arar."""
    
    def __init__(self):
        self.window_size = 50  # 50 mumluk desenleri ara
        self.future_horizon = 12  # Match sonrası 12 mum (1 saatlikte 12 saat)
        
    def find_fractal_match(self, closes: List[float], history_closes: List[float]) -> Optional[FractalMatch]:
        """
        closes: Şu anki son 50 kapanış
        history_closes: Geçmiş 2000+ kapanış
        """
        if len(closes) < self.window_size or len(history_closes) < self.window_size * 2:
            return None
            
        # 1. Normalize Query Pattern (Current)
        query = self._normalize(np.array(closes[-self.window_size:]))
        
        best_match = None
        best_score = -1.0
        
        # 2. Sliding Window Search
        # We search history UP TO the current pattern (don't overlap with current)
        history = np.array(history_closes[:-self.window_size])
        
        # Basit optimizasyon: Her adımda kaydır
        # Gerçek üretimde bu çok yavaş olabilir, o yüzden stride=1 yerine stride=5 kullanılabilir
        # veya Faiss gibi kütüphaneler. Ama burada basit numpy yeterli (<5000 mum için).
        
        for i in range(0, len(history) - self.window_size - self.future_horizon, 5): # Stride 5 for speed
            window = history[i : i + self.window_size]
            
            # Normalize Candidate
            candidate = self._normalize(window)
            
            # 3. Calculate Similarity (Cosine)
            # Dot product of normalized vectors
            score = np.dot(query, candidate) / (np.linalg.norm(query) * np.linalg.norm(candidate))
            
            if score > best_score:
                best_score = score
                # Calculate future outcome
                future_start_price = history[i + self.window_size - 1]
                future_end_price = history[i + self.window_size + self.future_horizon - 1]
                change_pct = ((future_end_price - future_start_price) / future_start_price) * 100
                
                best_match = FractalMatch(
                    match_index=i,
                    similarity=float(score),
                    future_change_pct=change_pct
                )
        
        # 4. Filter weak matches
        if best_match and best_match.similarity > 0.85: # %85 üzeri benzerlik
            logger.info(f"🧩 Fractal Found! Sim: {best_match.similarity:.2f}, Projecting: {best_match.future_change_pct:+.2f}%")
            return best_match
            
        return None
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-Max normalization to 0-1 range to compare shapes, not absolute prices"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

# Singleton
_analyzer = None
def get_fractal_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = FractalAnalyzer()
    return _analyzer
