
class Translator:
    """
    Kripto finansal terimlerini Türkçe açıklamalarıyla eşleştirir.
    Format: "Term (Açıklama)"
    """
    
    DICTIONARY = {
        # Yönler / Trendler
        "BULLISH": "Yükseliş (Boğa)",
        "BEARISH": "Düşüş (Ayı)",
        "NEUTRAL": "Nötr (Yatay)",
        "STRONG_BULLISH": "Güçlü Yükseliş",
        "STRONG_BEARISH": "Güçlü Düşüş",
        "UPTREND": "Yükseliş Trendi",
        "DOWNTREND": "Düşüş Trendi",
        "SIDEWAYS": "Yatay Piyasa",
        "RANGING": "Kanal Hareketi",
        
        # Piyasa Rejimleri
        "TRENDING_BULL": "Yükseliş Trendi (Boğa)",
        "TRENDING_BEAR": "Düşüş Trendi (Ayı)",
        "SCALPER_PARADISE": "Scalp Uygun (Hızlı Al-Sat)",
        "ACCUMULATION": "Toplama Evresi (Akümülasyon)",
        "DISTRIBUTION": "Dağıtım Evresi (Satış)",
        "EXPANSION": "Genişleme (Fiyat Kopuşu)",
        
        # Teknik Terimler
        "RSI": "RSI (Aşırı Alım/Satım Göstergesi)",
        "MACD": "MACD (Trend Takipçisi)",
        "Bollinger Bands": "Bollinger (Volatilite Bandı)",
        "Volume": "Hacim",
        "Open Interest": "Açık Pozisyonlar (Vadeli)",
        "Funding Rate": "Fonlama Oranı (Long/Short Dengesi)",
        "Order Book": "Emir Defteri",
        "Imbalance": "Dengesizlik (Alıcı/Satıcı Baskısı)",
        "Liquidation": "Likidasyon (Zorunlu Kapanış)",
        "Support": "Destek Seviyesi",
        "Resistance": "Direnç Seviyesi",
        "Pivot": "Dönüş Noktası",
        "Fibonacci": "Fibonacci (Altın Oran Seviyeleri)",
        "Divergence": "Uyumsuzluk (Fiyat/Osilatör Farkı)",
        
        # Formasyonlar
        "Double Bottom": "İkili Dip (Dönüş)",
        "Double Top": "İkili Tepe (Dönüş)",
        "Head & Shoulders": "Omuz Baş Omuz (Düşüş)",
        "Inv. Head & Shoulders": "Ters OBO (Yükseliş)",
        "Bull Flag": "Boğa Bayrağı (Devam)",
        "Bear Flag": "Ayı Bayrağı (Devam)",
        "Triangle": "Üçgen Sıkışması",
        "Wedge": "Kama Formasyonu",
        "Doji": "Doji (Kararsızlık Mumu)",
        "Engulfing": "Yutan Mum (Dönüş)",
        "Hammer": "Çekiç (Dip Dönüş)",
        
        
        # Dashboard Metrics (Dashboard Metrikleri)
        "Price": "Fiyat",
        "Confidence": "Güven",
        "Kelly Risk (%)": "Kelly Riski (Pozisyon Büyüklüğü)",
        "Fractal Score": "Fraktal Skoru (Zaman Dilimi Uyumu)",
        "Hurst Exp": "Hurst Üssü (Trend Gücü)",
        "Whale Support": "Balina Desteği (Büyük Alım Emri)",
        "Whale Resistance": "Balina Direnci (Büyük Satım Emri)",
        "OB Imbalance": "Emir Dengesi (Alıcı/Satıcı Farkı)",
        "Wyckoff": "Wyckoff Fazı (Akümülasyon/Dağıtım)",
        "Bias": "Eğilim (Yön)",
        "On-Chain": "Zincir Üstü (Blockchain Analizi)",
        "Strategy": "Strateji",
        
        # Portfolio & Trading (Portföy & İşlemler)
        "Total Equity": "Toplam Sermaye",
        "Cash Balance": "Nakit Bakiye",
        "Entry Price": "Giriş Fiyatı",
        "Current Price": "Güncel Fiyat",
        "Unrealized P&L": "Gerçekleşmemiş Kar/Zarar",
        "Side": "Yön",
        "Size": "Miktar",
        
        # Backtest & Performance (Geçmiş Test & Performans)
        "ROI": "Yatırım Getirisi",
        "Win Rate": "Kazanma Oranı",
        "Total Trades": "Toplam İşlem",
        
        # Visual Cortex
        "Visual Score": "Görsel Skor",
        "Visual Trend": "Görsel Trend",
        "Consensus Score": "Uzlaşma Skoru",
        "Consensus Trend": "Uzlaşma Trendi",
        "Agreement": "Uyum",
        
        # General
        "PnL": "Kar/Zarar",
        "Consensus": "Ortak Karar"
    }
    
    @staticmethod
    def t(text):
        """
        Metin içindeki terimleri bulur ve Türkçe açıklamasını parantez içine ekler.
        Örnek: "STRONG_BULLISH" -> "STRONG_BULLISH (Güçlü Yükseliş)"
        """
        if not isinstance(text, str): return text
        
        # Tam eşleşme kontrolü (Önce bunu dene)
        if text in Translator.DICTIONARY:
            return f"{text} ({Translator.DICTIONARY[text].split('(')[-1][:-1]})"
            
        # Parçalı eşleşme (Cümle içindeki kelimeleri tarama - Basitçe replace)
        # Not: Bu biraz riskli olabilir, ama Dashboard'da genelde kısa keywordler var.
        # Şimdilik sadece tam eşleşme veya anahtar kelime lookup yapalım.
        
        return text

    @staticmethod
    def get_desc(key):
        """Direkt açıklamayı döndürür"""
        val = Translator.DICTIONARY.get(key, key)
        if "(" in val:
            return val.split("(")[1].replace(")", "")
        return val
