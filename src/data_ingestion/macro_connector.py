import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("MACRO_DATA_CONNECTOR")

class MacroConnector:
    """
    KÜRESEL PİYASA GÖZLEMCİSİ
    Bitcoin'i etkileyen dış faktörleri (DXY, SPX, GOLD, VIX) çeker.
    """
    
    def __init__(self):
        # Takip edilecek küresel semboller
        self.tickers = {
            "DXY": "DX-Y.NYB",  # Dolar Endeksi (Ters Korelasyon)
            "SPX": "^GSPC",     # S&P 500 (Pozitif Korelasyon)
            "VIX": "^VIX",      # Korku Endeksi (Risk Algısı)
            "GOLD": "GC=F"      # Altın
        }

    async def fetch_macro_data(self, period="2y", interval="1h"):
        """Tüm makro verileri çeker ve birleştirir."""
        logger.info("Fetching Global Macro Economics Data...")
        
        master_df = pd.DataFrame()
        
        try:
            for name, ticker in self.tickers.items():
                # Yahoo Finance'den veri çek
                data = yf.download(ticker, period=period, interval=interval, progress=False)
                
                if data.empty:
                    logger.warning(f"No data for {name}")
                    continue
                
                # Sadece kapanış fiyatını al ve yeniden adlandır
                df = data[['Close']].rename(columns={'Close': f'macro_{name}'})
                
                # Saat dilimini kaldır (UTC uyumu için)
                df.index = df.index.tz_localize(None)
                
                if master_df.empty:
                    master_df = df
                else:
                    # Tarihe göre birleştir
                    master_df = master_df.join(df, how='outer')
            
            # Eksik verileri (Piyasa tatilleri vb.) doldur
            master_df = master_df.ffill().bfill()
            
            # Timestamp sütunu oluştur
            master_df['timestamp'] = master_df.index.astype('int64') // 10**6
            
            logger.info(f"Macro Data Fetched. Shape: {master_df.shape}")
            return master_df
            
        except Exception as e:
            logger.error(f"Macro Data Error: {e}")
            return None
