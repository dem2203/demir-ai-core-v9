import yfinance as yf
import pandas as pd
import logging

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
                # interval='1h' için max period ~730 gündür (2 yıl)
                data = yf.download(ticker, period=period, interval=interval, progress=False)
                
                if data.empty:
                    logger.warning(f"No data for {name}")
                    continue
                
                # Sadece kapanış fiyatını al ve yeniden adlandır
                # yfinance bazen MultiIndex döner, onu düzeltelim:
                if isinstance(data.columns, pd.MultiIndex):
                    df = data['Close'] # Eğer multiindex ise Close'u seç
                else:
                    df = data[['Close']] # Değilse DataFrame olarak al

                # Sütun ismini düzelt (Örn: macro_DXY)
                df.columns = [f'macro_{name}']
                
                # Saat dilimini kaldır (UTC uyumu için)
                df.index = df.index.tz_localize(None)
                
                if master_df.empty:
                    master_df = df
                else:
                    # Tarihe göre birleştir
                    master_df = master_df.join(df, how='outer')
            
            # Eksik verileri (Piyasa tatilleri vb.) doldur
            master_df = master_df.ffill().bfill()
            
            # Timestamp sütunu oluştur (Unix ms formatında)
            master_df['timestamp'] = master_df.index.astype('int64') // 10**6
            
            logger.info(f"Macro Data Fetched. Shape: {master_df.shape}")
            return master_df
            
        except Exception as e:
            logger.error(f"Macro Data Error: {e}")
            return None
