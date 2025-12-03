import pandas_datareader.data as web
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("MACRO_DATA_CONNECTOR")

class MacroConnector:
    """
    KÜRESEL PİYASA GÖZLEMCİSİ (FULL SPECTRUM)
    
    Kripto piyasasını etkileyen 5 Temel Unsuru takip eder:
    1. DXY: Dolar Gücü (Ters Korelasyon)
    2. VIX: Korku Endeksi (Ters Korelasyon)
    3. SPX: Amerikan Borsası S&P 500 (Pozitif Korelasyon)
    4. NDQ: Teknoloji Borsası Nasdaq (Yüksek Pozitif Korelasyon)
    5. TNX: 10 Yıllık Tahvil Faizi (Ters Korelasyon - Risk İştahı)
    """
    
    def __init__(self):
        # Stooq Sembolleri
        self.tickers = {
            "SPX": "^SPX",      # S&P 500 (Risk İştahı)
            "NDQ": "^NDQ",      # Nasdaq (Teknoloji Hisseleri)
            "VIX": "^VIX",      # Volatilite/Korku
            "DXY": "DXY",       # Dolar Endeksi (Nakit Kraldır)
            "TNX": "^TNX",      # 10 Yıllık Tahvil Faizi (Paranın Maliyeti)
        }

    async def fetch_macro_data(self, period="2y", interval="1h"):
        """
        Tüm küresel verileri çeker ve birleştirir.
        """
        logger.info("Fetching Full Spectrum Global Macro Data...")
        
        start_date = datetime.now() - timedelta(days=730)
        master_df = pd.DataFrame()
        
        try:
            for name, ticker in self.tickers.items():
                try:
                    # Stooq'tan çek (Günlük Veri)
                    df = web.DataReader(ticker, 'stooq', start=start_date)
                    
                    if df.empty: 
                        logger.warning(f"Missing data for {name}")
                        continue

                    # Sadece Kapanış Fiyatını Al
                    df = df[['Close']].rename(columns={'Close': f'macro_{name}'})
                    
                    # Birleştir
                    if master_df.empty:
                        master_df = df
                    else:
                        master_df = master_df.join(df, how='outer')
                        
                except Exception as e:
                    logger.warning(f"Could not fetch {name}: {e}")
            
            if master_df.empty:
                logger.error("No Macro Data fetched.")
                return None

            # 2. Veri Temizliği
            master_df.index = pd.to_datetime(master_df.index)
            master_df = master_df.sort_index()
            
            # Eksikleri Doldur (Forward Fill -> Backward Fill)
            master_df = master_df.ffill().bfill()

            # 3. Saatliğe Çevir (Kripto ile eşleşmesi için)
            master_df = master_df.resample('1h').ffill()
            
            # Timestamp Ekle
            master_df['timestamp'] = master_df.index.astype('int64') // 10**6
            
            logger.info(f"Global Macro Data Ready. Shape: {master_df.shape}")
            return master_df
            
        except Exception as e:
            logger.error(f"Macro Connector Critical Fail: {e}")
            return None
