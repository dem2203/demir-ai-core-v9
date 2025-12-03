import pandas_datareader.data as web
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("MACRO_DATA_CONNECTOR")

class MacroConnector:
    """
    KÜRESEL PİYASA GÖZLEMCİSİ (FULL SPECTRUM v2)
    
    Kripto piyasasını etkileyen TÜM majör varlıkları izler.
    Veri Kaynağı: Stooq (Engelsiz)
    """
    
    def __init__(self):
        # Stooq Sembolleri
        self.tickers = {
            "SPX": "^SPX",      # S&P 500
            "NDQ": "^NDQ",      # Nasdaq
            "VIX": "^VIX",      # Korku Endeksi
            "DXY": "DXY",       # Dolar Endeksi
            "TNX": "^TNX",      # 10 Yıllık Tahvil
            "GOLD": "GC.F",     # Altın Futures (YENİ)
            "SILVER": "SI.F",   # Gümüş Futures (YENİ)
            "OIL": "CL.F"       # Ham Petrol (YENİ)
        }

    async def fetch_macro_data(self, period="2y", interval="1h"):
        logger.info("Fetching Global Macro Data via Stooq...")
        
        start_date = datetime.now() - timedelta(days=730)
        master_df = pd.DataFrame()
        
        try:
            for name, ticker in self.tickers.items():
                try:
                    # Stooq'tan günlük veri çek
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
                        
                except Exception:
                    pass 
            
            if master_df.empty:
                return None

            # Temizlik ve Saatliğe Çevirme
            master_df.index = pd.to_datetime(master_df.index)
            master_df = master_df.sort_index()
            master_df = master_df.ffill().bfill()
            
            # Günlük -> Saatlik
            master_df = master_df.resample('1h').ffill()
            
            # Timestamp Ekle
            master_df['timestamp'] = master_df.index.astype('int64') // 10**6
            
            logger.info(f"Expanded Macro Data Fetched. Shape: {master_df.shape}")
            return master_df
            
        except Exception as e:
            logger.error(f"Macro Connector Fail: {e}")
            return None
