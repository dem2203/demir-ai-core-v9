import pandas_datareader.data as web
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("MACRO_DATA_CONNECTOR")

class MacroConnector:
    """
    KÜRESEL PİYASA GÖZLEMCİSİ (STOOQ EDITION)
    Yahoo Finance engeline takılmamak için 'Stooq' veri tabanını kullanır.
    """
    
    def __init__(self):
        # Stooq Sembolleri (Wall Street Standartları)
        self.tickers = {
            "SPX": "^SPX",      # S&P 500
            "VIX": "^VIX",      # Volatility Index
            "DXY": "DXY",       # US Dollar Index (Stooq'ta bazen yoktur, yedeği FRED olabilir)
        }

    async def fetch_macro_data(self, period="2y", interval="1h"):
        """
        Makro verileri Stooq'tan çeker.
        """
        logger.info("Fetching Global Macro Data via Stooq...")
        
        # Stooq son 5 yılı otomatik verir
        start_date = datetime.now() - timedelta(days=730)
        
        master_df = pd.DataFrame()
        
        try:
            for name, ticker in self.tickers.items():
                try:
                    # Stooq'tan veri çekme (pandas_datareader)
                    # Not: Stooq sadece GÜNLÜK veri verir.
                    df = web.DataReader(ticker, 'stooq', start=start_date)
                    
                    if df.empty: continue

                    # Sadece Kapanış Fiyatını Al
                    df = df[['Close']].rename(columns={'Close': f'macro_{name}'})
                    
                    # Tarihe göre birleştir
                    if master_df.empty:
                        master_df = df
                    else:
                        master_df = master_df.join(df, how='outer')
                        
                except Exception as e:
                    logger.warning(f"Could not fetch {name} from Stooq: {e}")
            
            if master_df.empty:
                logger.error("No Macro Data fetched.")
                return None

            # 2. Veri Temizliği ve Saatliğe Çevirme (Resample)
            # Kripto 7/24 çalışır, Makro veriler günlük.
            # Günlük veriyi saatlik formata yayıyoruz (Forward Fill)
            master_df.index = pd.to_datetime(master_df.index)
            master_df = master_df.sort_index()
            
            # Günlük -> Saatlik (1H) Upsampling
            master_df = master_df.resample('1h').ffill()
            
            # Timestamp (Unix MS) ekle
            master_df['timestamp'] = master_df.index.astype('int64') // 10**6
            
            logger.info(f"Macro Data Fetched via Stooq. Shape: {master_df.shape}")
            return master_df
            
        except Exception as e:
            logger.error(f"Macro Connector Critical Fail: {e}")
            return None
