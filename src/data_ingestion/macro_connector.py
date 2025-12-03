import pandas_datareader.data as web
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("MACRO_DATA_CONNECTOR")

class MacroConnector:
    """
    KÜRESEL PİYASA GÖZLEMCİSİ (PRO STOOQ EDITION)
    
    Yahoo Finance engeline takılmamak için 'Stooq' ve 'FRED' veri tabanlarını kullanır.
    Günlük makro veriyi çeker ve saatlik formata uyarlar (Regime Detection).
    """
    
    def __init__(self):
        # Stooq Sembolleri (Profesyonel Kodlar)
        self.tickers = {
            "SPX": "^SPX",      # S&P 500 Index
            "VIX": "^VIX",      # Volatility Index
            "DXY": "DXY",       # US Dollar Index (Genelde Stooq'ta DXY olarak geçer)
            # Eğer Stooq'ta DXY bulamazsa, FRED'den çekecek yedekleme eklenebilir.
        }

    async def fetch_macro_data(self, period="2y", interval="1h"):
        """
        Makro verileri çeker.
        Not: Stooq sadece Günlük (Daily) veri verir.
        Biz bunu alıp saatlik veriye 'Forward Fill' yöntemiyle yayacağız.
        Bu, 'Bugünkü piyasa rejimi nedir?' sorusunun cevabıdır.
        """
        logger.info("Fetching Global Macro Data via Stooq...")
        
        # Stooq son 5 yılı otomatik verir
        start_date = datetime.now() - timedelta(days=730)
        
        master_df = pd.DataFrame()
        
        try:
            # 1. Stooq üzerinden verileri çek (Toplu veya tek tek)
            # pandas_datareader senkrondur, bu yüzden direkt çağırıyoruz.
            for name, ticker in self.tickers.items():
                try:
                    # Stooq'tan veri çekme
                    df = web.DataReader(ticker, 'stooq', start=start_date)
                    
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
                logger.error("No Macro Data fetched from Stooq.")
                return None

            # 2. Veri Temizliği ve Düzenleme
            master_df.index = pd.to_datetime(master_df.index)
            master_df = master_df.sort_index() # Eskiden yeniye sırala
            
            # Eksik günleri doldur (Hafta sonları vs.)
            master_df = master_df.ffill().bfill()

            # 3. GÜNLÜK VERİYİ SAATLİĞE ÇEVİRME (Upsampling)
            # Botumuz saatlik çalıştığı için, günlük makro veriyi saatlere yaymalıyız.
            # Örn: Pazartesi DXY 104 ise, Pazartesi saat 01:00, 02:00... hepsinde 104 kabul edilir.
            master_df = master_df.resample('1h').ffill()
            
            # Timestamp (Unix MS) ekle - Füzyon için gerekli
            master_df['timestamp'] = master_df.index.astype('int64') // 10**6
            
            # Son 2 yılın verisini al (veya istenen period kadar)
            # master_df = master_df.tail(24 * 365 * 2) 
            
            logger.info(f"Macro Data Fetched & Resampled. Shape: {master_df.shape}")
            return master_df
            
        except Exception as e:
            logger.error(f"Macro Connector Critical Fail: {e}")
            return None
