import os
import time
import json
import logging
import pandas as pd
import numpy as np
import ccxt
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import random

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GeneticOptimizer")

# --- CONFIG & CONSTANTS ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# --- DATA INTEGRITY CHECKER (SIMULATED IMPORT BASED ON REPORT) ---
class RealDataVerifier:
    """
    Raporlanan Data Integrity kuralına uygun doğrulayıcı.
    Mock veriyi, hatalı OHLCV yapısını reddeder.
    """
    @staticmethod
    def verify_candle_integrity(df: pd.DataFrame) -> bool:
        if df.empty:
            logger.error("Data Integrity Fail: DataFrame is empty.")
            return False
        
        # Check High >= Low
        if not (df['high'] >= df['low']).all():
            logger.error("Data Integrity Fail: Found High < Low in real data.")
            return False
            
        # Check No NaNs in critical columns
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            logger.warning("Data Integrity Warning: NaNs found. Filling with ffill.")
            df.fillna(method='ffill', inplace=True)
            
        return True

# --- STRATEGY ENGINE (THE CORE LOGIC TO OPTIMIZE) ---
class StrategyEngine:
    """
    Bu sınıf, verilen parametre setiyle (genom) gerçek veriyi işler
    ve al-sat simülasyonu yapar.
    """
    def __init__(self, data: pd.DataFrame, initial_balance: float = 1000.0):
        self.data = data.copy()
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position = None # None or 'LONG'
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []

    def calculate_indicators(self, params: Dict[str, Any]):
        """
        Optimize edilecek indikatörleri hesaplar.
        """
        df = self.data
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=int(params['rsi_period'])).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=int(params['rsi_period'])).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=int(params['macd_fast']), adjust=False).mean()
        exp2 = df['close'].ewm(span=int(params['macd_slow']), adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=int(params['macd_signal']), adjust=False).mean()

        # Bollinger Bands
        df['sma'] = df['close'].rolling(window=int(params['bb_period'])).mean()
        df['std'] = df['close'].rolling(window=int(params['bb_period'])).std()
        df['upper_bb'] = df['sma'] + (df['std'] * params['bb_std'])
        df['lower_bb'] = df['sma'] - (df['std'] * params['bb_std'])

        self.data = df.dropna()

    def run_backtest(self, params: Dict[str, Any]) -> float:
        """
        Verilen parametrelerle backtest çalıştırır ve Fitness Skorunu (Net Kar) döner.
        """
        self.calculate_indicators(params)
        
        balance = self.initial_balance
        btc_amount = 0.0
        
        # Basit bir strateji mantığı (Raporlanan matematik motoruna benzer)
        # RSI < Alt Sınır ve MACD > Sinyal -> AL
        # RSI > Üst Sınır veya Stop Loss -> SAT
        
        for index, row in self.data.iterrows():
            current_price = row['close']
            
            # ALIM KOŞULU
            if self.position is None:
                if row['rsi'] < params['rsi_lower'] and row['macd'] > row['signal']:
                    # %98 bakiye ile al (komisyon payı)
                    amount_to_spend = balance * 0.98
                    btc_amount = amount_to_spend / current_price
                    balance -= amount_to_spend
                    self.position = 'LONG'
                    self.entry_price = current_price
                    self.trades.append({'type': 'BUY', 'price': current_price, 'time': index})
            
            # SATIM KOŞULU
            elif self.position == 'LONG':
                # Kar/Zarar hesabı
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                
                # Stop Loss veya Take Profit veya Teknik Satış
                if (pnl_pct < -params['stop_loss']) or \
                   (pnl_pct > params['take_profit']) or \
                   (row['rsi'] > params['rsi_upper']):
                    
                    sale_value = btc_amount * current_price
                    balance += sale_value
                    btc_amount = 0
                    self.position = None
                    self.trades.append({'type': 'SELL', 'price': current_price, 'time': index, 'pnl': pnl_pct})

            # Equity kaydı
            current_val = balance + (btc_amount * current_price)
            self.equity_curve.append(current_val)

        # Son durumda maldaysak satıp nakite dönelim
        if self.position == 'LONG':
            balance += btc_amount * self.data.iloc[-1]['close']
        
        net_profit = ((balance - self.initial_balance) / self.initial_balance) * 100
        
        # Fitness Fonksiyonu: Sadece Kar değil, İşlem sayısı da önemli (Aşırı işlemden kaçın)
        trade_count = len([t for t in self.trades if t['type'] == 'SELL'])
        if trade_count < 3: # Çok az işlem yapan stratejiyi cezalandır
            return -50.0
            
        return net_profit

# --- GENETIC ALGORITHM OPTIMIZER ---
class GeneticOptimizer:
    """
    Gerçek veriler üzerinde Genetik Algoritma çalıştırarak en iyi parametreleri bulur.
    Zero-Mock Policy: Parametre aralıkları matematiksel limitlere göredir.
    """
    def __init__(self, symbol: str = 'BTC/USDT', timeframe: str = '4h', population_size: int = 20):
        self.symbol = symbol
        self.timeframe = timeframe
        self.population_size = population_size
        self.exchange = None
        self.raw_data = None
        
        # Genom Tanımı (Optimize edilecek parametreler ve aralıkları)
        self.gene_space = {
            'rsi_period': (10, 24),    # Int
            'rsi_lower': (20, 40),     # Int
            'rsi_upper': (60, 80),     # Int
            'macd_fast': (8, 16),      # Int
            'macd_slow': (20, 30),     # Int
            'macd_signal': (5, 15),    # Int
            'bb_period': (15, 25),     # Int
            'bb_std': (1.5, 2.5),      # Float
            'stop_loss': (0.01, 0.10), # Float %
            'take_profit': (0.02, 0.20)# Float %
        }
        
        self.connect_exchange()

    def connect_exchange(self):
        """Gerçek Binance bağlantısı."""
        if not BINANCE_API_KEY:
            raise ValueError("BINANCE_API_KEY not found in environment variables.")
        
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True
        })

    def fetch_real_data(self, limit: int = 500):
        """Binance'den gerçek OHLCV verisi çeker."""
        logger.info(f"Fetching real data for {self.symbol}...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            verifier = RealDataVerifier()
            if verifier.verify_candle_integrity(df):
                self.raw_data = df
                logger.info(f"Data fetched successfully: {len(df)} candles.")
            else:
                raise ValueError("Data verification failed.")
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def create_individual(self) -> Dict[str, Any]:
        """Rastgele bir genom (parametre seti) oluşturur."""
        individual = {}
        for key, (min_val, max_val) in self.gene_space.items():
            if isinstance(min_val, int):
                individual[key] = random.randint(min_val, max_val)
            else:
                individual[key] = round(random.uniform(min_val, max_val), 2)
        return individual

    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutasyon: Bir geni rastgele değiştirir."""
        gene_to_mutate = random.choice(list(self.gene_space.keys()))
        min_val, max_val = self.gene_space[gene_to_mutate]
        
        if isinstance(min_val, int):
            individual[gene_to_mutate] = random.randint(min_val, max_val)
        else:
            individual[gene_to_mutate] = round(random.uniform(min_val, max_val), 2)
        return individual

    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Çaprazlama: İki ebeveynden yeni bir çocuk üretir."""
        child = {}
        for key in self.gene_space.keys():
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child

    def optimize(self, generations: int = 5, callback=None) -> Tuple[Dict, float, List]:
        """
        Ana Optimizasyon Döngüsü.
        """
        if self.raw_data is None:
            self.fetch_real_data()

        # 1. Popülasyonu Başlat
        population = [self.create_individual() for _ in range(self.population_size)]
        best_solution = None
        best_fitness = -9999.0
        history = []

        logger.info(f"Starting optimization: {generations} generations, {self.population_size} population.")

        for gen in range(generations):
            gen_scores = []
            
            # 2. Her birey için Fitness hesapla
            ranked_population = []
            for ind in population:
                engine = StrategyEngine(self.raw_data)
                fitness = engine.run_backtest(ind)
                ranked_population.append((fitness, ind))
                gen_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = ind
            
            # Sırala (En yüksek fitness en üstte)
            ranked_population.sort(key=lambda x: x[0], reverse=True)
            
            # Log ve Callback (UI update için)
            avg_score = sum(gen_scores) / len(gen_scores)
            log_msg = f"Gen {gen+1}/{generations} | Best: {ranked_population[0][0]:.2f}% | Avg: {avg_score:.2f}%"
            logger.info(log_msg)
            
            history_entry = {
                "generation": gen + 1,
                "best_fitness": ranked_population[0][0],
                "avg_fitness": avg_score
            }
            history.append(history_entry)
            
            if callback:
                callback(gen + 1, generations, best_fitness, avg_score)

            # 3. Seçilim (İlk %20'yi koru - Elitizm)
            elite_count = int(self.population_size * 0.2)
            next_generation = [ind for _, ind in ranked_population[:elite_count]]

            # 4. Çaprazlama ve Mutasyon ile yeni bireyler üret
            while len(next_generation) < self.population_size:
                parent1 = random.choice(ranked_population[:elite_count])[1]
                parent2 = random.choice(ranked_population[:elite_count])[1] # Basit turnuva yerine elit havuzdan seçim
                
                child = self.crossover(parent1, parent2)
                
                if random.random() < 0.3: # %30 Mutasyon şansı
                    child = self.mutate(child)
                
                next_generation.append(child)
            
            population = next_generation

        return best_solution, best_fitness, history

if __name__ == "__main__":
    # Test bloğu - Railway ortamında doğrudan çalıştırıldığında test eder
    print("DEMIR AI Optimizer Modülü Başlatılıyor...")
    opt = GeneticOptimizer(symbol='ETH/USDT', timeframe='1h', population_size=10)
    best_params, score, _ = opt.optimize(generations=3)
    print(f"\nOPTIMİZASYON TAMAMLANDI.\nEn İyi Skor: %{score:.2f}\nParametreler: {json.dumps(best_params, indent=2)}")
