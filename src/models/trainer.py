# -*- coding: utf-8 -*-
"""
DEMIR AI v11 - MODEL TRAINER
=============================
Profesyonel ML model eğitimi ve walk-forward validation.

ÖZELLİKLER:
- LightGBM (hızlı, güçlü, memory-efficient)
- Walk-forward validation (geçmişte tutarlı mı?)
- Feature importance analizi
- Model versiyonlama

KURAL: Model sadece BACKTEST ile kanıtlandıktan sonra canlıya alınır!

Author: DEMIR AI Team
Date: 2026-01-03
"""
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("MODEL_TRAINER")

# Model kayıt dizini
MODELS_DIR = Path("data/models")


@dataclass
class TrainingResult:
    """Eğitim sonucu."""
    model_name: str
    version: str
    timestamp: str
    train_accuracy: float
    val_accuracy: float
    feature_count: int
    sample_count: int
    best_features: List[str]
    walk_forward_results: Dict


@dataclass
class BacktestResult:
    """Backtest sonucu."""
    total_trades: int
    win_rate: float
    avg_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float


class QuantModelTrainer:
    """
    Profesyonel ML model eğitici.
    
    LightGBM kullanır çünkü:
    - Hızlı eğitim
    - Düşük memory kullanımı
    - Categorical feature desteği
    - Feature importance
    """
    
    def __init__(self, model_name: str = "quant_model"):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.model = None
        self.feature_columns = []
        self.version = "v1"
    
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "label_4h",
        test_size: float = 0.2,
        n_estimators: int = 500,
        max_depth: int = 7,
        learning_rate: float = 0.05
    ) -> TrainingResult:
        """
        Model eğit.
        
        Args:
            df: Feature'ları içeren DataFrame
            target_col: Hedef kolon (label_4h, label_1h, vs.)
            test_size: Validation için ayrılacak oran
            n_estimators: Ağaç sayısı
            max_depth: Max derinlik
            learning_rate: Öğrenme hızı
            
        Returns:
            TrainingResult
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            logger.error("LightGBM not installed! Run: pip install lightgbm")
            raise
        
        logger.info(f"🧠 Training model: {self.model_name}")
        
        # Feature seçimi (hedef ve timestamp hariç)
        exclude_cols = ['timestamp', 'label_1h', 'label_4h', 'label_4h_triple', 
                       'future_return_60', 'future_return_240']
        self.feature_columns = [c for c in df.columns if c not in exclude_cols]
        
        # NaN içeren satırları temizle
        df_clean = df.dropna(subset=[target_col] + self.feature_columns)
        
        if len(df_clean) < 1000:
            raise ValueError(f"Not enough data: {len(df_clean)} samples (need 1000+)")
        
        X = df_clean[self.feature_columns]
        y = df_clean[target_col]
        
        # Time-series split (son %20 validation)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"  Training: {len(X_train)} samples | Validation: {len(X_val)} samples")
        
        # Model oluştur ve eğit
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lambda env: None]  # Suppress output
        )
        
        # Accuracy hesapla
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_acc = (train_pred == y_train).mean()
        val_acc = (val_pred == y_val).mean()
        
        logger.info(f"  Train Accuracy: {train_acc:.2%}")
        logger.info(f"  Validation Accuracy: {val_acc:.2%}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        best_features = importance.head(20)['feature'].tolist()
        logger.info(f"  Top 5 Features: {importance.head(5)['feature'].tolist()}")
        
        # Walk-forward validation
        wf_results = self._walk_forward_validation(df_clean, target_col, n_splits=5)
        
        # Versiyon güncelle
        self.version = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Sonuç
        result = TrainingResult(
            model_name=self.model_name,
            version=self.version,
            timestamp=datetime.now().isoformat(),
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            feature_count=len(self.feature_columns),
            sample_count=len(df_clean),
            best_features=best_features,
            walk_forward_results=wf_results
        )
        
        # Kaydet
        self._save_model(result)
        
        return result
    
    def _walk_forward_validation(
        self,
        df: pd.DataFrame,
        target_col: str,
        n_splits: int = 5
    ) -> Dict:
        """
        Walk-forward validation.
        
        Her split için:
        - Önceki veriyle eğit
        - Sonraki dönemi test et
        
        Tutarlı performans = Güvenilir model
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return {"error": "LightGBM not installed"}
        
        logger.info(f"📊 Running {n_splits}-fold walk-forward validation...")
        
        results = []
        split_size = len(df) // (n_splits + 1)
        
        for i in range(n_splits):
            # Train: 0 → (i+1) * split_size
            # Test: (i+1) * split_size → (i+2) * split_size
            train_end = (i + 1) * split_size
            test_end = min((i + 2) * split_size, len(df))
            
            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]
            
            if len(train_df) < 500 or len(test_df) < 100:
                continue
            
            X_train = train_df[self.feature_columns]
            y_train = train_df[target_col]
            X_test = test_df[self.feature_columns]
            y_test = test_df[target_col]
            
            # Temp model
            temp_model = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                verbose=-1
            )
            temp_model.fit(X_train, y_train)
            
            # Test accuracy
            test_pred = temp_model.predict(X_test)
            accuracy = (test_pred == y_test).mean()
            
            results.append({
                "fold": i + 1,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "accuracy": accuracy
            })
            
            logger.info(f"  Fold {i+1}: {accuracy:.2%}")
        
        # Ortalama accuracy
        if results:
            avg_accuracy = np.mean([r["accuracy"] for r in results])
            std_accuracy = np.std([r["accuracy"] for r in results])
            
            logger.info(f"  Average: {avg_accuracy:.2%} (±{std_accuracy:.2%})")
            
            return {
                "folds": results,
                "avg_accuracy": avg_accuracy,
                "std_accuracy": std_accuracy,
                "is_consistent": std_accuracy < 0.05  # %5'ten az varyans
            }
        
        return {"error": "Not enough data for walk-forward"}
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tahmin yap.
        
        Returns:
            (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained!")
        
        # Eksik feature'ları kontrol et
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        X = features[self.feature_columns]
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def backtest(
        self,
        df: pd.DataFrame,
        target_col: str = "label_4h",
        initial_capital: float = 10000,
        position_size: float = 0.1,  # %10 pozisyon
        min_confidence: float = 0.60
    ) -> BacktestResult:
        """
        Backtest yap.
        
        Basit strateji:
        - Confidence > 0.60 → İşlem aç
        - 4 saat sonra kapat
        """
        if self.model is None:
            raise ValueError("Model not trained!")
        
        logger.info("📈 Running backtest...")
        
        # Predictions
        X = df[self.feature_columns]
        predictions, probabilities = self.predict(X)
        
        # Trade simulation
        capital = initial_capital
        trades = []
        peak_capital = capital
        max_drawdown = 0
        
        for i in range(len(df) - 240):  # 4 saat sonrasını bilmeliyiz
            confidence = max(probabilities[i])
            
            if confidence < min_confidence:
                continue
            
            predicted_class = predictions[i]
            actual_return = (df['close'].iloc[i + 240] - df['close'].iloc[i]) / df['close'].iloc[i]
            
            # Trade
            if predicted_class == 1:  # LONG
                pnl = actual_return * capital * position_size
            else:  # SHORT  
                pnl = -actual_return * capital * position_size
            
            capital += pnl
            trades.append({
                "pnl": pnl,
                "return": actual_return,
                "predicted": predicted_class,
                "confidence": confidence
            })
            
            # Drawdown
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
        
        if not trades:
            return BacktestResult(0, 0, 0, 0, 0, 0)
        
        # Metrics
        pnls = [t["pnl"] for t in trades]
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / len(trades)
        avg_return = np.mean(pnls)
        
        # Sharpe (günlük normalize edilmiş)
        if np.std(pnls) > 0:
            sharpe = np.sqrt(252) * np.mean(pnls) / np.std(pnls)
        else:
            sharpe = 0
        
        # Profit factor
        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        result = BacktestResult(
            total_trades=len(trades),
            win_rate=win_rate,
            avg_return=avg_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor
        )
        
        logger.info(f"  Trades: {result.total_trades}")
        logger.info(f"  Win Rate: {result.win_rate:.2%}")
        logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"  Sharpe: {result.sharpe_ratio:.2f}")
        
        return result
    
    def _save_model(self, result: TrainingResult):
        """Model ve metadata kaydet."""
        model_path = MODELS_DIR / f"{self.model_name}_{self.version}.joblib"
        meta_path = MODELS_DIR / f"{self.model_name}_{self.version}_meta.json"
        
        # Model kaydet
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'version': self.version
        }, model_path)
        
        # Metadata kaydet
        with open(meta_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        logger.info(f"💾 Model saved: {model_path}")
    
    def load_model(self, version: Optional[str] = None):
        """Kayıtlı modeli yükle."""
        if version:
            model_path = MODELS_DIR / f"{self.model_name}_{version}.joblib"
        else:
            # Son sürümü bul
            models = list(MODELS_DIR.glob(f"{self.model_name}_*.joblib"))
            if not models:
                raise FileNotFoundError(f"No model found for {self.model_name}")
            model_path = sorted(models)[-1]
        
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.version = data['version']
        
        logger.info(f"📂 Model loaded: {model_path}")


# Singleton
_trainer: Optional[QuantModelTrainer] = None


def get_model_trainer(name: str = "quant_model") -> QuantModelTrainer:
    """Get or create trainer singleton."""
    global _trainer
    if _trainer is None or _trainer.model_name != name:
        _trainer = QuantModelTrainer(name)
    return _trainer


# CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("""
    Usage:
    
    1. Prepare data:
       python src/data_pipeline/collector.py
       
    2. Generate features:
       python src/features/technical.py
       
    3. Train model:
       from src.models.trainer import get_model_trainer
       trainer = get_model_trainer()
       result = trainer.train(df)
       
    4. Backtest:
       backtest = trainer.backtest(df)
    """)
