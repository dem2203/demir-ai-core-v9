"""
Auto-Retrain Pipeline
Automatically retrains AI models on a weekly schedule with performance monitoring

Features:
- Weekly scheduled retraining
- Performance degradation detection
- Automatic rollback on poor performance
- Training metrics logging
- Model versioning
"""
import logging
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import shutil

logger = logging.getLogger("AUTO_RETRAIN")


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    model_name: str
    version: str
    timestamp: str
    sharpe_ratio: float
    total_return: float
    win_rate: float
    max_drawdown: float
    training_time_seconds: int
    total_timesteps: int
    status: str  # "success", "failed", "rolled_back"


@dataclass
class ModelPerformance:
    """Current model performance tracking."""
    model_name: str
    last_7d_sharpe: float
    last_30d_sharpe: float
    prediction_accuracy: float
    last_retrain: str
    version: str


class AutoRetrainPipeline:
    """
    Professional Auto-Retrain Pipeline for AI models.
    
    Features:
    - Weekly scheduled retraining
    - Performance monitoring
    - Automatic rollback
    - Model versioning
    """
    
    def __init__(
        self,
        models_dir: str = "src/brain/rl_agent/storage",
        metrics_file: str = "training_metrics.json",
        retrain_interval_days: int = 7,
        min_sharpe_threshold: float = 0.0,  # Don't deploy if Sharpe negative
        performance_drop_threshold: float = 0.3,  # Alert if 30% drop
        max_retrain_hours: int = 4,  # Max training time
        enable_auto_deploy: bool = False,  # Manual approval by default
        backup_count: int = 3  # Keep last 3 versions
    ):
        self.models_dir = models_dir
        self.metrics_file = os.path.join(models_dir, metrics_file)
        self.retrain_interval_days = retrain_interval_days
        self.min_sharpe_threshold = min_sharpe_threshold
        self.performance_drop_threshold = performance_drop_threshold
        self.max_retrain_hours = max_retrain_hours
        self.enable_auto_deploy = enable_auto_deploy
        self.backup_count = backup_count
        
        # Coins and their models
        self.model_configs = {
            "BTC/USDT": {"name": "ppo_btc", "current_version": "v5"},
            "ETH/USDT": {"name": "ppo_eth", "current_version": "v5"},
            "LTC/USDT": {"name": "ppo_ltc", "current_version": "v5"},
            "SOL/USDT": {"name": "ppo_sol", "current_version": "v5"}
        }
        
        # Performance history
        self.performance_history: Dict[str, List[TrainingMetrics]] = {}
        self._load_metrics()
        
        logger.info(f"AutoRetrainPipeline initialized: interval={retrain_interval_days}d, auto_deploy={enable_auto_deploy}")
    
    def check_retrain_needed(self) -> Dict[str, bool]:
        """Check which models need retraining based on schedule AND live performance."""
        from src.v10.performance_tracker import get_performance_tracker # Local import to avoid circular dependency
        
        needs_retrain = {}
        now = datetime.now()
        tracker = get_performance_tracker()
        report = tracker.get_report()
        
        # Check Live Performance (Real Feedback Loop)
        live_win_rate = report.win_rate
        # Filter completed only
        has_enough_data = report.completed_signals >= 5
        
        for symbol, config in self.model_configs.items():
            model_name = config["name"]
            
            # 1. Schedule Check
            last_train = self._get_last_training(model_name)
            
            if last_train:
                days_since = (now - datetime.fromisoformat(last_train.timestamp)).days
                schedule_trigger = days_since >= self.retrain_interval_days
            else:
                schedule_trigger = True  # Never trained
            
            # 2. Performance Check (Degradation in Training)
            degradation = self._check_performance_degradation(model_name)
            training_drop_trigger = degradation > self.performance_drop_threshold
            
            # 3. Live Trading Failure Check (Real Learning Loop)
            # If live win rate is below 40% (and we have sample size), trigger retrain!
            live_fail_trigger = has_enough_data and live_win_rate < 40.0
            
            if live_fail_trigger:
                logger.warning(f"🚨 {symbol} LIVE PERFORMANCE CRITICAL (WR: {live_win_rate:.1f}%), triggering EMERGENCY RETRAIN")
            
            needs_retrain[symbol] = schedule_trigger or training_drop_trigger or live_fail_trigger
            
            if needs_retrain[symbol]:
                reason = []
                if schedule_trigger: reason.append("schedule")
                if training_drop_trigger: reason.append("training_drop")
                if live_fail_trigger: reason.append("live_performance_fail")
                logger.info(f"Retrain needed for {symbol}: {', '.join(reason)}")
        
        return needs_retrain
    
    async def retrain_model(
        self,
        symbol: str,
        timesteps: int = 500000,
        num_candles: int = 43800  # 5 years
    ) -> Optional[TrainingMetrics]:
        """
        Retrain a single model.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timesteps: Training timesteps
            num_candles: Historical data to use
            
        Returns:
            TrainingMetrics if successful, None if failed
        """
        if symbol not in self.model_configs:
            logger.error(f"Unknown symbol: {symbol}")
            return None
        
        config = self.model_configs[symbol]
        model_name = config["name"]
        current_version = config["current_version"]
        
        # Calculate next version
        version_num = int(current_version[1:]) + 1
        new_version = f"v{version_num}"
        new_model_name = f"{model_name}_{new_version}"
        
        logger.info(f"Starting retrain: {symbol} -> {new_model_name}")
        start_time = datetime.now()
        
        try:
            # Backup current model
            self._backup_model(model_name, current_version)
            
            # Import trainer
            from src.brain.rl_agent.trainer import RLTrainer
            
            # Create trainer and run
            trainer = RLTrainer(symbol)
            sharpe, total_return = await trainer.train(
                total_timesteps=timesteps,
                save_name=new_model_name,
                num_candles=num_candles
            )
            
            training_time = (datetime.now() - start_time).seconds
            
            # Create metrics
            metrics = TrainingMetrics(
                model_name=model_name,
                version=new_version,
                timestamp=datetime.now().isoformat(),
                sharpe_ratio=sharpe,
                total_return=total_return,
                win_rate=0,  # Will be updated from evaluation
                max_drawdown=0,
                training_time_seconds=training_time,
                total_timesteps=timesteps,
                status="success"
            )
            
            # Check if new model is better
            if sharpe >= self.min_sharpe_threshold:
                if self.enable_auto_deploy:
                    self._deploy_model(model_name, new_version)
                    logger.info(f"Auto-deployed {new_model_name}")
                else:
                    logger.info(f"Training complete: {new_model_name} (Sharpe={sharpe:.2f}). Manual deployment required.")
            else:
                logger.warning(f"New model {new_model_name} has poor Sharpe ({sharpe:.2f}), not deploying")
                metrics.status = "not_deployed"
            
            # Save metrics
            self._save_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Retrain failed for {symbol}: {e}")
            
            # Rollback
            self._rollback_model(model_name, current_version)
            
            metrics = TrainingMetrics(
                model_name=model_name,
                version=new_version,
                timestamp=datetime.now().isoformat(),
                sharpe_ratio=0,
                total_return=0,
                win_rate=0,
                max_drawdown=0,
                training_time_seconds=(datetime.now() - start_time).seconds,
                total_timesteps=timesteps,
                status="failed"
            )
            self._save_metrics(metrics)
            
            return None
    
    async def retrain_all(self, force: bool = False) -> Dict[str, TrainingMetrics]:
        """
        Retrain all models that need updating.
        
        Args:
            force: Force retrain all models regardless of schedule
            
        Returns:
            Dict of symbol -> TrainingMetrics
        """
        results = {}
        
        if force:
            symbols_to_retrain = list(self.model_configs.keys())
        else:
            needs_retrain = self.check_retrain_needed()
            symbols_to_retrain = [s for s, need in needs_retrain.items() if need]
        
        if not symbols_to_retrain:
            logger.info("No models need retraining")
            return results
        
        logger.info(f"Retraining {len(symbols_to_retrain)} models: {symbols_to_retrain}")
        
        for symbol in symbols_to_retrain:
            metrics = await self.retrain_model(symbol)
            if metrics:
                results[symbol] = metrics
        
        return results
    
    def get_status(self) -> Dict:
        """Get current pipeline status."""
        needs_retrain = self.check_retrain_needed()
        
        model_status = {}
        for symbol, config in self.model_configs.items():
            model_name = config["name"]
            last_train = self._get_last_training(model_name)
            
            model_status[symbol] = {
                "current_version": config["current_version"],
                "last_retrain": last_train.timestamp if last_train else "Never",
                "needs_retrain": needs_retrain.get(symbol, True),
                "last_sharpe": last_train.sharpe_ratio if last_train else 0
            }
        
        return {
            "retrain_interval_days": self.retrain_interval_days,
            "auto_deploy_enabled": self.enable_auto_deploy,
            "models": model_status
        }
    
    def _backup_model(self, model_name: str, version: str):
        """Backup current model before retraining."""
        current_path = os.path.join(self.models_dir, f"{model_name}_{version}.zip")
        backup_path = os.path.join(self.models_dir, "backups", f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d')}.zip")
        
        if os.path.exists(current_path):
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(current_path, backup_path)
            logger.info(f"Backed up {current_path} to {backup_path}")
            
            # Clean old backups
            self._cleanup_old_backups(model_name)
    
    def _cleanup_old_backups(self, model_name: str):
        """Keep only the last N backups."""
        backup_dir = os.path.join(self.models_dir, "backups")
        if not os.path.exists(backup_dir):
            return
        
        # Find backups for this model
        backups = [f for f in os.listdir(backup_dir) if f.startswith(model_name)]
        backups.sort(reverse=True)  # Newest first
        
        # Remove old backups
        for old_backup in backups[self.backup_count:]:
            os.remove(os.path.join(backup_dir, old_backup))
            logger.info(f"Removed old backup: {old_backup}")
    
    def _rollback_model(self, model_name: str, version: str):
        """Rollback to previous version if retrain fails."""
        backup_dir = os.path.join(self.models_dir, "backups")
        current_path = os.path.join(self.models_dir, f"{model_name}_{version}.zip")
        
        # Find latest backup
        if os.path.exists(backup_dir):
            backups = [f for f in os.listdir(backup_dir) if f.startswith(model_name)]
            if backups:
                latest_backup = sorted(backups, reverse=True)[0]
                backup_path = os.path.join(backup_dir, latest_backup)
                shutil.copy2(backup_path, current_path)
                logger.info(f"Rolled back to {latest_backup}")
    
    def _deploy_model(self, model_name: str, new_version: str):
        """Update model version in config."""
        for symbol, config in self.model_configs.items():
            if config["name"] == model_name:
                config["current_version"] = new_version
                break
    
    def _get_last_training(self, model_name: str) -> Optional[TrainingMetrics]:
        """Get the last training metrics for a model."""
        if model_name not in self.performance_history:
            return None
        
        history = self.performance_history[model_name]
        if not history:
            return None
        
        return history[-1]
    
    def _check_performance_degradation(self, model_name: str) -> float:
        """Check if model performance has degraded."""
        if model_name not in self.performance_history:
            return 0.0
        
        history = self.performance_history[model_name]
        if len(history) < 2:
            return 0.0
        
        # Compare last training to previous
        current = history[-1].sharpe_ratio
        previous = history[-2].sharpe_ratio
        
        if previous <= 0:
            return 0.0
        
        return (previous - current) / previous
    
    def _load_metrics(self):
        """Load metrics from file."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    for model_name, metrics_list in data.items():
                        self.performance_history[model_name] = [
                            TrainingMetrics(**m) for m in metrics_list
                        ]
                logger.info(f"Loaded training metrics from {self.metrics_file}")
            except Exception as e:
                logger.warning(f"Could not load metrics: {e}")
    
    def _save_metrics(self, metrics: TrainingMetrics):
        """Save metrics to file."""
        if metrics.model_name not in self.performance_history:
            self.performance_history[metrics.model_name] = []
        
        self.performance_history[metrics.model_name].append(metrics)
        
        # Save to file
        try:
            data = {
                model_name: [asdict(m) for m in metrics_list]
                for model_name, metrics_list in self.performance_history.items()
            }
            
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved training metrics to {self.metrics_file}")
        except Exception as e:
            logger.warning(f"Could not save metrics: {e}")


# Convenience function for scheduled runs
async def run_weekly_retrain():
    """Run weekly retrain check and execution."""
    pipeline = AutoRetrainPipeline()
    status = pipeline.get_status()
    
    print("=== Auto-Retrain Pipeline Status ===")
    for symbol, info in status["models"].items():
        print(f"{symbol}: v{info['current_version']} | Last: {info['last_retrain']} | Needs: {info['needs_retrain']}")
    
    # Run retraining for models that need it
    results = await pipeline.retrain_all()
    
    print(f"\n=== Retrain Results ===")
    for symbol, metrics in results.items():
        print(f"{symbol}: {metrics.status} | Sharpe={metrics.sharpe_ratio:.2f}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_weekly_retrain())
