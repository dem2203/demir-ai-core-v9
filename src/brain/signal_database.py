# -*- coding: utf-8 -*-
"""
DEMIR AI - Signal Database (SQLite)
Sinyal geçmişi ve öğrenme için kalıcı veritabanı.

PHASE 117: Self-Learning Database
- Sinyal geçmişi kayıt
- İşlem sonuçları takip
- ML eğitim verisi
- Modül performans analizi
"""
import logging
import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger("SIGNAL_DB")


class SignalDatabase:
    """
    Sinyal Veritabanı (SQLite)
    
    Tüm sinyalleri ve sonuçları kalıcı olarak saklar.
    ML modeli bu verilerden öğrenir.
    """
    
    DB_FILE = "signals.db"
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or self.DB_FILE
        self._init_database()
        logger.info(f"✅ Signal Database initialized: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe connection."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """Veritabanı tablolarını oluştur."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sinyaller tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL,
                    tp_price REAL,
                    sl_price REAL,
                    modules_data TEXT,
                    status TEXT DEFAULT 'ACTIVE',
                    exit_price REAL,
                    exit_time TEXT,
                    result TEXT,
                    pnl_pct REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Modül performans tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS module_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_name TEXT UNIQUE NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    correct_signals INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0,
                    long_accuracy REAL DEFAULT 0,
                    short_accuracy REAL DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    last_updated TEXT
                )
            """)
            
            # Öğrenilmiş ağırlıklar tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_name TEXT UNIQUE NOT NULL,
                    base_weight REAL DEFAULT 0.05,
                    learned_weight REAL DEFAULT 0.05,
                    adjustment_reason TEXT,
                    last_updated TEXT
                )
            """)
            
            # Piyasa rejimleri tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    volatility REAL,
                    trend_strength REAL,
                    btc_price REAL
                )
            """)
            
            logger.info("📊 Database tables initialized")
    
    # ==================== SIGNAL OPERATIONS ====================
    
    def save_signal(self, signal: Dict) -> int:
        """Yeni sinyal kaydet."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO signals (
                    timestamp, symbol, direction, confidence,
                    entry_price, tp_price, sl_price, modules_data, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                signal.get('symbol', 'BTCUSDT'),
                signal.get('direction', 'NEUTRAL'),
                signal.get('confidence', 0),
                signal.get('entry_price', 0),
                signal.get('tp_price', 0),
                signal.get('sl_price', 0),
                json.dumps(signal.get('modules', [])),
                'ACTIVE'
            ))
            
            signal_id = cursor.lastrowid
            logger.info(f"📝 Signal saved: #{signal_id} {signal.get('direction')}")
            return signal_id
    
    def update_signal_result(self, signal_id: int, result: str, 
                            exit_price: float, pnl_pct: float):
        """Sinyal sonucunu güncelle."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE signals SET
                    status = 'CLOSED',
                    result = ?,
                    exit_price = ?,
                    exit_time = ?,
                    pnl_pct = ?
                WHERE id = ?
            """, (result, exit_price, datetime.now().isoformat(), pnl_pct, signal_id))
            
            logger.info(f"📊 Signal #{signal_id} updated: {result} ({pnl_pct:+.2f}%)")
            
            # Modül performansını güncelle
            self._update_module_performance(signal_id, result)
    
    def get_active_signals(self) -> List[Dict]:
        """Aktif sinyalleri al."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM signals WHERE status = 'ACTIVE'
                ORDER BY timestamp DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_signal_history(self, limit: int = 100) -> List[Dict]:
        """Sinyal geçmişini al."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM signals 
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """Genel istatistikleri al."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Toplam sinyaller
            cursor.execute("SELECT COUNT(*) FROM signals")
            total = cursor.fetchone()[0]
            
            # Kapalı sinyaller
            cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'CLOSED'")
            closed = cursor.fetchone()[0]
            
            # Kazançlı
            cursor.execute("SELECT COUNT(*) FROM signals WHERE result = 'WIN'")
            wins = cursor.fetchone()[0]
            
            # Kayıplı
            cursor.execute("SELECT COUNT(*) FROM signals WHERE result = 'LOSS'")
            losses = cursor.fetchone()[0]
            
            # Win rate
            win_rate = (wins / closed * 100) if closed > 0 else 0
            
            # Toplam PnL
            cursor.execute("SELECT SUM(pnl_pct) FROM signals WHERE status = 'CLOSED'")
            total_pnl = cursor.fetchone()[0] or 0
            
            # Son 7 gün
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT COUNT(*), SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END)
                FROM signals WHERE timestamp > ? AND status = 'CLOSED'
            """, (week_ago,))
            row = cursor.fetchone()
            weekly_total = row[0] or 0
            weekly_wins = row[1] or 0
            weekly_win_rate = (weekly_wins / weekly_total * 100) if weekly_total > 0 else 0
            
            return {
                'total_signals': total,
                'closed_signals': closed,
                'active_signals': total - closed,
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 1),
                'total_pnl_pct': round(total_pnl, 2),
                'weekly_signals': weekly_total,
                'weekly_win_rate': round(weekly_win_rate, 1),
                'learning_ready': closed >= 50  # 50+ sinyal = öğrenmeye hazır
            }
    
    # ==================== MODULE PERFORMANCE ====================
    
    def _update_module_performance(self, signal_id: int, result: str):
        """Sinyal sonucuna göre modül performansını güncelle."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sinyalin modül verilerini al
            cursor.execute("SELECT modules_data, direction FROM signals WHERE id = ?", (signal_id,))
            row = cursor.fetchone()
            if not row:
                return
            
            try:
                modules = json.loads(row['modules_data'] or '[]')
                signal_direction = row['direction']
            except:
                return
            
            is_win = result == 'WIN'
            
            for module in modules:
                module_name = module.get('name', module.get('module_name', ''))
                module_direction = module.get('direction', '')
                
                if not module_name:
                    continue
                
                # Modül sinyalin gerçek yönüyle aynı mıydı?
                correct = (module_direction == signal_direction) and is_win
                
                # Upsert
                cursor.execute("""
                    INSERT INTO module_performance (module_name, total_signals, correct_signals, last_updated)
                    VALUES (?, 1, ?, ?)
                    ON CONFLICT(module_name) DO UPDATE SET
                        total_signals = total_signals + 1,
                        correct_signals = correct_signals + ?,
                        accuracy = CAST(correct_signals + ? AS REAL) / (total_signals + 1) * 100,
                        last_updated = ?
                """, (
                    module_name,
                    1 if correct else 0,
                    datetime.now().isoformat(),
                    1 if correct else 0,
                    1 if correct else 0,
                    datetime.now().isoformat()
                ))
    
    def get_module_performance(self) -> List[Dict]:
        """Tüm modül performanslarını al."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM module_performance 
                ORDER BY accuracy DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_top_performing_modules(self, min_signals: int = 10) -> List[Dict]:
        """En iyi performans gösteren modülleri al."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM module_performance 
                WHERE total_signals >= ?
                ORDER BY accuracy DESC
                LIMIT 10
            """, (min_signals,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== LEARNED WEIGHTS ====================
    
    def save_learned_weights(self, weights: Dict[str, float], reason: str = ""):
        """Öğrenilmiş ağırlıkları kaydet."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for module_name, weight in weights.items():
                cursor.execute("""
                    INSERT INTO learned_weights (module_name, learned_weight, adjustment_reason, last_updated)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(module_name) DO UPDATE SET
                        learned_weight = ?,
                        adjustment_reason = ?,
                        last_updated = ?
                """, (
                    module_name, weight, reason, datetime.now().isoformat(),
                    weight, reason, datetime.now().isoformat()
                ))
            
            logger.info(f"💾 Saved {len(weights)} learned weights")
    
    def get_learned_weights(self) -> Dict[str, float]:
        """Öğrenilmiş ağırlıkları al."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT module_name, learned_weight FROM learned_weights")
            return {row['module_name']: row['learned_weight'] for row in cursor.fetchall()}
    
    # ==================== MARKET REGIME ====================
    
    def save_market_regime(self, regime: str, volatility: float, 
                          trend_strength: float, btc_price: float):
        """Piyasa rejimini kaydet."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO market_regimes (timestamp, regime, volatility, trend_strength, btc_price)
                VALUES (?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), regime, volatility, trend_strength, btc_price))
    
    def get_recent_regimes(self, hours: int = 24) -> List[Dict]:
        """Son X saatteki rejimleri al."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            cursor.execute("""
                SELECT * FROM market_regimes 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (since,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== TELEGRAM FORMAT ====================
    
    def format_stats_for_telegram(self) -> str:
        """Telegram formatında istatistikler."""
        stats = self.get_statistics()
        
        learning_emoji = "🧠" if stats['learning_ready'] else "📚"
        learning_text = "ÖĞRENMEYE HAZIR!" if stats['learning_ready'] else f"Öğrenme için {50 - stats['closed_signals']} sinyal daha gerekli"
        
        win_emoji = "✅" if stats['win_rate'] >= 55 else "⚠️" if stats['win_rate'] >= 45 else "❌"
        pnl_emoji = "📈" if stats['total_pnl_pct'] >= 0 else "📉"
        
        msg = f"""
🗄️ SİNYAL VERİTABANI
━━━━━━━━━━━━━━━━━━━━━━
📊 Toplam Sinyal: {stats['total_signals']}
📍 Aktif: {stats['active_signals']} | Kapalı: {stats['closed_signals']}
━━━━━━━━━━━━━━━━━━━━━━
{win_emoji} Kazanma Oranı: %{stats['win_rate']}
✅ Kazanç: {stats['wins']} | ❌ Kayıp: {stats['losses']}
{pnl_emoji} Toplam PnL: %{stats['total_pnl_pct']:+.2f}
━━━━━━━━━━━━━━━━━━━━━━
📅 Son 7 Gün: {stats['weekly_signals']} sinyal
📈 Haftalık Win Rate: %{stats['weekly_win_rate']}
━━━━━━━━━━━━━━━━━━━━━━
{learning_emoji} {learning_text}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_db = None

def get_signal_database() -> SignalDatabase:
    """Get or create signal database instance."""
    global _db
    if _db is None:
        _db = SignalDatabase()
    return _db
