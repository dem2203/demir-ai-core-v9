# -*- coding: utf-8 -*-
"""
Dashboard Helpers Module
=========================
Centralized data fetching for Dashboard System Modules.
Ensures reliable, real-time data from direct API sources.
"""
import requests
import json
import logging
import re
import platform
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger("DASH_HELPERS")

# ==========================================
# 1. WEB INTELLIGENCE (Fear & Greed, News)
# ==========================================

def fetch_fear_and_greed() -> Dict[str, Any]:
    """Fetch Fear & Greed Index from alternative.me API."""
    try:
        resp = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
        if resp.status_code == 200:
            data = resp.json()['data'][0]
            return {
                'value': int(data['value']),
                'classification': data['value_classification'],
                'timestamp': int(data['timestamp'])
            }
    except Exception as e:
        logger.error(f"F&G fetch error: {e}")
    return {'value': 50, 'classification': 'Neutral', 'timestamp': 0}

def fetch_crypto_news(limit: int = 10) -> List[Dict[str, str]]:
    """Fetch latest crypto news from Google News RSS."""
    news_items = []
    try:
        url = "https://news.google.com/rss/search?q=cryptocurrency+trading&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall('.//item')[:limit]:
                title = item.find('title').text
                link = item.find('link').text
                pubDate = item.find('pubDate').text
                
                # Cleanup title (remove source)
                clean_title = title.split(' - ')[0] if ' - ' in title else title
                
                news_items.append({
                    'title': clean_title,
                    'link': link,
                    'published': pubDate,
                    'source': 'Google News'
                })
    except Exception as e:
        logger.error(f"News fetch error: {e}")
        # Fallback to simple static list if completely fails
        news_items.append({'title': 'Live news stream unavailable', 'link': '#', 'published': '', 'source': 'System'})
    return news_items

# ==========================================
# 2. AI PREDICTIONS (Detailed Technical Analysis)
# ==========================================

def fetch_detailed_ta(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """Calculate detailed technical indicators from verified Binance data."""
    result = {
        'rsi': 50, 'rsi_status': 'Neutral',
        'macd': 0, 'macd_signal': 0, 'macd_hist': 0, 'macd_status': 'Neutral',
        'ema_fast': 0, 'ema_slow': 0, 'trend': 'Neutral',
        'bollinger_upper': 0, 'bollinger_lower': 0, 'bollinger_signal': 'In Range',
        'pivot_point': 0, 'r1': 0, 's1': 0
    }
    
    try:
        # Fetch 100 candles (1h)
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": 100}
        resp = requests.get(url, params=params, timeout=5)
        
        if resp.status_code == 200:
            klines = resp.json()
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            current = closes[-1]
            
            # --- RSI (14) ---
            if len(closes) > 14:
                gains, losses = [], []
                for i in range(1, len(closes)):
                    delta = closes[i] - closes[i-1]
                    gains.append(max(delta, 0))
                    losses.append(abs(min(delta, 0)))
                
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                result['rsi'] = rsi
                result['rsi_status'] = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"

            # --- MACD (12, 26, 9) ---
            if len(closes) > 26:
                ema12 = _calc_ema(closes, 12)[-1]
                ema26 = _calc_ema(closes, 26)[-1]
                macd_line = ema12 - ema26
                # Signal line needs historic MACD values, approximates here for simplicity or needs full array calculation
                # Simplified for dashboard display:
                result['macd'] = macd_line
                result['macd_status'] = "Bullish" if macd_line > 0 else "Bearish"

            # --- EMA Trend (9, 21) ---
            if len(closes) > 21:
                ema9 = _calc_ema(closes, 9)[-1]
                ema21 = _calc_ema(closes, 21)[-1]
                result['ema_fast'] = ema9
                result['ema_slow'] = ema21
                result['trend'] = "UP" if current > ema9 > ema21 else "DOWN" if current < ema9 < ema21 else "Choppy"

            # --- Bollinger Bands (20, 2) ---
            if len(closes) > 20:
                sma20 = sum(closes[-20:]) / 20
                variance = sum((x - sma20) ** 2 for x in closes[-20:]) / 20
                std_dev = variance ** 0.5
                upper = sma20 + (2 * std_dev)
                lower = sma20 - (2 * std_dev)
                
                result['bollinger_upper'] = upper
                result['bollinger_lower'] = lower
                
                if current > upper: result['bollinger_signal'] = "Overbought (Band Top)"
                elif current < lower: result['bollinger_signal'] = "Oversold (Band Bottom)"
                
    except Exception as e:
        logger.error(f"TA calc error: {e}")
        
    return result

def _calc_ema(data, period):
    if len(data) < period: return data
    c = 2.0 / (period + 1)
    current_ema = sum(data[:period]) / period
    ema_values = [current_ema]
    for value in data[period:]:
        current_ema = (c * value) + ((1 - c) * current_ema)
        ema_values.append(current_ema)
    return ema_values

# ==========================================
# 3. NEURAL BRAIN MONITOR (System Stats)
# ==========================================

def fetch_system_health() -> Dict[str, Any]:
    """Fetch real system metrics using psutil."""
    stats = {
        'cpu_usage': 0, 'ram_usage': 0, 'disk_usage': 0,
        'boot_time': 'Unknown', 'os_info': platform.system() + " " + platform.release()
    }
    
    try:
        import psutil
        stats['cpu_usage'] = psutil.cpu_percent(interval=None)
        stats['ram_usage'] = psutil.virtual_memory().percent
        stats['disk_usage'] = psutil.disk_usage('/').percent
        bt = datetime.fromtimestamp(psutil.boot_time())
        stats['boot_time'] = bt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.warning(f"System stats error: {e}")
        stats['cpu_usage'] = 0
        stats['ram_usage'] = 0
        
    return stats

def get_engine_status() -> Dict[str, str]:
    """Read latest engine status from json."""
    status = {'status': 'UNKNOWN', 'last_update': 'Never'}
    try:
        if platform.system() == "Windows":
             # Local dev - maybe read log file or json
             pass
             
        # Try reading dashboard_data for proof of life
        try:
            with open("dashboard_data.json", 'r') as f:
                data = json.load(f)
                # If we can read it, engine is likely alive writing to it
                status['status'] = 'RUNNING'
                # Check BTC timestamp
                if 'BTCUSDT' in data:
                    status['last_update'] = data['BTCUSDT'].get('timestamp', 'N/A')
        except:
            status['status'] = 'INITIALIZING'
            
    except Exception:
        pass
    return status


# ==========================================
# 4. THINKING BRAIN STATUS (NEW!)
# ==========================================

def fetch_thinking_brain_stats() -> Dict[str, Any]:
    """Fetch Thinking Brain status for dashboard."""
    stats = {
        'status': 'INACTIVE',
        'weights': {'rl': 35, 'claude': 30, 'rules': 35},
        'performance': {'rl': 0, 'claude': 0, 'rules': 0},
        'history_count': 0,
        'rl_agent_loaded': False,
        'current_regime': 'UNKNOWN'
    }
    
    try:
        from src.brain.thinking_brain import get_thinking_brain
        brain = get_thinking_brain()
        
        stats['status'] = 'ACTIVE'
        
        # Weights (as percentages)
        stats['weights'] = {
            'rl': int(brain._weights['rl'] * 100),
            'claude': int(brain._weights['claude'] * 100),
            'rules': int(brain._weights['rules'] * 100)
        }
        
        # Performance (win rates)
        perf = brain._performance_by_source
        for source in ['rl', 'claude', 'rules']:
            if perf[source]['total'] > 0:
                stats['performance'][source] = int(
                    perf[source]['wins'] / perf[source]['total'] * 100
                )
        
        # History
        stats['history_count'] = len(brain._decision_history)
        
        # RL Agent
        stats['rl_agent_loaded'] = bool(brain._rl_agent and brain._rl_agent.model)
        
        # Current regime from last decision
        if brain._decision_history:
            stats['current_regime'] = brain._decision_history[-1].get('regime', 'UNKNOWN')
        
    except Exception as e:
        logger.debug(f"TB stats error: {e}")
    
    return stats

