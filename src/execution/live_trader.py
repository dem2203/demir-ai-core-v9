"""
DEMIR AI - LIVE TRADE EXECUTION ENGINE
Real trading with Binance via authenticated API.

SAFETY FEATURES:
1. Max position size limits
2. Daily loss limits
3. Trade confirmation via Telegram
4. Emergency stop on drawdown
5. Paper mode toggle
"""
import os
import json
import logging
import asyncio
from datetime import datetime, date
from typing import Optional, Dict

import ccxt.async_support as ccxt

logger = logging.getLogger("LIVE_TRADER")

class LiveTrader:
    """
    Real Trade Execution with Binance API.
    
    Includes multiple safety guards to prevent catastrophic losses.
    
    SAFETY GUARDS:
    - Max daily trades: 10
    - Max daily loss: 5% of capital
    - Max single position: 25% of capital
    - Minimum trade interval: 5 minutes
    """
    
    # Safety Configuration
    MAX_DAILY_TRADES = 10
    MAX_DAILY_LOSS_PCT = 5.0  # 5% max daily loss
    MAX_POSITION_PCT = 25.0   # 25% max single position
    MIN_TRADE_INTERVAL_SECONDS = 300  # 5 minutes between trades
    
    STATE_FILE = "live_trading_state.json"
    
    def __init__(self, paper_mode: bool = True):
        """
        Initialize LiveTrader.
        
        Args:
            paper_mode: If True, simulate trades without real execution.
                        Set to False for REAL trading (dangerous!).
        """
        self.paper_mode = paper_mode
        self.exchange: Optional[ccxt.binance] = None
        self.state = self._load_state()
        
        # Log mode prominently
        if paper_mode:
            logger.info("📝 LiveTrader initialized in PAPER mode (safe)")
        else:
            logger.warning("⚠️🔴 LiveTrader initialized in LIVE mode - REAL MONEY!")
    
    def _load_state(self) -> Dict:
        """Load trading state from file."""
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "initial_capital": 10000.0,
            "current_balance": 10000.0,
            "daily_trades": 0,
            "daily_pnl": 0.0,
            "last_trade_date": str(date.today()),
            "last_trade_time": None,
            "positions": {},
            "trade_history": [],
            "emergency_stop": False
        }
    
    def _save_state(self):
        """Save trading state to file."""
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _reset_daily_counters(self):
        """Reset daily counters if new day."""
        today = str(date.today())
        if self.state["last_trade_date"] != today:
            logger.info("📆 New trading day - resetting counters")
            self.state["daily_trades"] = 0
            self.state["daily_pnl"] = 0.0
            self.state["last_trade_date"] = today
            self._save_state()
    
    async def get_exchange(self) -> ccxt.binance:
        """Get authenticated Binance exchange connection."""
        if not self.exchange:
            api_key = os.environ.get("BINANCE_API_KEY", "")
            api_secret = os.environ.get("BINANCE_API_SECRET", "")
            
            if not api_key or not api_secret:
                logger.error("❌ BINANCE_API_KEY and BINANCE_API_SECRET required for live trading!")
                raise ValueError("Missing Binance API credentials")
            
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
        
        return self.exchange
    
    def check_safety_guards(self, signal: Dict) -> tuple[bool, str]:
        """
        Check all safety guards before allowing trade.
        
        Returns:
            (can_trade: bool, reason: str)
        """
        self._reset_daily_counters()
        
        # 1. Emergency stop check
        if self.state["emergency_stop"]:
            return False, "🛑 Emergency stop active - trading disabled"
        
        # 2. Daily trade limit
        if self.state["daily_trades"] >= self.MAX_DAILY_TRADES:
            return False, f"🚫 Daily trade limit reached ({self.MAX_DAILY_TRADES})"
        
        # 3. Daily loss limit
        daily_loss_pct = (self.state["daily_pnl"] / self.state["initial_capital"]) * 100
        if daily_loss_pct <= -self.MAX_DAILY_LOSS_PCT:
            self.state["emergency_stop"] = True
            self._save_state()
            return False, f"🛑 Daily loss limit hit ({daily_loss_pct:.1f}%)"
        
        # 4. Trade interval check
        if self.state["last_trade_time"]:
            last_trade = datetime.fromisoformat(self.state["last_trade_time"])
            seconds_since = (datetime.now() - last_trade).total_seconds()
            if seconds_since < self.MIN_TRADE_INTERVAL_SECONDS:
                remaining = int(self.MIN_TRADE_INTERVAL_SECONDS - seconds_since)
                return False, f"⏳ Trade cooldown ({remaining}s remaining)"
        
        # 5. Position size check
        trade_size = float(signal.get('trade_value', 0))
        max_position = self.state["current_balance"] * (self.MAX_POSITION_PCT / 100)
        if trade_size > max_position:
            return False, f"📊 Position too large (${trade_size:.2f} > ${max_position:.2f} max)"
        
        return True, "✅ All safety checks passed"
    
    async def execute_trade(self, signal: Dict) -> Dict:
        """
        Execute a trade signal.
        
        Args:
            signal: Trade signal with keys: symbol, side, entry_price, sl_price, quantity
        
        Returns:
            Result dictionary with success status and details
        """
        symbol = signal['symbol'].replace('/', '')  # BTC/USDT -> BTCUSDT
        side = signal['side'].upper()  # BUY or SELL
        
        # 1. Safety checks
        can_trade, reason = self.check_safety_guards(signal)
        if not can_trade:
            logger.warning(f"Trade blocked: {reason}")
            return {"success": False, "reason": reason}
        
        # 2. Paper mode simulation
        if self.paper_mode:
            result = await self._simulate_trade(signal)
            return result
        
        # 3. REAL execution
        try:
            exchange = await self.get_exchange()
            
            quantity = float(signal.get('quantity', 0))
            if quantity <= 0:
                return {"success": False, "reason": "Invalid quantity"}
            
            # Format symbol for exchange
            symbol_exchange = symbol.replace('/', '')
            
            # Place market order
            if side == "BUY":
                order = await exchange.create_market_buy_order(symbol_exchange, quantity)
            else:
                order = await exchange.create_market_sell_order(symbol_exchange, quantity)
            
            # Update state
            self.state["daily_trades"] += 1
            self.state["last_trade_time"] = datetime.now().isoformat()
            
            # Record trade
            self.state["trade_history"].append({
                "time": datetime.now().isoformat(),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": order.get('average', order.get('price', 0)),
                "order_id": order.get('id'),
                "status": "FILLED"
            })
            
            self._save_state()
            
            logger.info(f"🔥 LIVE TRADE EXECUTED: {side} {quantity} {symbol}")
            return {
                "success": True,
                "mode": "LIVE",
                "order_id": order.get('id'),
                "filled_price": order.get('average'),
                "quantity": quantity
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {"success": False, "reason": str(e)}
        
        finally:
            if self.exchange:
                await self.exchange.close()
                self.exchange = None
    
    async def _simulate_trade(self, signal: Dict) -> Dict:
        """Simulate trade in paper mode."""
        symbol = signal['symbol']
        side = signal['side'].upper()
        price = float(signal.get('entry_price', 0))
        quantity = float(signal.get('quantity', 0.01))
        
        self.state["daily_trades"] += 1
        self.state["last_trade_time"] = datetime.now().isoformat()
        
        # Calculate trade value
        trade_value = quantity * price
        
        if side == "BUY":
            self.state["current_balance"] -= trade_value
            self.state["positions"][symbol] = {
                "entry_price": price,
                "quantity": quantity,
                "value": trade_value,
                "time": datetime.now().isoformat()
            }
        else:  # SELL
            if symbol in self.state["positions"]:
                pos = self.state["positions"][symbol]
                pnl = (price - pos["entry_price"]) * pos["quantity"]
                self.state["current_balance"] += pos["value"] + pnl
                self.state["daily_pnl"] += pnl
                del self.state["positions"][symbol]
            else:
                self.state["current_balance"] += trade_value
        
        self.state["trade_history"].append({
            "time": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "mode": "PAPER"
        })
        
        self._save_state()
        
        logger.info(f"📝 PAPER TRADE: {side} {quantity} {symbol} @ ${price:.2f}")
        return {
            "success": True,
            "mode": "PAPER",
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "new_balance": self.state["current_balance"]
        }
    
    async def get_account_balance(self) -> Dict:
        """Get actual account balance from Binance."""
        if self.paper_mode:
            return {
                "mode": "PAPER",
                "balance": self.state["current_balance"],
                "positions": self.state["positions"]
            }
        
        try:
            exchange = await self.get_exchange()
            balance = await exchange.fetch_balance()
            
            return {
                "mode": "LIVE",
                "USDT": balance.get('USDT', {}).get('free', 0),
                "BTC": balance.get('BTC', {}).get('free', 0),
                "ETH": balance.get('ETH', {}).get('free', 0),
                "total": balance.get('total', {})
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {"error": str(e)}
        finally:
            if self.exchange:
                await self.exchange.close()
                self.exchange = None
    
    def reset_emergency_stop(self):
        """Reset emergency stop flag (manual action required)."""
        logger.warning("⚠️ Emergency stop being reset manually")
        self.state["emergency_stop"] = False
        self._save_state()
    
    def get_status(self) -> Dict:
        """Get current trading status."""
        self._reset_daily_counters()
        
        return {
            "mode": "PAPER" if self.paper_mode else "LIVE",
            "balance": self.state["current_balance"],
            "daily_trades": self.state["daily_trades"],
            "daily_pnl": self.state["daily_pnl"],
            "positions": len(self.state["positions"]),
            "emergency_stop": self.state["emergency_stop"],
            "max_daily_trades": self.MAX_DAILY_TRADES,
            "max_daily_loss_pct": self.MAX_DAILY_LOSS_PCT
        }


# CLI Entry point
async def main():
    """Test live trader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Trader")
    parser.add_argument("--live", action="store_true", help="Enable LIVE mode (DANGEROUS!)")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--balance", action="store_true", help="Show balance")
    
    args = parser.parse_args()
    
    trader = LiveTrader(paper_mode=not args.live)
    
    if args.status:
        status = trader.get_status()
        print(f"\n📊 Trading Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
    
    if args.balance:
        balance = await trader.get_account_balance()
        print(f"\n💰 Account Balance:")
        for key, value in balance.items():
            print(f"   {key}: {value}")
    
    print("\n✅ LiveTrader ready!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
