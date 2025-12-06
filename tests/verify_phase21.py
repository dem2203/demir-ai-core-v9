import unittest
import sys
import os
import asyncio
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.signal_filter import SignalFilter
from src.utils.notifications import NotificationManager
# For engine, we might need to mock more, but let's test components first.

class TestPhase21(unittest.TestCase):
    def setUp(self):
        self.signal_filter = SignalFilter(quality_threshold=0) 
        self.notifier = NotificationManager()
        # Mock telegram token to bypass early return
        self.notifier.telegram_token = "MOCK_TOKEN"
        self.notifier.telegram_chat_id = "MOCK_ID"
        
        # Mock telegram sending
        self.sent_messages = []
        self.notifier.send_message_raw = self.mock_send_message_raw
        
    async def mock_send_message_raw(self, text):
        self.sent_messages.append(text)
        
    def test_strict_filtering(self):
        """Test that signals with confidence < 85 are rejected."""
        print("\nTesting Strict Filtering...")
        
        # Snapshot mock
        snapshot = {
            'brain_state': {'htf_direction': 1},
            'whale_support': 0, 'whale_resistance': 0
        }
        
        # Low confidence signal
        low_conf_signal = {
            'symbol': 'BTC/USDT', 'side': 'BUY',
            'confidence': 80, 'quality': 'MODERATE',
            'entry_price': 100, 'tp_price': 110, 'sl_price': 90
        }
        
        # High confidence signal
        high_conf_signal = {
            'symbol': 'BTC/USDT', 'side': 'BUY',
            'confidence': 86, 'quality': 'STRONG',
            'entry_price': 100, 'tp_price': 110, 'sl_price': 90
        }
        
        # We need to test specific logic in SignalFilter.should_send_signal
        # NOTE: After we modify SignalFilter, this test is expected to pass.
        
        should_send_low, _, reason_low = self.signal_filter.should_send_signal(low_conf_signal, snapshot)
        print(f"Low Conf (80%) Result: {should_send_low}, Reason: {reason_low}")
        
        should_send_high, _, reason_high = self.signal_filter.should_send_signal(high_conf_signal, snapshot)
        print(f"High Conf (86%) Result: {should_send_high}, Reason: {reason_high}")
        
        self.assertFalse(should_send_low, "Signal with 80% confidence should be rejected")
        self.assertTrue(should_send_high, "Signal with 86% confidence should be accepted")

    def test_deduplication(self):
        """Test that duplicate signals are not sent."""
        print("\nTesting Deduplication...")
        
        signal = {
            'symbol': 'BTC/USDT', 'side': 'BUY',
            'confidence': 90, 'quality': 'STRONG',
            'entry_price': 100, 'tp_price': 110, 'sl_price': 90,
            'pattern': 'Test'
        }
        snapshot = {'brain_state': {}}
        
        # We need to run async methods
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Use a fresh notifier for this test to rely on its internal cache
        notifier = NotificationManager()
        notifier.telegram_token = "MOCK_TOKEN"
        notifier.telegram_chat_id = "MOCK_ID"
        notifier.send_message_raw = self.mock_send_message_raw
        notifier.signal_filter.should_send_signal = lambda s, sn: (True, 90, "OK") # Bypass filter level
        
        # First send
        loop.run_until_complete(notifier.send_signal(signal, snapshot))
        count_after_first = len(self.sent_messages)
        print(f"Sent messages after first call: {count_after_first}")
        
        # Second send (Duplicate)
        loop.run_until_complete(notifier.send_signal(signal, snapshot))
        count_after_second = len(self.sent_messages)
        print(f"Sent messages after second call: {count_after_second}")
        
        self.assertEqual(count_after_first, 1, "First signal should be sent")
        self.assertEqual(count_after_second, 1, "Duplicate signal should NOT be sent")

        # Different signal (should pass)
        diff_signal = signal.copy()
        diff_signal['symbol'] = 'ETH/USDT'
        loop.run_until_complete(notifier.send_signal(diff_signal, snapshot))
        print(f"Sent messages after different signal: {len(self.sent_messages)}")
        self.assertEqual(len(self.sent_messages), 2, "Different signal should be sent")
        loop.close()

if __name__ == '__main__':
    unittest.main()
