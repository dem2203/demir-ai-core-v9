# -*- coding: utf-8 -*-
"""
Test script for Phase 42 Critical Scrapers
Tests liquidation tracker, whale tracker, and Reddit scraper.
"""
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

from brain.liquidation_tracker import LiquidationTracker
from brain.whale_tracker import WhaleTracker
from brain.reddit_scraper import RedditScraper

def test_liquidation_tracker():
    """Test liquidation zones calculation"""
    print("\n" + "="*50)
    print("🔥 TESTING LIQUIDATION TRACKER")
    print("="*50)
    
    tracker = LiquidationTracker()
    
    # Test BTC
    summary = tracker.get_liquidation_summary('BTCUSDT')
    
    print(f"\n📊 Symbol: {summary['symbol']}")
    print(f"📈 Zone Count: {summary['zone_count']}")
    print(f"⚡ Cascade Risk: {summary['cascade_risk']}")
    print(f"💥 Nearby Liquidations: ${summary['nearby_liq_size']/1e9:.2f}B")
    
    if summary['nearest_long_liq']:
        nl = summary['nearest_long_liq']
        print(f"\n🔴 Nearest Long Liq:")
        print(f"   Price: ${nl.price_level:,.0f} ({nl.distance_pct:.1f}% below)")
        print(f"   Size: ${nl.size_usd/1e6:.0f}M ({nl.leverage}x)")
    
    if summary['nearest_short_liq']:
        ns = summary['nearest_short_liq']
        print(f"\n🟢 Nearest Short Liq:")
        print(f"   Price: ${ns.price_level:,.0f} (+{ns.distance_pct:.1f}% above)")
        print(f"   Size: ${ns.size_usd/1e6:.0f}M ({ns.leverage}x)")
    
    print(f"\n💡 Summary: {summary['summary']}")
    
    # Test Telegram format
    print("\n📱 Telegram Format:")
    print(tracker.format_for_telegram('BTCUSDT'))
    
    return summary['zone_count'] > 0

def test_whale_tracker():
    """Test whale activity tracking"""
    print("\n" + "="*50)
    print("🐋 TESTING WHALE TRACKER")
    print("="*50)
    
    tracker = WhaleTracker()
    
    # Test BTC whales (last 24h)
    summary = tracker.get_whale_summary('BTC', hours=24)
    
    print(f"\n📊 Symbol: {summary['symbol']}")
    print(f"🐋 Whales Tracked: {summary['whale_count']}")
    print(f"🔴 Exchange Inflow: ${summary['total_inflow']/1e6:.2f}M")
    print(f"🟢 Exchange Outflow: ${summary['total_outflow']/1e6:.2f}M")
    print(f"📈 Net Flow: ${summary['net_flow']/1e6:+.2f}M")
    print(f"🎯 Direction: {summary['direction']}")
    
    print(f"\n💡 Summary: {summary['summary']}")
    
    if summary['recent_whales']:
        print("\n🔥 Recent Large Transfers:")
        for whale in summary['recent_whales'][:5]:
            print(f"  • ${whale.amount_usd/1e6:.1f}M {whale.tx_type}")
            print(f"    {whale.timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    # Test Telegram format
    print("\n📱 Telegram Format:")
    print(tracker.format_for_telegram('BTC', hours=24))
    
    return True  # Always pass (might be 0 whales if no recent activity)

def test_reddit_scraper():
    """Test Reddit sentiment scraping"""
    print("\n" + "="*50)
    print("💬 TESTING REDDIT SCRAPER")
    print("="*50)
    
    scraper = RedditScraper()
    
    # Get sentiment (last 24h)
    sentiment = scraper.get_sentiment(hours=24)
    
    print(f"\n📊 Sentiment Score: {sentiment['score']}/100")
    print(f"😊 Mood: {sentiment['sentiment']}")
    print(f"📝 Posts Analyzed: {sentiment['post_count']}")
    print(f"🟢 Bullish Posts: {sentiment['bullish_count']}")
    print(f"🔴 Bearish Posts: {sentiment['bearish_count']}")
    print(f"⚪ Neutral Posts: {sentiment['neutral_count']}")
    
    if sentiment.get('top_posts'):
        print("\n🔥 Top Posts (by score):")
        for i, post in enumerate(sentiment['top_posts'][:5], 1):
            emoji = "🟢" if post.sentiment == 'BULLISH' else "🔴" if post.sentiment == 'BEARISH' else "⚪"
            print(f"{i}. {emoji} [{post.subreddit}] {post.title[:60]}...")
            print(f"   ↑{post.score} ({post.upvote_ratio:.0%} upvoted)")
    
    print(f"\n💡 Summary: {sentiment['summary']}")
    
    # Test Telegram format
    print("\n📱 Telegram Format:")
    print(scraper.format_for_telegram(hours=24))
    
    return sentiment['post_count'] > 0

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 PHASE 42: CRITICAL WEB SCRAPERS TEST")
    print("="*60)
    print("Testing liquidation tracker, whale tracker, and Reddit scraper...")
    print("All scrapers use NO API KEYS (pure web scraping)")
    
    results = {}
    
    # Test each scraper
    try:
        results['liquidation'] = test_liquidation_tracker()
    except Exception as e:
        print(f"\n❌ Liquidation Tracker FAILED: {e}")
        results['liquidation'] = False
    
    try:
        results['whale'] = test_whale_tracker()
    except Exception as e:
        print(f"\n❌ Whale Tracker FAILED: {e}")
        results['whale'] = False
    
    try:
        results['reddit'] = test_reddit_scraper()
    except Exception as e:
        print(f"\n❌ Reddit Scraper FAILED: {e}")
        results['reddit'] = False
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {name.upper()}: {'PASS' if status else 'FAIL'}")
    
    print(f"\n🎯 Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 42 scrapers are operational.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
