#!/usr/bin/env python3
import time
import sys
import os
import datetime
import feedparser
from config import FEED_URLS, SINGLE_RUN_MODE, RECOMMENDATIONS_DIR, BACKTEST_DIR, PARALLELISM, MIN_OVERALL_SCORE, MODEL_THINKING, logger, BATCH_SIZE, ENABLE_BACKTESTING
from utils import load_processed_articles, save_processed_articles, load_daily_stats, save_daily_stats, update_watchlist_performance, is_recent
from data import fetch_full_text
from analysis import batch_generic_check, batch_analyze_headlines, get_ticker_symbol, analyze_company_context, is_valid_ticker
from trade import evaluate_trade_opportunity, save_enhanced_recommendation, generate_trade_confidence
from backtest import update_backtests, simulate_trade_performance
from api import check_market_events
from concurrent.futures import ThreadPoolExecutor, as_completed


def debug_feeds():
    """Test parsing each feed to identify issues"""
    logger.info("Debug mode: Testing feed parsing...")
    
    for feed_url in FEED_URLS:
        try:
            logger.info(f"Testing feed: {feed_url}")
            feed = feedparser.parse(feed_url, timeout=15)
            
            # Log feed status
            if hasattr(feed, 'status'):
                logger.info(f"  Status: {feed.status}")
            
            # Log feed bozo (error flag)
            if hasattr(feed, 'bozo') and feed.bozo:
                logger.error(f"  Feed parse error: {feed.get('bozo_exception', 'Unknown error')}")
            
            # Log entry count
            logger.info(f"  Entries: {len(feed.entries)}")
            
            # Test first entry if available
            if feed.entries:
                entry = feed.entries[0]
                logger.info(f"  First entry title: {entry.get('title', 'No title')}")
                logger.info(f"  First entry date fields: {[k for k in entry.keys() if 'date' in k.lower() or 'time' in k.lower() or 'pub' in k.lower()]}")
                
                if 'pubDate' in entry:
                    logger.info(f"  pubDate: {entry.pubDate}")
                
                # Test is_recent function
                is_rec = is_recent(entry)
                logger.info(f"  is_recent result: {is_rec}")
            else:
                logger.warning(f"  No entries found in feed")
        
        except Exception as e:
            logger.error(f"Error testing feed {feed_url}: {e}", exc_info=True)
    
    logger.info("Feed testing complete")

# Removed local ENABLE_BACKTESTING definition - use the one from config.py instead
def check_feeds_availability():
    """Check if the RSS feeds are available and contain entries"""
    import requests
    from bs4 import BeautifulSoup
    
    logger.info("Checking feed availability...")
    
    for feed_url in FEED_URLS:
        try:
            # First, check if the URL is accessible
            response = requests.get(feed_url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Feed URL returned status code {response.status_code}: {feed_url}")
                continue
                
            # Check the content
            content_length = len(response.content)
            logger.info(f"Feed URL accessible, content length: {content_length} bytes: {feed_url}")
            
            # Parse the raw content (bypass feedparser for testing)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all(['item', 'entry'])
            
            if items:
                logger.info(f"Found {len(items)} raw items in feed: {feed_url}")
                # Print a sample title if available
                for item in items[:1]:
                    title_tag = item.find(['title'])
                    if title_tag:
                        logger.info(f"Sample title: {title_tag.text}")
            else:
                logger.warning(f"No items found in raw feed content: {feed_url}")
                
            # Try with feedparser
            parsed = feedparser.parse(feed_url)
            
            # Debug the entire feedparser output for the first feed
            if feed_url == FEED_URLS[0]:
                import pprint
                with open("feed_debug.txt", "w") as f:
                    f.write(pprint.pformat(parsed))
                logger.info(f"Wrote complete debug info for first feed to feed_debug.txt")
                
            if parsed.entries:
                logger.info(f"Feedparser found {len(parsed.entries)} entries in feed: {feed_url}")
            else:
                logger.warning(f"Feedparser found no entries in feed: {feed_url}")
                
        except Exception as e:
            logger.error(f"Error checking feed {feed_url}: {e}")

def process_feed(processed_articles, daily_stats, run_timestamp):
    today_str = datetime.date.today().isoformat()
    batch_items = []
    for feed_url in FEED_URLS:
        logger.info(f"Processing feed: {feed_url}")
        try:
            # Add timeout to feedparser call
            feed = feedparser.parse(feed_url, timeout=10)
            
            # Check for feedparser errors
            if hasattr(feed, 'bozo') and feed.bozo:
                logger.error(f"Feed parse error: {feed.get('bozo_exception', 'Unknown error')}")
            
            if not feed.entries:
                logger.warning(f"No entries found in feed: {feed_url}")
                # Log feed structure to help diagnose
                logger.debug(f"Feed structure: {feed.keys()}")
                if 'feed' in feed:
                    logger.debug(f"Feed title: {feed.feed.get('title', 'No title')}")
                continue
                
            logger.info(f"Found {len(feed.entries)} entries in feed: {feed_url}")
            new_articles_found = False
            
            for entry in feed.entries:
                # Log entry structure for debugging
                if feed.entries.index(entry) == 0:
                    logger.debug(f"First entry keys: {entry.keys()}")
                
                entry_title = entry.get("title", "No title")
                logger.debug(f"Checking entry: {entry_title[:50]}...")
                
                if not is_recent(entry):
                    logger.debug(f"Entry not recent: {entry_title[:50]}...")
                    continue
                    
                logger.debug(f"Entry is recent: {entry_title[:50]}...")
                
                entry_id = entry.get("id", entry.get("link", entry.get("title")))
                if entry_id in processed_articles:
                    logger.debug(f"Entry already processed: {entry_title[:50]}...")
                    continue
                    
                processed_articles.add(entry_id)
                headline = entry.get("title", "").strip()
                summary = entry.get("summary", "").strip()
                full_text = fetch_full_text(entry)
                
                if headline:
                    batch_items.append({"headline": headline, "summary": summary, "full_text": full_text})
                    new_articles_found = True
                    logger.info(f"Added to batch: {headline[:50]}...")
                
                if len(batch_items) >= BATCH_SIZE:
                    process_batch(batch_items, today_str, run_timestamp, processed_articles, daily_stats)
                    batch_items = []
            
            if batch_items:
                process_batch(batch_items, today_str, run_timestamp, processed_articles, daily_stats)
                batch_items = []
                
            save_processed_articles(processed_articles)
            save_daily_stats(daily_stats)
            logger.info(f"{'New articles found' if new_articles_found else 'No new recent articles'} in {feed_url}")
        except Exception as e:
            logger.error(f"Error processing feed {feed_url}: {e}", exc_info=True)

def process_batch(items, today_str, run_timestamp, processed_articles, daily_stats):
    """Process a batch of news items"""
    if today_str not in daily_stats:
        daily_stats[today_str] = {"total": 0, "generic": 0, "failed": 0}
    
    daily_stats[today_str]["total"] += len(items)
    headlines = [item["headline"] for item in items]
    
    # First step: Filter out generic headlines
    generic_classification = batch_generic_check(headlines)
    non_generic_items = [item for item in items if not generic_classification.get(item["headline"], False)]
    daily_stats[today_str]["generic"] += len(items) - len(non_generic_items)
    
    if not non_generic_items:
        logger.info("No non-generic items found in this batch")
        return
    
    # Second step: Analyze non-generic headlines
    logger.info(f"Analyzing {len(non_generic_items)} non-generic headlines")
    analysis_results = batch_analyze_headlines(non_generic_items)
    
    with ThreadPoolExecutor(max_workers=PARALLELISM) as executor:
        futures = []
        
        for item in non_generic_items:
            headline = item["headline"]
            
            if headline not in analysis_results:
                daily_stats[today_str]["failed"] += 1
                continue
                
            analysis = analysis_results[headline]
            confidence = analysis.get("confidence", "low").lower().replace(" ", "-")
            
            # Track confidence levels in stats
            if confidence in ['very-high', 'high', 'medium', 'low']:
                daily_stats[today_str].setdefault(confidence, 0)
                daily_stats[today_str][confidence] += 1
                
            logger.info(
                f"Analyzed headline: {headline}, Confidence: {confidence}, "
                f"Catalyst: {analysis.get('market_catalyst')}, Sentiment: {analysis.get('sentiment')} "
                f"(Score: {analysis.get('sentiment_score')}), Timing: {analysis.get('impact_timing')}, "
                f"Impact: {analysis.get('price_impact_range')}"
            )
            
            # Only process medium or higher confidence analyses
            if confidence in ['very-high', 'high', 'medium']:
                # First try to use ticker from batch analysis
                ticker = None
                
                # Check if analysis already provided a ticker and it's valid
                if analysis.get('ticker') and analysis.get('ticker') != "null" and is_valid_ticker(analysis.get('ticker')):
                    ticker = analysis.get('ticker')
                    logger.info(f"Using ticker {ticker} from batch analysis for '{headline}'")
                else:
                    # If not, extract ticker with the improved get_ticker_symbol function
                    ticker = get_ticker_symbol(headline, analysis.get("analysis", ""))
                
                if ticker:
                    futures.append(executor.submit(process_ticker_analysis, ticker, headline, analysis, item["summary"], run_timestamp))
                else:
                    logger.info(f"No valid ticker found for '{headline}'")
        
        # Wait for all analysis tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")

def process_ticker_analysis(ticker, headline, analysis, summary, run_timestamp):
    """Process analysis for a specific ticker"""
    try:
        from data import get_company_fundamentals  # Avoid circular import
        
        # Get fundamental data for the company
        fundamentals = get_company_fundamentals(ticker)
        
        # Analyze how the news relates to company context
        company_context = analyze_company_context(ticker, headline, analysis.get("analysis", ""), fundamentals)
        
        # Evaluate trading opportunity
        trade_evaluation = evaluate_trade_opportunity(ticker, analysis)
        trade_evaluation['company_context'] = company_context
        
        # Generate trade confidence score
        trade_evaluation['trade_confidence'] = generate_trade_confidence(
            ticker, 
            headline, 
            analysis, 
            trade_evaluation.get('factors', {}).get('technical', {}).get('details', {})
        )
        
        # Adjust position size based on company fundamentals if necessary
        if company_context.get('risk_adjustment') and 'strategy' in trade_evaluation and company_context.get('risk_adjustment') < 0:
            strategy = trade_evaluation['strategy']
            pos_size = strategy.get('position_size', '')
            if '3-5%' in pos_size:
                strategy['position_size'] = '1-2% of portfolio (reduced due to fundamentals)'
            elif '2-3%' in pos_size:
                strategy['position_size'] = '1% of portfolio (reduced due to fundamentals)'
        
        # Save recommendation if score meets minimum threshold
        if trade_evaluation['overall_score'] >= MIN_OVERALL_SCORE:
            save_enhanced_recommendation(run_timestamp, headline, analysis, summary, trade_evaluation)
            
            # Setup backtest if enabled
            if ENABLE_BACKTESTING and 'strategy' in trade_evaluation:
                simulate_trade_performance(ticker, trade_evaluation['strategy'])
        else:
            logger.info(f"Skipping low-score ({trade_evaluation['overall_score']:.1f}/10) opportunity for {ticker}")
            
    except Exception as e:
        logger.error(f"Error processing ticker analysis for {ticker}: {e}")

def run_maintenance_tasks():
    """Run periodic maintenance tasks"""
    try:
        # Update performance of stocks in watchlist
        logger.info("Starting watchlist performance update")
        update_watchlist_performance()
        
        # Update backtests if enabled
        if ENABLE_BACKTESTING:
            logger.info("Starting backtest update")
            update_backtests()
        
        # Check for significant market events
        logger.info("Starting market events check")
        try:
            events_info = check_market_events(MODEL_THINKING)
            
            # Validate the response
            if not isinstance(events_info, dict):
                logger.error(f"Invalid market events response: {type(events_info)}")
                events_info = {'events': [], 'market_risk': 'low'}
                
            market_risk = events_info.get('market_risk', 'low')
            
            # Adjust trading threshold if market risk is elevated
            if market_risk in ['medium', 'high']:
                for event in events_info.get('events', []):
                    logger.warning(f"Market event detected: {event}")
                    
                if market_risk == 'high':
                    global MIN_OVERALL_SCORE
                    MIN_OVERALL_SCORE = min(8.0, MIN_OVERALL_SCORE + 1.0)
                    logger.warning(f"Increased trading threshold to {MIN_OVERALL_SCORE} due to high market risk")
        except Exception as e:
            logger.error(f"Error in market events check: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error running maintenance tasks: {e}", exc_info=True)

def main():
    """Main entry point for the trading system"""
    logger.info("Starting multi-factor newsfeed trading system...")
    debug_feeds()
    check_feeds_availability()

    # Ensure directories exist
    os.makedirs(RECOMMENDATIONS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RECOMMENDATIONS_DIR, "by_ticker"), exist_ok=True)
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    
    # Run maintenance tasks
    logger.info("Running maintenance tasks...")
    run_maintenance_tasks()
    
    # Load processed articles and stats
    processed_articles = load_processed_articles()
    daily_stats = load_daily_stats()
    
    # Create timestamp for this run
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Process news feeds
    process_feed(processed_articles, daily_stats, run_timestamp)
    
    # Exit if single run mode
    if SINGLE_RUN_MODE:
        logger.info("Single run completed. Exiting.")
        sys.exit(0)
    
    # Continue processing in loop
    while True:
        try:
            logger.info("Waiting 5 minutes before next scan...")
            time.sleep(300)
            run_maintenance_tasks()
            run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            process_feed(processed_articles, daily_stats, run_timestamp)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.info("Waiting 10 minutes before retry due to error...")
            time.sleep(600)

if __name__ == "__main__":
    main()