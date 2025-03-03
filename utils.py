import json
import os
import datetime
from config import PROCESSED_FILE, STATS_FILE, WATCHLIST_FILE, logger

def load_processed_articles():
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE, "r") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading processed articles: {e}")
            return set()
    return set()

def save_processed_articles(processed_articles):
    try:
        with open(PROCESSED_FILE, "w") as f:
            json.dump(list(processed_articles), f)
    except IOError as e:
        logger.error(f"Error saving processed articles: {e}")

def load_daily_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading daily stats: {e}")
            return {}
    return {}

def save_daily_stats(daily_stats):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(daily_stats, f, indent=4)
    except IOError as e:
        logger.error(f"Error saving daily stats: {e}")

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading watchlist: {e}")
            return {}
    return {}

def save_watchlist(watchlist):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(watchlist, f, indent=4)
    except IOError as e:
        logger.error(f"Failed to save watchlist: {e}")

def add_to_watchlist(ticker, reason, expected_impact, timeframe, score=None, get_current_price_func=None):
    from data import get_current_price  # Import here to avoid circular imports
    watchlist = load_watchlist()
    today = datetime.date.today().isoformat()
    price = get_current_price(ticker) if get_current_price_func is None else get_current_price_func(ticker)
    if price is None or price <= 0:
        logger.error(f"Cannot add {ticker} to watchlist: unable to fetch valid price")
        return
    if ticker not in watchlist:
        watchlist[ticker] = []
    watchlist[ticker].append({
        "date_added": today,
        "reason": reason,
        "expected_impact": expected_impact,
        "timeframe": timeframe,
        "score": score,
        "status": "watching",
        "price_at_entry": price
    })
    save_watchlist(watchlist)
    logger.info(f"Added {ticker} to watchlist: {reason} (Score: {score})")

def update_watchlist_performance(get_current_price_func=None):
    from data import get_current_price  # Import here to avoid circular imports
    watchlist = load_watchlist()
    updated = False
    for ticker, entries in watchlist.items():
        current_price = get_current_price(ticker) if get_current_price_func is None else get_current_price_func(ticker)
        if current_price is None or current_price <= 0:
            logger.warning(f"Skipping performance update for {ticker}: no valid current price")
            continue
        for entry in entries:
            if entry["status"] == "watching" and "price_at_entry" in entry:
                entry_price = entry["price_at_entry"]
                if entry_price is None or entry_price <= 0:
                    logger.warning(f"Invalid entry price for {ticker}: {entry_price}")
                    continue
                try:
                    percent_change = ((current_price - entry_price) / entry_price) * 100
                    entry["current_performance"] = percent_change
                    updated = True
                    entry_date = datetime.date.fromisoformat(entry["date_added"])
                    days_passed = (datetime.date.today() - entry_date).days
                    if days_passed > 5:
                        entry["status"] = "closed"
                        entry["final_performance"] = percent_change
                        entry["close_reason"] = "time_elapsed"
                        entry["close_date"] = datetime.date.today().isoformat()
                        logger.info(f"Auto-closed watchlist item for {ticker}: {percent_change:.2f}% after {days_passed} days")
                except ZeroDivisionError:
                    logger.warning(f"Zero division error for {ticker}: entry_price is zero")
                    continue
    if updated:
        save_watchlist(watchlist)
        logger.info("Updated watchlist performance metrics")

def is_recent(entry):
    """
    Check if an RSS feed entry is recent (within the last 24 hours).
    Handles various date formats and missing date information.
    """
    # Debug information
    logger.debug(f"Checking recency for entry: {entry.get('title', 'Unknown')}")
    
    # If no published_parsed, check alternative date fields
    if 'published_parsed' not in entry:
        # Try alternative date fields
        for field in ['updated_parsed', 'created_parsed', 'pubDate']:
            if field in entry:
                try:
                    if field == 'pubDate':
                        # Parse string date format
                        from email.utils import parsedate_to_datetime
                        publish_time = parsedate_to_datetime(entry[field])
                    else:
                        # Parse tuple format
                        publish_time = datetime.datetime(*entry[field][:6])
                    
                    age_seconds = (datetime.datetime.now() - publish_time).total_seconds()
                    logger.debug(f"Using {field}, age: {age_seconds/3600:.1f} hours")
                    return age_seconds <= 86400  # 24 hours
                except Exception as e:
                    logger.debug(f"Failed to parse {field}: {e}")
        
        # If we couldn't find a date, assume it's recent
        logger.info(f"No date information found for: {entry.get('title', 'Unknown')}, assuming recent")
        return True
    
    try:
        # Standard case with published_parsed
        publish_time = datetime.datetime(*entry.published_parsed[:6])
        age_seconds = (datetime.datetime.now() - publish_time).total_seconds()
        
        # Handle potential timezone issues
        if age_seconds < 0:
            logger.warning(f"Article appears to be in the future, assuming recent: {entry.get('title', 'Unknown')}")
            return True
            
        logger.debug(f"Entry age: {age_seconds/3600:.1f} hours, recent: {age_seconds <= 86400}")
        return age_seconds <= 86400  # 24 hours
    except Exception as e:
        logger.warning(f"Error determining if entry is recent: {e}, assuming recent")
        return True