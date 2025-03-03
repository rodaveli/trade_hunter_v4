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
    Check if an RSS feed entry is recent (within the last 48 hours).
    Handles various date formats and missing date information.
    """
    logger.debug(f"Checking recency for entry: {entry.get('title', 'Unknown')}")
    
    # Try to find a date field in the entry
    date_fields = ['published_parsed', 'updated_parsed', 'pubDate']
    
    for field in date_fields:
        if field in entry:
            try:
                if field == 'pubDate':
                    # Handle string date formats
                    from email.utils import parsedate_to_datetime
                    try:
                        # Standard RFC822 format
                        publish_time = parsedate_to_datetime(entry[field])
                    except (ValueError, TypeError):
                        # Try custom parsing for other formats
                        date_str = entry[field]
                        if 'UT' in date_str:  # Business Wire format
                            date_str = date_str.replace('UT', 'GMT')
                        try:
                            publish_time = datetime.datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
                        except ValueError:
                            # Try more formats if needed
                            formats = [
                                '%a, %d %b %Y %H:%M:%S %z',
                                '%Y-%m-%dT%H:%M:%S%z',
                                '%Y-%m-%d %H:%M:%S'
                            ]
                            for fmt in formats:
                                try:
                                    publish_time = datetime.datetime.strptime(date_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                logger.warning(f"Could not parse date: {date_str}")
                                return True  # Assume recent if we can't parse
                else:
                    # Handle tuple format
                    publish_time = datetime.datetime(*entry[field][:6])
                
                # Calculate age
                age_seconds = (datetime.datetime.now() - publish_time).total_seconds()
                
                # Handle timezone issues (negative age)
                if age_seconds < 0:
                    logger.warning(f"Entry appears to be in the future: {entry.get('title', 'Unknown')}")
                    return True
                
                # Consider entries from the last 48 hours as recent (increased from 24 to be safe)
                logger.debug(f"Entry age: {age_seconds/3600:.1f} hours")
                return age_seconds <= 172800  # 48 hours
                
            except Exception as e:
                logger.warning(f"Error parsing date for {field}: {e}")
                # Continue to try other date fields
    
    # If we get here, we couldn't find or parse any date field
    logger.info(f"No valid date information found for: {entry.get('title', 'Unknown')}, assuming recent")
    return True  # Assume recent if we can't determine the date