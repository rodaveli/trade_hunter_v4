import json
import os
import datetime
import re
from config import BACKTEST_DIR, ENABLE_BACKTESTING, logger
from data import get_current_price

def simulate_trade_performance(ticker, strategy, days_to_monitor=10):
    try:
        if not ENABLE_BACKTESTING:
            return None
        current_price = get_current_price(ticker)
        if current_price is None or current_price <= 0:
            logger.error(f"Cannot simulate trade for {ticker}: no valid current price")
            return None
        strategy_type = strategy.get('type', 'unknown')
        is_bullish = strategy_type.startswith('bullish')
        backtest_record = {
            'ticker': ticker,
            'strategy_type': strategy_type,
            'entry_date': datetime.datetime.now().isoformat(),
            'entry_price': current_price,
            'is_bullish': is_bullish,
            'target_price': None,
            'stop_price': None,
            'daily_results': [],
            'final_result': None,
            'max_gain': 0,
            'max_loss': 0,
            'hit_target': False,
            'hit_stop': False
        }
        
        # Improved price extraction with regex
        try:
            target_text = strategy.get('target', '')
            target_match = re.search(r'\$(\d+\.\d+)', target_text)
            if target_match:
                backtest_record['target_price'] = float(target_match.group(1))
                
            stop_text = strategy.get('stop_loss', '')
            stop_match = re.search(r'\$(\d+\.\d+)', stop_text)
            if stop_match:
                backtest_record['stop_price'] = float(stop_match.group(1))
                
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not parse target/stop prices for {ticker}: {e}")
            
        backtest_file = os.path.join(BACKTEST_DIR, f"{ticker}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(backtest_file, "w") as f:
            json.dump(backtest_record, f, indent=4)
        logger.info(f"Initialized backtest for {ticker}")
        return backtest_record
    except Exception as e:
        logger.error(f"Error simulating trade performance for {ticker}: {e}")
        return None

def update_backtests():
    if not ENABLE_BACKTESTING:
        return
    try:
        backtest_files = [f for f in os.listdir(BACKTEST_DIR) if f.endswith('.json')]
        updated = 0
        for file in backtest_files:
            file_path = os.path.join(BACKTEST_DIR, file)
            try:
                with open(file_path, "r") as f:
                    backtest = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error reading {file}: {e}")
                continue
            if backtest.get('final_result') is not None:
                continue
            ticker = backtest.get('ticker')
            current_price = get_current_price(ticker)
            if current_price is None or current_price <= 0:
                logger.warning(f"Skipping backtest update for {ticker}: no valid current price")
                continue
            entry_price = backtest.get('entry_price')
            target_price = backtest.get('target_price')
            stop_price = backtest.get('stop_price')
            is_bullish = backtest.get('is_bullish')
            if entry_price <= 0:
                logger.warning(f"Invalid entry price for {ticker}: {entry_price}")
                continue
            percent_change = ((current_price - entry_price) / entry_price * 100) if is_bullish else ((entry_price - current_price) / entry_price * 100)
            hit_target = target_price and (current_price >= target_price if is_bullish else current_price <= target_price)
            hit_stop = stop_price and (current_price <= stop_price if is_bullish else current_price >= stop_price)
            max_gain = max(backtest.get('max_gain', 0), percent_change if percent_change > 0 else 0)
            max_loss = min(backtest.get('max_loss', 0), percent_change if percent_change < 0 else 0)
            daily_results = backtest.get('daily_results', [])
            daily_results.append({
                'date': datetime.datetime.now().isoformat(),
                'price': current_price,
                'percent_change': percent_change
            })
            entry_date = datetime.datetime.fromisoformat(backtest.get('entry_date'))
            days_elapsed = (datetime.datetime.now() - entry_date).days
            is_complete = days_elapsed >= 10 or hit_target or hit_stop
            backtest.update({
                'daily_results': daily_results,
                'max_gain': max_gain,
                'max_loss': max_loss,
                'hit_target': hit_target,
                'hit_stop': hit_stop
            })
            if is_complete:
                backtest.update({
                    'final_result': percent_change,
                    'final_price': current_price,
                    'close_date': datetime.datetime.now().isoformat(),
                    'close_reason': 'target_hit' if hit_target else 'stop_hit' if hit_stop else 'time_elapsed'
                })
            with open(file_path, "w") as f:
                json.dump(backtest, f, indent=4)
            updated += 1
            if is_complete:
                logger.info(f"Completed backtest for {ticker}: {percent_change:.2f}% ({backtest['close_reason']})")
        if updated > 0:
            logger.info(f"Updated {updated} backtest records")
    except Exception as e:
        logger.error(f"Error in backtest update: {e}")