import os
import json
import datetime
import re
import yfinance as yf
from api import robust_api_call
from config import RECOMMENDATIONS_DIR, MIN_OVERALL_SCORE, ENABLE_BACKTESTING, logger
from data import get_current_price, get_extended_market_context, get_stock_data
from analysis import perform_technical_analysis, detect_unusual_options_activity
from utils import add_to_watchlist
from config import logger

def evaluate_trade_opportunity(ticker, news_analysis=None):
    """
    Evaluates a trade opportunity for a given ticker based on technical analysis, options activity, and news.
    
    Args:
        ticker (str): Stock ticker symbol.
        news_analysis (dict, optional): Analysis of news including sentiment and confidence.
    
    Returns:
        dict: Trade evaluation with scores, recommendation, and strategy if applicable.
    """
    results = {
        'ticker': ticker,
        'timestamp': datetime.datetime.now().isoformat(),
        'factors': {},
        'overall_score': 0,
        'max_score': 10,
        'recommendation': "No Trade"
    }
    
    # Technical Analysis
    tech_analysis = perform_technical_analysis(ticker)
    if tech_analysis['technical_rating'] == "No Data":
        results['factors']['technical'] = {'score': 0, 'rating': "No Data", 'details': tech_analysis}
        return results
    tech_score = min(5, (tech_analysis['technical_score'] / tech_analysis['max_score']) * 5)
    results['factors']['technical'] = {'score': tech_score, 'rating': tech_analysis['technical_rating'], 'details': tech_analysis}
    results['overall_score'] += tech_score
    
    # Options Activity
    options_activity = detect_unusual_options_activity(ticker)
    options_score = min(3, options_activity['unusual_score'] * 0.5)
    results['factors']['options_activity'] = {'score': options_score, 'assessment': options_activity['assessment'], 'details': options_activity}
    results['overall_score'] += options_score
    
    # News Analysis
    if news_analysis:
        news_confidence = news_analysis.get('confidence', 'low')
        news_score = {'very-high': 2.0, 'high': 1.5, 'medium': 1.0, 'low': 0.5}.get(news_confidence.lower().replace(' ', '-'), 0)
        # Adjust news score based on market cap (favor smaller companies)
        stock_data = get_stock_data(ticker, get_cache_timestamp())
        market_cap = stock_data.get('marketCap', 0) / 1e9  # Convert to billions
        if market_cap > 0 and market_cap < 1:  # Small cap (< $1B)
            news_score *= 1.5
        elif market_cap < 10:  # Mid cap (< $10B)
            news_score *= 1.2
        results['factors']['news_catalyst'] = {
            'score': news_score,
            'confidence': news_confidence,
            'catalyst': news_analysis.get('market_catalyst', 'Unknown'),
            'sentiment': news_analysis.get('sentiment', 'Unknown'),
            'details': news_analysis
        }
        results['overall_score'] += news_score
    
    # Market Context and Recommendation
    results['market_context'] = get_extended_market_context()
    results['recommendation'] = (
        "Strong Buy" if results['overall_score'] >= 7 else
        "Buy" if results['overall_score'] >= 5 else
        "Watch" if results['overall_score'] >= 3 else
        "No Trade"
    )
    
    # Generate Strategy for Buy Recommendations
    if results['recommendation'] in ["Strong Buy", "Buy"]:
        use_options = options_score > 1.5 or (news_analysis and news_analysis.get('impact_timing') == 'immediate')
        results['strategy'] = generate_trade_strategy(ticker, tech_analysis, options_activity, news_analysis, results['market_context'], use_options)
    
    return results

def generate_trade_strategy(ticker, tech_analysis, options_activity, news_analysis, market_context, use_options=True):
    """
    Generates a trading strategy using an AI model based on technicals, options, news, and market context.
    The model has full freedom to recommend the best strategy (or no trade), considering trader preferences as guidance.

    Args:
        ticker (str): Stock ticker symbol.
        tech_analysis (dict): Technical analysis data.
        options_activity (dict): Options activity data.
        news_analysis (dict): News analysis data.
        market_context (dict): Market context data.
        use_options (bool): Preference for using options (default True, but model can override).

    Returns:
        dict: Trading strategy details or a "no trade" recommendation with explanation.
    """
    try:
        # Fetch current price
        current_price = get_current_price(ticker)
        if current_price is None:
            logger.error(f"No current price available for {ticker}")
            return {"type": "error", "strategy": "No current price available"}

        # Extract nearest support and resistance from technical analysis
        support_resistance = tech_analysis.get('support_resistance', {})
        supports = support_resistance.get('supports', [])
        resistances = support_resistance.get('resistances', [])
        nearest_support = min(supports, key=lambda x: abs(x - current_price)) if supports else current_price * 0.9
        nearest_resistance = min(resistances, key=lambda x: abs(x - current_price)) if resistances else current_price * 1.1

        # Summarize technical analysis concisely
        tech_summary = f"Rating: {tech_analysis.get('technical_rating', 'Unknown')}, Score: {tech_analysis.get('technical_score', 0)}/{tech_analysis.get('max_score', 14)}"
        daily_signals = tech_analysis.get('signals', {}).get('daily', {})
        if daily_signals:
            if daily_signals.get('price_above_sma50'): tech_summary += ", Price > 50-day MA"
            if daily_signals.get('macd_bullish'): tech_summary += ", MACD Bullish"
            rsi = daily_signals.get('rsi_value', 0)
            if rsi > 70: tech_summary += ", RSI Overbought"
            elif rsi < 30: tech_summary += ", RSI Oversold"

        # Summarize options activity
        options_summary = options_activity.get('assessment', 'No options data available')
        if options_activity.get('unusual_activity'):
            call_count = sum(1 for act in options_activity['unusual_activity'] if act['type'] == 'call')
            put_count = sum(1 for act in options_activity['unusual_activity'] if act['type'] == 'put')
            options_summary += f", {call_count} calls, {put_count} puts"

        # Summarize news analysis
        news_summary = "No news data" if not news_analysis else (
            f"Sentiment: {news_analysis.get('sentiment', 'Unknown')} (Score: {news_analysis.get('sentiment_score', 'N/A')}), "
            f"Impact: {news_analysis.get('price_impact_range', 'Unknown')}, "
            f"Timing: {news_analysis.get('impact_timing', 'Unknown')}"
        )

        # Summarize market context
        market_summary = (
            f"Trend: {market_context.get('trend', 'Unknown')}, "
            f"SPY: {market_context.get('spy_price', 'N/A')} ({market_context.get('spy_change_1d', 0):.1f}%), "
            f"VIX: {market_context.get('vix', 0):.1f}"
        )

        # Construct the prompt for the AI model
        prompt = f"""
Act as an expert trading strategist. Use the following information to recommend the best trading strategy for {ticker}:

Trader Preferences (consider these as guidance, not strict rules):
- Prefers options for asymmetric risk-reward and shorter time frames.
- Prefers lower cost of capital.
- Uses puts instead of shorting stock.

Current Price: ${current_price:.2f}

Technical Analysis:
- Support: ${nearest_support:.2f}
- Resistance: ${nearest_resistance:.2f}
- {tech_summary}

Options Activity: {options_summary}

News Analysis: {news_summary}

Market Context: {market_summary}

Analyze all this data and decide on the best trading strategy—or recommend no trade if conditions aren’t favorable. You may deviate from the trader’s preferences if justified (e.g., higher cost of capital for a compelling opportunity). 

If recommending a trade, return a JSON object with:
- 'strategy_type': e.g., 'bullish_options', 'bearish_stock'
- 'entry': entry point or range (e.g., '$50-$51')
- 'target': profit target (e.g., '$55')
- 'stop_loss': stop loss level (e.g., '$48')
- 'position_size': suggested size (e.g., '2% of portfolio')
- 'options_details': specific options if applicable (e.g., 'Buy 1 $50 call expiring Dec 20')
- 'risk_reward': estimated risk-reward ratio (e.g., 3.5)
- 'market_aligned': true/false (aligns with market trend?)
- 'explanation': why this strategy makes sense

If no trade is recommended, return a JSON object with:
- 'recommendation': 'No Trade'
- 'explanation': why no trade is advised

Return your response as a valid JSON object.
"""

        # Define models in the desired fallback order
        models = [
            "claude-3-7-sonnet-20250219",
            "deepseek-reasoner",
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-2.0-flash"
        ]

        # Make the API call
        success, response = robust_api_call(models, prompt, config={'response_mime_type': 'application/json'}, max_tokens=4000)

        if not success:
            logger.error(f"API call failed for {ticker}: {response}")
            return {"type": "error", "strategy": "Failed to generate strategy"}

        # Parse the model’s JSON response
        try:
            strategy_data = json.loads(response)

            if strategy_data.get('recommendation') == 'No Trade':
                return {
                    "type": "no_trade",
                    "strategy": "No trade recommended",
                    "position_size": "N/A",
                    "entry": "N/A",
                    "target": "N/A",
                    "stop_loss": "N/A",
                    "risk_reward": 0,
                    "market_aligned": False,
                    "explanation": strategy_data.get('explanation', 'No explanation provided')
                }
            else:
                return {
                    "type": strategy_data.get('strategy_type', 'unknown'),
                    "strategy": f"{strategy_data.get('strategy_type', 'unknown').replace('_', ' ').title()}: {strategy_data.get('options_details', '')}",
                    "position_size": strategy_data.get('position_size', 'N/A'),
                    "entry": strategy_data.get('entry', 'N/A'),
                    "target": strategy_data.get('target', 'N/A'),
                    "stop_loss": strategy_data.get('stop_loss', 'N/A'),
                    "risk_reward": float(strategy_data.get('risk_reward', 0)),
                    "market_aligned": bool(strategy_data.get('market_aligned', False)),
                    "explanation": strategy_data.get('explanation', 'No explanation provided')
                }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response for {ticker}: {response}")
            return {"type": "error", "strategy": "Failed to parse strategy"}

    except Exception as e:
        logger.error(f"Error generating strategy for {ticker}: {e}")
        return {"type": "error", "strategy": "Unable to generate strategy"}

def generate_risk_reward_chart(ticker, strategy, run_timestamp):
    """
    Generates a risk/reward chart in JSON format based on the trade strategy.
    
    Args:
        ticker (str): Stock ticker symbol.
        strategy (dict): Trading strategy details.
        run_timestamp (str): Timestamp of the run for file storage.
    """
    try:
        entry = strategy.get('entry', '')
        current_price_match = re.search(r'\$(\d+\.\d+)', entry)
        if not current_price_match:
            logger.warning(f"No valid entry price found for {ticker}")
            return
        current_price = float(current_price_match.group(1))
        
        target_match = re.search(r'\$(\d+\.\d+)', strategy.get('target', ''))
        stop_match = re.search(r'\$(\d+\.\d+)', strategy.get('stop_loss', ''))
        if not target_match or not stop_match:
            logger.warning(f"Missing target or stop price for {ticker}")
            return
        
        target_price = float(target_match.group(1))
        stop_price = float(stop_match.group(1))
        
        potential_gain = abs(target_price - current_price) / current_price * 100
        potential_loss = abs(stop_price - current_price) / current_price * 100
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
        
        chart_data = {
            'ticker': ticker,
            'current_price': current_price,
            'target_price': target_price,
            'stop_price': stop_price,
            'potential_gain_pct': potential_gain,
            'potential_loss_pct': potential_loss,
            'risk_reward_ratio': risk_reward,
            'strategy_type': strategy.get('type', 'unknown')
        }
        
        os.makedirs(os.path.join(RECOMMENDATIONS_DIR, run_timestamp), exist_ok=True)
        chart_file = os.path.join(RECOMMENDATIONS_DIR, run_timestamp, f"{ticker}_risk_reward.json")
        with open(chart_file, "w") as f:
            json.dump(chart_data, f, indent=4)
            
    except Exception as e:
        logger.error(f"Error generating risk/reward chart for {ticker}: {e}")

def generate_trade_confidence(ticker, headline, analysis, tech_analysis):
    """
    Generates a confidence score for the trade using an API call to a thinking model.
    
    Args:
        ticker (str): Stock ticker symbol.
        headline (str): News headline.
        analysis (dict): News analysis data.
        tech_analysis (dict): Technical analysis data.
    
    Returns:
        dict: Confidence score and reasoning.
    """
    try:
        prompt = (
            f"Act as a professional hedge fund manager evaluating this trade idea. "
            f"Based on the news and technical analysis, rate your confidence in this trade from 0-10 "
            f"and explain your reasoning in 2-3 sentences.\n\n"
            f"Ticker: {ticker}\n"
            f"Headline: {headline}\n"
            f"Analysis: {analysis.get('analysis', '')}\n"
            f"Technical Rating: {tech_analysis.get('technical_rating', 'Unknown')}\n"
            f"Key Signals: {tech_analysis.get('technical_score', 0)} out of {tech_analysis.get('max_score', 14)} points\n\n"
            f"Return a JSON object with 'confidence_score' (number from 0 to 10) and 'reasoning' (string)."
        )
        models = ["gemini-2.0-flash"]
        config = {'response_mime_type': 'application/json'}
        success, response = robust_api_call(models, prompt, config, max_tokens=4000, retries=2)
        if success and 'confidence_score' in response and 'reasoning' in response:
            return response
        logger.error("Failed to generate trade confidence")
        return {'confidence_score': 5, 'reasoning': "Unable to generate confidence assessment"}
    except Exception as e:
        logger.error(f"Error in trade confidence generation: {e}")
        return {'confidence_score': 5, 'reasoning': "Error in analysis"}

def save_enhanced_recommendation(run_timestamp, headline, analysis, summary, trade_evaluation):
    """
    Saves an enhanced trade recommendation to files.
    
    Args:
        run_timestamp (str): Timestamp of the run.
        headline (str): News headline.
        analysis (dict): News analysis data.
        summary (str): Summary of the trade (unused here but kept for compatibility).
        trade_evaluation (dict): Trade evaluation data.
    """
    try:
        ticker = trade_evaluation.get('ticker', 'unknown')
        score = trade_evaluation.get('overall_score', 0)
        recommendation = trade_evaluation.get('recommendation', 'No Trade')
        if score < MIN_OVERALL_SCORE:
            logger.info(f"Skipping low-score ({score:.1f}) recommendation for {ticker}: {headline}")
            return
        
        run_dir = os.path.join(RECOMMENDATIONS_DIR, run_timestamp)
        ticker_dir = os.path.join(RECOMMENDATIONS_DIR, "by_ticker")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(ticker_dir, exist_ok=True)
        
        filename = os.path.join(run_dir, f"{ticker}_{recommendation.lower().replace(' ', '_')}.txt")
        ticker_file = os.path.join(ticker_dir, f"{ticker}.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        trade_confidence = trade_evaluation.get('trade_confidence', {})
        confidence_text = f"""
TRADE CONFIDENCE:
- Score: {trade_confidence.get('confidence_score', 'N/A')}/10
- Assessment: {trade_confidence.get('reasoning', 'No assessment available')}
"""
        strategy = trade_evaluation.get('strategy', {})
        strategy_text = f"""
TRADE STRATEGY:
- Type: {strategy.get('type', 'Unknown')}
- Strategy: {strategy.get('strategy', 'No strategy available')}
- Position Size: {strategy.get('position_size', 'Unknown')}
- Entry: {strategy.get('entry', 'Unknown')}
- Target: {strategy.get('target', 'Unknown')}
- Stop Loss: {strategy.get('stop_loss', 'Unknown')}
- Risk/Reward Ratio: {strategy.get('risk_reward', 0):.2f}
- Market Aligned: {'Yes' if strategy.get('market_aligned', False) else 'No'}
"""
        recommendation_text = f"""
=================================================================
TRADE RECOMMENDATION: {ticker} - {recommendation} (Score: {score:.1f}/10)
=================================================================
Date: {timestamp}
Headline: {headline}

NEWS ANALYSIS:
- Catalyst: {analysis.get('market_catalyst', 'Unknown')}
- Sentiment: {analysis.get('sentiment', 'Unknown')} (Score: {analysis.get('sentiment_score', 'N/A')})
- Expected Impact: {analysis.get('price_impact_range', 'Unknown')}
- Timing: {analysis.get('impact_timing', 'Unknown')}
- Sell The News: {'Yes' if analysis.get('sell_the_news', False) else 'No'}
- Analysis: {analysis.get('analysis', 'No analysis available')}

TECHNICAL ANALYSIS:
- Rating: {trade_evaluation.get('factors', {}).get('technical', {}).get('rating', 'Unknown')}
- Key Signals: {summarize_technical_signals(trade_evaluation)}

OPTIONS ACTIVITY:
- Assessment: {trade_evaluation.get('factors', {}).get('options_activity', {}).get('assessment', 'Unknown')}
- Unusual Activity: {summarize_options_activity(trade_evaluation)}

MARKET CONTEXT:
- Market Trend: {trade_evaluation.get('market_context', {}).get('trend', 'Unknown')}
- SPY: ${trade_evaluation.get('market_context', {}).get('spy_price', 0):.2f} ({trade_evaluation.get('market_context', {}).get('spy_change_1d', 0):.1f}%)
- VIX: {trade_evaluation.get('market_context', {}).get('vix', 0):.1f} ({trade_evaluation.get('market_context', {}).get('volatility', 'Unknown')})
{confidence_text}
{strategy_text}

=================================================================
"""
        with open(filename, "w") as f:
            f.write(recommendation_text)
        with open(ticker_file, "a") as f:
            f.write(recommendation_text)
        
        add_to_watchlist(
            ticker,
            headline,
            analysis.get('price_impact_range', 'Unknown'),
            analysis.get('impact_timing', 'Unknown'),
            score
        )
        if 'strategy' in trade_evaluation:
            generate_risk_reward_chart(ticker, strategy, run_timestamp)
        logger.info(f"Saved enhanced recommendation for {ticker}: {recommendation} (Score: {score:.1f}/10)")
    except Exception as e:
        logger.error(f"Error saving enhanced recommendation: {e}")

def summarize_technical_signals(trade_evaluation):
    """
    Summarizes key technical signals from the trade evaluation.
    
    Args:
        trade_evaluation (dict): Trade evaluation data.
    
    Returns:
        str: Summary of technical signals.
    """
    try:
        daily_signals = trade_evaluation.get('factors', {}).get('technical', {}).get('details', {}).get('signals', {}).get('daily', {})
        if not daily_signals:
            return "No technical signals available"
        key_signals = []
        if daily_signals.get('price_above_sma20'): key_signals.append("Price above 20-day MA")
        if daily_signals.get('price_above_sma50'): key_signals.append("Price above 50-day MA")
        if daily_signals.get('price_above_sma200'): key_signals.append("Price above 200-day MA")
        if daily_signals.get('sma50_above_sma200'): key_signals.append("SMA50 above SMA200 (Golden Cross)")
        if daily_signals.get('roc_positive'): key_signals.append(f"Positive ROC ({daily_signals.get('roc', 0):.2f})")
        if daily_signals.get('macd_bullish_crossover'): key_signals.append("Bullish MACD crossover")
        if daily_signals.get('rsi_value', 0) > 70: key_signals.append(f"RSI overbought ({daily_signals.get('rsi_value', 0):.1f})")
        elif daily_signals.get('rsi_value', 0) < 30: key_signals.append(f"RSI oversold ({daily_signals.get('rsi_value', 0):.1f})")
        if daily_signals.get('obv_increasing'): key_signals.append("Rising on-balance volume")
        if daily_signals.get('bb_breakout_up'): key_signals.append("Bollinger Band bullish breakout")
        return ", ".join(key_signals) or "No significant technical signals detected"
    except Exception as e:
        logger.error(f"Error summarizing technical signals: {e}")
        return "Error processing technical signals"

def summarize_options_activity(trade_evaluation):
    """
    Summarizes options activity from the trade evaluation.
    
    Args:
        trade_evaluation (dict): Trade evaluation data.
    
    Returns:
        str: Summary of options activity.
    """
    try:
        options_details = trade_evaluation.get('factors', {}).get('options_activity', {}).get('details', {})
        unusual_activity = options_details.get('unusual_activity', [])
        if not unusual_activity:
            return "No unusual options activity detected"
        call_count = sum(1 for act in unusual_activity if act.get('type') == 'call')
        put_count = sum(1 for act in unusual_activity if act.get('type') == 'put')
        highest_volume = max(unusual_activity, key=lambda x: x.get('volume', 0), default=None)
        highest_vol_desc = f"{highest_volume['type'].upper()} {highest_volume['strike']} exp {highest_volume['expiration']} (Vol: {highest_volume['volume']})" if highest_volume else "None"
        return f"{call_count} bullish / {put_count} bearish signals. Highest volume: {highest_vol_desc}"
    except Exception as e:
        logger.error(f"Error summarizing options activity: {e}")
        return "Error processing options activity"

def get_cache_timestamp():
    """
    Returns a timestamp for caching purposes (hourly granularity).
    
    Returns:
        int: Unix timestamp divided by 3600 (hours).
    """
    return int(datetime.datetime.now().timestamp() // 3600)