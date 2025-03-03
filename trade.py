import os
import json
import datetime
import re
import yfinance as yf
from api import robust_api_call
from config import RECOMMENDATIONS_DIR, MIN_OVERALL_SCORE, ENABLE_BACKTESTING, logger
from data import get_current_price, get_extended_market_context
from analysis import perform_technical_analysis, detect_unusual_options_activity
from utils import add_to_watchlist

def evaluate_trade_opportunity(ticker, news_analysis=None):
    results = {
        'ticker': ticker,
        'timestamp': datetime.datetime.now().isoformat(),
        'factors': {},
        'overall_score': 0,
        'max_score': 10,
        'recommendation': "No Trade"
    }
    tech_analysis = perform_technical_analysis(ticker)
    if tech_analysis['technical_rating'] == "No Data":
        results['factors']['technical'] = {'score': 0, 'rating': "No Data", 'details': tech_analysis}
        return results
    tech_score = min(5, (tech_analysis['technical_score'] / tech_analysis['max_score']) * 5)
    results['factors']['technical'] = {'score': tech_score, 'rating': tech_analysis['technical_rating'], 'details': tech_analysis}
    results['overall_score'] += tech_score
    options_activity = detect_unusual_options_activity(ticker)
    options_score = min(3, options_activity['unusual_score'] * 0.5)
    results['factors']['options_activity'] = {'score': options_score, 'assessment': options_activity['assessment'], 'details': options_activity}
    results['overall_score'] += options_score
    if news_analysis:
        news_confidence = news_analysis.get('confidence', 'low')
        news_score = {'very-high': 2.0, 'high': 1.5, 'medium': 1.0, 'low': 0.5}.get(news_confidence.lower().replace(' ', '-'), 0)
        results['factors']['news_catalyst'] = {
            'score': news_score,
            'confidence': news_confidence,
            'catalyst': news_analysis.get('market_catalyst', 'Unknown'),
            'sentiment': news_analysis.get('sentiment', 'Unknown'),
            'details': news_analysis
        }
        results['overall_score'] += news_score
    results['market_context'] = get_extended_market_context()
    results['recommendation'] = (
        "Strong Buy" if results['overall_score'] >= 7 else
        "Buy" if results['overall_score'] >= 5 else
        "Watch" if results['overall_score'] >= 3 else
        "No Trade"
    )
    if results['recommendation'] in ["Strong Buy", "Buy"]:
        use_options = options_score > 1.5 or (news_analysis and news_analysis.get('impact_timing') == 'immediate')
        results['strategy'] = generate_trade_strategy(ticker, tech_analysis, options_activity, news_analysis, results['market_context'], use_options)
    return results

def generate_trade_strategy(ticker, tech_analysis, options_activity, news_analysis, market_context, use_options=True):
    try:
        current_price = get_current_price(ticker)
        if current_price is None:
            return {"type": "error", "strategy": "No current price available"}
        support_resistance = tech_analysis.get('support_resistance', {})
        supports = support_resistance.get('supports', [])
        resistances = support_resistance.get('resistances', [])
        nearest_support = min(supports, key=lambda x: abs(x - current_price)) if supports else (current_price * 0.9)
        nearest_resistance = min(resistances, key=lambda x: abs(x - current_price)) if resistances else (current_price * 1.1)
        potential_upside = ((nearest_resistance - current_price) / current_price) * 100
        potential_downside = ((current_price - nearest_support) / current_price) * 100
        risk_reward = potential_upside / potential_downside if potential_downside > 0 else 0
        sentiment = news_analysis.get('sentiment', 'neutral') if news_analysis else 'neutral'
        sentiment_score = news_analysis.get('sentiment_score', 0) if news_analysis else 0
        impact_timing = news_analysis.get('impact_timing', 'unknown') if news_analysis else 'unknown'
        market_trend = market_context.get('trend', 'unknown')
        market_alignment = (
            (sentiment_score > 0 and market_trend in ['bullish', 'strongly bullish']) or
            (sentiment_score < 0 and market_trend in ['bearish', 'strongly bearish'])
        )
        daily_signals = tech_analysis.get('signals', {}).get('daily', {})
        default_stop_pct = -7.5 if sentiment_score > 0 else 7.5
        default_stop_price = round(current_price * (1 + default_stop_pct / 100), 2)
        stock = yf.Ticker(ticker)
        stock_options = stock.options
        best_expiry = None
        call_strikes = []
        put_strikes = []
        if stock_options:
            for exp in stock_options:
                exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.datetime.now()).days
                if impact_timing == "immediate" and days_to_exp >= 7:
                    best_expiry = exp
                    break
                elif impact_timing in ["1-3 days", "1-2 weeks"] and 14 <= days_to_exp <= 35:
                    best_expiry = exp
                    break
                elif 30 <= days_to_exp <= 60:
                    best_expiry = exp
                    break
            if not best_expiry:
                best_expiry = stock_options[0]
            chain = stock.option_chain(best_expiry)
            calls_df = chain.calls
            puts_df = chain.puts
            calls_itm = calls_df[calls_df['strike'] < current_price].sort_values('strike', ascending=False)
            calls_otm = calls_df[calls_df['strike'] >= current_price].sort_values('strike')
            if not calls_itm.empty:
                call_strikes.append(calls_itm.iloc[0]['strike'])
            if len(calls_otm) >= 2:
                call_strikes.extend([calls_otm.iloc[0]['strike'], calls_otm.iloc[1]['strike']])
            elif not calls_otm.empty:
                call_strikes.append(calls_otm.iloc[0]['strike'])
            puts_itm = puts_df[puts_df['strike'] > current_price].sort_values('strike')
            puts_otm = puts_df[puts_df['strike'] <= current_price].sort_values('strike', ascending=False)
            if not puts_itm.empty:
                put_strikes.append(puts_itm.iloc[0]['strike'])
            if len(puts_otm) >= 2:
                put_strikes.extend([puts_otm.iloc[0]['strike'], puts_otm.iloc[1]['strike']])
            elif not puts_otm.empty:
                put_strikes.append(puts_otm.iloc[0]['strike'])
        if use_options:
            bull_options = [opt for opt in options_activity.get('unusual_activity', []) if opt.get('type') == 'call' and sentiment_score >= 0]
            bear_options = [opt for opt in options_activity.get('unusual_activity', []) if opt.get('type') == 'put' and sentiment_score <= 0]
            if sentiment_score > 0.3 and bull_options and daily_signals.get('obv_increasing', False):
                strategy_type = "bullish_options"
                best_options = sorted(bull_options, key=lambda x: x.get('vol_oi_ratio', 0), reverse=True)[:3]
                expiry_days = [(datetime.datetime.strptime(opt['expiration'], '%Y-%m-%d') - datetime.datetime.now()).days for opt in best_options]
                recommend_call = (
                    f"{best_options[0]['strike']} strike, expiring {best_options[0]['expiration']}" if best_options else
                    f"{call_strikes[0] if call_strikes else round(current_price * 1.05, 2)} strike, expiring {best_expiry}"
                )
                if impact_timing == 'immediate' and any(days < 14 for days in expiry_days):
                    strategy = f"Short-term call options - {recommend_call}"
                    stop = f"Stop loss: Close position if underlying drops below ${nearest_support:.2f} or if option loses 40% of premium"
                elif impact_timing in ['1-3 days', '1-2 weeks'] and any(14 <= days <= 45 for days in expiry_days):
                    strategy = f"Medium-term call options - {recommend_call}"
                    stop = f"Stop loss: Close position if underlying drops below ${nearest_support:.2f} or if option loses 40% of premium"
                else:
                    strategy = f"Long call + stock position - {recommend_call}"
                    stop = f"Stop loss: Close position if underlying drops below ${nearest_support:.2f} or if option loses 40% of premium"
            elif sentiment_score < -0.3 and bear_options:
                strategy_type = "bearish_options"
                best_options = sorted(bear_options, key=lambda x: x.get('vol_oi_ratio', 0), reverse=True)[:3]
                recommend_put = (
                    f"{best_options[0]['strike']} strike, expiring {best_options[0]['expiration']}" if best_options else
                    f"{put_strikes[0] if put_strikes else round(current_price * 0.95, 2)} strike, expiring {best_expiry}"
                )
                if impact_timing in ['immediate', '1-3 days']:
                    strategy = f"Put options - {recommend_put}"
                    stop = f"Stop loss: Close position if underlying rises above ${nearest_resistance:.2f} or if option loses 40% of premium"
                else:
                    strategy = f"Bear put spread - Buy {put_strikes[0]} puts and sell {put_strikes[1]} puts, expiring {best_expiry}"
                    stop = f"Stop loss: Close position if underlying rises above ${nearest_resistance:.2f} or if spread loses 40% of premium"
            elif abs(sentiment_score) > 0.5:
                strategy_type = "directional_options"
                if sentiment_score > 0:
                    strategy = f"Long call options - {call_strikes[0] if call_strikes else round(current_price * 1.03, 2)} strike, expiring {best_expiry}"
                    stop = f"Stop loss: Close position if underlying drops below ${nearest_support:.2f} or if option loses 45% of premium"
                else:
                    strategy = f"Long put options - {put_strikes[0] if put_strikes else round(current_price * 0.97, 2)} strike, expiring {best_expiry}"
                    stop = f"Stop loss: Close position if underlying rises above ${nearest_resistance:.2f} or if option loses 45% of premium"
            else:
                strategy_type = "neutral_options"
                if sentiment_score > 0:
                    strategy = f"Bull call spread - Buy {call_strikes[0]} calls and sell {call_strikes[1]} calls, expiring {best_expiry}"
                    stop = f"Stop loss: Close position if underlying drops below ${(current_price * 0.95):.2f} or if spread loses 45% of premium"
                else:
                    strategy = f"Bear put spread - Buy {put_strikes[0]} puts and sell {put_strikes[1]} puts, expiring {best_expiry}"
                    stop = f"Stop loss: Close position if underlying rises above ${(current_price * 1.05):.2f} or if spread loses 45% of premium"
        else:
            if sentiment_score > 0.3 and daily_signals.get('obv_increasing', False):
                strategy_type = "bullish_stock"
                strategy = "Long stock position with stop at nearest support"
                stop = f"Stop loss: ${nearest_support:.2f} (-{potential_downside:.1f}%)"
            elif sentiment_score < -0.3:
                strategy_type = "bearish_stock"
                strategy = "Short stock position with stop at nearest resistance"
                stop = f"Stop loss: ${nearest_resistance:.2f} (+{potential_upside:.1f}%)"
            else:
                strategy_type = "neutral_stock"
                strategy = "Wait for confirmation before entry"
                stop = f"Stop loss: ${default_stop_price:.2f} ({default_stop_pct:.1f}%)"
        position_size = "1-2% of portfolio" if risk_reward <= 2 else "2-3% of portfolio" if risk_reward <= 3 else "3-5% of portfolio"
        if strategy_type.startswith('bullish'):
            entry = f"Enter at current price (${current_price:.2f}) or on pullback to ${(current_price * 0.98):.2f}"
            target = f"Target: ${nearest_resistance:.2f} (+{potential_upside:.1f}%)"
        elif strategy_type.startswith('bearish'):
            entry = f"Enter at current price (${current_price:.2f}) or on bounce to ${(current_price * 1.02):.2f}"
            target = f"Target: ${nearest_support:.2f} (-{potential_downside:.1f}%)"
        else:
            entry = f"Enter on confirmation at ${current_price:.2f} with 25% position, add on momentum"
            target = f"Target: Initial target ${(current_price * (1 + (sentiment_score * 10))):.2f} (~{abs(sentiment_score * 10):.1f}% move)"
        return {
            "type": strategy_type,
            "strategy": strategy,
            "position_size": position_size,
            "entry": entry,
            "target": target,
            "stop_loss": stop,
            "risk_reward": risk_reward,
            "market_aligned": market_alignment
        }
    except Exception as e:
        logger.error(f"Error generating trade strategy for {ticker}: {e}")
        return {"type": "error", "strategy": "Unable to generate strategy"}

def generate_risk_reward_chart(ticker, strategy, run_timestamp):
    try:
        entry = strategy.get('entry', '')
        
        # Improved price extraction with regex
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
        
        # Ensure directory exists
        os.makedirs(os.path.join(RECOMMENDATIONS_DIR, run_timestamp), exist_ok=True)
        
        chart_file = os.path.join(RECOMMENDATIONS_DIR, run_timestamp, f"{ticker}_risk_reward.json")
        with open(chart_file, "w") as f:
            json.dump(chart_data, f, indent=4)
            
    except Exception as e:
        logger.error(f"Error generating risk/reward chart for {ticker}: {e}")

def generate_trade_confidence(ticker, headline, analysis, tech_analysis):
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
        THINKING_MODELS = [
            "deepseek-reasoner",
            "gemini-2.0-flash-thinking-exp-01-21",
            "claude-3-7-sonnet-20250219"
        ]
        config = {'response_mime_type': 'application/json'} if "gemini" in THINKING_MODELS[0] else None
        success, response = robust_api_call(THINKING_MODELS, prompt, config, max_tokens=4000, retries=2)
        if success and 'confidence_score' in response and 'reasoning' in response:
            return response
        logger.error("All thinking models failed for generate_trade_confidence")
        return {'confidence_score': 5, 'reasoning': "Unable to generate confidence assessment"}
    except Exception as e:
        logger.error(f"Error in trade confidence generation: {e}")
        return {'confidence_score': 5, 'reasoning': "Error in analysis"}

def save_enhanced_recommendation(run_timestamp, headline, analysis, summary, trade_evaluation):
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