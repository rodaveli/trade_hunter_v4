import talib
import pandas as pd
import numpy as np
import yfinance as yf
from api import robust_api_call
from config import MODEL_JSON, MODEL_TICKER, logger
from data import is_valid_ticker

def batch_generic_check(headlines, model=MODEL_JSON):
    prompt = (
        "Classify each headline as 'generic' or 'not generic'. "
        "Generic headlines are routine or minor updates such as regular earnings reports, scheduled executive changes, or small product launches. "
        "Not generic headlines involve significant, potentially market-moving events like mergers, acquisitions, unexpected earnings surprises, major product breakthroughs, or high-impact regulatory changes. "
        "IMPORTANT: If there's any doubt, classify as 'not generic' to avoid missing potentially valuable trading opportunities. Be conservative - only classify as 'not generic' if the headline strongly indicates a significant market-moving event."
        "Return JSON: [{'headline': <string>, 'answer': <string>}].\n\n"
        "Examples:\n"
        "- 'Company X Reports Q2 Earnings In Line With Expectations' -> generic\n"
        "- 'Company X Earnings Beat Expectations by 5%' -> not generic\n"
        "- 'Company Y Announces Merger with Z' -> not generic\n"
        "- 'Company A Appoints New CFO As Planned' -> generic\n"
        "- 'Company A Unexpectedly Replaces CFO After Accounting Issues' -> not generic\n"
        "- 'Company B Beats Earnings Estimates by 10%' -> not generic\n"
        "- 'Company C Launches New Smartphone Model' -> generic\n"
        "- 'Company C's New Smartphone Features Breakthrough Battery Technology' -> not generic\n"
        "- 'Company D Receives FDA Approval for Revolutionary Drug' -> not generic\n\n"
        "Headlines:\n"
        + "\n".join(f"{i}. {h}" for i, h in enumerate(headlines, 1))
    )
    config = {'response_mime_type': 'application/json'}
    success, response = robust_api_call([model], prompt, config, max_tokens=2000)
    if success and isinstance(response, list):
        return {item["headline"]: (item["answer"].lower() == "generic") for item in response if 'headline' in item and 'answer' in item}
    logger.error("Failed to classify headlines")
    return {}

def batch_analyze_headlines(items, model=MODEL_JSON):
    prompt = (
        "Analyze each item using the headline and full text. Check if it's about a public company and a market catalyst. "
        "If it's about a public company, include the stock ticker symbol. "
        "Also determine sentiment (positive, negative, neutral) with a score (-1 to 1). "
        "Evaluate the potential timing of impact (immediate, 1-3 days, 1-2 weeks, 1+ month). "
        "Estimate approximate price impact percentage range. "
        "Detect if this is likely a 'sell the_news' event where positive headlines might cause selling. "
        "Assign confidence as 'very-high', 'high', 'medium', or 'low' based on the following criteria: "
        "'very-high' for clear, significant market-moving events (e.g., major mergers, acquisitions, unexpected earnings surprises); "
        "'high' for likely impactful events; 'medium' for possibly relevant events; 'low' for generic or unrelated news. "
        "Return JSON: [{'headline': <string>, 'is_public_company': <bool>, 'ticker': <string or null>, 'market_catalyst': <string>, "
        "'confidence': <'very-high'|'high'|'medium'|'low'>, 'sentiment': <string>, 'sentiment_score': <float>, "
        "'impact_timing': <string>, 'price_impact_range': <string>, 'sell_the_news': <bool>, 'analysis': <string>}].\n\n"
        "If it's a public company, include the stock ticker symbol in the 'ticker' field. If no ticker can be determined, set 'ticker' to null.\n\nItems:\n"
        + "\n".join(f"{i}. Headline: {item['headline']}\n   Full Text: {item['full_text'][:2000]}..." for i, item in enumerate(items, 1))
    )
    config = {'response_mime_type': 'application/json'}
    success, response = robust_api_call([model], prompt, config, max_tokens=2000)
    if success and isinstance(response, list):
        required_keys = ['headline', 'is_public_company', 'ticker', 'market_catalyst', 'confidence', 'sentiment', 'sentiment_score', 'impact_timing', 'price_impact_range', 'sell_the_news', 'analysis']
        return {item["headline"]: item for item in response if isinstance(item, dict) and all(k in item for k in required_keys)}
    logger.error("Failed to analyze headlines")
    return {}

def get_ticker_symbol(headline: str, analysis_text: str, model=MODEL_TICKER) -> str:
    """
    Extract the stock ticker symbol from a headline and analysis using Google Gemini with grounding.
    
    Args:
        headline: The news headline
        analysis_text: Additional analysis text that might contain company information
        model: The model to use for extraction (default is MODEL_TICKER from config)
        
    Returns:
        A valid ticker symbol or None if no valid ticker is found
    """
    try:
        # First attempt with Google Search grounding
        prompt = (
            f"TASK: Extract the stock ticker symbol from the following headline and analysis.\n\n"
            f"HEADLINE: {headline}\n\n"
            f"ANALYSIS: {analysis_text}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Identify the public company or companies mentioned in the headline or analysis\n"
            f"2. Return ONLY the most relevant stock ticker symbol as a JSON object with key 'ticker'\n"
            f"3. If multiple companies are mentioned, choose the main subject of the headline\n"
            f"4. If no ticker can be confidently determined, return {{'ticker': null}}\n"
            f"5. Do NOT include any explanation, just return the JSON object\n"
            f"6. Only include standard US stock tickers (NYSE, NASDAQ, etc.)\n"
            f"7. If the ticker is uncertain, use Google Search to verify it\n\n"
            f"RESPONSE FORMAT: {{'ticker': 'SYMBOL'}} or {{'ticker': null}} if no ticker found"
        )
        
        # Configure Google Search as a tool as shown in the documentation
        google_search_config = {
            'response_mime_type': 'application/json',
            'tools': [{'google_search': {}}]
        }
        
        # Try with search grounding first
        success, response = robust_api_call([model], prompt, google_search_config)
        
        if success and isinstance(response, dict) and 'ticker' in response:
            ticker = response['ticker']
            if is_valid_ticker(ticker):  # FIXED: Use is_valid_ticker instead of ticker_is_valid
                logger.info(f"Found and verified ticker (with grounding): {ticker}")
                return ticker
        
        # Fall back to standard prompt if grounding doesn't work
        config = {'response_mime_type': 'application/json'}
        success, response = robust_api_call([model], prompt, config)
        
        if success and isinstance(response, dict) and 'ticker' in response:
            ticker = response['ticker']
            if is_valid_ticker(ticker):  # FIXED: Use is_valid_ticker instead of ticker_is_valid
                logger.info(f"Found and verified ticker: {ticker}")
                return ticker
            
        logger.info(f"No valid ticker found for headline: '{headline}'")
        return None
    except Exception as e:
        logger.error(f"Error extracting ticker symbol: {e}")
        return None

# This function is no longer needed since we're using is_valid_ticker
# def ticker_is_valid(ticker):
#     """Helper function to validate a ticker symbol"""
#     if ticker is None:
#         return False
#         
#     ticker = str(ticker).strip().upper()
#     
#     # Quick validation checks
#     if not ticker or len(ticker) > 5 or not ticker.isalnum():
#         return False
#     
#     # Validate ticker using yfinance
#     return is_valid_ticker(ticker)

def analyze_company_context(ticker, headline, analysis_text, fundamentals):
    try:
        if not fundamentals:
            return {
                "relevant": False,
                "impact_assessment": "No fundamental data available to assess impact",
                "financial_strength": "Unknown",
                "risk_adjustment": 0
            }
        metrics_text = "\n".join([f"{k}: {v}" for k, v in fundamentals.items() if v is not None])
        prompt = (
            f"As a financial analyst, evaluate how the following news relates to {ticker}'s fundamentals.\n\n"
            f"News Headline: {headline}\n\n"
            f"News Analysis: {analysis_text}\n\n"
            f"Company Financials:\n{metrics_text}\n\n"
            f"Analyze:\n"
            f"1. Is this news materially significant given the company's size and business? (Yes/No)\n"
            f"2. Briefly analyze how this news relates to the company's financials (2-3 sentences)\n"
            f"3. Assess company's financial position as 'strong', 'moderate', or 'weak'\n"
            f"4. Suggest a position size adjustment factor from -2 to +2 (-2 = much smaller position due to weak fundamentals, 0 = no adjustment, +2 = increase position size due to strong fundamentals)\n\n"
            f"Provide a JSON response with keys: 'relevant', 'impact_assessment', 'financial_strength', 'risk_adjustment'."
        )
        THINKING_MODELS = [
            "deepseek-reasoner",
            "gemini-2.0-flash-thinking-exp-01-21",
            "claude-3-7-sonnet-20250219"
        ]
        config = {'response_mime_type': 'application/json'} if "gemini" in THINKING_MODELS[0] else None
        success, response = robust_api_call(THINKING_MODELS, prompt, config, max_tokens=4000, retries=2)
        if success:
            return response
        logger.error("All thinking models failed for analyze_company_context")
        return {
            "relevant": False,
            "impact_assessment": "Failed to get valid response",
            "financial_strength": "Unknown",
            "risk_adjustment": 0
        }
    except Exception as e:
        logger.error(f"Error in company context analysis: {e}")
        return {
            "relevant": False,
            "impact_assessment": "Error in analysis",
            "financial_strength": "Unknown",
            "risk_adjustment": 0
        }

def perform_technical_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        timeframes = {
            'intraday': {'period': "5d", 'interval': "15m"},
            'daily': {'period': "120d", 'interval': "1d"},
            'weekly': {'period': "1y", 'interval': "1wk"}
        }
        data = {}
        for tf_name, tf_params in timeframes.items():
            df = stock.history(**tf_params)
            if not df.empty and len(df) > 30:
                data[tf_name] = df
        if not data:
            logger.error(f"No data available for technical analysis of {ticker}")
            return {
                'ticker': ticker,
                'technical_score': 0,
                'max_score': 14,
                'technical_rating': "No Data",
                'signals': {},
                'support_resistance': {}
            }
        signals = {}
        tech_score = 0
        max_score = 14
        for timeframe, df in data.items():
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            signals[timeframe] = {}
            sma20 = talib.SMA(close, timeperiod=20)
            sma50 = talib.SMA(close, timeperiod=50)
            sma200 = talib.SMA(close, timeperiod=200) if len(close) >= 200 else pd.Series([np.nan] * len(close))
            price_above_sma20 = close.iloc[-1] > sma20.iloc[-1]
            price_above_sma50 = close.iloc[-1] > sma50.iloc[-1]
            price_above_sma200 = False if np.isnan(sma200.iloc[-1]) else close.iloc[-1] > sma200.iloc[-1]
            ma_alignment_bullish = sma20.iloc[-1] > sma50.iloc[-1]
            ma_alignment_strongly_bullish = ma_alignment_bullish and (not np.isnan(sma200.iloc[-1]) and sma50.iloc[-1] > sma200.iloc[-1])
            signals[timeframe].update({
                'price_above_sma20': price_above_sma20,
                'price_above_sma50': price_above_sma50,
                'price_above_sma200': price_above_sma200,
                'ma_alignment_bullish': ma_alignment_bullish,
                'ma_alignment_strongly_bullish': ma_alignment_strongly_bullish
            })
            if timeframe in ['daily', 'weekly']:
                if price_above_sma20: tech_score += 0.5
                if price_above_sma50: tech_score += 0.5
                if price_above_sma200 and timeframe == 'daily': tech_score += 1
                if ma_alignment_strongly_bullish: tech_score += 1
            if timeframe == 'daily':
                sma50_above_sma200 = False if np.isnan(sma200.iloc[-1]) else sma50.iloc[-1] > sma200.iloc[-1]
                signals[timeframe]['sma50_above_sma200'] = sma50_above_sma200
                if sma50_above_sma200:
                    tech_score += 1
                roc = talib.ROC(close, timeperiod=12)
                roc_value = roc.iloc[-1] if not np.isnan(roc.iloc[-1]) else 0
                roc_positive = roc_value > 0
                signals[timeframe]['roc'] = roc_value
                signals[timeframe]['roc_positive'] = roc_positive
                if roc_positive:
                    tech_score += 0.5
            macd, macd_signal, macd_hist = talib.MACD(close)
            macd_bullish = macd_hist.iloc[-1] > 0
            macd_bullish_crossover = macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0
            signals[timeframe].update({
                'macd_bullish': macd_bullish,
                'macd_bullish_crossover': macd_bullish_crossover
            })
            if timeframe == 'daily' and macd_bullish: tech_score += 1
            if timeframe in ['daily', 'weekly'] and macd_bullish_crossover: tech_score += 1
            rsi = talib.RSI(close, timeperiod=14)
            rsi_value = rsi.iloc[-1]
            rsi_bullish = 40 <= rsi_value <= 70
            rsi_overbought = rsi_value > 70
            rsi_oversold = rsi_value < 30
            signals[timeframe].update({
                'rsi_value': rsi_value,
                'rsi_bullish': rsi_bullish,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold
            })
            if timeframe == 'daily':
                if rsi_bullish: tech_score += 0.5
                if rsi_oversold: tech_score += 0.5
            slowk, slowd = talib.STOCH(high, low, close)
            stoch_bullish_crossover = slowk.iloc[-1] > slowd.iloc[-1] and slowk.iloc[-2] <= slowd.iloc[-2]
            stoch_oversold = slowk.iloc[-1] < 20 and slowd.iloc[-1] < 20
            signals[timeframe].update({
                'stoch_k': slowk.iloc[-1],
                'stoch_d': slowd.iloc[-1],
                'stoch_bullish_crossover': stoch_bullish_crossover,
                'stoch_oversold': stoch_oversold
            })
            if timeframe == 'daily' and stoch_bullish_crossover: tech_score += 1
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            bb_width = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
            bb_squeeze = bb_width < talib.SMA((upper - lower) / middle, timeperiod=20).iloc[-1] * 0.85
            bb_breakout_up = close.iloc[-1] > upper.iloc[-1] and close.iloc[-2] <= upper.iloc[-2]
            signals[timeframe].update({
                'bb_width': bb_width,
                'bb_squeeze': bb_squeeze,
                'bb_breakout_up': bb_breakout_up,
                'bb_percent_b': (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            })
            if timeframe == 'daily':
                if bb_squeeze: tech_score += 0.5
                if bb_breakout_up: tech_score += 1
            obv = talib.OBV(close, volume)
            obv_increasing = obv.iloc[-1] > obv.iloc[-20:].mean()
            signals[timeframe]['obv_increasing'] = obv_increasing
            if timeframe == 'daily' and obv_increasing: tech_score += 1
            adl = talib.AD(high, low, close, volume)
            adl_increasing = adl.iloc[-1] > adl.iloc[-20:].mean()
            signals[timeframe]['adl_increasing'] = adl_increasing
            if timeframe == 'daily' and adl_increasing: tech_score += 1
            cmf_period = min(20, len(df) - 1)
            mf_multiplier = ((close.iloc[-cmf_period:] - low.iloc[-cmf_period:]) - (high.iloc[-cmf_period:] - close.iloc[-cmf_period:])) / (high.iloc[-cmf_period:] - low.iloc[-cmf_period:])
            mf_volume = mf_multiplier * volume.iloc[-cmf_period:]
            cmf = mf_volume.sum() / volume.iloc[-cmf_period:].sum()
            signals[timeframe].update({'cmf': cmf, 'cmf_positive': cmf > 0})
            if timeframe == 'daily' and cmf > 0: tech_score += 1
        technical_rating = (
            "Very Bullish" if tech_score >= 10 else
            "Bullish" if tech_score >= 7 else
            "Neutral" if tech_score >= 4 else
            "Bearish" if tech_score >= 2 else
            "Very Bearish"
        )
        support_resistance = calculate_pivot_points(data.get('daily', pd.DataFrame())) if 'daily' in data else {}
        return {
            'ticker': ticker,
            'technical_score': tech_score,
            'max_score': max_score,
            'technical_rating': technical_rating,
            'signals': signals,
            'support_resistance': support_resistance
        }
    except Exception as e:
        logger.error(f"Error in technical analysis for {ticker}: {e}")
        return {
            'ticker': ticker,
            'technical_score': 0,
            'max_score': 14,
            'technical_rating': "Error",
            'signals': {},
            'support_resistance': {}
        }

def calculate_pivot_points(df):
    try:
        if df.empty:
            return {}
        high = df['High'].max()
        low = df['Low'].min()
        close = df['Close'].iloc[-1]
        pp = (high + low + close) / 3
        s1, s2, s3 = (2 * pp) - high, pp - (high - low), low - 2 * (high - pp)
        r1, r2, r3 = (2 * pp) - low, pp + (high - low), high + 2 * (pp - low)
        return {
            'current_price': close,
            'pivot': pp,
            'supports': [s1, s2, s3],
            'resistances': [r1, r2, r3]
        }
    except Exception as e:
        logger.error(f"Error calculating pivot points: {e}")
        return {}

def detect_unusual_options_activity(ticker):
    """
    Detect unusual options activity for a given ticker symbol.
    This function should be implemented based on your requirements.
    """
    # Implementation would go here
    return {
        'unusual_score': 0,
        'assessment': 'No unusual options activity detected',
        'unusual_activity': []
    }