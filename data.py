import feedparser
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
from functools import lru_cache
from config import logger

def fetch_full_text(entry):
    return entry.get("content", [{"value": ""}])[0].get("value", "") or entry.get("summary", "")[:2000]

@lru_cache(maxsize=1000)
def is_valid_ticker(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        history = ticker_obj.history(period="1d")
        return not history.empty and 'Close' in history.columns and not history['Close'].isna().all()
    except Exception as e:
        logger.warning(f"Ticker validation failed for {ticker}: {e}")
        return False

def get_current_price(ticker, retries=3, delay=2):
    for attempt in range(retries):
        try:
            ticker_obj = yf.Ticker(ticker)
            history = ticker_obj.history(period="5d")
            if history.empty or 'Close' not in history.columns:
                logger.warning(f"No data for {ticker} on attempt {attempt + 1}")
                if attempt < retries - 1:
                    time.sleep(delay)
                continue
            for i in range(len(history) - 1, -1, -1):
                close_price = history['Close'].iloc[i]
                if not pd.isna(close_price) and close_price > 0:
                    logger.info(f"Fetched valid price ${close_price:.2f} for {ticker} from {history.index[i].date()}")
                    return close_price
            logger.warning(f"No valid close price found for {ticker}")
            return None
        except Exception as e:
            logger.error(f"Error fetching price for {ticker} on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error(f"Failed to fetch price for {ticker} after {retries} attempts")
    return None

def get_cache_timestamp():
    return int(datetime.datetime.now().timestamp() // 3600)

@lru_cache(maxsize=100)
def get_stock_data(ticker: str, cache_timestamp: int) -> dict:
    try:
        stock = yf.Ticker(ticker)
        data = stock.info
        hist = stock.history(period="60d")
        if not hist.empty and len(hist) > 20:
            data['historical_volatility'] = hist['Close'].pct_change().std() * np.sqrt(252) * 100
        return data if data and 'currentPrice' in data else {}
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {e}")
        return {}

@lru_cache(maxsize=100)
def get_options_chain_data(ticker: str, cache_timestamp: int) -> dict:
    try:
        ticker_obj = yf.Ticker(ticker)
        expiration_dates = ticker_obj.options[:5]
        if not expiration_dates:
            return {}
        options_data = {}
        for expiry in expiration_dates:
            chain = ticker_obj.option_chain(expiry)
            calls_df = chain.calls.copy()
            puts_df = chain.puts.copy()
            expiry_date = datetime.datetime.strptime(expiry, '%Y-%m-%d')
            days_to_expiry = (expiry_date - datetime.datetime.now()).days
            stock_price = get_current_price(ticker)
            if stock_price is None:
                continue
            for df in [calls_df, puts_df]:
                df['theta_per_day'] = (df['lastPrice'] * 0.1) / max(days_to_expiry, 1)
                if 'calls' in df.columns:
                    df['breakeven'] = df['strike'] + df['lastPrice']
                else:
                    df['breakeven'] = df['strike'] - df['lastPrice']
                df['pct_to_breakeven'] = abs((df['breakeven'] - stock_price) / stock_price * 100)
            options_data[expiry] = {
                "calls": calls_df.to_dict(orient="records"),
                "puts": puts_df.to_dict(orient="records")
            }
        return options_data
    except Exception as e:
        logger.error(f"Error fetching options data for {ticker}: {e}")
        return {}

def get_company_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        if not income_stmt.empty and not balance_sheet.empty:
            recent_yr = income_stmt.columns[0]
            prev_yr = income_stmt.columns[1] if len(income_stmt.columns) > 1 else None
            revenue = income_stmt.loc['TotalRevenue', recent_yr] if 'TotalRevenue' in income_stmt.index else None
            revenue_growth = ((revenue / income_stmt.loc['TotalRevenue', prev_yr]) - 1) * 100 if prev_yr and revenue else None
            net_income = income_stmt.loc['NetIncome', recent_yr] if 'NetIncome' in income_stmt.index else None
            profit_margin = (net_income / revenue) * 100 if revenue and net_income else None
            total_assets = balance_sheet.loc['TotalAssets', recent_yr] if 'TotalAssets' in balance_sheet.index else None
            total_debt = balance_sheet.loc['TotalDebt', recent_yr] if 'TotalDebt' in balance_sheet.index else None
            cash = balance_sheet.loc['Cash', recent_yr] if 'Cash' in balance_sheet.index else None
            debt_to_assets = (total_debt / total_assets) if total_assets and total_debt else None
            operating_cf = cashflow.loc['OperatingCashFlow', recent_yr] if not cashflow.empty and 'OperatingCashFlow' in cashflow.index else None
            fcf = operating_cf - cashflow.loc['CapitalExpenditures', recent_yr] if not cashflow.empty and 'CapitalExpenditures' in cashflow.index and operating_cf else None
            info = stock.info
            market_cap = info.get('marketCap')
            beta = info.get('beta')
            pe_ratio = info.get('trailingPE')
            forward_pe = info.get('forwardPE')
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            return {
                "ticker": ticker,
                "market_cap_billions": round(market_cap / 1e9, 2) if market_cap else None,
                "revenue_billions": round(revenue / 1e9, 2) if revenue else None,
                "revenue_growth_pct": round(revenue_growth, 2) if revenue_growth else None,
                "profit_margin_pct": round(profit_margin, 2) if profit_margin else None,
                "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                "forward_pe": round(forward_pe, 2) if forward_pe else None,
                "debt_to_assets": round(debt_to_assets, 2) if debt_to_assets else None,
                "cash_billions": round(cash / 1e9, 2) if cash else None,
                "free_cash_flow_billions": round(fcf / 1e9, 2) if fcf else None,
                "dividend_yield_pct": round(dividend_yield, 2),
                "beta": round(beta, 2) if beta else None
            }
    except Exception as e:
        logger.error(f"Error getting fundamentals for {ticker}: {e}")
    return {}

def get_market_context():
    try:
        spy = yf.Ticker("SPY")
        qqq = yf.Ticker("QQQ")
        vix = yf.Ticker("^VIX")
        spy_data = spy.history(period="60d")
        qqq_data = qqq.history(period="60d")
        vix_data = vix.history(period="60d")
        if spy_data.empty or qqq_data.empty or vix_data.empty:
            logger.warning("Market context data is empty")
            return {}
        spy_current = spy_data["Close"].iloc[-1]
        qqq_current = qqq_data["Close"].iloc[-1]
        vix_current = vix_data["Close"].iloc[-1]
        import talib  # Import here to avoid circular dependencies with analysis.py
        spy_sma20 = talib.SMA(spy_data["Close"], timeperiod=20).iloc[-1]
        spy_sma50 = talib.SMA(spy_data["Close"], timeperiod=50).iloc[-1]
        qqq_sma20 = talib.SMA(qqq_data["Close"], timeperiod=20).iloc[-1]
        spy_rsi = talib.RSI(spy_data["Close"], timeperiod=14).iloc[-1]
        market_trend = (
            "strongly bullish" if spy_current > spy_sma20 and spy_sma20 > spy_sma50 else
            "bullish" if spy_current > spy_sma20 else
            "strongly bearish" if spy_current < spy_sma20 and spy_sma20 < spy_sma50 else
            "bearish" if spy_current < spy_sma20 else
            "neutral"
        )
        vix_sma20 = talib.SMA(vix_data["Close"], timeperiod=20).iloc[-1]
        volatility = (
            "very low" if vix_current < 15 else
            "low" if vix_current < 20 else
            "moderate" if vix_current < 30 else
            "high" if vix_current < 40 else
            "extreme"
        )
        return {
            "spy_price": spy_current,
            "spy_change_1d": ((spy_current / spy_data["Close"].iloc[-2]) - 1) * 100,
            "spy_rsi": spy_rsi,
            "trend": market_trend,
            "vix": vix_current,
            "volatility": volatility,
            "volatility_trend": "increasing" if vix_current > vix_sma20 else "decreasing",
            "market_breadth": "narrow" if qqq_current / qqq_data["Close"].iloc[-10] > spy_current / spy_data["Close"].iloc[-10] else "broad"
        }
    except Exception as e:
        logger.error(f"Error getting market context: {e}")
        return {}

def get_extended_market_context():
    try:
        market_context = get_market_context()
        indices = {'DJIA': '^DJI', 'NASDAQ': '^IXIC', 'Russell2000': '^RUT', 'TLT': 'TLT', 'DXY': 'DX-Y.NYB'}
        for name, symbol in indices.items():
            try:
                idx = yf.Ticker(symbol)
                hist = idx.history(period="5d")
                if not hist.empty:
                    market_context[f'{name}_change_1d'] = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
            except Exception as e:
                logger.warning(f"Error getting {name} data: {e}")
        return market_context
    except Exception as e:
        logger.error(f"Error in extended market context: {e}")
        return {}