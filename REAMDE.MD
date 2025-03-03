Multi-Factor Newsfeed Trading System
This project is a sophisticated trading system that analyzes news headlines from multiple RSS feeds, evaluates their potential impact on stock prices, and generates trading recommendations. It combines news sentiment analysis, technical indicators, and market context, leveraging AI models (such as Google Gemini, Anthropic’s Claude, and optionally DeepSeek) to process data and make informed trading decisions. The system also supports backtesting to evaluate trade performance over time.

The project is organized into modular Python files, each responsible for a specific aspect of the system. This README provides a detailed explanation of each file, its purpose, and its major functions, along with instructions for setup and usage.

Table of Contents
Project Structure
File Descriptions
config.py
utils.py
api.py
data.py
analysis.py
trade.py
backtest.py
main.py
How the System Works
Usage
Dependencies
Environment Variables
Notes
Project Structure
The project is structured into the following files, each with a distinct role:

config.py: Centralizes configuration settings, API keys, and constants.
utils.py: Provides utility functions for data management and time operations.
api.py: Handles interactions with AI APIs and rate limiting.
data.py: Manages data retrieval and preprocessing (e.g., stock prices, RSS feeds).
analysis.py: Conducts headline and technical analysis.
trade.py: Evaluates trade opportunities and generates strategies.
backtest.py: Simulates and tracks trade performance.
main.py: Orchestrates the system as the main entry point.
This modular design enhances readability, maintainability, and scalability.

File Descriptions
config.py
Purpose: Centralizes all configuration settings, including logging, file paths, API keys, and system-wide constants.

Major Components:

Logging Setup: Configures the logging format and level (e.g., INFO, DEBUG).
File Paths: Defines directories for recommendations, processed articles, daily stats, watchlists, and backtest results.
API Keys: Loads API keys for Gemini, Anthropic, and (optionally) DeepSeek from environment variables.
RSS Feeds: Lists URLs of RSS feeds to monitor (e.g., financial news sources).
Constants: Defines settings like MIN_OVERALL_SCORE (minimum score for a trade recommendation), PARALLELISM (for batch processing), and ENABLE_BACKTESTING.
Rate Limits: Sets per-minute and daily limits for API calls to prevent quota exhaustion.
utils.py
Purpose: Contains helper functions for file I/O, watchlist management, and time-related operations.

Major Functions:

load_processed_articles(): Reads a JSON file containing IDs of already processed articles.
save_processed_articles(processed_articles): Writes the set of processed article IDs to a JSON file.
load_daily_stats(): Loads daily statistics (e.g., articles processed) from a JSON file.
save_daily_stats(daily_stats): Saves updated daily stats to a JSON file.
load_watchlist(): Retrieves the current watchlist from a JSON file.
save_watchlist(watchlist): Saves the watchlist to a JSON file.
add_to_watchlist(ticker, reason, expected_impact, timeframe, score=None): Adds a ticker to the watchlist with details like reason and expected price impact.
update_watchlist_performance(): Updates watchlist entries with current stock prices and performance metrics.
is_recent(entry): Checks if an RSS feed entry was published within the last 24 hours.
api.py
Purpose: Manages API calls to AI models (Gemini, Claude, DeepSeek) and enforces rate limits.

Major Functions:

rate_limit(model): Ensures API calls stay within per-minute limits for the specified model.
check_rate_limits(model): Verifies that daily API call limits are not exceeded.
clean_response_text(response_text): Formats and cleans raw API response text.
robust_api_call(models, prompt, config=None, max_tokens=4000, thinking_budget=None, retries=3, initial_delay=2): Executes API calls with retry logic and fallback to a default model if errors occur.
parse_market_events_text(text): Extracts market events and risk levels from API responses.
check_market_events(model): Queries the AI for significant market-moving events and assesses overall market risk.
data.py
Purpose: Fetches and preprocesses data, including stock prices, RSS feed content, and company fundamentals.

Major Functions:

fetch_full_text(entry): Retrieves the full text of an RSS feed entry.
is_valid_ticker(ticker): Validates a ticker symbol by checking its historical data availability.
get_current_price(ticker, retries=3, delay=2): Fetches the latest closing price for a ticker with retry logic.
get_cache_timestamp(): Creates a timestamp for caching data.
get_stock_data(ticker, cache_timestamp): Retrieves and caches stock info and historical volatility.
get_options_chain_data(ticker, cache_timestamp): Fetches and caches options chain data.
get_company_fundamentals(ticker): Gathers financial fundamentals (e.g., P/E ratio, revenue).
get_market_context(): Collects market data (e.g., S&P 500, NASDAQ, VIX).
get_extended_market_context(): Adds additional indices (e.g., DJIA, Russell 2000) to market context.
analysis.py
Purpose: Analyzes news headlines, sentiment, and stock technicals to inform trading decisions.

Major Functions:

batch_generic_check(headlines, model=MODEL_JSON): Classifies headlines as 'generic' or 'not generic'.
batch_analyze_headlines(items, model=MODEL_JSON): Analyzes headlines for company mentions, sentiment, and impact timing.
get_ticker_symbol(headline, analysis_text, model=MODEL_TICKER): Extracts ticker symbols from headlines and analysis.
analyze_company_context(ticker, headline, analysis_text, fundamentals): Assesses how news aligns with a company’s fundamentals.
perform_technical_analysis(ticker): Computes technical indicators (e.g., SMA, MACD, RSI).
calculate_pivot_points(df): Determines pivot points for support and resistance levels.
detect_unusual_options_activity(ticker): Identifies unusual options trading volume or patterns.
trade.py
Purpose: Evaluates trading opportunities and generates actionable strategies.

Major Functions:

evaluate_trade_opportunity(ticker, news_analysis=None): Assesses trade viability using technicals, options, and news sentiment.
generate_trade_strategy(ticker, tech_analysis, options_activity, news_analysis, market_context, use_options=True): Creates a strategy with entry points, targets, and stop losses.
generate_risk_reward_chart(ticker, strategy, run_timestamp): Produces a JSON file for risk/reward visualization.
generate_trade_confidence(ticker, headline, analysis, tech_analysis): Rates confidence in a trade using an AI model.
save_enhanced_recommendation(run_timestamp, headline, analysis, summary, trade_evaluation): Saves detailed trade recommendations as text files.
summarize_technical_signals(trade_evaluation): Summarizes key technical signals.
summarize_options_activity(trade_evaluation): Summarizes unusual options activity.
backtest.py
Purpose: Simulates and monitors trade performance over time.

Major Functions:

simulate_trade_performance(ticker, strategy, days_to_monitor=10): Starts a backtest record for a trade strategy.
update_backtests(): Updates backtest records with current prices and checks for target/stop triggers.
main.py
Purpose: Serves as the entry point, coordinating the entire workflow.

Major Functions:

process_feed(processed_articles, daily_stats, run_timestamp): Fetches and filters recent RSS feed articles.
process_batch(items, today_str, run_timestamp, processed_articles, daily_stats): Processes batches of headlines for analysis.
process_ticker_analysis(ticker, headline, analysis, summary, run_timestamp): Handles ticker-specific analysis and trade evaluation.
run_maintenance_tasks(): Updates watchlists, backtests, and checks market events.
main(): Runs the system in a loop or as a single execution.
How the System Works
Setup:
Loads configurations, initializes API clients, and sets up directories.
Maintenance:
Updates watchlist performance and backtest records; checks market events.
News Fetching:
Pulls recent articles from RSS feeds and filters out processed ones.
Analysis:
Classifies headlines, extracts tickers, and analyzes sentiment and impact.
Trade Evaluation:
Combines news analysis, technicals, and options data to assess trades.
Generates strategies for high-confidence opportunities.
Output:
Saves recommendations if they meet the score threshold.
Adds tickers to the watchlist.
Backtesting:
Simulates trade performance and updates results periodically.
Execution:
Runs continuously (every 5 minutes) or as a single pass.
Usage
Ensure dependencies are installed and environment variables are set. Run the system with:

bash

Collapse

Wrap

Copy
python3 main.py
For a single execution:

bash

Collapse

Wrap

Copy
python3 main.py --single-run
Dependencies
feedparser: Parses RSS feeds.
yfinance: Fetches stock and market data.
talib: Calculates technical indicators.
pandas: Handles data manipulation.
numpy: Supports numerical operations.
google.generativeai: Interfaces with Gemini API.
anthropic: Interfaces with Claude API.
openai: Optional for DeepSeek or other OpenAI-compatible APIs.
Install with:

bash

Collapse

Wrap

Copy
pip install feedparser yfinance TA-Lib pandas numpy google-generativeai anthropic openai
Note: TA-Lib may require manual installation; see its documentation.

Environment Variables
Set these in your environment:

GEMINI_API_KEY: Google Gemini API key.
ANTHROPIC_API_KEY: Anthropic Claude API key.
DEEPSEEK_API_KEY (optional): DeepSeek API key.
Example:

bash

Collapse

Wrap

Copy
export GEMINI_API_KEY='your_gemini_key'
export ANTHROPIC_API_KEY='your_anthropic_key'
export DEEPSEEK_API_KEY='your_deepseek_key'  # Optional
Notes
Modularity: Each file has a single responsibility, easing updates and debugging.
Rate Limiting: Prevents API quota issues with per-minute and daily checks.
Caching: Reduces redundant API calls for stock and options data.
Error Handling: Includes retries and logging for robustness.
Extensibility: Designed for easy addition of new features or models.
This README provides a complete guide to understanding and using the trading system effectively.