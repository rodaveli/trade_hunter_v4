import os
import logging
import google.genai as genai
import anthropic
from openai import OpenAI

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECOMMENDATIONS_DIR = os.path.join(SCRIPT_DIR, "recommendations")
PROCESSED_FILE = os.path.join(SCRIPT_DIR, "processed_articles.json")
STATS_FILE = os.path.join(SCRIPT_DIR, "daily_stats.json")
WATCHLIST_FILE = os.path.join(SCRIPT_DIR, "watchlist.json")
BACKTEST_DIR = os.path.join(SCRIPT_DIR, "backtest_results")

# Ensure directories exist
os.makedirs(RECOMMENDATIONS_DIR, exist_ok=True)
os.makedirs(os.path.join(RECOMMENDATIONS_DIR, "by_ticker"), exist_ok=True)
os.makedirs(BACKTEST_DIR, exist_ok=True)

# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    logger.warning("DEEPSEEK_API_KEY environment variable is not set")

# Model Configuration
USE_SONNET_FOR_THINKING = True
MODEL_ANALYSIS = "gemini-2.0-flash"
MODEL_THINKING = "claude-3-7-sonnet-20250219" if USE_SONNET_FOR_THINKING else "gemini-2.0-flash"
MODEL_TICKER = "gemini-2.0-flash"  # Using Gemini 2.0 Flash for ticker extraction with grounding
MODEL_JSON = MODEL_ANALYSIS

# Single run mode flag (for command-line control)
SINGLE_RUN_MODE = len(os.sys.argv) > 1 and os.sys.argv[1] == "--single-run"

# Initialize API clients
client = genai.Client(api_key=GEMINI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None

# RSS Feed URLs
FEED_URLS = [
    "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEFpRXEMGSQ5STlFQGEZdE0RPAwoBThFNXBBdVVM=",
    "https://rss.app/feeds/_kbmkWNuMUCeQeZ29.xml",
    "https://www.prnewswire.com/rss/news-releases-list.rss",
    "https://seekingalpha.com/market_currents.xml",
    "https://www.benzinga.com/feed/",
    "https://www.investors.com/category/news/rss/",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
]

# Configuration Options
MIN_OVERALL_SCORE = 6.0
PARALLELISM = 4
ENABLE_BACKTESTING = True
BATCH_SIZE = 20
ANTHROPIC_TOKEN_LIMIT = 8000  # tokens per minute

# Rate Limits
RATE_LIMITS = {
    MODEL_ANALYSIS: {'MAX_RPM': 30, 'MAX_RPD': 1500},
    MODEL_THINKING: {'MAX_RPM': 10, 'MAX_RPD': 500},
    MODEL_TICKER: {'MAX_RPM': 30, 'MAX_RPD': 1500},
    MODEL_JSON: {'MAX_RPM': 30, 'MAX_RPD': 1500},
    "deepseek-reasoner": {'MAX_RPM': 1000, 'MAX_RPD': 100000},
    "gemini-2.0-flash-thinking-exp-01-21": {'MAX_RPM': 30, 'MAX_RPD': 1500},
    "claude-3-7-sonnet-20250219": {'MAX_RPM': 10, 'MAX_RPD': 500},
    "gemini-2.0-flash": {'MAX_RPM': 30, 'MAX_RPD': 1500}
}