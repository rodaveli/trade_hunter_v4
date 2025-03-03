import time
import json
import html
import re
from functools import lru_cache
from config import RATE_LIMITS, ANTHROPIC_TOKEN_LIMIT, client, anthropic_client, deepseek_client, logger
from collections import deque
from threading import Lock
import datetime

# Rate limiting variables
daily_request_counts = {model: 0 for model in RATE_LIMITS}
last_request_date = datetime.date.today()
MODEL_REQUEST_TIMES = {model: deque() for model in RATE_LIMITS}
anthropic_token_tracker = deque()
anthropic_lock = Lock()

def rate_limit(model):
    if model not in MODEL_REQUEST_TIMES:
        return
    now = time.time()
    while MODEL_REQUEST_TIMES[model] and MODEL_REQUEST_TIMES[model][0] < now - 60:
        MODEL_REQUEST_TIMES[model].popleft()
    if len(MODEL_REQUEST_TIMES[model]) >= RATE_LIMITS[model]['MAX_RPM']:
        wait_until = MODEL_REQUEST_TIMES[model][0] + 60
        sleep_time = wait_until - now
        if sleep_time > 0:
            logger.info(f"Rate limit reached for {model}. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
    MODEL_REQUEST_TIMES[model].append(time.time())

def check_rate_limits(model):
    global daily_request_counts, last_request_date
    current_date = datetime.date.today()
    if current_date != last_request_date:
        daily_request_counts = {m: 0 for m in RATE_LIMITS}
        last_request_date = current_date
    if daily_request_counts[model] >= RATE_LIMITS[model]['MAX_RPD']:
        now = datetime.datetime.now()
        tomorrow = datetime.datetime.combine(current_date + datetime.timedelta(days=1), datetime.time.min)
        sleep_seconds = (tomorrow - now).total_seconds()
        logger.warning(f"Daily limit reached for {model}. Sleeping for {int(sleep_seconds)}s")
        time.sleep(sleep_seconds)
        daily_request_counts[model] = 0
        last_request_date = current_date

def clean_response_text(response_text: str) -> str:
    try:
        response_text = response_text.strip()
        response_text = html.unescape(response_text)
        response_text = re.sub(r'[\x00-\x1F\x7F]', '', response_text)
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()
        return response_text
    except Exception as e:
        logger.error(f"Error in clean_response_text: {e}")
        raise

@lru_cache(maxsize=1000)
def robust_api_call(models, prompt, config=None, max_tokens=4000, thinking_budget=None, retries=3, initial_delay=2):
    if isinstance(models, str):
        models = [models]
    for model in models:
        for attempt in range(retries):
            try:
                rate_limit(model)
                if model == "deepseek-reasoner":
                    if deepseek_client is None:
                        raise ValueError("Deepseek client is not initialized")
                    check_rate_limits(model)
                    daily_request_counts[model] += 1
                    response = deepseek_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=1,
                    )
                    response_text = response.choices[0].message.content
                    response_text = clean_response_text(response_text)
                    try:
                        parsed = json.loads(response_text)
                        return True, parsed
                    except json.JSONDecodeError:
                        logger.error(f"DeepSeek returned invalid JSON: {response_text}")
                        return False, response_text
                elif model.startswith("claude-"):
                    with anthropic_lock:
                        now = time.time()
                        while anthropic_token_tracker and anthropic_token_tracker[0][0] < now - 60:
                            anthropic_token_tracker.popleft()
                        current_total = sum(tokens for _, tokens in anthropic_token_tracker)
                        if current_total + max_tokens > ANTHROPIC_TOKEN_LIMIT:
                            if anthropic_token_tracker:
                                oldest_time = anthropic_token_tracker[0][0]
                                sleep_time = (oldest_time + 60) - now
                                if sleep_time > 0:
                                    logger.info(f"Token limit near for {model}, sleeping for {sleep_time:.2f} seconds")
                                    time.sleep(sleep_time)
                            else:
                                logger.error(f"Single request exceeds token limit: max_tokens={max_tokens}")
                                return False, None
                        check_rate_limits(model)
                        daily_request_counts[model] += 1
                        if thinking_budget is None:
                            thinking_budget = max(1024, max_tokens - 1000)
                        else:
                            if thinking_budget >= max_tokens:
                                raise ValueError(f"thinking_budget ({thinking_budget}) must be less than max_tokens ({max_tokens})")
                        thinking_params = {"type": "enabled", "budget_tokens": thinking_budget}
                        message = anthropic_client.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            temperature=1,
                            thinking=thinking_params,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        for block in message.content:
                            if block.type == "text":
                                response_text = block.text
                                break
                        else:
                            raise ValueError("No text content block found")
                        actual_tokens = message.usage.output_tokens
                        anthropic_token_tracker.append((time.time(), actual_tokens))
                        logger.info(f"Used {actual_tokens} output tokens for {model}")
                    response_text = clean_response_text(response_text)
                    try:
                        parsed = json.loads(response_text)
                        return True, parsed
                    except json.JSONDecodeError:
                        return False, response_text
                else:
                    check_rate_limits(model)
                    daily_request_counts[model] += 1
                    
                    # Check if config contains Google Search tool
                    has_search_tool = False
                    if config and 'tools' in config:
                        for tool in config['tools']:
                            if 'google_search' in tool:
                                has_search_tool = True
                                break
                    
                    # Process the config for Gemini API
                    gemini_config = None
                    if config:
                        gemini_config = {}
                        if 'response_mime_type' in config:
                            gemini_config['response_mime_type'] = config['response_mime_type']
                        
                        # Handle Google Search as a tool
                        if has_search_tool:
                            from google.genai.types import Tool, GoogleSearch
                            tools = []
                            for tool_config in config['tools']:
                                if 'google_search' in tool_config:
                                    tools.append(Tool(google_search=GoogleSearch()))
                            if tools:
                                gemini_config['tools'] = tools
                        
                        # Copy other configs if needed
                        if 'response_modalities' in config:
                            gemini_config['response_modalities'] = config['response_modalities']
                    
                    # Generate content with appropriate config
                    if gemini_config:
                        from google.genai.types import GenerateContentConfig
                        generate_config = GenerateContentConfig(**gemini_config)
                        response = client.models.generate_content(model=model, contents=prompt, config=generate_config)
                    else:
                        response = client.models.generate_content(model=model, contents=prompt)
                    
                    # Process grounding metadata if available
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                            logger.info(f"Response includes grounding metadata from Google Search")
                    
                    response_text = clean_response_text(response.text)
                    try:
                        parsed = json.loads(response_text)
                        if hasattr(response, 'usage_metadata'):
                            logger.info(f"Used {response.usage_metadata.total_token_count} tokens for {model}")
                        return True, parsed
                    except json.JSONDecodeError:
                        if 'response_mime_type' in config and config['response_mime_type'] == 'application/json':
                            logger.error(f"Gemini returned invalid JSON: {response_text}")
                        return False, response_text
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {model}: {e}")
                if attempt < retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    continue
    fallback_model = "gemini-2.0-flash"
    logger.warning(f"All reasoning models failed. Falling back to {fallback_model}")
    try:
        rate_limit(fallback_model)
        check_rate_limits(fallback_model)
        daily_request_counts[fallback_model] += 1
        response = client.models.generate_content(model=fallback_model, contents=prompt)
        response_text = clean_response_text(response.text)
        parsed = json.loads(response_text)
        if hasattr(response, 'usage_metadata'):
            logger.info(f"Used {response.usage_metadata.total_token_count} tokens for {fallback_model}")
        return True, parsed
    except Exception as e:
        logger.error(f"Fallback to {fallback_model} failed: {e}")
        return False, None

def parse_market_events_text(text):
    """Parse market events text when JSON parsing fails"""
    events = []
    market_risk = 'low'
    lines = text.split('\n')
    in_events_section = False
    for line in lines:
        if line.startswith('# Market-Moving Events'):
            continue
        if line.startswith('## Key Events'):
            in_events_section = True
            continue
        if line.startswith('## Market Risk Assessment'):
            in_events_section = False
            risk_match = re.search(r'\*\*(\w+)\*\*', line)
            if risk_match:
                market_risk = risk_match.group(1).lower()
            continue
        if in_events_section and line.strip():
            events.append(line.strip())
    return {'events': events, 'market_risk': market_risk}

def check_market_events(model):
    """
    Check for significant market-moving events in the past 24 hours.
    Uses Google Search grounding when available for more accurate and up-to-date information.
    
    Args:
        model: The model to use for market event detection
        
    Returns:
        Dict with 'events' (list of strings) and 'market_risk' (string)
    """
    prompt = (
        "What are the most significant market-moving events in the past 24 hours? "
        "Focus on Fed announcements, economic data releases, geopolitical events, "
        "or other factors that could significantly impact stock prices. "
        "Rate the market risk as low, medium, or high. "
        "Please provide the response in JSON format with keys 'events' (list of strings) and 'market_risk' (string)."
    )
    
    # Try with Google Search grounding first
    try:
        from google.genai.types import Tool, GoogleSearch
        google_search_config = {
            'response_mime_type': 'application/json',
            'tools': [{'google_search': {}}],
            'response_modalities': ["TEXT"],
        }
        
        success, response = robust_api_call([model], prompt, google_search_config)
        if success and isinstance(response, dict):
            events = response.get('events', [])
            # Ensure events is a list
            if not isinstance(events, list):
                events = [str(events)] if events else []
            market_risk = response.get('market_risk', 'low')
            logger.info(f"Market events check completed with Google Search grounding: {len(events)} events, risk: {market_risk}")
            return {'events': events, 'market_risk': market_risk}
    except Exception as e:
        logger.warning(f"Failed to use Google Search grounding for market events: {e}")
    
    # Fallback to standard call
    success, response = robust_api_call([model], prompt)
    if success:
        if isinstance(response, dict):
            events = response.get('events', [])
            # Ensure events is a list
            if not isinstance(events, list):
                events = [str(events)] if events else []
            market_risk = response.get('market_risk', 'low')
        elif isinstance(response, list):
            # If response is a list, use it as events
            logger.warning("Response is a list instead of dict")
            events = [str(item) for item in response]
            market_risk = 'low'
        else:
            logger.warning(f"Unexpected response type: {type(response)}")
            events = []
            market_risk = 'low'
    else:
        if response and isinstance(response, str):
            parsed = parse_market_events_text(response)
            events = parsed['events']
            market_risk = parsed['market_risk']
        else:
            logger.error("Failed to get market events")
            events = []
            market_risk = 'low'
    
    return {'events': events, 'market_risk': market_risk}