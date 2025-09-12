"""
Bitcoin Investment Advisory Pipeline Agent

This module implements a complete pipeline for Bitcoin investment advisory:
1. Collects news from the web with optimal structure
2. Summarizes news for short-term and long-term effects
3. Analyzes effects using a trained model
4. Forecasts Bitcoin prices for the next 10 days
5. Generates a comprehensive investment advisory

Features:
- Comprehensive RSS and web scraping for Bitcoin news
- Advanced news categorization into short-term and long-term impact
- Integration with trained models for news summarization, effects analysis, and price forecasting
- Configurable API endpoints and model parameters
- Robust error handling and retries

Usage:
    python pipeline_agent.py --config config.json
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
import datetime as dt
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

# New prompt templates module
try:
    from .prompt_templates import (
        summarization_prompt,
        advisory_json_prompt,
        advisory_narrative_prompt,
        validate_summary_payload,
    )
except Exception:
    # Fallback to relative import when executed as script
    try:
        from prompt_templates import (
            summarization_prompt,
            advisory_json_prompt,
            advisory_narrative_prompt,
            validate_summary_payload,
        )
    except Exception as e:  # pragma: no cover
        print(f"Warning: prompt_templates import failed: {e}")
        summarization_prompt = advisory_json_prompt = advisory_narrative_prompt = None
        def validate_summary_payload(x):
            return x
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

try:
    import requests
    import feedparser
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from newspaper import Article
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install the required packages using: pip install -r requirements.txt")
    sys.exit(1)

# Try to import optional packages
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not available. Install with: pip install openai")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google Generative AI package not available. Install with: pip install google-generativeai")


# -----------------------------
# Config
# -----------------------------
DEFAULT_CONFIG = {
    "api_keys": {
        "gemini": [],
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "openai_base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    },
    "models": {
        "summarization_model": "gemini-2.0-flash-exp",
        "effects_model": "your-effects-model-endpoint",
        "forecast_model": "your-forecast-model-endpoint",
        "advisory_model": "gpt-4o"
    },
    "news": {
        "max_articles": 30,
        "short_term_count": 10,
        "long_term_count": 10,
        "sources": [
            "https://news.google.com/rss/search?q=Bitcoin+price&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=BTC+crypto&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Cryptocurrency+market&hl=en-US&gl=US&ceid=US:en"
        ]
    },
    "output_dir": "Files/Final Project/outputs/pipeline",
    "prompt_style": "comprehensive"
}

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36"
HEADERS = {"User-Agent": USER_AGENT}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bitcoin_agent.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pipeline_agent")


# -----------------------------
# Data Structures
# -----------------------------

@dataclass
class NewsArticle:
    """News article data structure with relevant fields"""
    title: str
    url: str
    content: str
    source: str
    published_date: Optional[str]
    summary: str = ""
    impact_type: str = ""  # 'short_term', 'long_term', or 'both'


class BitcoinPipelineAgent:
    """
    Pipeline agent that orchestrates the entire workflow for Bitcoin investment advisory
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Bitcoin Pipeline Agent with optional configuration
        
        Args:
            config_path: Path to JSON configuration file
        """
        # Default configuration
        self.config = DEFAULT_CONFIG.copy()
        
        # Load custom configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    self._update_nested_dict(self.config, custom_config)
                logger.info(f"Loaded custom configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Initialize session for web requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT
        })
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Initialize API clients based on configuration
        self._init_api_clients()
        
        logger.info("Bitcoin Pipeline Agent initialized successfully")
    
    def _update_nested_dict(self, d: dict, u: dict) -> dict:
        """Recursively update nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _init_api_clients(self):
        """Initialize API clients for the different models"""
        # Initialize Gemini API if keys are available
        self.gemini_api_keys = self.config["api_keys"]["gemini"]
        self.current_key_index = 0
        
        if self.gemini_api_keys and GEMINI_AVAILABLE:
            logger.info(f"Initialized Gemini API with {len(self.gemini_api_keys)} keys")
        elif self.gemini_api_keys:
            logger.warning("Gemini API keys provided but google-generativeai package not installed")
        else:
            logger.warning("No Gemini API keys provided!")
        
        # Initialize OpenAI client if API key is provided
        self.openai_api_key = self.config["api_keys"]["openai"]
        if self.openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = openai.OpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.config["api_keys"]["openai_base_url"]
                )
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                self.openai_client = None
        elif self.openai_api_key:
            logger.warning("OpenAI API key provided but openai package not installed")
            self.openai_client = None
        else:
            logger.warning("No OpenAI API key provided!")
            self.openai_client = None
    
    def _get_next_gemini_key(self) -> str:
        """Get next Gemini API key for load balancing"""
        if not self.gemini_api_keys:
            raise ValueError("No Gemini API keys available")
        
        key = self.gemini_api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
        return key
    
    def _clean_html(self, text: str) -> str:
        """Clean HTML tags from text"""
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(" ", strip=True)
    
    def _fetch_url(self, url: str, timeout: int = 15) -> Optional[str]:
        """Fetch URL content with error handling"""
        try:
            resp = self.session.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _call_trained_model(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call a trained model API endpoint"""
        try:
            logger.info(f"Calling trained model at {endpoint}")
            response = self.session.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error calling trained model at {endpoint}: {e}")
            raise
    
    # -----------------------------
    # Stage 1: Crawl/collect news
    # -----------------------------
    
    def collect_news(self, max_articles: int = None, target_date: str = None) -> List[NewsArticle]:
        """
        Collect Bitcoin-related news articles from various sources
        
        Args:
            max_articles: Maximum number of articles to collect
            target_date: Target date for news collection (default: today)
            
        Returns:
            List of collected news articles
        """
        if max_articles is None:
            max_articles = self.config["news"]["max_articles"]
        
        if target_date is None:
            target_date = dt.datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Collecting Bitcoin news for {target_date}")
        
        # Get sources from config
        news_sources = self.config["news"].get("sources", [])
        if not news_sources:
            logger.warning("No news sources configured, using default RSS feeds")
            news_sources = [
                "https://news.google.com/rss/search?q=Bitcoin+price&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=BTC+crypto&hl=en-US&gl=US&ceid=US:en"
            ]
        
        all_articles = []
        
        # Collect from RSS feeds
        for feed_url in news_sources:
            try:
                logger.debug(f"Processing feed: {feed_url}")
                parsed = feedparser.parse(feed_url)
                for entry in parsed.entries:
                    try:
                        article = NewsArticle(
                            title=entry.get("title", ""),
                            url=entry.get("link", ""),
                            content=self._clean_html(entry.get("summary", "")),
                            source=feed_url.split('/')[2] if '/' in feed_url else feed_url,
                            published_date=entry.get("published", "")
                        )
                        all_articles.append(article)
                    except Exception as e:
                        logger.warning(f"Error processing feed entry: {e}")
            except Exception as e:
                logger.warning(f"Feed parse failed {feed_url}: {e}")
        
        # Dedup by URL
        unique_articles = []
        seen_urls = set()
        
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        # Limit total articles
        unique_articles = unique_articles[:max_articles]
        
        logger.info(f"Found {len(unique_articles)} unique articles")
        
        # Scrape content for each article using ThreadPoolExecutor for parallelism
        with ThreadPoolExecutor(max_workers=5) as executor:
            scraped_articles = list(executor.map(self._scrape_article_content, unique_articles))
        
        logger.info(f"Successfully scraped {len(scraped_articles)} articles")
        
        return scraped_articles
    
    def _scrape_article_content(self, article: NewsArticle) -> NewsArticle:
        """Enhanced content scraping using newspaper3k and BeautifulSoup"""
        try:
            # Method 1: newspaper3k
            news_article = Article(article.url)
            news_article.download()
            news_article.parse()
            
            if news_article.text and len(news_article.text) > 100:
                article.content = news_article.text[:5000]  # Limit content length
                if not article.published_date and news_article.publish_date:
                    article.published_date = news_article.publish_date.strftime('%Y-%m-%d')
                return article
                
        except Exception as e:
            logger.debug(f"newspaper3k failed for {article.url}: {e}")
        
        try:
            # Method 2: BeautifulSoup fallback
            response = self.session.get(article.url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extract text from common content containers
            content_selectors = ['article', '.content', '.post-content', '.entry-content', 'main', '.story-body']
            content = ""
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(strip=True, separator=' ')
                    if len(content) > 100:
                        break
            
            if not content:
                # Fallback to all paragraphs
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            article.content = content[:5000] if content else "Content not available"
            
        except Exception as e:
            logger.warning(f"BeautifulSoup fallback failed for {article.url}: {e}")
            if not article.content:
                article.content = f"Failed to scrape content: {e}"
        
        return article
    
    def bucket_news(self, articles: List[NewsArticle]) -> Dict[str, List[NewsArticle]]:
        """
        Bucket news articles into short-term and long-term impact categories
        
        Args:
            articles: List of news articles to categorize
            
        Returns:
            Dictionary with short_term and long_term article lists
        """
        logger.info(f"Bucketing {len(articles)} articles into impact categories")
        
        # This should ideally use an AI model to categorize articles
        # For now, we'll use a simple heuristic based on keywords
        
        short_term_keywords = [
            'price', 'crash', 'rally', 'surge', 'plunge', 'today', 'breaking',
            'buy', 'sell', 'trading', 'market', 'volatility', 'jump', 'drop'
        ]
        
        long_term_keywords = [
            'regulation', 'adoption', 'infrastructure', 'institutional',
            'development', 'technology', 'policy', 'future', 'innovation',
            'investment', 'strategy', 'long-term'
        ]
        
        short_term_articles = []
        long_term_articles = []
        
        for article in articles:
            title_lower = article.title.lower()
            content_lower = article.content.lower()
            
            short_term_score = sum(1 for kw in short_term_keywords if kw in title_lower or kw in content_lower)
            long_term_score = sum(1 for kw in long_term_keywords if kw in title_lower or kw in content_lower)
            
            if short_term_score > long_term_score:
                article.impact_type = 'short_term'
                short_term_articles.append(article)
            elif long_term_score > short_term_score:
                article.impact_type = 'long_term'
                long_term_articles.append(article)
            else:
                article.impact_type = 'both'
                # Add to both buckets
                short_term_articles.append(article)
                long_term_articles.append(article)
        
        # Limit to configured counts
        short_term_limit = self.config["news"]["short_term_count"]
        long_term_limit = self.config["news"]["long_term_count"]
        
        short_term_articles = short_term_articles[:short_term_limit]
        long_term_articles = long_term_articles[:long_term_limit]
        
        logger.info(f"Bucketed articles: {len(short_term_articles)} short-term, {len(long_term_articles)} long-term")
        
        return {
            'short_term': short_term_articles,
            'long_term': long_term_articles
        }


    def summarize_items(self, bucketed_news: Dict[str, List[NewsArticle]], style: str = None) -> Dict[str, Any]:
        """
        Summarize news items using the trained summarization model
        
        Args:
            bucketed_news: Dictionary with short_term and long_term news lists
            
        Returns:
            Dictionary with summarized news data
        """
        logger.info("Summarizing news items using AI model")
        
        short_term_news = bucketed_news['short_term']
        long_term_news = bucketed_news['long_term']
        
        # Get today's date for the analysis
        analysis_date = dt.datetime.now().strftime('%Y-%m-%d')
        
        # Prepare news data for summarization
        short_term_data = []
        for i, article in enumerate(short_term_news, 1):
            short_term_data.append({
                "id": i,
                "title": article.title,
                "source": article.source,
                "url": article.url,
                "content": article.content[:1000],  # Limit content length
                "published_date": article.published_date
            })
        
        long_term_data = []
        for i, article in enumerate(long_term_news, 1):
            long_term_data.append({
                "id": i,
                "title": article.title,
                "source": article.source,
                "url": article.url,
                "content": article.content[:1000],  # Limit content length
                "published_date": article.published_date
            })
        
        try:
            # For short-term news summary
            short_term_summary = self._call_summarization_model(
                short_term_data,
                analysis_date,
                "short_term",
                style=style
            )
            
            # For long-term news summary
            long_term_summary = self._call_summarization_model(
                long_term_data,
                analysis_date,
                "long_term",
                style=style
            )
            
            # Combine summaries into a complete news analysis
            news_analysis = {
                'date': analysis_date,
                'short_term_count': len(short_term_news),
                'long_term_count': len(long_term_news),
                'total_news_items': len(short_term_news) + len(long_term_news),
                'short_term_summary': short_term_summary.get('summary', ''),
                'long_term_summary': long_term_summary.get('summary', ''),
                'bullish_ratio': short_term_summary.get('bullish_ratio', 0.5),
                'bearish_ratio': short_term_summary.get('bearish_ratio', 0.3),
                'neutral_ratio': short_term_summary.get('neutral_ratio', 0.2),
                'high_impact_count': short_term_summary.get('high_impact_count', 0) + long_term_summary.get('high_impact_count', 0),
                'avg_confidence': (short_term_summary.get('confidence', 0.5) + long_term_summary.get('confidence', 0.5)) / 2,
                'key_events': short_term_summary.get('key_events', []) + long_term_summary.get('key_events', []),
                'daily_view': {
                    'summary': short_term_summary.get('daily_summary', '') + "\n\n" + long_term_summary.get('daily_summary', ''),
                    'sentiment': short_term_summary.get('sentiment', 'neutral'),
                    'market_impact': short_term_summary.get('market_impact', 'medium'),
                    'key_risks': short_term_summary.get('risk_factors', []) + long_term_summary.get('risk_factors', []),
                    'watch_items': short_term_summary.get('watch_items', []) + long_term_summary.get('opportunities', []),
                    'recommendation_short_term': {
                        'action': short_term_summary.get('recommendation', 'HOLD'),
                        'probability': short_term_summary.get('recommendation_confidence', 0.5)
                    },
                    'recommendation_long_term': {
                        'action': long_term_summary.get('recommendation', 'HOLD'),
                        'probability': long_term_summary.get('recommendation_confidence', 0.5)
                    }
                }
            }
            
            logger.info("Successfully summarized news items")
            return news_analysis
            
        except Exception as e:
            logger.error(f"Error summarizing news: {e}")
            return {
                'date': analysis_date,
                'error': str(e)
            }
    
    def _call_summarization_model(self, news_data: List[Dict], analysis_date: str, impact_type: str, style: str = None) -> Dict[str, Any]:
        """Call the summarization model API with the prepared news data"""
        # This would be replaced with your actual trained summarization model API call
        
        # Import Gemini API if keys are available
        if self.gemini_api_keys and GEMINI_AVAILABLE:
            try:
                # Configure with the next available key
                api_key = self._get_next_gemini_key()
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(self.config["models"]["summarization_model"])
                
                # Prepare prompt using new template system
                prompt = summarization_prompt(
                    news_items=news_data,
                    analysis_date=analysis_date,
                    impact_type=impact_type,
                    style=style or self.config.get("prompt_style", "comprehensive")
                ) if summarization_prompt else ""
                
                # Call the model
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_output_tokens": 4096,
                    }
                )
                
                # Parse JSON response
                text = response.text.strip()
                
                # Clean up common JSON formatting issues
                text = text.replace('```json', '').replace('```', '').strip()
                
                # Parse the JSON
                try:
                    result = json.loads(text)
                except Exception:
                    logger.warning("Model output not valid JSON; attempting recovery")
                    # crude recovery: extract between first { and last }
                    if '{' in text and '}' in text:
                        candidate = text[text.find('{'): text.rfind('}')+1]
                        try:
                            result = json.loads(candidate)
                        except Exception:
                            raise
                    else:
                        raise
                # validate/normalize
                result = validate_summary_payload(result)
                return result
                
            except Exception as e:
                logger.error(f"Error calling Gemini API for summarization: {e}")
                # Fall back to mock summarization
        
        # Mock summarization as fallback
        bullish_ratio = 0.6 if impact_type == 'short_term' else 0.55
        bearish_ratio = 0.3 if impact_type == 'short_term' else 0.25
        neutral_ratio = 1 - bullish_ratio - bearish_ratio
        
        mock = {
            'summary': f"Mock {impact_type} summary of {len(news_data)} articles for {analysis_date}.",
            'daily_summary': f"Mock daily summary for {analysis_date} with {len(news_data)} {impact_type} news items.",
            'sentiment': 'bullish' if bullish_ratio > bearish_ratio else 'bearish',
            'market_impact': 'medium',
            'key_events': [f"{impact_type} key event {i}" for i in range(1, 4)],
            'risk_factors': [f"{impact_type} risk factor {i}" for i in range(1, 4)],
            'watch_items': [f"{impact_type} watch item {i}" for i in range(1, 4)],
            'opportunities': [f"{impact_type} opportunity {i}" for i in range(1, 4)],
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'neutral_ratio': neutral_ratio,
            'high_impact_count': len(news_data) // 3,
            'confidence': 0.85,
            'recommendation': 'BUY' if bullish_ratio > 0.5 else 'HOLD',
            'recommendation_confidence': bullish_ratio
        }
        return validate_summary_payload(mock)
    
    def analyze_effects(self, news_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze effects using the trained effects model
        
        Args:
            news_analysis: Summarized news data
            
        Returns:
            Dictionary with effects analysis
        """
        logger.info("Analyzing news effects using trained model")
        
        # Get Bitcoin price data for context
        btc_data = self._get_bitcoin_price_data()
        
        # Add price data to the analysis
        news_analysis.update(btc_data)
        
        # Prepare payload for the effects model, including all relevant analysis
        payload = {
            "news_analysis": news_analysis
        }
        
        endpoint = self.config["models"]["effects_model"]
        
        try:
            # Call the trained effects model
            effects_analysis = self._call_trained_model(endpoint, payload)
            logger.info("Successfully analyzed news effects")
        except Exception as e:
            logger.warning(f"Effects model call failed: {e}. Using mock analysis.")
            # Mock effects analysis as fallback
            effects_analysis = {
                'bull_prob': 0.65,
                'bear_prob': 0.25,
                'base_prob': 0.10,
                'scenarios': {
                    'bullish': 0.65,
                    'bearish': 0.25,
                    'base': 0.10
                }
            }
        
        # Update news analysis with effects data
        news_analysis.update(effects_analysis)
        
        return news_analysis
    
    def _get_bitcoin_price_data(self) -> Dict[str, Any]:
        """Get Bitcoin price data from Yahoo Finance"""
        try:
            # Download Bitcoin data
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=70)  # Extra buffer for 60 days
            
            btc = yf.Ticker("BTC-USD")
            hist = btc.download(start=start_date, end=end_date, progress=False)
            
            if hist.empty:
                raise ValueError("No price data received")
            
            # Get the last 60 days of closing prices
            prices = hist['Close'].dropna().tolist()[-60:]
            
            # Generate next 10 days of prices (mock forecast)
            # In reality, this would come from your forecast model
            current_price = prices[-1]
            next_10_day_prices = [current_price]
            for i in range(9):
                # Simple random walk for demonstration
                change = np.random.normal(0.002, 0.02)  # Mean daily return and volatility
                next_price = next_10_day_prices[-1] * (1 + change)
                next_10_day_prices.append(next_price)
            
            return {
                'price_history_60d': prices,
                'current_price': prices[-1],
                'next_10_day_prices': next_10_day_prices,
            }
            
        except Exception as e:
            logger.error(f"Error fetching Bitcoin price data: {e}")
            return {
                'price_history_60d': [50000] * 60,
                'current_price': 50000,
                'next_10_day_prices': [50000 + (i * 100) for i in range(10)],
            }


    def forecast_next_10_days(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forecast Bitcoin prices for the next 10 days using the trained forecast model
        
        Args:
            analysis_data: News and effects analysis data
            
        Returns:
            Dictionary with forecasted prices
        """
        logger.info("Forecasting Bitcoin prices for next 10 days")
        
        # Prepare payload for the forecast model, including all relevant analysis
        payload = {
            "analysis_data": analysis_data
        }
        
        endpoint = self.config["models"]["forecast_model"]
        
        try:
            # Call the trained forecast model
            forecast_result = self._call_trained_model(endpoint, payload)
            analysis_data['next_10_day_prices'] = forecast_result.get('next_10_day_prices')
            logger.info("Successfully forecasted Bitcoin prices")
        except Exception as e:
            logger.warning(f"Forecast model call failed: {e}. Using mock forecast.")
            # Fallback to mock forecast
            if 'next_10_day_prices' not in analysis_data:
                current_price = analysis_data.get('current_price', 50000)
                bull_prob = analysis_data.get('bull_prob', 0.5)
                bear_prob = analysis_data.get('bear_prob', 0.3)
                trend_factor = (bull_prob - bear_prob) * 0.005
                
                next_10_day_prices = [current_price]
                for i in range(9):
                    volatility = 0.02
                    change = np.random.normal(trend_factor, volatility)
                    next_price = next_10_day_prices[-1] * (1 + change)
                    next_10_day_prices.append(next_price)
                
                analysis_data['next_10_day_prices'] = next_10_day_prices
        
        return analysis_data
    
    def generate_investment_advisory(self, complete_analysis: Dict[str, Any]) -> str:
        """
        Generate comprehensive investment advisory using the trained advisory model
        
        Args:
            complete_analysis: Complete analysis data including news, effects, and forecast
            
        Returns:
            Comprehensive investment advisory text
        """
        logger.info("Generating comprehensive investment advisory")
        
        # In a real implementation, this would call the BitcoinInvestmentAdvisor from your notebook
        # For now, we'll create a simplified implementation
        
        if self.openai_client:
            try:
                # Use the investment advisor prompt creation logic
                prompt = self._create_investment_advisory_prompt(complete_analysis)
                
                # Call OpenAI API
                response = self.openai_client.chat.completions.create(
                    model=self.config["models"]["advisory_model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1500,
                    timeout=60,
                )
                
                advisory = response.choices[0].message.content.strip()
                logger.info(f"Successfully generated investment advisory ({len(advisory)} chars)")
                return advisory
                
            except Exception as e:
                logger.error(f"Error generating investment advisory: {e}")
                return self._create_simple_advisory(complete_analysis)
        else:
            logger.info("OpenAI client not available, using simple advisory")
            return self._create_simple_advisory(complete_analysis)
    
    def _create_investment_advisory_prompt(self, daily_data: Dict[str, Any]) -> str:
        """Create a comprehensive investment advisory prompt"""
        date_str = daily_data.get('date', dt.datetime.now().strftime('%Y-%m-%d'))
        next_10_day_prices = daily_data.get('next_10_day_prices', [])
        
        if not next_10_day_prices:
            current_price = daily_data.get('current_price', 50000)
            next_10_day_prices = [current_price * (1 + 0.005 * i) for i in range(10)]
        
        # Format summaries
        short_term_summary = daily_data.get('short_term_summary', 'No short-term summary available.')
        long_term_summary = daily_data.get('long_term_summary', 'No long-term summary available.')
        
        # Extract scenario probabilities
        bull_prob = daily_data.get('bull_prob', 0.5)
        bear_prob = daily_data.get('bear_prob', 0.3)
        base_prob = daily_data.get('base_prob', 0.2)
        
        # Get daily view data
        daily_view = daily_data.get('daily_view', {})
        
        # Format next 10-day price predictions
        newline = "\n"
        price_predictions = ""
        for i, price in enumerate(next_10_day_prices, 1):
            price_predictions += f"Day {i}: ${price:.2f}{newline}"
        
        price_change_10d = ((next_10_day_prices[-1] / next_10_day_prices[0]) - 1) * 100
        
        prompt = f"""You are an elite Bitcoin investment advisor with deep expertise in cryptocurrency markets, institutional trading strategies, and comprehensive financial analysis. Provide an extensive, institutional-grade investment advisory for Bitcoin based on the comprehensive market intelligence below.

DATE: {date_str}

MARKET INTELLIGENCE SUMMARY:
• Total News Items Analyzed: {daily_data.get('total_news_items', 0)}
• Long-term Impact News: {daily_data.get('long_term_count', 0)} items
• Short-term Impact News: {daily_data.get('short_term_count', 0)} items
• High Impact News: {daily_data.get('high_impact_count', 0)} items
• Market Sentiment Distribution: {daily_data.get('bullish_ratio', 0.5):.1%} Bullish, {daily_data.get('bearish_ratio', 0.3):.1%} Bearish, {daily_data.get('neutral_ratio', 0.2):.1%} Neutral
• Average Analyst Confidence: {daily_data.get('avg_confidence', 0.5):.2%}

NEXT 10-DAY PRICE PREDICTIONS:
{price_predictions}
Total 10-Day Price Change: {price_change_10d:+.2f}%

MARKET SCENARIO PROBABILITIES:
• Bullish Scenario: {bull_prob:.1%}
• Base Case Scenario: {base_prob:.1%}  
• Bearish Scenario: {bear_prob:.1%}

CURRENT MARKET RECOMMENDATIONS:
• Short-term Action: {daily_view.get('recommendation_short_term', {}).get('action', 'N/A')} (Probability: {daily_view.get('recommendation_short_term', {}).get('probability', 0):.1%})
• Long-term Action: {daily_view.get('recommendation_long_term', {}).get('action', 'N/A')} (Probability: {daily_view.get('recommendation_long_term', {}).get('probability', 0):.1%})

LONG-TERM IMPACT NEWS ANALYSIS:
{long_term_summary}

SHORT-TERM IMPACT NEWS ANALYSIS:
{short_term_summary}

KEY MARKET RISKS:
{newline.join(f"• {risk}" for risk in daily_view.get('key_risks', []))}

CRITICAL WATCH ITEMS:
{newline.join(f"• {item}" for item in daily_view.get('watch_items', []))}

DAILY MARKET SUMMARY:
{daily_view.get('summary', 'No summary available')}

Based on this comprehensive market intelligence and the predicted next 10-day price movements, provide an EXTENSIVE institutional-grade Bitcoin investment advisory that includes:

1. **Executive Summary & Market Overview** (200+ words)
2. **Investment Recommendation** (specific position sizes, entry/exit points, timeframes)
3. **Risk Assessment & Management** (detailed risk analysis, hedging strategies)
4. **Price Targets & Scenarios** (incorporating the 10-day predictions, multiple scenarios)
5. **Trading Strategy & Execution** (entry strategies, portfolio allocation, timing)
6. **Market Outlook & Catalysts** (short/medium/long-term outlook)
7. **Technical Analysis Integration** (support/resistance, momentum indicators)  
8. **Fundamental Analysis** (adoption trends, regulatory landscape, institutional flows)
9. **Risk-Reward Analysis** (expected returns, maximum drawdown, Sharpe ratios)
10. **Alternative Scenarios** (black swan events, regulatory changes)
11. **Portfolio Integration** (correlation with other assets, diversification)
12. **Actionable Investment Thesis** (clear rationale, conviction level)

Make your analysis comprehensive, data-driven, and suitable for institutional investors managing significant Bitcoin allocations. Use the next 10-day price predictions to inform your near-term tactical recommendations while maintaining focus on long-term strategic positioning."""

        return prompt
    
    def _create_simple_advisory(self, daily_data: Dict[str, Any]) -> str:
        """Create a simplified investment advisory"""
        date_str = daily_data.get('date', dt.datetime.now().strftime('%Y-%m-%d'))
        next_10_day_prices = daily_data.get('next_10_day_prices', [])
        
        if not next_10_day_prices:
            return f"Unable to generate investment advisory due to missing price forecast data for {date_str}."
        
        price_change = ((next_10_day_prices[-1] / next_10_day_prices[0]) - 1) * 100
        current_price = next_10_day_prices[0]
        final_price = next_10_day_prices[-1]
        
        daily_view = daily_data.get('daily_view', {})
        sentiment = daily_view.get('sentiment', 'neutral')
        market_impact = daily_view.get('market_impact', 'medium')
        
        short_term_rec = daily_view.get('recommendation_short_term', {}).get('action', 'HOLD')
        long_term_rec = daily_view.get('recommendation_long_term', {}).get('action', 'HOLD')
        
        advisory = f"""# Bitcoin Investment Advisory for {date_str}

## Executive Summary
Based on the comprehensive analysis of {daily_data.get('total_news_items', 0)} news items and current market conditions, the Bitcoin market currently shows {sentiment} sentiment with {market_impact} impact expected. Our 10-day price forecast projects a {price_change:+.2f}% move from ${current_price:.2f} to ${final_price:.2f}.

## Investment Recommendations
- **Short-term recommendation**: {short_term_rec}
- **Long-term recommendation**: {long_term_rec}

## Key Risk Factors:
"""
        
        for risk in daily_view.get('key_risks', []):
            advisory += f"- {risk}\n"
        
        advisory += "\n## Critical Watch Items:\n"
        for item in daily_view.get('watch_items', []):
            advisory += f"- {item}\n"
        
        advisory += f"\n## Market Summary:\n{daily_view.get('summary', 'No summary available.')}"
        
        return advisory


    def run(self, target_date: str = None) -> Dict[str, Any]:
        """
        Run the complete Bitcoin investment advisory pipeline
        
        Args:
            target_date: Target date for analysis (default: today)
            
        Returns:
            Complete results dictionary
        """
        if target_date is None:
            target_date = dt.datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Starting Bitcoin investment advisory pipeline for {target_date}")
        
        results = {
            'target_date': target_date,
            'start_time': dt.datetime.now().isoformat(),
            'status': 'running'
        }
        
        try:
            # Step 1: Collect news
            news_articles = self.collect_news(target_date=target_date)
            results['news_articles_count'] = len(news_articles)
            
            # Step 2: Bucket news into short-term and long-term impact
            bucketed_news = self.bucket_news(news_articles)
            results['short_term_count'] = len(bucketed_news['short_term'])
            results['long_term_count'] = len(bucketed_news['long_term'])
            
            # Step 3: Summarize news items
            news_analysis = self.summarize_items(bucketed_news)
            results['news_analysis'] = news_analysis
            
            # Step 4: Analyze effects
            effects_analysis = self.analyze_effects(news_analysis)
            results['effects_analysis'] = effects_analysis
            
            # Step 5: Forecast next 10 days
            forecast_data = self.forecast_next_10_days(effects_analysis)
            results['forecast'] = forecast_data.get('next_10_day_prices')
            
            # Step 6: Generate investment advisory
            advisory = self.generate_investment_advisory(forecast_data)
            results['advisory'] = advisory
            
            results['status'] = 'completed'
            results['end_time'] = dt.datetime.now().isoformat()
            
            # Save results to file
            self._save_results(results, target_date)
            
            logger.info(f"Bitcoin investment advisory pipeline completed successfully for {target_date}")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = dt.datetime.now().isoformat()
        
        return results
    
    def _save_results(self, results: Dict[str, Any], target_date: str):
        """Save results to output directory"""
        output_path = os.path.join(self.config['output_dir'], f"bitcoin_advisory_{target_date}.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description='Bitcoin Investment Advisory Pipeline Agent')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--date', type=str, help='Target date for analysis (YYYY-MM-DD)')
    args = parser.parse_args()
    
    agent = BitcoinPipelineAgent(config_path=args.config)
    results = agent.run(target_date=args.date)
    
    if results['status'] == 'completed':
        print("\n===== BITCOIN INVESTMENT ADVISORY =====")
        print(f"Date: {results['target_date']}")
        print(f"Total news articles: {results['news_articles_count']}")
        print("\nADVISORY:")
        print("-" * 50)
        print(results['advisory'][:500] + "..." if len(results['advisory']) > 500 else results['advisory'])
        print("-" * 50)
        print(f"Full results saved to {os.path.join(agent.config['output_dir'], f'bitcoin_advisory_{results['target_date']}.json')}")
    else:
        print(f"Pipeline failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
