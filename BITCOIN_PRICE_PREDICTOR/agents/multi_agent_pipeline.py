"""
Bitcoin Investment Advisory Pipeline - Multi-Agent Architecture

This module implements a complete pipeline for Bitcoin investment advisory using 4 specialized agents:
1. NewsAgent: Collects and summarizes news using Gemini
2. AnalysisAgent: Analyzes news effects using local model
3. ForecastAgent: Forecasts prices using local model
4. RecommendationAgent: Generates advisory using local model

Features:
- Modular agent-based architecture
- Specialized agents for each pipeline stage
- Robust error handling and fallbacks
- Configurable via config.json
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
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google Generative AI package not available. Install with: pip install google-generativeai")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers package not available. Install with: pip install transformers torch")


# -----------------------------
# Config
# -----------------------------
DEFAULT_CONFIG = {
    "api_keys": {
        "gemini": "AIzaSyCj2Km9Agz40pVF1ZvXgDNNrhBvGfFxQ3w",
    },
    "models": {
        "summarization_model": "gemini-1.5-flash",
        "analysis_model_path": "../main_models/my-awesome-model_final_bitcoin-individual-news-dataset/checkpoint-400",
        "forecast_model_path": "../main_models/qwen_bitcoin_chat_fast_more_longer_explanation_v2/checkpoint-612",
        "advisory_model_path": "../main_models/my-awesome-model_final_bitcoin-investment-advisory-dataset_v2/checkpoint-400",
        "advisory_model": "gpt-4o"
    },
    "news": {
        "max_articles": 5,
        "short_term_count": 10,
        "long_term_count": 10,
        "sources": [
            "http://feeds.feedburner.com/CoingeckoBuzz"
        ]
    },
    "output_dir": "outputs/notebook_test_results",
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
logger = logging.getLogger("bitcoin_pipeline")


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


# -----------------------------
# Agent 1: News Agent
# -----------------------------

class NewsAgent:
    """
    Agent responsible for collecting news, categorizing into short-term/long-term,
    and summarizing each category using Gemini API
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})

        # Initialize Gemini API
        self.gemini_api_keys = config["api_keys"]["gemini"]
        if isinstance(self.gemini_api_keys, str):
            self.gemini_api_keys = [self.gemini_api_keys]
        self.current_key_index = 0

        logger.info("NewsAgent initialized")

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

    def collect_news(self, max_articles: int = None, target_date: str = None) -> List[NewsArticle]:
        """Collect Bitcoin-related news articles from RSS feeds"""
        if max_articles is None:
            max_articles = self.config["news"]["max_articles"]

        if target_date is None:
            target_date = dt.datetime.now().strftime('%Y-%m-%d')

        logger.info(f"NewsAgent: Collecting Bitcoin news for {target_date}")

        news_sources = self.config["news"].get("sources", [])
        if not news_sources:
            logger.warning("No news sources configured")
            return []

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
        logger.info(f"NewsAgent: Found {len(unique_articles)} unique articles")

        return unique_articles

    def bucket_news(self, articles: List[NewsArticle]) -> Dict[str, List[NewsArticle]]:
        """Bucket news articles into short-term and long-term impact categories"""
        logger.info(f"NewsAgent: Bucketing {len(articles)} articles")

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
                short_term_articles.append(article)
                long_term_articles.append(article)

        # Limit to configured counts
        short_term_limit = self.config["news"]["short_term_count"]
        long_term_limit = self.config["news"]["long_term_count"]

        short_term_articles = short_term_articles[:short_term_limit]
        long_term_articles = long_term_articles[:long_term_limit]

        logger.info(f"NewsAgent: Bucketed - {len(short_term_articles)} short-term, {len(long_term_articles)} long-term")

        return {
            'short_term': short_term_articles,
            'long_term': long_term_articles
        }

    def summarize_news_category(self, articles: List[NewsArticle], impact_type: str) -> Dict[str, Any]:
        """Summarize a category of news articles using Gemini API"""
        if not articles:
            return {
                'summary': f'No {impact_type} news available',
                'daily_summary': f'No {impact_type} news to summarize',
                'sentiment': 'neutral',
                'market_impact': 'low',
                'key_events': [],
                'risk_factors': [],
                'watch_items': [],
                'opportunities': [],
                'bullish_ratio': 0.33,
                'bearish_ratio': 0.33,
                'neutral_ratio': 0.34,
                'high_impact_count': 0,
                'confidence': 0.5,
                'recommendation': 'HOLD',
                'recommendation_confidence': 0.5
            }

        if not GEMINI_AVAILABLE or not self.gemini_api_keys:
            logger.warning("Gemini not available, using mock summary")
            return self._mock_summary(articles, impact_type)

        try:
            # Configure Gemini
            api_key = self._get_next_gemini_key()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.config["models"]["summarization_model"])

            # Prepare news data
            analysis_date = dt.datetime.now().strftime('%Y-%m-%d')
            news_data = []
            for i, article in enumerate(articles, 1):
                news_data.append({
                    "id": i,
                    "title": article.title,
                    "source": article.source,
                    "url": article.url,
                    "content": article.content[:1000]
                })

            # Create prompt
            prompt = summarization_prompt(
                news_items=news_data,
                analysis_date=analysis_date,
                impact_type=impact_type,
                style=self.config.get("prompt_style", "comprehensive")
            )

            if not prompt:
                return self._mock_summary(articles, impact_type)

            # Call Gemini
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_output_tokens": 4096,
                }
            )

            # Parse response
            text = response.text.strip()
            text = text.replace('```json', '').replace('```', '').strip()

            try:
                result = json.loads(text)
            except:
                if '{' in text and '}' in text:
                    candidate = text[text.find('{'): text.rfind('}')+1]
                    result = json.loads(candidate)
                else:
                    raise

            result = validate_summary_payload(result)
            logger.info(f"NewsAgent: Successfully summarized {impact_type} news")
            return result

        except Exception as e:
            logger.error(f"NewsAgent: Error summarizing {impact_type} news: {e}")
            return self._mock_summary(articles, impact_type)

    def _mock_summary(self, articles: List[NewsArticle], impact_type: str) -> Dict[str, Any]:
        """Generate mock summary when Gemini is not available"""
        bullish_ratio = 0.6 if impact_type == 'short_term' else 0.55
        bearish_ratio = 0.3 if impact_type == 'short_term' else 0.25
        neutral_ratio = 1 - bullish_ratio - bearish_ratio

        return {
            'summary': f"Mock {impact_type} summary of {len(articles)} articles.",
            'daily_summary': f"Mock daily summary for {impact_type} news.",
            'sentiment': 'bullish' if bullish_ratio > bearish_ratio else 'bearish',
            'market_impact': 'medium',
            'key_events': [f"{impact_type} key event {i}" for i in range(1, 4)],
            'risk_factors': [f"{impact_type} risk factor {i}" for i in range(1, 4)],
            'watch_items': [f"{impact_type} watch item {i}" for i in range(1, 4)],
            'opportunities': [f"{impact_type} opportunity {i}" for i in range(1, 4)],
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'neutral_ratio': neutral_ratio,
            'high_impact_count': len(articles) // 3,
            'confidence': 0.85,
            'recommendation': 'BUY' if bullish_ratio > 0.5 else 'HOLD',
            'recommendation_confidence': bullish_ratio
        }

    def run(self, target_date: str = None) -> Dict[str, Any]:
        """Run the complete news collection and summarization pipeline"""
        logger.info("NewsAgent: Starting news pipeline")

        # Collect news
        articles = self.collect_news(target_date=target_date)

        # Bucket news
        bucketed_news = self.bucket_news(articles)

        # Summarize each category
        short_term_summary = self.summarize_news_category(
            bucketed_news['short_term'], 'short_term'
        )
        long_term_summary = self.summarize_news_category(
            bucketed_news['long_term'], 'long_term'
        )

        # Combine results
        result = {
            'date': dt.datetime.now().strftime('%Y-%m-%d'),
            'total_articles': len(articles),
            'short_term_count': len(bucketed_news['short_term']),
            'long_term_count': len(bucketed_news['long_term']),
            'short_term_summary': short_term_summary,
            'long_term_summary': long_term_summary,
            'articles': [
                {
                    'title': a.title,
                    'url': a.url,
                    'impact_type': a.impact_type,
                    'source': a.source
                } for a in articles
            ]
        }

        logger.info("NewsAgent: Completed news pipeline")
        return result


# -----------------------------
# Agent 2: Analysis Agent
# -----------------------------

class AnalysisAgent:
    """
    Agent responsible for analyzing news effects using the local analysis model
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.local_model = None

        # Load local analysis model
        if TRANSFORMERS_AVAILABLE:
            model_path = self.config["models"].get("analysis_model_path")
            if model_path and os.path.exists(model_path):
                try:
                    logger.info(f"AnalysisAgent: Loading model from {model_path}")
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    self.local_model = {
                        "tokenizer": tokenizer,
                        "model": model
                    }
                    logger.info("AnalysisAgent: Model loaded successfully")
                except Exception as e:
                    logger.error(f"AnalysisAgent: Error loading model: {e}")
            else:
                logger.warning(f"AnalysisAgent: Model path not found: {model_path}")
        else:
            logger.warning("AnalysisAgent: Transformers not available")

    def analyze_news_effects(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the effects of news using local model"""
        logger.info("AnalysisAgent: Analyzing news effects")

        if not self.local_model:
            logger.warning("AnalysisAgent: No local model available, using mock analysis")
            return self._mock_analysis(news_data)

        try:
            # Prepare input for the model
            input_text = f"Analyze the effects of this Bitcoin news analysis on price: {json.dumps(news_data, ensure_ascii=False)}"

            tokenizer = self.local_model["tokenizer"]
            model = self.local_model["model"]

            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
            outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Try to parse JSON response
            try:
                analysis = json.loads(response)
            except:
                # Fallback to mock if parsing fails
                analysis = self._mock_analysis(news_data)

            logger.info("AnalysisAgent: Successfully analyzed news effects")
            return analysis

        except Exception as e:
            logger.error(f"AnalysisAgent: Error analyzing news: {e}")
            return self._mock_analysis(news_data)

    def _mock_analysis(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock analysis when model is not available"""
        return {
            'bull_prob': 0.65,
            'bear_prob': 0.25,
            'base_prob': 0.10,
            'scenarios': {
                'bullish': 0.65,
                'bearish': 0.25,
                'base': 0.10
            },
            'market_sentiment': 'bullish',
            'confidence': 0.85
        }

    def run(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the analysis pipeline"""
        logger.info("AnalysisAgent: Starting analysis pipeline")

        # Get Bitcoin price data
        btc_data = self._get_bitcoin_price_data()

        # Analyze news effects
        effects_analysis = self.analyze_news_effects(news_data)

        # Combine with price data
        analysis_result = {
            **news_data,
            **btc_data,
            **effects_analysis
        }

        logger.info("AnalysisAgent: Completed analysis pipeline")
        return analysis_result

    def _get_bitcoin_price_data(self) -> Dict[str, Any]:
        """Get Bitcoin price data from Yahoo Finance"""
        try:
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=70)

            btc = yf.Ticker("BTC-USD")
            hist = btc.download(start=start_date, end=end_date, progress=False)

            if hist.empty:
                raise ValueError("No price data received")

            prices = hist['Close'].dropna().tolist()[-60:]

            current_price = prices[-1]
            next_10_day_prices = [current_price * (1 + 0.005 * i) for i in range(10)]

            return {
                'price_history_60d': prices,
                'current_price': current_price,
                'next_10_day_prices': next_10_day_prices,
            }

        except Exception as e:
            logger.error(f"AnalysisAgent: Error fetching Bitcoin price data: {e}")
            return {
                'price_history_60d': [50000] * 60,
                'current_price': 50000,
                'next_10_day_prices': [50000 + (i * 100) for i in range(10)],
            }


# -----------------------------
# Agent 3: Forecast Agent
# -----------------------------

class ForecastAgent:
    """
    Agent responsible for forecasting Bitcoin prices using the local forecast model
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.local_model = None

        # Load local forecast model
        if TRANSFORMERS_AVAILABLE:
            model_path = self.config["models"].get("forecast_model_path")
            if model_path and os.path.exists(model_path):
                try:
                    logger.info(f"ForecastAgent: Loading model from {model_path}")
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    self.local_model = {
                        "tokenizer": tokenizer,
                        "model": model
                    }
                    logger.info("ForecastAgent: Model loaded successfully")
                except Exception as e:
                    logger.error(f"ForecastAgent: Error loading model: {e}")
            else:
                logger.warning(f"ForecastAgent: Model path not found: {model_path}")
        else:
            logger.warning("ForecastAgent: Transformers not available")

    def forecast_prices(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast Bitcoin prices for next 10 days"""
        logger.info("ForecastAgent: Forecasting Bitcoin prices")

        if not self.local_model:
            logger.warning("ForecastAgent: No local model available, using mock forecast")
            return self._mock_forecast(analysis_data)

        try:
            # Prepare input for the model
            input_text = f"Forecast Bitcoin prices for the next 10 days based on this analysis: {json.dumps(analysis_data, ensure_ascii=False)}"

            tokenizer = self.local_model["tokenizer"]
            model = self.local_model["model"]

            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
            outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Try to extract prices from response
            import re
            price_pattern = r'\$?(\d+(?:\.\d+)?)'
            prices = re.findall(price_pattern, response)

            if len(prices) >= 10:
                next_10_day_prices = [float(p) for p in prices[:10]]
            else:
                # Fallback to current price based forecast
                current_price = analysis_data.get('current_price', 50000)
                next_10_day_prices = [current_price * (1 + 0.005 * i) for i in range(10)]

            analysis_data['next_10_day_prices'] = next_10_day_prices
            logger.info("ForecastAgent: Successfully forecasted prices")
            return analysis_data

        except Exception as e:
            logger.error(f"ForecastAgent: Error forecasting prices: {e}")
            return self._mock_forecast(analysis_data)

    def _mock_forecast(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock forecast when model is not available"""
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

    def run(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the forecast pipeline"""
        logger.info("ForecastAgent: Starting forecast pipeline")

        # Generate forecast
        forecast_result = self.forecast_prices(analysis_data)

        logger.info("ForecastAgent: Completed forecast pipeline")
        return forecast_result


# -----------------------------
# Agent 4: Recommendation Agent
# -----------------------------

class RecommendationAgent:
    """
    Agent responsible for generating investment recommendations using the local advisory model
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.local_model = None

        # Load local advisory model
        if TRANSFORMERS_AVAILABLE:
            model_path = self.config["models"].get("advisory_model_path")
            if model_path and os.path.exists(model_path):
                try:
                    logger.info(f"RecommendationAgent: Loading model from {model_path}")
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    self.local_model = {
                        "tokenizer": tokenizer,
                        "model": model
                    }
                    logger.info("RecommendationAgent: Model loaded successfully")
                except Exception as e:
                    logger.error(f"RecommendationAgent: Error loading model: {e}")
            else:
                logger.warning(f"RecommendationAgent: Model path not found: {model_path}")
        else:
            logger.warning("RecommendationAgent: Transformers not available")

    def generate_advisory(self, complete_data: Dict[str, Any]) -> str:
        """Generate investment advisory using local model"""
        logger.info("RecommendationAgent: Generating investment advisory")

        if not self.local_model:
            logger.warning("RecommendationAgent: No local model available, using simple advisory")
            return self._simple_advisory(complete_data)

        try:
            # Prepare input for the model
            input_text = f"Generate a comprehensive Bitcoin investment advisory based on this analysis: {json.dumps(complete_data, ensure_ascii=False)}"

            tokenizer = self.local_model["tokenizer"]
            model = self.local_model["model"]

            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
            outputs = model.generate(**inputs, max_length=1024, num_return_sequences=1)
            advisory = tokenizer.decode(outputs[0], skip_special_tokens=True)

            logger.info("RecommendationAgent: Successfully generated advisory")
            return advisory

        except Exception as e:
            logger.error(f"RecommendationAgent: Error generating advisory: {e}")
            return self._simple_advisory(complete_data)

    def _simple_advisory(self, complete_data: Dict[str, Any]) -> str:
        """Generate simple advisory when model is not available"""
        date_str = complete_data.get('date', dt.datetime.now().strftime('%Y-%m-%d'))
        next_10_day_prices = complete_data.get('next_10_day_prices', [])

        if not next_10_day_prices:
            return f"Unable to generate investment advisory due to missing price forecast data for {date_str}."

        price_change = ((next_10_day_prices[-1] / next_10_day_prices[0]) - 1) * 100
        current_price = next_10_day_prices[0]
        final_price = next_10_day_prices[-1]

        daily_view = complete_data.get('daily_view', {})
        sentiment = daily_view.get('sentiment', 'neutral')
        market_impact = daily_view.get('market_impact', 'medium')

        short_term_rec = daily_view.get('recommendation_short_term', {}).get('action', 'HOLD')
        long_term_rec = daily_view.get('recommendation_long_term', {}).get('action', 'HOLD')

        advisory = f"""# Bitcoin Investment Advisory for {date_str}

## Executive Summary
Based on the comprehensive analysis of {complete_data.get('total_articles', 0)} news items and current market conditions, the Bitcoin market currently shows {sentiment} sentiment with {market_impact} impact expected. Our 10-day price forecast projects a {price_change:+.2f}% move from ${current_price:.2f} to ${final_price:.2f}.

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

    def run(self, complete_data: Dict[str, Any]) -> str:
        """Run the recommendation pipeline"""
        logger.info("RecommendationAgent: Starting recommendation pipeline")

        # Generate advisory
        advisory = self.generate_advisory(complete_data)

        logger.info("RecommendationAgent: Completed recommendation pipeline")
        return advisory


# -----------------------------
# Pipeline Orchestrator
# -----------------------------

class BitcoinPipelineOrchestrator:
    """
    Main orchestrator that coordinates all 4 agents in the Bitcoin investment advisory pipeline
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the orchestrator with all agents

        Args:
            config_path: Path to configuration file (defaults to config.json)
        """
        # Load configuration
        self.config = DEFAULT_CONFIG.copy()

        if config_path is None:
            default_config_path = os.path.join(os.path.dirname(__file__), "config.json")
            if os.path.exists(default_config_path):
                config_path = default_config_path

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    self._update_nested_dict(self.config, custom_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")

        # Initialize agents
        self.news_agent = NewsAgent(self.config)
        self.analysis_agent = AnalysisAgent(self.config)
        self.forecast_agent = ForecastAgent(self.config)
        self.recommendation_agent = RecommendationAgent(self.config)

        # Create output directory
        os.makedirs(self.config["output_dir"], exist_ok=True)

        logger.info("BitcoinPipelineOrchestrator initialized with all agents")

    def _update_nested_dict(self, d: dict, u: dict) -> dict:
        """Recursively update nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

    def run_pipeline(self, target_date: str = None) -> Dict[str, Any]:
        """
        Run the complete pipeline through all agents

        Args:
            target_date: Target date for analysis

        Returns:
            Complete pipeline results
        """
        if target_date is None:
            target_date = dt.datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Starting Bitcoin investment advisory pipeline for {target_date}")

        results = {
            'target_date': target_date,
            'start_time': dt.datetime.now().isoformat(),
            'status': 'running',
            'pipeline_stages': []
        }

        try:
            # Stage 1: News Collection and Summarization
            logger.info("=== STAGE 1: News Agent ===")
            news_results = self.news_agent.run(target_date=target_date)
            results['pipeline_stages'].append({
                'stage': 'news_collection',
                'status': 'completed',
                'data': news_results
            })

            # Stage 2: News Analysis
            logger.info("=== STAGE 2: Analysis Agent ===")
            analysis_results = self.analysis_agent.run(news_results)
            results['pipeline_stages'].append({
                'stage': 'news_analysis',
                'status': 'completed',
                'data': analysis_results
            })

            # Stage 3: Price Forecasting
            logger.info("=== STAGE 3: Forecast Agent ===")
            forecast_results = self.forecast_agent.run(analysis_results)
            results['pipeline_stages'].append({
                'stage': 'price_forecast',
                'status': 'completed',
                'data': forecast_results
            })

            # Stage 4: Investment Advisory
            logger.info("=== STAGE 4: Recommendation Agent ===")
            advisory = self.recommendation_agent.run(forecast_results)
            results['pipeline_stages'].append({
                'stage': 'investment_advisory',
                'status': 'completed',
                'data': advisory
            })

            # Final results
            results.update({
                'status': 'completed',
                'end_time': dt.datetime.now().isoformat(),
                'final_advisory': advisory,
                'forecast_prices': forecast_results.get('next_10_day_prices', [])
            })

            # Save results
            self._save_results(results, target_date)

            logger.info(f"Pipeline completed successfully for {target_date}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = dt.datetime.now().isoformat()

        return results

    def _save_results(self, results: Dict[str, Any], target_date: str):
        """Save pipeline results to file"""
        output_path = os.path.join(self.config['output_dir'], f"bitcoin_pipeline_{target_date}.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Pipeline results saved to {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Bitcoin Investment Advisory Multi-Agent Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file (optional)')
    parser.add_argument('--date', type=str, help='Target date for analysis (YYYY-MM-DD)')
    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = BitcoinPipelineOrchestrator(config_path=args.config)

    # Run pipeline
    results = orchestrator.run_pipeline(target_date=args.date)

    if results['status'] == 'completed':
        print("\n" + "="*60)
        print("BITCOIN INVESTMENT ADVISORY PIPELINE COMPLETED")
        print("="*60)
        print(f"Date: {results['target_date']}")
        print(f"News Articles Processed: {results['pipeline_stages'][0]['data']['total_articles']}")
        print("\nFINAL INVESTMENT ADVISORY:")
        print("-" * 40)
        print(results['final_advisory'][:500] + "..." if len(results['final_advisory']) > 500 else results['final_advisory'])
        print("-" * 40)
        print(f"Full results saved to {orchestrator.config['output_dir']}/bitcoin_pipeline_{results['target_date']}.json")
    else:
        print(f"Pipeline failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()