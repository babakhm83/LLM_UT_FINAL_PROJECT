# Bitcoin Pipeline Agent - Complete Investment Advisory System

## 🚀 Overview

The Bitcoin Pipeline Agent is a comprehensive, end-to-end investment advisory system that combines real-time news analysis, market sentiment evaluation, price forecasting, and professional investment recommendations. This sophisticated system leverages multiple AI models and data sources to provide institutional-grade Bitcoin investment insights.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Pipeline Stages](#pipeline-stages)
- [Models and Training](#models-and-training)
- [Output Examples](#output-examples)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Contributing](#contributing)

## ✨ Features

### 🔄 Complete Pipeline Automation

- **Automated News Collection**: Scrapes Bitcoin-related news from multiple RSS feeds and web sources
- **Intelligent News Categorization**: Separates news into short-term and long-term impact categories
- **AI-Powered Summarization**: Uses advanced language models to summarize and analyze news impact
- **Market Effects Analysis**: Employs trained models to assess news effects on Bitcoin price
- **Price Forecasting**: Predicts Bitcoin prices for the next 10 days using specialized forecasting models
- **Investment Advisory Generation**: Creates comprehensive, institutional-grade investment recommendations

### 🧠 Advanced AI Integration

- **Multi-Model Architecture**: Integrates Gemini, OpenAI, and custom trained models
- **Sentiment Analysis**: Analyzes market sentiment from news and social indicators
- **Risk Assessment**: Evaluates multiple risk factors and scenarios
- **Scenario Planning**: Provides bullish, bearish, and base case projections

### 📊 Data Sources

- **Real-time News**: Google News RSS feeds, cryptocurrency news sources
- **Market Data**: Yahoo Finance API for historical and current Bitcoin prices
- **Web Scraping**: Advanced content extraction using newspaper3k and BeautifulSoup
- **API Integration**: Support for multiple external data providers

### 🎯 Professional Output

- **Institutional-Grade Reports**: Detailed investment advisories suitable for professional use
- **Risk Management**: Comprehensive risk analysis and hedging strategies
- **Portfolio Integration**: Guidance on Bitcoin allocation within broader portfolios
- **Multiple Timeframes**: Short-term tactical and long-term strategic recommendations

## 🏗️ Architecture

The system follows a modular, pipeline-based architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   News Sources  │────│  Data Pipeline  │────│  AI Processing  │
│                 │    │                 │    │                 │
│ • RSS Feeds     │    │ • Collection    │    │ • Summarization │
│ • Web Scraping  │    │ • Deduplication │    │ • Categorization│
│ • APIs          │    │ • Filtering     │    │ • Sentiment     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Final Output  │────│   Advisory Gen  │────│ Effects Analysis│
│                 │    │                 │    │                 │
│ • JSON Results  │    │ • OpenAI GPT-4  │    │ • Trained Model │
│ • Formatted     │    │ • Professional  │    │ • Probabilities │
│ • Advisory Text │    │ • Reports       │    │ • Scenarios     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Price Forecast  │────│  Market Data    │
                       │                 │    │                 │
                       │ • 10-day Pred   │    │ • Yahoo Finance │
                       │ • Trained Model │    │ • Price History │
                       │ • Scenarios     │    │ • Volume Data   │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection for API access

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd bitcoin-pipeline-agent

# Or download the pipeline_agent.py file directly
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Required Packages

If `requirements.txt` is not available, install manually:

```bash
pip install requests feedparser beautifulsoup4 pandas numpy yfinance newspaper3k openai google-generativeai
```

### Step 4: Set Up Environment Variables

```bash
# Create a .env file or set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## ⚙️ Configuration

### Configuration File Structure

Create a `config.json` file with the following structure:

```json
{
  "api_keys": {
    "gemini": ["your-gemini-api-key-1", "your-gemini-api-key-2"],
    "openai": "your-openai-api-key",
    "openai_base_url": "https://api.openai.com/v1"
  },
  "models": {
    "summarization_model": "gemini-2.0-flash-exp",
    "effects_model": "https://your-effects-model-endpoint.com/predict",
    "forecast_model": "https://your-forecast-model-endpoint.com/predict",
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
```

### Key Configuration Options

#### API Keys

- **gemini**: List of Gemini API keys for load balancing
- **openai**: OpenAI API key for advisory generation
- **openai_base_url**: Custom OpenAI API endpoint (optional)

#### Models

- **summarization_model**: Gemini model for news summarization
- **effects_model**: Your trained model endpoint for effects analysis
- **forecast_model**: Your trained model endpoint for price forecasting
- **advisory_model**: OpenAI model for generating investment advisories

#### News Settings

- **max_articles**: Maximum number of articles to collect
- **short_term_count**: Number of articles for short-term analysis
- **long_term_count**: Number of articles for long-term analysis
- **sources**: List of RSS feed URLs

## 📖 Usage

### Command Line Interface

#### Basic Usage

```bash
python pipeline_agent.py
```

#### With Configuration File

```bash
python pipeline_agent.py --config config.json
```

#### Specific Date Analysis

```bash
python pipeline_agent.py --config config.json --date 2025-09-07
```

### Python API Usage

#### Basic Implementation

```python
from pipeline_agent import BitcoinPipelineAgent

# Initialize the agent
agent = BitcoinPipelineAgent(config_path="config.json")

# Run the complete pipeline
results = agent.run()

# Access results
print(f"Status: {results['status']}")
print(f"Advisory: {results['advisory']}")
```

#### Advanced Usage with Custom Parameters

```python
from pipeline_agent import BitcoinPipelineAgent

# Initialize with custom configuration
agent = BitcoinPipelineAgent(config_path="custom_config.json")

# Run pipeline for specific date
results = agent.run(target_date="2025-09-07")

# Access detailed results
if results['status'] == 'completed':
    print(f"Analyzed {results['news_articles_count']} articles")
    print(f"Price forecast: {results['forecast']}")
    print(f"Investment advisory:\n{results['advisory']}")
else:
    print(f"Pipeline failed: {results['error']}")
```

### Jupyter Notebook Integration

The agent is fully integrated into the `bitcoin_prediction_agent.ipynb` notebook:

```python
# In your Jupyter notebook
from pipeline_agent import BitcoinPipelineAgent

# Initialize the pipeline agent
pipeline_agent = BitcoinPipelineAgent(config_path="config.json")

# Run the complete pipeline
results = pipeline_agent.run()

# Display results
display(results['advisory'])
```

## 🔌 API Integration

### Supported APIs

#### 1. Gemini API (Google)

- **Purpose**: News summarization and analysis
- **Models**: gemini-2.0-flash-exp, gemini-pro
- **Rate Limits**: Handled with key rotation
- **Authentication**: API key required

#### 2. OpenAI API

- **Purpose**: Investment advisory generation
- **Models**: gpt-4o, gpt-4, gpt-3.5-turbo
- **Rate Limits**: Built-in retry logic
- **Authentication**: API key required

#### 3. Yahoo Finance API

- **Purpose**: Bitcoin price data
- **Data**: Historical prices, volume, market cap
- **Rate Limits**: None (free tier)
- **Authentication**: None required

#### 4. Custom Trained Models

- **Purpose**: Effects analysis and price forecasting
- **Protocol**: HTTP POST with JSON payload
- **Authentication**: As per your model deployment

### Model Endpoint Specifications

#### Effects Model Endpoint

```
POST /predict
Content-Type: application/json

{
  "news_analysis": {
    "date": "2025-09-07",
    "short_term_summary": "...",
    "long_term_summary": "...",
    "price_history_60d": [...],
    "current_price": 50000,
    ...
  }
}

Response:
{
  "bull_prob": 0.65,
  "bear_prob": 0.25,
  "base_prob": 0.10,
  "scenarios": {...}
}
```

#### Forecast Model Endpoint

```
POST /predict
Content-Type: application/json

{
  "analysis_data": {
    "news_analysis": {...},
    "effects_analysis": {...},
    "price_history_60d": [...],
    ...
  }
}

Response:
{
  "next_10_day_prices": [50000, 51000, 52000, ...]
}
```

## 🔄 Pipeline Stages

### Stage 1: News Collection

**Function**: `collect_news()`

**Process**:

1. Fetches RSS feeds from configured sources
2. Parses XML/RSS content using feedparser
3. Extracts article metadata (title, URL, date)
4. Deduplicates articles by URL
5. Scrapes full article content using newspaper3k and BeautifulSoup
6. Handles failures gracefully with fallback methods

**Output**: List of `NewsArticle` objects with full content

### Stage 2: News Categorization

**Function**: `bucket_news()`

**Process**:

1. Analyzes article titles and content for keywords
2. Categorizes into short-term vs long-term impact
3. Uses keyword matching algorithm:
   - Short-term: price, crash, rally, trading, etc.
   - Long-term: regulation, adoption, technology, etc.
4. Handles articles that fit both categories
5. Limits articles per category based on configuration

**Output**: Dictionary with categorized article lists

### Stage 3: News Summarization

**Function**: `summarize_items()`

**Process**:

1. Calls Gemini API for each category (short-term, long-term)
2. Generates comprehensive summaries with structured output
3. Extracts sentiment, market impact, key events
4. Calculates bullish/bearish ratios
5. Provides confidence scores and recommendations
6. Combines results into unified news analysis

**Output**: Structured news analysis with sentiment and probabilities

### Stage 4: Effects Analysis

**Function**: `analyze_effects()`

**Process**:

1. Enriches news analysis with Bitcoin price data
2. Calls trained effects model with complete analysis
3. Receives market scenario probabilities
4. Integrates effects data with news analysis
5. Handles model failures with fallback logic

**Output**: Enhanced analysis with effects probabilities

### Stage 5: Price Forecasting

**Function**: `forecast_next_10_days()`

**Process**:

1. Calls trained forecast model with complete analysis data
2. Receives 10-day price predictions
3. Validates forecast data
4. Handles model failures with mock forecasts
5. Integrates predictions into analysis

**Output**: Analysis with 10-day price forecasts

### Stage 6: Investment Advisory Generation

**Function**: `generate_investment_advisory()`

**Process**:

1. Creates comprehensive prompt with all analysis data
2. Calls OpenAI API for advisory generation
3. Generates institutional-grade investment recommendations
4. Includes risk assessment, scenarios, and strategies
5. Formats output for professional use

**Output**: Comprehensive investment advisory text

## 🤖 Models and Training

### Model Requirements

#### 1. News Summarization Model

- **Provider**: Google Gemini
- **Model**: gemini-2.0-flash-exp
- **Input**: Array of news articles with metadata
- **Output**: Structured JSON with summaries, sentiment, probabilities
- **Training**: Pre-trained, no custom training required

#### 2. Effects Analysis Model

- **Type**: Custom trained model
- **Input**: Complete news analysis with price data
- **Output**: Market scenario probabilities (bull/bear/base)
- **Training**: Your custom training pipeline
- **Deployment**: REST API endpoint

#### 3. Forecast Model

- **Type**: Custom trained model
- **Input**: Complete analysis data including effects
- **Output**: 10-day price predictions
- **Training**: Your custom training pipeline
- **Deployment**: REST API endpoint

#### 4. Investment Advisory Model

- **Provider**: OpenAI
- **Model**: gpt-4o
- **Input**: Complete analysis with price forecasts
- **Output**: Professional investment advisory text
- **Training**: Pre-trained, no custom training required

### Training Data Format

#### For Effects Model

```json
{
  "news_analysis": {
    "date": "2025-09-07",
    "short_term_summary": "Market shows bullish sentiment...",
    "long_term_summary": "Regulatory developments suggest...",
    "bullish_ratio": 0.6,
    "bearish_ratio": 0.3,
    "neutral_ratio": 0.1,
    "price_history_60d": [50000, 51000, ...],
    "current_price": 52000
  }
}
```

#### For Forecast Model

```json
{
  "analysis_data": {
    "news_analysis": {...},
    "bull_prob": 0.65,
    "bear_prob": 0.25,
    "base_prob": 0.10,
    "price_history_60d": [...],
    "current_price": 52000
  }
}
```

## 📊 Output Examples

### Complete Pipeline Results

```json
{
  "target_date": "2025-09-07",
  "status": "completed",
  "news_articles_count": 28,
  "short_term_count": 10,
  "long_term_count": 10,
  "news_analysis": {
    "date": "2025-09-07",
    "total_news_items": 20,
    "bullish_ratio": 0.6,
    "bearish_ratio": 0.3,
    "neutral_ratio": 0.1,
    "short_term_summary": "...",
    "long_term_summary": "..."
  },
  "effects_analysis": {
    "bull_prob": 0.65,
    "bear_prob": 0.25,
    "base_prob": 0.10
  },
  "forecast": [52000, 53000, 54000, ...],
  "advisory": "# Bitcoin Investment Advisory for 2025-09-07..."
}
```

### Sample Investment Advisory

```markdown
# Bitcoin Investment Advisory for 2025-09-07

## Executive Summary

Based on comprehensive analysis of 28 news items and current market conditions,
Bitcoin shows bullish sentiment with medium-high impact expected. Our 10-day
price forecast projects a +8.5% move from $52,000 to $56,420.

## Investment Recommendations

- **Short-term recommendation**: BUY (75% confidence)
- **Long-term recommendation**: HOLD (65% confidence)
- **Position size**: 3-5% of portfolio for aggressive investors
- **Entry strategy**: Dollar-cost average over 3-5 days

## Risk Assessment & Management

### Primary Risks:

- Regulatory uncertainty in key markets
- Macroeconomic headwinds affecting risk assets
- Technical resistance at $55,000 level

### Risk Mitigation:

- Set stop-loss at $48,000 (-8%)
- Take partial profits at $57,000 (+10%)
- Monitor regulatory developments closely

## Price Targets & Scenarios

- **Bullish Scenario (65% probability)**: $58,000-$62,000
- **Base Case (25% probability)**: $50,000-$55,000
- **Bearish Scenario (10% probability)**: $45,000-$48,000

...
```

## 🔧 Troubleshooting

### Common Issues

#### 1. API Key Errors

**Problem**: "No Gemini API keys available" or "OpenAI authentication failed"
**Solution**:

- Verify API keys in configuration file
- Check environment variables
- Ensure keys have sufficient credits/quota
- Test keys with simple API calls

#### 2. News Collection Failures

**Problem**: "Failed to fetch" or "No articles collected"
**Solution**:

- Check internet connection
- Verify RSS feed URLs are accessible
- Review rate limiting and user agent settings
- Check firewall/proxy settings

#### 3. Model Endpoint Errors

**Problem**: "Error calling trained model" or timeout errors
**Solution**:

- Verify model endpoints are accessible
- Check authentication for custom models
- Review payload format and data types
- Test endpoints independently
- Implement fallback logic

#### 4. Memory/Performance Issues

**Problem**: High memory usage or slow execution
**Solution**:

- Reduce max_articles in configuration
- Implement article content length limits
- Use ThreadPoolExecutor for parallel processing
- Monitor system resources

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger("pipeline_agent").setLevel(logging.DEBUG)
```

### Configuration Validation

```python
# Validate configuration before running
agent = BitcoinPipelineAgent(config_path="config.json")
if not agent.gemini_api_keys:
    print("Warning: No Gemini API keys configured")
if not agent.openai_client:
    print("Warning: OpenAI client not initialized")
```

## 🛠️ Development

### Project Structure

```
bitcoin-pipeline-agent/
├── pipeline_agent.py          # Main pipeline implementation
├── config.json                # Configuration file
├── requirements.txt           # Python dependencies
├── btc_agent_readme.md        # This documentation
├── outputs/                   # Generated reports
│   └── pipeline/
├── logs/                      # Application logs
│   └── bitcoin_agent.log
└── tests/                     # Unit tests
    ├── test_pipeline.py
    ├── test_news_collection.py
    └── test_models.py
```

### Adding New Features

#### 1. New Data Sources

```python
def add_custom_news_source(self, source_url: str, parser_func):
    """Add a custom news source with parsing function"""
    # Implementation here
```

#### 2. Additional Models

```python
def integrate_new_model(self, model_endpoint: str, model_type: str):
    """Integrate a new AI model into the pipeline"""
    # Implementation here
```

#### 3. Enhanced Analytics

```python
def add_technical_analysis(self, price_data: List[float]):
    """Add technical analysis indicators"""
    # Implementation here
```

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_news_collection.py

# Integration tests
python -m pytest tests/test_pipeline.py::test_full_pipeline
```

### Code Quality

```bash
# Format code
black pipeline_agent.py

# Lint code
flake8 pipeline_agent.py

# Type checking
mypy pipeline_agent.py
```

## 🤝 Contributing

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure code quality (black, flake8, mypy)
5. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd bitcoin-pipeline-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Areas for Contribution

- Additional news sources and parsers
- Enhanced sentiment analysis algorithms
- New AI model integrations
- Performance optimizations
- Extended test coverage
- Documentation improvements
- Bug fixes and error handling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or feature requests:

1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information
4. For urgent matters, contact the development team

## 🙏 Acknowledgments

- Google Gemini API for news summarization
- OpenAI for investment advisory generation
- Yahoo Finance for market data
- The open-source community for the excellent libraries used
- Contributors and testers who helped improve the system

---

**Note**: This system is for informational purposes only and should not be considered as financial advice. Always consult with qualified financial advisors before making investment decisions.
