# Bitcoin News Effects Dataset

A comprehensive dataset for training AI models to predict the impact of news events on Bitcoin prices. This dataset transforms individual news items and market context into structured predictions about price direction, impact strength, and timeframe.

## Dataset Overview

The Bitcoin News Effects Dataset provides a structured approach to news-based cryptocurrency price prediction by:

1. **Isolating Individual News Items**: Treats each news article as a separate training sample
2. **Providing Market Context**: Includes daily market sentiment and recommendations
3. **Structured Output Format**: Consistent JSON prediction format for each news item
4. **Fine-Grained Analysis**: Classifies direction, strength, timeframe, and confidence

## Data Sources

This dataset is created from existing analyzed Bitcoin news data:

- **News Source**: Individual news items from `/outputs_btc_effects/per_date/*.json`
- **Time Period**: Coverage from multiple years of Bitcoin market activity
- **Granularity**: Each individual news article gets its own training sample
- **Volume**: Approximately 100 news items per day across thousands of days

## Dataset Creation Process

The `bitcoin_news_effects_dataset.ipynb` notebook follows these key steps:

1. **Data Collection**:

   - Loads existing analyzed news data from local JSON files
   - Extracts both long-term and short-term news items
   - Preserves daily market context for each item

2. **Data Processing**:

   - Separates each news item into an individual training sample
   - Preserves the original analysis including direction, magnitude, and confidence
   - Maintains connection to market context from that day

3. **Feature Engineering**:

   - Transforms news direction (bullish/bearish) to price direction (up/down)
   - Maps magnitude classifications to impact strength
   - Determines appropriate timeframe based on news category
   - Calculates confidence scores from multiple factors

4. **Training Data Preparation**:
   - Creates consistent instruction format for all samples
   - Formats input with news title, summary, and market context
   - Generates structured JSON outputs for prediction

## Dataset Structure

Each sample in the dataset follows this format:

### Instruction

```
Analyze Bitcoin news and predict price impact. Return JSON with this exact structure:

{
  "sentiment": "bullish|neutral|bearish",
  "price_direction": "up|sideways|down",
  "impact_strength": "high|medium|low",
  "timeframe": "immediate|short_term|medium_term",
  "confidence": 0.75,
  "key_reason": "Brief explanation of main factor"
}
```

### Input

```
News Title: [Article Title]

News Summary: [Article Summary]

Impact Tags: [Tags associated with the news]

Market Context:
Bull 65% | Base 25% | Bear 10%

Daily Recommendations:
Short-term: Buy
Long-term: Hold
```

### Output

```json
{
  "sentiment": "bullish",
  "price_direction": "up",
  "impact_strength": "medium",
  "timeframe": "short_term",
  "confidence": 0.85,
  "key_reason": "Increased institutional adoption signals growing mainstream acceptance"
}
```

## Data Fields

### Input Fields

- **News Title**: The headline of the news article
- **News Summary**: A detailed summary of the article's content
- **Impact Tags**: Categories or keywords associated with the news
- **Market Context**: Overall market sentiment probabilities
- **Daily Recommendations**: Short and long-term trading recommendations

### Output Fields

- **sentiment**: Overall market sentiment (bullish, bearish, neutral)
- **price_direction**: Expected price movement (up, down, sideways)
- **impact_strength**: Magnitude of the expected impact (high, medium, low)
- **timeframe**: When the impact is expected (immediate, short_term, medium_term)
- **confidence**: Confidence level in the prediction (0.0-1.0)
- **key_reason**: Brief explanation of the main driver of the effect

## Dataset Statistics

- **Time Period**: Multiple years of Bitcoin market data
- **Total Samples**: Thousands to potentially 100,000+ individual news items
- **News Items Per Day**: ~100 news articles on average
- **Format**: Instruction-tuning format (instruction, input, output)
- **Timeframes**: Mix of short-term and long-term impact predictions

## Distribution Analysis

The dataset aims for balanced representation across:

- **Sentiment**: Distribution across bullish, bearish, and neutral assessments
- **Direction**: Mix of up, down, and sideways price predictions
- **Impact Strength**: Range of high, medium, and low impact assessments
- **Timeframes**: Variety of immediate, short-term, and medium-term horizons

## Usage

This dataset is designed for fine-tuning language models to predict Bitcoin price movements based on news:

1. **News-Based Price Prediction**: Training models to forecast price effects from news articles
2. **Sentiment Analysis**: Learning to extract market sentiment from news content
3. **Impact Assessment**: Predicting the magnitude and timeframe of market reactions
4. **Trading Signal Generation**: Converting news analysis into actionable trading signals

## Hugging Face Integration

The dataset is available on Hugging Face:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("tahamajs/bitcoin-individual-news-dataset")

# Access a sample
sample = dataset['train'][0]
print(f"Instruction: {sample['instruction']}")
print(f"Input: {sample['input']}")
print(f"Output: {sample['output']}")
```

## Training Applications

This dataset is well-suited for:

- **Large Language Models**: Fine-tuning for Bitcoin news analysis
- **Specialized Trading Models**: Developing news-based trading strategies
- **Sentiment Analysis Systems**: Training for crypto-specific sentiment extraction
- **Market Impact Predictors**: Learning patterns between news and price movements

## Dataset Creation Tools

This dataset was created using:

- **pandas**: For data manipulation and preprocessing
- **HuggingFace Datasets**: For dataset management and publishing
- **tqdm**: For progress tracking during processing
- **Python's json module**: For handling JSON data structures

## Citation

If you use this dataset in your research or project, please cite:

```
@dataset{bitcoin_news_effects_dataset,
  author = {Tahamajs},
  title = {Bitcoin News Effects Dataset},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/tahamajs/bitcoin-individual-news-dataset}
}
```

## License

This dataset is provided for research and educational purposes only. Use at your own risk for trading or investment purposes.

## Acknowledgments

- The original news data used to create this dataset
- The HuggingFace team for their dataset infrastructure
