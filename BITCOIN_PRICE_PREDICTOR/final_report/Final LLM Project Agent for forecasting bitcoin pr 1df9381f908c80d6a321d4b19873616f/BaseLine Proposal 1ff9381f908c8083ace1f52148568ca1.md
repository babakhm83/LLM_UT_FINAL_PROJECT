# BaseLine Proposal :

Taha Majlesi 810101504

Mahdi Naeni 810101536

Babak hosseini Mohtasham 810101408

**LLM-Based Bitcoin Price Forecasting Using Tweet Events**

**1. Introduction**

Time series forecasting is a cornerstone of data-driven decision-making in finance, energy, and other domains. Traditional numerical methods excel when historical patterns persist but struggle to adapt to sudden, event-driven shifts. Recent research by Wang et al. (NeurIPS 2024) introduced a paradigm that integrates unstructured news events into forecasting models via Large Language Models (LLMs) and generative agents, achieving significant improvements in domains such as electricity demand, exchange rates, and bitcoin prices. This proposal adapts that framework to social media signals, specifically tweets, to predict Bitcoin price fluctuations. By leveraging the immediacy and richness of Twitter data, we aim to enhance forecasting accuracy in a volatile, sentiment-driven market.

**2. Motivation and Objectives**

Cryptocurrency markets are highly sensitive to public sentiment, news, and social media discourse. Tweets from influential accounts, breaking news, and viral trends can trigger rapid price movements. While prior work has shown the benefit of filtering and integrating news articles, tweets offer several advantages:

- **Real-time Signals**: Tweets are published continuously and may reflect market sentiment before traditional news outlets report an event.
- **Diverse Perspectives**: Twitter hosts a wide range of users, from retail traders to institutional analysts, providing a multifaceted view of market mood.
- **Rich Metadata**: Tweets include likes, retweets, geolocation, and user metadata that can inform impact analysis.

Our primary objectives are:

1. **Data Integration**: Collect and preprocess historical Bitcoin price series alongside time-synchronized tweets.
2. **Event Filtering Agent**: Develop an LLM-based reasoning agent to identify tweets likely to influence price movements, categorizing them by short- and long-term impact.
3. **Prompt Construction**: Design contextual prompts that combine numerical price tokens, tweet summaries, and supplementary metadata.
4. **Fine-Tuning**: Adapt a pre-trained LLM (e.g., LLaMa 2) with Low-Rank Adaptation (LoRA) to forecast the next-price sequence conditioned on tweet context.
5. **Evaluation & Reflection**: Implement an evaluation agent to detect missed tweet events and iteratively refine the tweet-selection logic.
6. **Performance Analysis**: Compare the proposed method against baselines—numerical-only LLM forecasting, sentiment-augmented models, and standard time series architectures (e.g., ARIMA, LSTM).

**3. Related Work**

- *News-to-Forecast Framework*: Wang et al. (2024) demonstrated that integrating filtered news through LLM agents yields superior forecasting accuracy across diverse domains. Their pipeline includes an information retrieval module, reasoning/filtering agents, prompt integration, LLM fine-tuning, and an evaluation loop.
- *Social Media in Finance*: Previous studies have incorporated Twitter sentiment scores into regression or deep learning models for stock and cryptocurrency forecasting, often relying on keyword counts or aggregated sentiment indexes. However, they lack the deep contextual understanding enabled by LLMs and iterative event reflection.
- *LLM-Based Forecasting*: Emerging work frames forecasting as next-token prediction using LLMs (Gruver et al., 2023), showing promising few-shot capabilities. The integration of external events—tweets in this case—into LLM input has yet to be fully explored.

**4. Methodology**

***4.1 Data Collection***

- **Bitcoin Price Series**: Obtain historical BTC–USD price data (e.g., daily close, high, low, volume) from APIs such as CoinGecko or Binance covering the past five years.
- **Twitter Data**: Use Twitter Academic API to retrieve tweets containing keywords (#Bitcoin, BTC, crypto) and from influential accounts (exchanges, analysts) timestamped concurrently with price data. Collect metadata: retweet counts, likes, user follower counts, and geolocation when available.

***4.2 Preprocessing***

- **Price Tokens**: Convert price values into sequences of digit tokens and delimiters (e.g., “1”, “2”, “0”, “5”, “0”, “.”, “3”, “0”) for LLM input.
- **Tweet Filtering**: Apply basic filters for language (English), removal of spam/bots (using user verification flags, follower thresholds), and near-duplicate removal.

***4.3 Reasoning Agent for Tweet Selection***

- **Initial Logic**: Craft few-shot prompts to guide an LLM agent (GPT-4 Turbo) to categorize tweets by likely impact: short-term (e.g., breaking news), long-term (e.g., regulatory announcements), or negligible.
- **Chain-of-Thought (CoT)**: Leverage multi-step CoT prompting so the agent explains why each tweet is selected, ensuring reasoning transparency.
- **JSON Output**: Require the agent to output a structured JSON list: tweet text, user handle, timestamp, impact category, and rationale.

***4.4 Prompt Integration***

- **Combined Prompt**: For each forecast, assemble input text as:
    
    ```
    Historical BTC prices: [token sequence];
    Selected Tweets (impact and rationale): [JSON block];
    Supplementary Info: [e.g., trading volume, on-chain metrics];
    Predict next 7 days of BTC prices.
    
    ```
    
- **Supplementary Metadata**: Optionally include indicators such as Google Trends for “Bitcoin” or S&P 500 index values.

***4.5 LLM Fine-Tuning***

- **Model Selection**: Use a decoder-only LLM (e.g. LLaMa 2 7B) and fine-tune via LoRA on the prepared dataset of prompts and ground-truth next-token sequences.
- **Training Setup**: Allocate 80% of data for training, 10% for validation (used by evaluation agent), and 10% for testing. Use learning rate scheduling and early stopping based on validation loss.

***4.6 Evaluation Agent and Iterative Refinement***

- **Error Analysis**: After fine-tuning each iteration, apply the model on the validation set. Compute forecasting errors (RMSE, MAE, MAPE) and identify time windows with high discrepancies.
- **Missed Tweets Detection**: Input all tweets from the validation window to an LLM evaluation agent that flags potentially influential tweets overlooked in the initial selection.
- **Logic Update**: The agent suggests refinements to tweet-selection logic, which is then used in the next iteration’s reasoning agent.
- **Convergence**: Continue for up to 3 iterations or until validation performance plateaus.

**6. Expected Outcomes**

- Demonstrate that LLMs fine-tuned with filtered tweet context outperform numerical-only and sentiment-baseline models in Bitcoin price forecasting (targeting ≥5% relative RMSE improvement).
- Provide an open-source pipeline (reasoning and evaluation prompts, fine-tuning scripts) for future event-driven forecasting research.
- Quantify the contribution of social media signals and identify which tweet features (e.g., influencer posts, sentiment extremes) have the greatest impact.

**7. Resources and Requirements**

- **Computing**: Access to GPUs (e.g., A100) for LoRA fine-tuning.
- **APIs**: Twitter Academic API access, CoinGecko/Binance data.
- **Software**: Python, PyTorch, Hugging Face Transformers, Python packages for data processing (pandas, numpy).

---

## Datasets that we want to use:

Below are ten publicly available datasets—five tweet collections and five Bitcoin price sources—well suited for aligning Twitter events with Bitcoin market movements. The tweet datasets offer raw collections and sentiment annotations spanning millions of posts, while the price datasets provide high-resolution historical series (from minute-level to daily data) and APIs for real-time querying. Together, these resources enable event-driven forecasting experiments that integrate social signals with quantitative price patterns.

---

### Tweet Datasets

1. **Bitcoin Tweets**
    
    A Kaggle dataset collecting tweets containing “Bitcoin” across various time frames—ideal for keyword-based event retrieval and custom filtering pipelines.
    
2. **BTC Tweets Sentiment**
    
    Kaggle’s sentiment-annotated Twitter corpus focused on Bitcoin discussions, featuring polarity labels to train or evaluate sentiment filtering agents.
    
3. **Bitcoin Tweets (16M with Sentiment)**
    
    A massive 16 million-tweet corpus (2016–2019) scraped via Twint/Tweepy and tagged with sentiment scores, useful for historical impact analysis.
    
4. **English Tweets Mentioning Bitcoin (2021–2022)**
    
    Over 22 million English-language tweets mentioning “bitcoin” collected with snscrape from January 2021 through late 2022, enabling up-to-date social signal integration.
    
5. **Crypto-Related Tweets (30M, Oct 2020–Mar 2021)**
    
    A 30 million-tweet dataset covering broad cryptocurrency discourse—including Bitcoin—suitable for period-specific sentiment and volume studies.
    

---

### Bitcoin Price Datasets

1. **Bitcoin Historical Data (1 min intervals)**
    
    High-frequency price series at one-minute resolution from multiple exchanges, supporting fine-grained forecasting tasks.
    
2. **Binance BTCUSDT OHLCV**
    
    Daily, 4 h, 1 h, and 15 min OHLCV data for the BTC/USDT trading pair via Binance API—updated through 2024.
    
3. **CryptoDataDownload OHLCV**
    
    Free CSV downloads of daily, hourly, and minute-level Bitcoin OHLCV data across various exchanges—straightforward to integrate into analysis.
    
4. **CoinMarketCap API**
    
    A comprehensive API offering live and historical Bitcoin price, market cap, and volume data, ideal for real-time model inputs.
    
5. **CoinDesk API**
    
    Provides live and historical BTC pricing along with contextual news metadata—useful for building aligned event-price datasets.
    

---

### Updated Team Responsibilities:

Based on the team roles outlined in your new request, the division of responsibilities includes the following:

1. **Data Collection**: Babak will handle web crawling for tweet data and dataset aggregation due to his experience.
2. **Model Training**: Taha will focus on training and fine-tuning the models due to his extensive experience with neural networks.
3. **Project Management and Evaluation**: Mehdi will manage the overall project and perform evaluations, leveraging his creativity and leadership skills.

This collaborative approach will ensure that each member can leverage their strengths, ensuring a comprehensive and effective implementation.

![Screenshot 2025-04-24 at 8.38.12 PM.png](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/Screenshot_2025-04-24_at_8.38.12_PM.png)

---

Below are ten publicly available datasets—five tweet collections and five Bitcoin price sources—well suited for aligning Twitter events with Bitcoin market movements. The tweet datasets offer raw collections and sentiment annotations spanning millions of posts, while the price datasets provide high-resolution historical series (from minute-level to daily data) and APIs for real-time querying. Together, these resources enable event-driven forecasting experiments that integrate social signals with quantitative price patterns.

---

## Tweet Datasets

1. **Bitcoin Tweets**
    
    A Kaggle dataset collecting tweets containing “Bitcoin” across various time frames—ideal for keyword-based event retrieval and custom filtering pipelines. citeturn0search0
    
2. **BTC Tweets Sentiment**
    
    Kaggle’s sentiment-annotated Twitter corpus focused on Bitcoin discussions, featuring polarity labels to train or evaluate sentiment filtering agents. citeturn0search1
    
3. **Bitcoin Tweets (16M with Sentiment)**
    
    A massive 16 million-tweet corpus (2016–2019) scraped via Twint/Tweepy and tagged with sentiment scores, useful for historical impact analysis. citeturn0search4
    
4. **English Tweets Mentioning Bitcoin (2021v**
    
    Over 22 million English-language tweets mentioning “bitcoin” collected with snscrape from January 2021 through late 2022, enabling up-to-date social signal integration. citeturn0search14
    
5. **Crypto-Related Tweets (30M, Oct 2020–Mar 2021)**
    
    A 30 million-tweet dataset covering broad cryptocurrency discourse—including Bitcoin—suitable for period-specific sentiment and volume studies. citeturn0search7
    

---

## Bitcoin Price Datasets

1. **Bitcoin Historical Data (1 min intervals)**
    
    High-frequency price series at one-minute resolution from multiple exchanges, supporting fine-grained forecasting tasks. citeturn0search2
    
2. **Binance BTCUSDT OHLCV**
    
    Daily, 4 h, 1 h, and 15 min OHLCV data for the BTC/USDT trading pair via Binance API—updated through 2024. citeturn0search6
    
3. **CryptoDataDownload OHLCV**
    
    Free CSV downloads of daily, hourly, and minute-level Bitcoin OHLCV data across various exchanges—straightforward to integrate into analysis. citeturn0search25
    
4. **CoinMarketCap API**
    
    A comprehensive API offering live and historical Bitcoin price, market cap, and volume data, ideal for real-time model inputs. citeturn0search9
    
5. **CoinDesk API**
    
    Provides live and historical BTC pricing along with contextual news metadata—useful for building aligned event-price datasets. citeturn0search12
    

[https://developers.coindesk.com/](https://developers.coindesk.com/)

[https://coinmarketcap.com/api/?utm_source=chatgpt.com](https://coinmarketcap.com/api/?utm_source=chatgpt.com)

[https://www.cryptodatadownload.com/?utm_source=chatgpt.com](https://www.cryptodatadownload.com/?utm_source=chatgpt.com)

[https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024?utm_source=chatgpt.com](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024?utm_source=chatgpt.com)

[https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data?utm_source=chatgpt.com](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data?utm_source=chatgpt.com)

[https://paperswithcode.com/dataset/crypto-related-tweets-from-10-10-2020-to-3-3?utm_source=chatgpt.com](https://paperswithcode.com/dataset/crypto-related-tweets-from-10-10-2020-to-3-3?utm_source=chatgpt.com)

[https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022?utm_source=chatgpt.com](https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022?utm_source=chatgpt.com)

[https://www.kaggle.com/datasets/gauravduttakiit/bitcoin-tweets-16m-tweets-with-sentiment-tagged?utm_source=chatgpt.com](https://www.kaggle.com/datasets/gauravduttakiit/bitcoin-tweets-16m-tweets-with-sentiment-tagged?utm_source=chatgpt.com)

[https://www.kaggle.com/datasets/aisolutions353/btc-tweets-sentiment?utm_source=chatgpt.com](https://www.kaggle.com/datasets/aisolutions353/btc-tweets-sentiment?utm_source=chatgpt.com)

[https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets?utm_source=chatgpt.com](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets?utm_source=chatgpt.com)

**Tweet Datasets**

1. Bitcoin Tweets (Kaggle)
    
    [https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets) citeturn0search0
    
2. Bitcoin Tweets – 16 M Tweets with Sentiment (Kaggle)
    
    [https://www.kaggle.com/datasets/gauravduttakiit/bitcoin-tweets-16m-tweets-with-sentiment-tagged](https://www.kaggle.com/datasets/gauravduttakiit/bitcoin-tweets-16m-tweets-with-sentiment-tagged) citeturn0search1
    
3. BTC Tweets Sentiment (Kaggle)
    
    [https://www.kaggle.com/datasets/aisolutions353/btc-tweets-sentiment](https://www.kaggle.com/datasets/aisolutions353/btc-tweets-sentiment) citeturn0search4
    
4. Crypto-Related Tweets from 10.10.2020 to 3.3.2021 (30 M Tweets)
    
    [https://paperswithcode.com/dataset/crypto-related-tweets-from-10-10-2020-to-3-3](https://paperswithcode.com/dataset/crypto-related-tweets-from-10-10-2020-to-3-3) citeturn0search3
    
5. English Tweets Mentioning Bitcoin (2021–2022) (Kaggle)
    
    [https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022](https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022) citeturn0search9
    

**Bitcoin Price Data**

6. CryptoDataDownload – Binance OHLCV (Daily, Hourly, Minute)

[https://www.cryptodatadownload.com/data/binance/](https://www.cryptodatadownload.com/data/binance/) citeturn1search1

7. Binance Spot API (Kline/Candlestick data)

[https://binance-docs.github.io/apidocs/delivery_testnet/en/#kline-candlestick-bars](https://binance-docs.github.io/apidocs/delivery_testnet/en/#kline-candlestick-bars) citeturn2search2

8. CryptoDataDownload (Homepage – all exchanges & intervals)

[https://www.cryptodatadownload.com/](https://www.cryptodatadownload.com/) citeturn1search0

9. CoinMarketCap API Documentation v1

[https://coinmarketcap.com/api/documentation/v1/](https://coinmarketcap.com/api/documentation/v1/) citeturn3search0

10. CoinDesk Cryptocurrency Data API

[https://developers.coindesk.com/](https://developers.coindesk.com/) citeturn4search0

this are 2 main datasets that we have :

![Screenshot 2025-04-24 at 8.49.34 PM.png](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/Screenshot_2025-04-24_at_8.49.34_PM.png)

![Screenshot 2025-04-24 at 8.51.34 PM.png](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/Screenshot_2025-04-24_at_8.51.34_PM.png)

[https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets?utm_source=chatgpt.com](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets?utm_source=chatgpt.com)

![Screenshot 2025-04-24 at 8.50.07 PM.png](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/Screenshot_2025-04-24_at_8.50.07_PM.png)

![Screenshot 2025-04-24 at 8.50.49 PM.png](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/Screenshot_2025-04-24_at_8.50.49_PM.png)

[https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022?utm_source=chatgpt.com](https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022?utm_source=chatgpt.com)

![Screenshot 2025-04-24 at 8.49.49 PM.png](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/Screenshot_2025-04-24_at_8.49.49_PM.png)

![Screenshot 2025-04-24 at 8.51.11 PM.png](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/Screenshot_2025-04-24_at_8.51.11_PM.png)

[https://www.kaggle.com/datasets/gauravduttakiit/bitcoin-tweets-16m-tweets-with-sentiment-tagged?utm_source=chatgpt.com](https://www.kaggle.com/datasets/gauravduttakiit/bitcoin-tweets-16m-tweets-with-sentiment-tagged?utm_source=chatgpt.com)

[https://www.kaggle.com/datasets/behdadkarimi/persian-tweets-emotional-dataset](https://www.kaggle.com/datasets/behdadkarimi/persian-tweets-emotional-dataset)

![Screenshot 2025-05-28 at 1.55.33 PM.png](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/Screenshot_2025-05-28_at_1.55.33_PM.png)

## Major Public Crypto-News Datasets

### 1. Crypto News + (Kaggle)

A CSV of ≈4 MB containing structured news items from 2021-10-12 through 2023-12-19, with fields for **title**, **text**, **source**, **subject**, and **precomputed sentiment**. citeturn0search0

**Link:** [https://www.kaggle.com/datasets/oliviervha/crypto-news](https://www.kaggle.com/datasets/oliviervha/crypto-news)

### 2. 174 K CryptoPanic Articles (GitHub)

A dump of 174 000 news items from CryptoPanic.com covering 430 top coins, labeled by users as positive/negative/important and linked to specific symbols. Available in CSV and MySQL formats. citeturn0search1

**Link:** [https://github.com/soheilrahsaz/cryptoNewsDataset](https://github.com/soheilrahsaz/cryptoNewsDataset)

### 3. Crypto News Recent (Kaggle)

A 46 MB CSV (`crypto_news_api.csv`) of the freshest headlines fetched via a crypto-news API, updated regularly; ideal for real-time or rolling-window studies. citeturn0search15

**Link:** [https://www.kaggle.com/datasets/lukasschmidt/crypto-news-recent](https://www.kaggle.com/datasets/lukasschmidt/crypto-news-recent)

### 4. Crypto News Headlines & Market Prices by Date (Kaggle)

Pairs daily news headlines with cryptocurrency price data, facilitating direct correlation and sentiment analysis on a per-date basis. citeturn0search23

**Link:** [https://www.kaggle.com/datasets/aaroncbastian/crypto-news-headlines-and-market-prices-by-date](https://www.kaggle.com/datasets/aaroncbastian/crypto-news-headlines-and-market-prices-by-date)

### 5. CryptoNER – Crypto News + NER (Kaggle)

Contains 26 000+ headlines from various outlets, annotated with named-entity tags (coins, organizations, events), enabling fine-grained event extraction and categorization. citeturn0search22

**Link:** [https://www.kaggle.com/datasets/kaballa/cryptoner-ml-model](https://www.kaggle.com/datasets/kaballa/cryptoner-ml-model)

### 6. BitcoinNews Dataset (Kaggle)

A focused collection of Bitcoin-specific articles curated for sentiment analysis, useful for isolating BTC-related event signals. citeturn0search8

**Link:** [https://www.kaggle.com/datasets/ashirwadsangwan/bitcoinnews-dataset](https://www.kaggle.com/datasets/ashirwadsangwan/bitcoinnews-dataset)

### 7. Bitcoin Tweets (Kaggle)

Although not traditional “news,” this dataset archives tweets mentioning “Bitcoin”—a rapid signal channel that often precedes formal news reports. citeturn0search7

**Link:** [https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets)

### 8. edloginova/cryptodata (GitHub)

Covers **tweets/comments**, economic indicators, and Google-search trends for multiple cryptocurrencies from 2015-08-07 to 2019-04-07—ideal for multi-source event analysis. citeturn0search4

**Link:** [https://github.com/edloginova/cryptodata](https://github.com/edloginova/cryptodata)

### 9. CryptoPanic API Wrapper (GitHub)

A Python library that simplifies fetching live and historic crypto news from CryptoPanic, complete with rating metadata and coin tags, ready for time-series alignment. citeturn0search10

**Link:** [https://github.com/pAulseperformance/cryptopanic_API_Wrapper](https://github.com/pAulseperformance/cryptopanic_API_Wrapper)

### 10. Cryptocurrency Sentiment Analysis (rishikonapure/GitHub)

Real-time pipeline correlating tweet sentiment, retweets, likes, and follower counts with price movements across multiple coins—includes historic tweet dumps. citeturn0search9

**Link:** [https://github.com/rishikonapure/Cryptocurrency-Sentiment-Analysis](https://github.com/rishikonapure/Cryptocurrency-Sentiment-Analysis)

---

## Ad-Hoc News Collection Tools

### NewsAPI (Kaggle How-To)

A tutorial notebook demonstrating how to pull timestamped crypto-news via NewsAPI.org and assemble your own CSV dataset. citeturn0search6

**Link:** [https://www.kaggle.com/getting-started/462566](https://www.kaggle.com/getting-started/462566)

### CryptoPanic Scraper (GitHub)

An end-to-end scraper for CryptoPanic’s front end, enabling full-site archive dumps for custom time-range retrieval. citeturn0search5

**Link:** [https://github.com/pAulseperformance/cryptopanic_scraper](https://github.com/pAulseperformance/cryptopanic_scraper)

---

[From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/From%20News%20to%20Forecast%20Integrating%20Event%20Analysis%20i%201df9381f908c80ff8094e668b8613e5a.md)

[**Why GRPO is Important and How it Works**](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/Why%20GRPO%20is%20Important%20and%20How%20it%20Works%201df9381f908c80a7a60afa23e5e4550b.md)

[DeepSeekMath 7B Full Technical Report](BaseLine%20Proposal%201ff9381f908c8083ace1f52148568ca1/DeepSeekMath%207B%20Full%20Technical%20Report%201df9381f908c80dfa20cd8a7896b73c4.md)

Some references : 

[https://arxiv.org/abs/2504.09696](https://arxiv.org/abs/2504.09696)

[https://arxiv.org/abs/2502.18548](https://arxiv.org/abs/2502.18548)

[https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

[https://arxiv.org/pdf/2211.08469v1](https://arxiv.org/pdf/2211.08469v1)

[https://arxiv.org/pdf/2110.00061v3](https://arxiv.org/pdf/2110.00061v3)

[https://arxiv.org/pdf/2312.11043](https://arxiv.org/pdf/2312.11043)

[https://arxiv.org/pdf/2303.14884v1](https://arxiv.org/pdf/2303.14884v1)

[https://arxiv.org/pdf/2406.01326v2](https://arxiv.org/pdf/2406.01326v2)

**8. References**

- Wang, X., Feng, M., Qiu, J., Gu, J., Zhao, J. (2024). *From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection*. NeurIPS 2024.
- Gruver, M. et al. (2023). *Language Models as Probabilistic Forecasting Engines*. ACL.
- Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., Neubig, G. (2023). *Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing.*