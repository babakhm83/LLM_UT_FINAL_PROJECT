# Research Proposal :

# **Research Proposal: LLM-Based Dollar Price Forecasting Using Tweet Events in Iran**

## **1. Introduction**

Forecasting the USD/IRR exchange rate is a notoriously challenging task due to the inherent volatility of the Iranian economy, which is subject to complex geopolitical pressures, international sanctions, and opaque domestic economic policies.1 Traditional econometric models often struggle to capture the full spectrum of influences, particularly the rapid shifts in market sentiment driven by real-time events and public discourse. Social media platforms, especially Twitter (now X), have emerged as significant arenas for the dissemination of news, rumors, and opinions that can sway economic expectations and behaviors in Iran, a country with a unique information ecosystem where official narratives may differ from open market realities.3

This research proposes to develop and evaluate a novel framework for forecasting the USD/IRR exchange rate by leveraging Large Language Models (LLMs) to integrate and interpret event signals derived from Persian-language Tweets. The methodology will adapt the approach presented by Wang et al. (2024) 5, which integrates event analysis into LLM-based time series forecasting with a reflection mechanism. This adaptation will specifically focus on the Iranian context, addressing the unique challenges of Persian language processing, data acquisition in a sanctioned environment, and the interpretation of socio-political and economic narratives prevalent on Iranian Twitter.

The primary research questions are:

1. Can an LLM-based forecasting model, adapted to the Iranian context and incorporating Persian Tweet events, achieve superior accuracy in predicting the USD/IRR exchange rate compared to traditional time series models and simpler LLM baselines?
2. How can the reasoning and evaluation agent framework proposed by Wang et al. (2024) 5 be effectively tailored to filter, interpret, and learn from Persian Tweet events relevant to the Iranian Rial's fluctuations?
3. What types of Tweet-derived events and narratives demonstrate the most significant impact on USD/IRR movements, and how does the model's iterative refinement process enhance its ability to identify and utilize these signals?

The objectives of this research are to:

1. Develop a robust data acquisition pipeline for Iranian USD/IRR rates, Persian Tweets, and supplementary economic/political data.
2. Adapt and fine-tune a Persian-proficient LLM for USD/IRR forecasting, integrating processed Tweet events through advanced prompt engineering.
3. Design and implement specialized Reasoning and Evaluation Agents, based on LLMs, to dynamically select relevant Tweet events and iteratively refine the selection logic based on forecasting performance.
4. Rigorously evaluate the proposed framework against a comprehensive set of baseline models using appropriate metrics and conduct ablation studies to assess the contribution of each component.
5. Perform qualitative analysis to understand the model's behavior and the nature of influential Tweet events in the Iranian context.

This proposal outlines the literature review, proposed methodology including the adaptation of the Wang et al. (2024) 5 framework, data acquisition and processing strategies tailored for Iran, the LLM forecasting module, the design of the analytical agents, and a comprehensive evaluation plan. It also discusses expected outcomes, potential challenges, and a projected timeline.

## **2. Literature Review**

### **2.1. Large Language Models in Time Series Forecasting**

The application of Large Language Models (LLMs) to time series forecasting represents a paradigm shift from traditional statistical methods (like ARIMA, Exponential Smoothing) and earlier machine learning approaches (like SVR, Random Forests).5 While traditional models excel when time series distributions remain consistent, they often falter in capturing sudden disruptions or the impact of exogenous events.5 Deep learning models, including LSTMs and GRUs, offered improvements in handling non-linearities but still primarily relied on numerical historical data.

Recent advancements have demonstrated that LLMs, pre-trained on vast textual corpora, possess emergent capabilities in understanding patterns and can be adapted for time series forecasting.5 Works like Time-LLM 5 and the findings by Gruver et al. (2024) 5 (cited as in 5) show that LLMs can perform zero-shot or few-shot time series forecasting by reframing the task as a sequence-to-sequence or next-token prediction problem. The core idea in Wang et al. (2024) 5 is to treat numerical time series as sequences of digit tokens, allowing the LLM to predict future digits conditioned on past digits and, crucially, on textual information provided in prompts. This ability to seamlessly integrate textual context with numerical data is a key advantage for event-driven forecasting.

### **2.2. Event-Driven Forecasting and News/Social Media Impact**

Financial markets are significantly influenced by external events, including economic announcements, political developments, and unforeseen shocks.6 Event-driven forecasting aims to incorporate the impact of such occurrences into predictive models. News articles have long been recognized as a rich source of event information, providing context that numerical data alone cannot capture.5 Studies have shown that financial news sentiment can influence stock market behavior and volatility.9

With the proliferation of social media, platforms like Twitter have become crucial channels for real-time information dissemination, sentiment expression, and narrative formation.12 Research has explored the link between social media sentiment (often from Twitter) and financial market movements, including stock prices 14 and, to a lesser extent, exchange rates.13 For instance, analysis of Twitter data has been used to gauge investor sentiment and predict stock returns.15 The brevity and informality of tweets, along with the sheer volume of data, present unique NLP challenges.16 Frameworks for assessing social media impact often involve sentiment analysis, volume analysis, and network analysis.12 The methodology of event study analysis is also relevant, quantifying how specific events, including those highlighted in news or social media, impact security prices by comparing expected returns to abnormal returns.

### **2.3. Natural Language Processing for Persian Language**

The Persian language, while spoken by millions, is considered low-resource in the context of NLP model development compared to languages like English.18 This is due to relatively fewer large-scale, high-quality annotated datasets and pre-trained models. Challenges in Persian NLP include morphological richness, ambiguity, variations in script and spacing, and the prevalence of colloquialisms and cultural nuances in informal text like tweets.19

Despite these challenges, significant progress has been made. Several Persian-specific language models have been developed, such as:

- **ParsBERT**: A BERT-based model pre-trained on a large Persian corpus, showing strong performance on various NLU tasks like sentiment analysis and NER.23 It has been fine-tuned for domain-specific tasks, including financial sentiment analysis in the Iranian banking sector using Twitter data.26
- **PersianMind**: A cross-lingual Persian-English LLM that has reported state-of-the-art results on some Persian benchmarks and comparable performance to GPT-3.5-turbo on Persian reading comprehension.30
- **Fibonacci LLM**: An LLM specifically designed for the Persian language, aiming for cultural alignment and proficiency in tasks like content generation, translation, and Q&A.33

Multilingual LLMs like LLaMa, Gemma, and Mistral also offer potential, as they can transfer knowledge from high-resource languages, though their performance on nuanced Persian tasks without specific fine-tuning might be limited.18 Benchmarks for evaluating Persian LLMs are also emerging, such as ParsiNLU 37, Khayyam Challenge (PersianMMLU) 18, FaMTEB for text embeddings 38, and culturally nuanced datasets like PerCul 20 and SentiPersian.22 Fine-tuning techniques like LoRA are particularly important for adapting these models to specific tasks in low-resource settings.31

### **2.4. The Iranian Economic and Information Context**

Iran's economy operates under unique and severe pressures, primarily stemming from decades of international sanctions, particularly those targeting its oil exports and financial sector.1 This has led to high inflation, currency devaluation, and limited access to global financial markets.1 A critical feature of the Iranian currency market is the existence of multiple exchange rates: an official rate set by the Central Bank of Iran (CBI) 43, and one or more open market (or "free market") rates, which are significantly higher and more reflective of actual supply and demand dynamics for most individuals and businesses.44 The USD/IRR rate on the open market is highly sensitive to political news, sanction developments, oil price fluctuations, and public sentiment.3

Data transparency from official Iranian sources can be a challenge. While bodies like the Statistical Center of Iran (SCI) 49 and the CBI 43 publish economic data, there can be delays, and the methodologies or accessibility might differ from international standards. Unofficial sources like Bonbast.com 46 and Navasan.net 45 have become de facto references for open market exchange rates.

Social media, particularly Twitter, plays a vital role in Iran's information landscape despite state restrictions and filtering.52 It serves as a platform for news dissemination (by official, semi-official, and independent outlets), economic commentary by analysts and the public, and the spread of rumors that can quickly impact market sentiment.53 Hashtags such as #دلار (dollar), #نرخ_ارز (exchange_rate), #اقتصاد_ایران (Iran_economy), and #تحریم (sanctions) are commonly used to discuss economic conditions.3 This makes Twitter a potentially rich source of real-time event data for forecasting the open market USD/IRR rate. The dichotomy between official data and narratives versus the realities reflected in open market rates and social media discourse implies that a model attempting to forecast the Rial must be adept at navigating this complex information environment. Signals from Twitter reflecting open market sentiment or early indicators of inflation (e.g., discussions about rising prices using specific hashtags) might precede official data releases or capture dynamics not fully represented in them. Therefore, the ability of the proposed framework to identify and weigh these social media signals appropriately will be crucial.

## **3. Proposed Methodology (Adapted from Wang et al., 2024** 5**)**

### **3.1. Overall Framework Adaptation for Iran**

The methodology proposed in this research adapts the comprehensive framework introduced by Wang et al. (2024) 5 for integrating event analysis in LLM-based time series forecasting. The original framework 5 comprises five key stages, which will be tailored for the Iranian context:

- **(A) Information Retrieval:** This stage will focus on collecting historical USD/IRR exchange rate data (primarily open market rates), a corpus of Persian-language Tweets related to the Iranian economy and currency, and supplementary Iranian economic and political data.
- **(B) Reasoning Agent:** An LLM-based agent will be developed and prompted to filter the collected Tweets, identifying those that are most relevant and potentially impactful on the USD/IRR rate. This agent will be specifically attuned to the nuances of Persian language and the Iranian socio-economic context.
- **(C & D) Prompt Integration & LLM Finetuning:** The selected Tweet events, along with historical USD/IRR data and supplementary information, will be formatted into textual prompts. A Persian-proficient LLM will then be fine-tuned using these integrated data prompts to forecast future USD/IRR values.
- **(E) Evaluation Agent:** Another LLM-based agent will be tasked with reviewing the forecasting model's errors by comparing predictions against actual USD/IRR movements. This agent will analyze all available Tweet data (both selected and unselected by the Reasoning Agent) from the corresponding period to identify any "missed news" or overlooked signals that might explain the discrepancies. Based on this analysis, it will propose refinements to the Reasoning Agent's logic.

A core strength of the Wang et al. (2024) 5 framework is its iterative nature.5 This iterative loop, where the Evaluation Agent's feedback informs the Reasoning Agent's logic for subsequent rounds of data preparation and model fine-tuning, is particularly vital for the dynamic and often opaque Iranian information environment. The "Reflection" mechanism, embodied by the Evaluation Agent, allows the system to learn from its errors in interpreting subtle or evolving narratives within Persian Tweets. This is especially important given the potential limitations in the availability of large, pre-annotated datasets for initially training the agents' complex reasoning logic for the Iranian context. For example, an initial rule for the Reasoning Agent might be that "tweets about new international banking sanctions are negative for IRR." However, if the Evaluation Agent observes that after a specific set of sanctions, the IRR unexpectedly stabilized or even strengthened in the short term, while tweets simultaneously focused on domestic resilience narratives, it would flag this discrepancy. The Reasoning Agent would then be prompted to refine its logic to consider such counter-narratives, thereby improving its ability to capture the multifaceted responses to events in Iran.

### **3.2. Data Acquisition and Pre-processing for Iran**

### **3.2.1. USD/IRR Time Series Data**

Acquiring a consistent and reliable time series for the USD/IRR exchange rate is foundational. Given the significant divergence between official and open market rates in Iran, and the likelihood that public sentiment reflected in Tweets aligns more closely with open market conditions, this research will prioritize open market rates.

- **Sources:** Daily open market USD/IRR rates will be primarily sourced from publicly accessible platforms such as Bonbast.com (which has archived data 46), Navasan.net 45, and Alanchand.com.44 These sources are widely referenced for tracking the Rial's unofficial value. Official rates from the Central Bank of Iran (CBI) 43 will also be collected for comparative analysis and to understand the full spectrum of currency valuation, though their direct use in forecasting models driven by public sentiment may be limited. XE.com 57 will serve as a cross-referencing source.
- **Frequency:** Data will be collected at a daily frequency. While intra-day data would be ideal for capturing rapid event impacts, consistent and reliable intra-day open market rates for Iran are generally not available over long historical periods.
- **Challenges:** Potential challenges include discrepancies between different open market sources, missing data points for certain periods, and ensuring data consistency (e.g., Toman vs. Rial denomination) over the selected historical timeframe (target: 2019-2025).
- **Pre-processing:** Standard time series pre-processing techniques will be applied. Missing values will be handled using appropriate methods such as linear interpolation or forward/backward fill, with careful consideration of the data's nature. While the core LLM approach in Wang et al. (2024) 5 aims to work with original data scales, normalization (e.g., to a range) might be necessary for some baseline models used for comparison.

### **3.2.2. Iranian Tweet Event Data**

The collection of relevant Persian Tweets is central to this research.

- **Collection Strategy:**
- *Tools:* Access to the Twitter API (X API) will be sought, adhering to all terms of service. If direct API access is limited or cost-prohibitive for academic research, alternative open-source scraping tools like snscrape (which has been used for collecting tweets in other languages like Arabic and can be adapted for Persian 59) will be considered. Communalytic is another option but requires a paid API plan.60
- *Keywords & Hashtags:* A comprehensive and evolving list of Persian keywords and hashtags will be curated. This list will target discussions related to the Iranian economy, USD/IRR exchange rate, sanctions, inflation, economic policies, specific political figures known for economic commentary, major Iranian banks, and terms referring to market conditions. Initial examples include: #دلار (dollar), #نرخ_ارز (exchange_rate), #اقتصاد_ایران (Iran_economy), #تحریم (sanctions), #تورم (inflation), #گرانی (high_prices), #بازار_آزاد (open_market), as well as names of key economic institutions and figures.3 This list will be refined throughout the research.
- *Influential Accounts:* Tweets from influential Iranian economists, financial analysts, prominent news agencies with economic desks 54, and potentially key political figures or entities known to use Twitter for economic announcements or commentary will be specifically tracked.53
- *Time Period:* The target period for Tweet collection will be 2019-2025. This timeframe encompasses significant economic events, including the "maximum pressure" sanctions campaign, the COVID-19 pandemic's economic impact, various phases of JCPOA negotiations, and internal political transitions, all of which likely influenced the USD/IRR rate and were discussed on Twitter.
- **Ethical Considerations:** Data collection will strictly adhere to ethical guidelines. All user data will be anonymized to the extent possible during analysis. Only publicly available tweets will be collected. The research will be mindful of Twitter's Terms of Service and the sensitive nature of opinions expressed, particularly within the Iranian context.
- **Volume & Filtering:** A large volume of tweets is anticipated. Initial filtering will be performed based on the curated keywords and hashtags. The Reasoning Agent will subsequently perform more sophisticated semantic filtering to identify truly impactful event-related tweets.

### **3.2.3. Supplementary Iranian Economic and Political Data**

To provide broader context to the LLM, various supplementary data points will be collected and textualized.

- **Sources:**
- *Macroeconomic Indicators:* Inflation rates, Gross Domestic Product (GDP) figures, and unemployment statistics will be sourced from the Statistical Center of Iran (SCI) 49 and the Central Bank of Iran (CBI).43 These will be cross-referenced with data from international bodies like the World Bank and IMF where available and applicable.1
- *Oil Sector Data:* Data on Iranian oil prices (e.g., Iran Heavy) and production/export volumes will be gathered from the Iranian Oil Ministry 47, OPEC publications, and international commodity market trackers (e.g., Trading Economics 63).
- *Sanction Announcements:* Official announcements regarding the imposition, easing, or waiver of international sanctions pertaining to Iran will be collected from sources like the U.S. Department of the Treasury's Office of Foreign Assets Control (OFAC), European Union official journals, and United Nations Security Council resolutions. News reports from reputable international and Iranian news agencies covering these announcements will also be monitored.
- *Major Policy Shifts:* Significant economic or monetary policy announcements from Iranian governmental bodies, such as the Ministry of Economic Affairs and Finance 64, SCI 49, or the CBI, will be tracked.
- *Geopolitical Events:* A timeline of major regional and international geopolitical events that have clear and direct implications for Iran's economy (e.g., escalations in regional conflicts, breakthroughs or breakdowns in JCPOA negotiations) will be compiled from news archives.
- **Formatting:** All supplementary information will be converted into concise textual descriptions suitable for input into the LLM prompts, as demonstrated in Wang et al. (2024).5 For example, "On [date], the Statistical Center of Iran reported a monthly inflation rate of [X]% for [month, year]." or "On [date], new U.S. sanctions targeting Iran's petrochemical sector were announced by OFAC."

### **3.2.4. Data Pre-processing for Persian Text**

Processing Persian text from Tweets requires careful attention to its linguistic characteristics.

- **Normalization:** Persian text will be normalized to ensure consistency. This includes standardizing variations of characters (e.g., 'ی' vs. 'ي', 'ک' vs. 'ك'), and removing diacritics (e.g., Fatha, Kasra, Damma) unless they are determined to be crucial for disambiguating meaning in specific financial or economic contexts.
- **Tokenization:** A Persian-aware tokenizer will be employed. Standard tokenizers designed for English or other languages often perform poorly on Persian due to its morphological complexity, compound words, and common variations in word spacing (e.g., "می‌رود" vs. "می رود").19
- **Noise Reduction:** Tweets often contain noise that needs to be filtered. This includes removing URLs, excessive or irrelevant special characters, and standardizing common Twitter artifacts (e.g., @mentions, #hashtags if not part of the core message being analyzed). Strategies for handling retweets (e.g., considering only original tweets or weighting retweets differently) will be explored. Identifying and potentially filtering bot-generated content will be a significant challenge, requiring heuristic rules or specialized detection methods if feasible.
- **Language Identification:** While the focus is on Persian Tweets, some relevant information might be tweeted in English (e.g., by international news outlets or analysts). A language identification step (using libraries like langdetect 66 or langid 66) will be used to filter out non-Persian tweets, unless they are direct quotes of Persian sources or highly relevant international news directly impacting Iran.

A summary of data sources is provided in Table 1.

**Table 1: Data Sources for Iranian USD/IRR Forecasting**

| **Data Type** | **Specific Source(s)** | **Collection Method/API** | **Frequency** | **Key Variables** | **Time Period** | **Iranian Context Challenges & Mitigation** |
| --- | --- | --- | --- | --- | --- | --- |
| USD/IRR (Open Market) | Bonbast.com, Navasan.net, Alanchand.com | Web Scraping, Manual Collection (from archives) | Daily | Closing Rate (USD to IRR/Toman) | 2019-2025 | Data gaps, unofficial nature, source discrepancies. Mitigation: Cross-verification, imputation, focus on most consistent source. |
| USD/IRR (Official) | Central Bank of Iran (CBI) | Official Website/API (if available) | Daily | Official Exchange Rate | 2019-2025 | May not reflect market reality, limited public trading. Mitigation: Use for comparative analysis, acknowledge limitations. |
| Tweets | Twitter (X) Platform | Twitter API / snscrape / Academic Datasets | Real-time | Tweet text, timestamp, user info (anonymized), engagement metrics | 2019-2025 | API restrictions, censorship, bot activity, slang/coded language. Mitigation: Robust scraping, advanced filtering, Persian NLP expertise. |
| Inflation | Statistical Center of Iran (SCI), CBI | Official Websites | Monthly | Consumer Price Index (CPI), Inflation Rate (%) | 2019-2025 | Potential delays in reporting, methodological questions. Mitigation: Use latest available, cross-reference if possible. |
| GDP | Statistical Center of Iran (SCI), CBI | Official Websites | Quarterly/Annually | Gross Domestic Product, Growth Rate (%) | 2019-2025 | Infrequent updates. Mitigation: Use for broader context, textualize effectively. |
| Unemployment | Statistical Center of Iran (SCI) | Official Websites | Quarterly/Annually | Unemployment Rate (%) | 2019-2025 | Infrequent updates. Mitigation: Use for broader context. |
| Oil Prices/Production | Iran Oil Ministry, OPEC, International Commodity Sites | Official Reports, News, Data Services | Daily/Monthly | Price (Iran Heavy/Brent), Production Volume, Export Volume | 2019-2025 | Data on actual Iranian exports can be opaque due to sanctions. Mitigation: Use best available estimates, news reports. |
| Sanction Announcements | US Treasury (OFAC), EU, UN, News Agencies | Official Websites, News Archives | Event-driven | Details of sanctions (sector, entities), effective dates | 2019-2025 | Complex legal language. Mitigation: Summarize key impacts clearly. |
| Major Policy Shifts | Iranian Government Economic Bodies, News Agencies | Official Announcements, News Archives | Event-driven | Description of policy change (monetary, fiscal, trade) | 2019-2025 | Discerning actual impact vs. stated intent. Mitigation: Focus on concrete announced changes. |
| Key Geopolitical Events | Reputable International & Iranian News Archives | News Aggregation | Event-driven | Description of event, date, perceived relevance to Iran's economy | 2019-2025 | Subjectivity in relevance assessment. Mitigation: Focus on events with widely acknowledged economic implications for Iran. |

This structured approach to data acquisition, acknowledging the specific challenges within the Iranian context, is essential for building a credible and robust forecasting model. The transparency in sourcing and mitigating potential data issues will be a cornerstone of this research.

### **3.3. LLM-Based Forecasting Module for USD/IRR**

### **3.3.1. Selection of Base LLM**

The choice of the base LLM is critical for the success of this project, requiring a model with strong Persian language capabilities and amenability to fine-tuning.

- **Criteria:**
1. **Persian Language Proficiency:** The model must demonstrate a high level of understanding of Persian, including its nuances, syntax, and common vocabulary used in economic and social discourse. This is more critical than for the agent LLMs if tweets are processed directly in Persian by the forecasting LLM.
2. **Generation Quality:** For forecasting, the LLM will generate sequences of digits representing future rates; however, its underlying language generation capabilities reflect its pattern recognition strength.
3. **Fine-tuning Stability:** The model should be known to fine-tune well using techniques like LoRA, without catastrophic forgetting or instability.
4. **Performance on Persian Benchmarks:** Demonstrated performance on relevant Persian NLP benchmarks (e.g., ParsiNLU 37, Khayyam Challenge 18, FaMTEB 38) is a strong indicator.
5. **Model Size vs. Resources:** The parameter size of the LLM must be manageable with available computational resources (GPUs) for fine-tuning.
6. **Availability:** Preference for open-source models with permissive licenses for academic research.
- **Candidate Models:**
- **Persian-Specific Models:**
- *ParsBERT* 23: While primarily an encoder model (BERT architecture), its strong Persian understanding could be leveraged. For sequence generation, it might need to be part of an encoder-decoder setup or its embeddings used as features for another generative model. Fine-tuning ParsBERT for tasks like sentiment analysis has been explored.26
- *PersianMind* 30: A cross-lingual Persian-English LLM that claims state-of-the-art results on some Persian tasks. Its suitability for fine-tuning on this specific forecasting task needs investigation.
- *Fibonacci LLM* 33: Developed specifically for Persian with a focus on cultural alignment. Its architecture (e.g., based on deepseek-ai/Janus-Pro-7B 33) and fine-tunability with LoRA would need to be assessed.
- **Multilingual Models:**
- *LLaMa 2/3 Family*: LLaMa 2 was used in the original Wang et al. (2024) study.5 These models have strong multilingual capabilities and extensive community support for fine-tuning with LoRA.31
- *Gemma / Mistral Families*: Other powerful open-source multilingual models that have shown good performance and are amenable to fine-tuning.18
- **Evaluation of Candidates:** Before committing to a single model, preliminary experiments will be conducted. This will involve testing the zero-shot or few-shot capabilities of candidate LLMs on a small, curated set of Persian financial tweets and news snippets to assess their basic comprehension, ability to identify relevant economic concepts, and the coherence of their generated summaries or rationales in Persian (or English translation if the model is primarily English-focused). The model that demonstrates the best balance of Persian understanding, fine-tuning feasibility, and resource requirements will be selected.

### **3.3.2. Prompt Engineering for Iranian Data Integration**

The prompt serves as the primary interface for feeding integrated information to the LLM. The structure proposed by Wang et al. (2024) 5 will be adapted:

- **Instruction Component:** This will clearly state the task.
- Example: "Based on the provided historical USD/IRR open market rates, recent impactful Persian Tweet events, and supplementary economic data for Iran, predict the daily USD/IRR open market rate for the next 7 days."
- **Input Component:** This section will contain all the contextual information.
- *Historical USD/IRR data:* "The historical open market USD/IRR rates (in Iranian Toman per USD) are: 2024-03-01:58500, 2024-03-02:58700,..., [latest_date]:[latest_rate]."
- *Forecasting task details:* "Region: Iran. Currency Pair: USD/IRR (Open Market). Prediction Horizon: 7 days. Data Frequency: Daily. Current Date: [prediction_date]."
- *Supplementary Information (textualized):* This will include concise statements about the current economic environment.
- Example: "Sanction Status: Comprehensive US sanctions on banking and oil sectors remain active. Latest Inflation Report (Statistical Center of Iran, [report_date]): Year-on-year inflation at 45%. Recent Central Bank of Iran Policy ([policy_date]): Announced measures to increase USD supply to exchange bureaus. Global Oil Price (Brent Crude, [current_date]): $85 per barrel."
- *Selected Tweet Events (output from Reasoning Agent):* This will be a list of the most salient Tweet events. Each event will include:
- Example: "{'tweet_source': '@EconomistFarsi', 'tweet_publish_time': '2024-03-15 10:30:00', 'tweet_summary_persian': 'زمزمه‌های افزایش نرخ بهره توسط بانک مرکزی قوت گرفته است.', 'tweet_summary_english': 'Rumors of an interest rate hike by the central bank are strengthening.', 'predicted_impact_on_rial': 'Positive_for_Rial', 'impact_horizon': 'Short-term', 'rationale_for_impact': 'An interest rate hike would typically strengthen the currency by attracting capital and curbing inflation.'}" (Multiple such JSON objects for different tweets).
- **Output Component (Expected Format for LLM to Generate):**
- Example: "The predicted USD/IRR open market rates for the next 7 days are: [date+1]:[predicted_rate1], [date+2]:[predicted_rate2],..., [date+7]:[predicted_rate7]."
- **Handling Persian in Prompts:**
- If a Persian-native LLM (like Fibonacci or a well-fine-tuned PersianMind) is chosen, Persian tweet summaries can be directly included in the prompt.
- If a multilingual LLM with primary English training (like LLaMa) is used, Persian tweet summaries will likely need to be machine-translated into English before being included in the prompt. The quality of this translation will be crucial. Alternatively, the LLM must demonstrate robust mixed-language processing capabilities if Persian snippets are directly embedded. This decision will impact overall complexity and potential information loss during translation.

### **3.3.3. Fine-tuning Strategy with LoRA**

To adapt the selected base LLM for the specific task of USD/IRR forecasting using Iranian tweet events, a parameter-efficient fine-tuning strategy is essential.

- **LoRA (Low-Rank Adaptation):** This technique will be adopted as it significantly reduces the number of trainable parameters, making fine-tuning large models computationally feasible while preserving much of their pre-trained knowledge.5 LoRA involves injecting trainable low-rank matrices into the layers of the transformer model.
- **Parameters for LoRA:**
- *Rank (r):* Initial experiments will explore ranks such as 8 or 16, as suggested in.5
- *Alpha (α):* A scaling factor, often set to be similar to or double the rank (e.g., 16 or 32). 5 used an alpha of 16.
- *Learning Rate:* A small learning rate, such as 1×10−4 (0.0001) as used in 5, will be the starting point, subject to tuning.
- *Target Modules:* LoRA will typically be applied to attention mechanism matrices (query, key, value, output) within the transformer layers.
- **Training Data Construction:** The training dataset will consist of input-output pairs.
- *Input:* The structured prompt as described in section 3.3.2, containing historical USD/IRR data, textualized supplementary information, and selected Persian Tweet events with their rationales.
- *Output:* The actual sequence of USD/IRR rates for the N-day horizon following the input data's end date.
- **Loss Function:** The standard causal language modeling loss (e.g., cross-entropy) will be used. The LLM will be trained to predict the next token in the output sequence (which, for the rate prediction part, will be the digits of the future exchange rates).
- **Numerical Tokenization:** Exchange rates will be represented as sequences of individual digit tokens. For example, a rate of "58500" would be tokenized as "5", "8", "5", "0", "0". Following 5, numerical values will maintain a consistent number of significant figures (e.g., three, or a fixed decimal representation if applicable) to ensure uniformity and prevent an explosion in the token vocabulary for numbers.

A comparative overview of candidate LLMs is presented in Table 2.

**Table 2: Candidate LLMs for Forecasting and Agent Design**

| **Model Name** | **Base Architecture** | **Parameters (Approx.)** | **Primary Language(s)** | **Specific Persian Training Data/Corpora Known** | **Performance on Persian Benchmarks (e.g., Khayyam, FaMTEB, ParsiNLU)** | **Fine-tuning Feasibility (LoRA support, resource needs)** | **Availability (Open Source/API)** | **Potential Pros for Iranian Tweet Analysis** | **Potential Cons for Iranian Tweet Analysis** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLaMa 2 / LLaMa 3 | Transformer (Decoder-only) | 7B, 13B, 70B+ | Multilingual | General multilingual corpora likely include some Persian. | Varies; generally strong multilingual baseline. Specific Persian SOTA unlikely without fine-tuning. | Excellent LoRA support, extensive community resources. High resource needs for larger versions. | Open Source (LLaMa 2, some LLaMa 3 variants) | Proven fine-tuning capability.5 Large model capacity for complex patterns. | May require significant Persian-specific fine-tuning to capture nuances. Potential translation needs for prompts. |
| Gemma | Transformer (Decoder-only) | 2B, 7B | Multilingual | Trained on diverse multilingual text. | Good multilingual performance. | Good LoRA support. Moderate resource needs. | Open Source | Efficient for its size. Good starting point for multilingual tasks. | May require more Persian-specific fine-tuning than dedicated Persian models. |
| Mistral | Transformer (Decoder-only) | 7B, Mixtral 8x7B | Multilingual | Optimized for English, strong multilingual. | Strong multilingual performance, competitive with larger models. | Good LoRA support. Moderate to high resource needs. | Open Source (some variants) | High efficiency and performance for size. | Similar to Gemma, needs Persian-specific adaptation. |
| ParsBERT | BERT (Encoder-only) | ~110M | Persian | Large Persian corpus (news, books, web). | Strong on Persian NLU tasks (sentiment, NER).23 | Primarily for NLU, not generation. LoRA applicable to BERT. Low resource needs. | Open Source | Excellent Persian understanding. Good for initial tweet analysis if used in a pipeline. | Not inherently generative. Would require an encoder-decoder setup or different role in the pipeline. |
| PersianMind | Transformer | 7B / 13B (reported) | Persian, English | Details on specific corpora not fully public, but claims extensive Persian data.30 | Claims SOTA on Belebele (Persian), ParsiNLU QA.30 | Stated as fine-tunable. LoRA compatibility needs verification. Moderate to high resource needs. | Potentially Open (CC BY-NC-SA 4.0) 30 | Designed for Persian, cross-lingual capabilities. | Newer model, less community fine-tuning support compared to LLaMa. License restrictions for commercial use. |
| Fibonacci LLM | Transformer (e.g., Janus-Pro-7B based) | 1.7B, 7B | Persian, English | Claims diverse, localized Persian datasets.33 | Claims good accuracy and character-level performance for Persian.33 | Built on fine-tunable architectures. LoRA compatibility likely. Moderate resource needs. | Access via platform; some models on Hugging Face 33 | Specifically designed for Persian cultural and linguistic context. | Newer, less independent verification on broad benchmarks. Potential platform dependency. |

The selection process will prioritize models that offer a balance between strong out-of-the-box Persian understanding (or proven adaptability) and robust, well-documented fine-tuning capabilities using LoRA. The insights from 5 regarding LLaMa 2's successful fine-tuning provide a strong argument for considering similar multilingual architectures, while Persian-specific models offer the allure of deeper linguistic grounding if they prove equally adaptable.

### **3.4. Analytical Agent for Iranian Tweet Event Reasoning (Reasoning Agent)**

### **3.4.1. Reasoning Agent Design and LLM Choice**

The Reasoning Agent is a critical component responsible for sifting through the vast amount of pre-filtered Persian Tweets and identifying those that contain information pertinent to potential shifts in the USD/IRR exchange rate.

- **LLM Choice:** Given the complexity of understanding nuanced Persian text, inferring intent, and relating it to economic drivers, a highly capable LLM is required. GPT-4 Turbo, as used in Wang et al. (2024) 5, is a strong candidate due to its advanced reasoning and language understanding capabilities. If an open-source Persian LLM (e.g., advanced versions of PersianMind or Fibonacci) demonstrates comparable or superior performance on complex Persian reasoning tasks and cultural understanding, it would be a preferred alternative, potentially offering better contextual nuance and reducing reliance on proprietary models.
- **Function:** The primary function of the Reasoning Agent is to:
1. Receive a batch of Persian Tweets (pre-filtered by keywords, hashtags, and potentially influential accounts).
2. Analyze each tweet to determine its relevance to USD/IRR fluctuations.
3. For relevant tweets, categorize the potential impact (e.g., positive/negative for the Rial, short-term/long-term effect).
4. Generate a concise rationale explaining why the tweet is considered impactful.
5. Output this information in a structured format (e.g., JSON) for integration into the forecasting LLM's prompt.

### **3.4.2. Initial Reasoning Logic for Iranian Tweet Selection**

The agent's decision-making will be guided by an initial reasoning logic, which can be seeded with expert domain knowledge about the Iranian economy and subsequently refined through the iterative evaluation process. This logic adapts the principles from 5 (Figure 3 Part I, Appendix A.6.1).

- **Initial Categories of Impactful Events (derived from understanding of Iranian economic drivers):**
- *Official Announcements:* Tweets reporting or directly quoting official statements from the Central Bank of Iran, Ministry of Economy, President's office, or other key governmental bodies regarding currency policy, trade regulations, subsidy changes, or budget issues.
- *Sanctions News/Rumors:* Tweets discussing new international sanctions being imposed, existing sanctions being eased or waived, or significant developments in sanction enforcement. This includes both factual news and credible rumors from influential sources.
- *Oil Sector Developments:* Tweets related to major changes in Iran's oil export volumes, pricing, new oil contracts, or disruptions to oil infrastructure.
- *Public Sentiment & Market Behavior:* Tweets reflecting widespread public concern or shifts in behavior regarding inflation, commodity shortages, hoarding, or panic buying/selling of foreign currency. Trending hashtags related to economic hardship or market volatility are key indicators.
- *Geopolitical Events:* Tweets about significant geopolitical developments (e.g., regional conflicts, diplomatic breakthroughs or breakdowns, particularly concerning JCPOA) that have direct and foreseeable economic consequences for Iran.
- *Statements by Key International Actors:* Tweets reporting statements from influential international figures (e.g., US officials on sanctions policy) or organizations (e.g., IMF/World Bank on Iran's economy) that could impact investor confidence or international financial relations.
- **Impact Classification:**
- *Effect on Rial Value:* Positive_for_Rial (likely to strengthen the Rial or slow its depreciation), Negative_for_Rial (likely to weaken the Rial or accelerate its depreciation), Neutral/Uncertain.
- *Impact Horizon:* Short-Term (expected impact within days to a few weeks), Long-Term (expected impact over months or longer).

### **3.4.3. Adapting Prompting Method for Iranian Tweet Context**

To effectively guide the Reasoning Agent, a combination of advanced prompting techniques will be employed, tailored to the specifics of Persian tweets.

- **Few-Shot Prompting:** As suggested in 5 and supported by literature 68, the agent will be provided with a few high-quality examples of Persian tweets along with their desired analysis (impact category, rationale, sentiment). These examples will help the LLM understand the expected output format and the type of reasoning required. Examples will cover various scenarios, including subtle or sarcastically phrased tweets that are common in Iranian social media.
- **Chain-of-Thought (CoT) Prompting:** To encourage deeper, more structured reasoning, CoT prompting will be utilized.5 The prompt will guide the LLM through a sequence of intermediate reasoning steps for each tweet:
1. *"Accurately summarize the core message of this Persian tweet. Pay attention to any slang, cultural references, or indirect expressions that might alter the literal meaning."*
2. *"Identify key entities mentioned in the tweet (e.g., individuals, organizations, specific economic terms like 'نرخ بهره' - interest rate, 'نقدینگی' - liquidity, locations relevant to economic activity like 'بازار تهران' - Tehran Bazaar)."*
3. *"Assess the potential credibility or bias of the tweet's source, if discernible. Consider if it's an official news outlet, a known economic analyst, an anonymous account, or potentially part of a coordinated campaign. Note if the information appears to be factual, opinion, or rumor."* This step is particularly important in the Iranian context, where information can be polarized, and discerning credible signals from noise is challenging.73 For instance, a tweet from an anonymous account claiming a massive, uncorroborated policy change should be treated with more skepticism than a direct quote from the CBI governor tweeted by a reputable news agency. The CoT can guide the LLM to internally "ask": Is this claim verifiable? Does the language use hyperbole?
4. *"Connect the tweet's content to known drivers of the USD/IRR exchange rate in Iran (e.g., international sanctions, oil revenue fluctuations, domestic monetary policy, public confidence, inflation expectations)."*
5. *"Based on the above, infer the likely short-term and long-term impact on the stability and value of the Iranian Rial. Clearly state if the impact is expected to be positive (strengthening Rial), negative (weakening Rial), or neutral/uncertain."*
6. *"Assign an overall sentiment score or category specifically related to its implication for the Rial: e.g., 'Strongly Negative for Rial', 'Mildly Positive for Rial', 'Neutral'."*
7. *"Generate a concise rationale in English (or Persian, if the downstream forecasting LLM is Persian-native and handles mixed-language inputs effectively) explaining your reasoning for the predicted impact and sentiment."*
- **Structured Prompting:** To ensure the output from the Reasoning Agent is consistent and easily parsable by the downstream forecasting LLM, structured prompting techniques will be used.77 The agent will be instructed to provide its analysis in a specific format, preferably JSON, with clearly defined fields for the tweet summary, source (if available), publication time, impact classification, sentiment, and rationale.
- **Example Prompts for Reasoning Logic** 5**:**
- **Prompt 1 (Initial Logic Definition):**
    
    "Please summarize the logic for selecting and categorizing Persian Tweets that are likely to influence the USD/IRR open market exchange rate in Iran. Your logic should consider the following factors and output them in a structured JSON format:
    
    1.  Events/Sentiments Positive for Rial:
    
    a.  Short-Term Positive: (e.g., news of temporary sanctions waivers, successful central bank interventions to supply FX, positive breakthroughs in international negotiations).
    
    b.  Long-Term Positive: (e.g., concrete news of major sanctions lifting, significant increases in non-oil exports, successful implementation of structural economic reforms).
    
    2.  Events/Sentiments Negative for Rial:
    
    a.  Short-Term Negative: (e.g., rumors of new sanctions, political instability, sharp increases in inflation figures, negative statements by influential international bodies about Iran's economy).
    
    b.  Long-Term Negative: (e.g., imposition of severe new multilateral sanctions, sustained drops in global oil prices impacting Iran's revenue, failure of critical economic policies).
    
    3.  Other Factors: Consider the source of the tweet (official, analyst, public), the scale of discussion (trending hashtags), and the directness of the economic implication.
    
    Provide a rationale for each classification."
    
- **Prompt 2 (Tweet Selection and Analysis Task):**
    
    "The prediction date is. You are provided with a list of Persian Tweets published before this date. Based on the reasoning logic defined previously, please perform the following for each tweet that you deem impactful for the USD/IRR rate:
    
    1.  If the tweet is impactful, select it.
    
    2.  Provide a concise summary of the tweet in both Persian and English.
    
    3.  Identify the region as 'Iran'.
    
    4.  Note the 'publication_time'.
    
    5.  Predict the 'impact_on_rial' as 'Positive_for_Rial', 'Negative_for_Rial', or 'Neutral_Uncertain'.
    
    6.  Specify the 'impact_horizon' as 'Short-term' or 'Long-term'.
    
    7.  Provide a 'rationale_for_impact' explaining your reasoning.
    
    If no tweets in the list are deemed impactful, output an empty list.
    
    The input list of tweets is:.
    
    Please output your analysis as a valid JSON list of objects, where each object represents an impactful tweet."
    

The iterative refinement of this logic by the Evaluation Agent (discussed next) is what will allow the Reasoning Agent to adapt to the complex and evolving nature of Iranian economic discourse on Twitter.

### **3.5. Evaluation Agent for Refining Tweet Selection Logic**

### **3.5.1. Evaluation Agent Design and LLM Choice**

The Evaluation Agent plays a crucial role in the framework's reflective learning capability. Its primary function is to analyze the forecasting performance of the main LLM and, by examining the broader context of all available tweets during that period, identify potential reasons for forecast errors, particularly those related to missed or misinterpreted tweet-driven events.

- **LLM Choice:** Similar to the Reasoning Agent, the Evaluation Agent requires a highly capable LLM with strong analytical and reasoning skills. GPT-4 Turbo 5 is the default choice. An open-source model with comparable capabilities, especially in understanding Persian context if available, would be an alternative.
- **Function:**
1. Receive the forecasting model's predictions, the actual USD/IRR values, and thus the prediction errors for a specific period.
2. Access both the set of tweets *selected* by the Reasoning Agent for that period and the larger pool of *all potentially relevant* tweets collected for that period (i.e., those that passed initial keyword filtering but might have been discarded by the Reasoning Agent).
3. Identify "missed news" – i.e., tweets from the unselected pool that, in hindsight and given the forecast errors, appear to have been impactful and were erroneously ignored.
4. Analyze if selected tweets were misinterpreted by the Reasoning Agent (e.g., wrong impact direction or horizon).
5. Generate proposals for updating the Reasoning Agent's news selection logic to improve future performance.

### **3.5.2. Prompting for Identifying Missed Iranian Events and Updating Logic**

The prompting strategy for the Evaluation Agent will adapt the three-phase approach outlined in Wang et al. (2024).5

- **Phase 1: Evaluation Setup & Contextualization:**
- The agent is provided with the specifics of the forecasting task being evaluated.
- Example Prompt: *"You are an expert financial analyst specializing in the Iranian economy. Your task is to evaluate the USD/IRR open market exchange rate forecast for the period [start_date] to [end_date]. Background context for this period includes:. Please outline the steps you would take to analyze forecast errors and identify if any tweet-driven events were missed or misinterpreted by the primary forecasting model."*
- **Phase 2: Error Analysis & Missed News Identification:**
- The agent receives the actual and predicted USD/IRR series, the calculated errors, the list of tweets used by the forecasting model (selected by the Reasoning Agent), and access to the broader pool of all tweets collected for that period.
- Example Prompt: *"For the period [start_date] to [end_date], the actual USD/IRR rates were: [...]. The model's predicted rates were: [...]. This resulted in the following prediction errors (Actual - Predicted): [...]. The tweets selected by the Reasoning Agent and used for this forecast were:. All potentially relevant Persian tweets collected for this period are available in [reference to larger dataset/list]. Analyze the prediction errors. Were there any significant forecast deviations? By examining both the selected tweets and the full pool of available tweets, identify any specific Persian Tweet events that were missed by the Reasoning Agent (i.e., not selected) but could plausibly explain the observed forecast errors. For each missed tweet, provide:
- 'missed_tweet_summary_persian'
- 'missed_tweet_summary_english'
- 'publication_time'
- 'potential_impact_on_rial' (Positive/Negative/Neutral)
- 'reasoning_for_missed_impact' (Why it might have caused the forecast error and why it was likely missed by the initial logic). Output the missed news as a JSON list."*
- An example of a missed news output, adapted for Iran from 5 Appendix A.7: *"Missed News: Tweet from @ 'شنیده‌ها حاکی از تزریق سنگین ارز توسط بانک مرکزی در بازار آزاد برای کنترل نوسانات است. صرافی‌ها از افزایش عرضه دلار خبر می‌دهند.' (English: 'Rumors indicate heavy FX injection by the Central Bank in the open market to control volatility. Exchange offices report increased USD supply.') Occurred at:. Possible Reasoning: This tweet, although based on rumors, originated from a generally credible local analyst and hinted at a potential short-term supply-side intervention by the CBI. The forecasting model overpredicted Rial depreciation for this day, suggesting this supply news, which would strengthen the Rial, was missed. The initial logic might be too dismissive of 'rumors' even from known commentators."*
- **Phase 3: Logic Update Generation:**
- Based on the identified missed news or misinterpreted events, the agent proposes specific refinements to the Reasoning Agent's selection logic.
- Example Prompt: *"Based on the missed Tweet event(s) identified concerning, propose specific updates to the news selection logic for the Reasoning Agent. The goal is to improve its ability to capture such events for future USD/IRR forecasting in Iran. For instance, should the logic:
- Adjust thresholds for source credibility regarding rumors?
- Add new keywords or monitor specific emerging hashtags?
- Change the weighting of sentiment from certain types of accounts? Provide the updated logic rules or principles."*
- **Iterative Refinement Prompts:** The process of refining logic can be made more explicit by using prompts that encourage the LLM to reflect on its previous logic and the new evidence. Techniques described in studies on iterative prompting 79 can be adapted, where the agent is asked to compare its old logic with the new findings and explicitly state how the logic should change.

The power of this Evaluation Agent lies in its ability to act as a domain expert in hindsight. It's not feasible to pre-define every conceivable type of tweet that could impact the volatile Iranian Rial. New slang, new influential figures, or entirely novel types of economic or political events will emerge. The Evaluation Agent's strength is its capacity to connect *unforeseen* forecast errors back to *previously ignored or misinterpreted* tweet patterns. For instance, if a new, seemingly innocuous hashtag starts trending and consistently correlates with periods of market anxiety and forecast deviations (even if the words themselves are not overtly economic), the Evaluation Agent can detect this pattern. It can then suggest to the Reasoning Agent: "Increase vigilance for hashtag X and associated discussions, as it appears to co-occur with periods of unexplained Rial volatility not captured by current rules." This allows the system to adapt and learn about the evolving nature of influential narratives in the Iranian Twittersphere.

## **4. Evaluation Plan**

The proposed framework's performance will be rigorously evaluated through quantitative metrics, comparisons against baseline models, ablation studies to assess component contributions, and qualitative analysis of its behavior in the Iranian context.

### **4.1. Baseline Models for USD/IRR Forecasting**

To establish the efficacy of the proposed LLM-based framework incorporating Iranian Tweet events, its performance will be compared against a diverse set of baseline models. These baselines will range from traditional statistical methods to simpler machine learning and LLM-based approaches, allowing for a nuanced understanding of the sources of performance improvement.

The selected baseline models are:

- **Statistical Models:**
- *ARIMA (Autoregressive Integrated Moving Average):* A standard univariate time series model for capturing linear dependencies and trends in the USD/IRR data.
- *SARIMA (Seasonal ARIMA):* An extension of ARIMA to account for potential seasonality in the exchange rate data, if identified.
- **Traditional Machine Learning Models:**
- *Support Vector Regression (SVR):* A non-linear regression model capable of capturing complex relationships.
- *Random Forest Regressor:* An ensemble learning method known for its robustness and ability to handle non-linear data.
- **Basic Deep Learning Models:**
- *LSTM (Long Short-Term Memory Network):* A type of recurrent neural network well-suited for sequence data like time series, capable of learning long-term dependencies.
- *GRU (Gated Recurrent Unit Network):* A simpler variant of LSTM, often achieving comparable performance with fewer parameters.
- **Simpler LLM Baselines (to isolate the impact of different components of the proposed framework):**
- *LLM (Numerical Only):* The selected base LLM (from Section 3.3.1) fine-tuned solely on historical numerical USD/IRR data, without any textual prompts or Tweet event integration. This will establish the LLM's baseline forecasting capability on the raw time series.
- *LLM (Textual Prompt - No Tweets):* The LLM fine-tuned on USD/IRR data using generic textual prompts (e.g., describing the task and data format) but without any specific news or Tweet event data, similar to the "Textual Prompt without News" condition in Wang et al. (2024).5
- *LLM (Unfiltered Tweets):* The LLM fine-tuned on USD/IRR data combined with *all* keyword-matched Persian Tweets (i.e., before the Reasoning Agent's filtering). This will demonstrate the value of intelligent event selection compared to a naive inclusion of all potentially relevant social media noise, analogous to "Textual Prompt with Non-Filtered News" in Wang et al. (2024).5
- *LLM (Keyword/Simple Sentiment Filtered Tweets):* The LLM fine-tuned on USD/IRR data plus Tweets filtered using a simpler, non-agentic method (e.g., basic keyword matching combined with a standard Persian sentiment analyzer without the sophisticated reasoning or iterative refinement of the proposed agents). This provides a more advanced baseline than unfiltered tweets but less complex than the full proposed framework.
- **Proposed Full Framework:** The complete LLM-based forecasting model integrated with the Reasoning Agent (for intelligent Tweet selection) and the Evaluation Agent (for iterative logic refinement).

Table 3 provides a summary of these baseline models.

**Table 3: Baseline Models for Performance Comparison**

| **Model Type/Name** | **Brief Description** | **Key Inputs** | **Rationale for Inclusion as a Baseline** |
| --- | --- | --- | --- |
| ARIMA / SARIMA | Classical statistical models for time series forecasting. | Historical USD/IRR data only. | Standard benchmarks in time series forecasting; represent traditional linear modeling capabilities. |
| Support Vector Regression (SVR) | Machine learning model for regression tasks, capable of capturing non-linear relationships. | Historical USD/IRR data, potentially basic lagged features. | Represents traditional non-linear machine learning approaches. |
| Random Forest Regressor | Ensemble machine learning model using decision trees, robust to overfitting. | Historical USD/IRR data, potentially basic lagged features. | Represents robust ensemble machine learning approaches. |
| LSTM / GRU | Deep learning recurrent neural networks for sequence modeling. | Historical USD/IRR data only. | Standard deep learning baselines for time series forecasting, capturing temporal dependencies. |
| LLM (Numerical Only) | Selected base LLM fine-tuned only on historical USD/IRR numerical sequences. | Historical USD/IRR data only. | Establishes the LLM's raw numerical forecasting capability without textual or event context. |
| LLM (Textual Prompt - No Tweets) | LLM fine-tuned with generic textual prompts about the forecasting task but no specific Tweet/event data. | Historical USD/IRR data, generic textual prompts. | Tests the incremental benefit of using LLMs in a prompted fashion over purely numerical input, without event data.5 |
| LLM (Unfiltered Tweets) | LLM fine-tuned with USD/IRR data and all keyword-matched Persian Tweets (no sophisticated agent-based filtering). | Historical USD/IRR data, all pre-filtered (keyword-matched) Persian Tweets. | Highlights the necessity of intelligent event filtering; shows impact of social media noise if not properly curated.5 |
| LLM (Keyword/Simple Sentiment Filtered Tweets) | LLM fine-tuned with USD/IRR data and Tweets filtered by basic keywords and a standard sentiment analyzer. | Historical USD/IRR data, Tweets filtered by keywords and basic sentiment scores. | Provides a stronger baseline than unfiltered tweets, representing a common approach to integrating social media data without advanced agentic reasoning. |
| **Proposed Full Framework** | **LLM fine-tuned with USD/IRR data and Tweet events selected and refined by Reasoning and Evaluation Agents.** | **Historical USD/IRR data, supplementary info, Tweet events processed by the full agentic pipeline.** | **The primary model whose performance is being evaluated, incorporating all proposed methodological innovations.** |

This range of baselines will allow for a clear demonstration of the incremental benefits provided by each level of sophistication, from basic time series modeling to the full agent-based LLM framework.

### **4.2. Performance Metrics**

The forecasting performance of all models will be evaluated using standard regression metrics, consistent with Wang et al. (2024) 5, to ensure comparability and a comprehensive assessment:

- **Mean Squared Error (MSE):** Calculated as n1∑i=1n(yi−y^i)2, where yi is the actual value, y^i is the predicted value, and n is the number of predictions. MSE penalizes larger errors more heavily.
- **Root Mean Squared Error (RMSE):** Calculated as n1∑i=1n(yi−y^i)2. RMSE provides an error measure in the same units as the USD/IRR rate, making it more interpretable than MSE.
- **Mean Absolute Error (MAE):** Calculated as n1∑i=1n∣yi−y^i∣. MAE is less sensitive to outliers compared to MSE/RMSE and gives a straightforward measure of average error magnitude.
- **Mean Absolute Percentage Error (MAPE):** Calculated as n100∑i=1n∣yiyi−y^i∣. MAPE expresses the error as a percentage of the actual values, which is useful for understanding the relative error and comparing performance across different price levels or time periods.

In addition to these value-based metrics, **Directional Accuracy** will also be computed. This metric measures the percentage of times the model correctly predicts the direction of change (i.e., whether the Rial will appreciate, depreciate, or remain stable relative to the USD) for the next forecasting period (e.g., next day). For many practical applications in finance, correctly predicting the direction of movement can be more valuable than predicting the exact future value.

Statistical significance tests (e.g., Diebold-Mariano test) will be employed to compare the predictive accuracy of the proposed model against the baseline models.

### **4.3. Ablation Studies**

To rigorously assess the individual contributions of the key components within the proposed framework, a series of ablation studies will be conducted. This approach systematically removes or simplifies parts of the model to quantify their impact on performance, mirroring the structure of Table 1 in Wang et al. (2024).5 The following configurations will be compared:

1. **Numerical Only:** The selected base LLM fine-tuned using only historical numerical USD/IRR data. This serves as the most basic LLM configuration.
2. **Textual Prompt (No News/Tweets):** The LLM fine-tuned on USD/IRR data with generic textual prompts describing the task and data format, but without incorporating any specific Tweet events or news. This configuration is analogous to the "Textual Prompt without News" in.5
3. **Unfiltered Tweets:** The LLM fine-tuned on USD/IRR data combined with all Persian Tweets that pass an initial broad keyword/hashtag filtering stage, without the sophisticated selection and reasoning performed by the Reasoning Agent. This is comparable to the "Textual Prompt with Non-Filtered News" in 5 and will help quantify the impact of information overload and noise from irrelevant tweets.
4. **Filtered Tweets (Reasoning Agent Only - Initial Logic):** The LLM fine-tuned on USD/IRR data plus Tweet events selected by the Reasoning Agent using its *initial, pre-defined* logic, before any iterative refinement from the Evaluation Agent. This isolates the benefit of the initial intelligent filtering.
5. **Full Model (Filtered Tweets + Iterative Refinement):** The complete proposed framework, where the LLM is fine-tuned on USD/IRR data plus Tweet events selected by the Reasoning Agent whose logic has been iteratively refined through feedback from the Evaluation Agent. This corresponds to the "Textual Prompt with Filtered News" in.5

Comparing the performance metrics (Section 4.2) across these five configurations will allow for a quantitative assessment of the value added by:

- (a) Using textual prompts with the LLM (comparing configuration 2 vs. 1).
- (b) Integrating Tweet data in any form (comparing configuration 3 vs. 2).
- (c) Employing intelligent Tweet filtering via the Reasoning Agent (comparing configuration 4 vs. 3).
- (d) Incorporating iterative logic refinement via the Evaluation Agent (comparing configuration 5 vs. 4).

This systematic deconstruction is crucial for justifying the complexity of the proposed framework. If, for example, the performance gain from configuration 3 to 4 is minimal, it might suggest that the initial Reasoning Agent's logic is not significantly better than simpler filtering. Conversely, a substantial improvement from configuration 4 to 5 would strongly validate the critical role of the Evaluation Agent and the iterative learning process in adapting to the nuances of Iranian Tweet events.

### **4.4. Qualitative Analysis**

Beyond quantitative metrics, a qualitative analysis will be performed to gain deeper insights into the model's behavior and the nature of the information it leverages.

- **Case Studies:** Specific historical periods characterized by high USD/IRR volatility or significant economic/political events in Iran will be selected for in-depth analysis. Examples include periods surrounding major sanction announcements, key political speeches, election cycles, or sudden oil price shocks.
- **Analysis of Selected Tweets:** For these case study periods, the Persian Tweets selected as impactful by the Reasoning Agent (in its final, refined state) will be manually reviewed. The content, source (if identifiable), and sentiment of these tweets will be examined.
- **Evaluation of Rationales:** The rationales generated by the Reasoning Agent for why these specific tweets were deemed impactful will be assessed for their coherence, economic plausibility, and alignment with expert understanding of the Iranian context.
- **Review of "Missed News" and Logic Updates:** The "missed news" (i.e., initially overlooked but retrospectively important tweets) identified by the Evaluation Agent during the iterative refinement process will be analyzed. The corresponding updates made to the Reasoning Agent's logic will be examined to understand how the system learned and adapted.
- **Interpretability Assessment:** An overall assessment will be made of the model's interpretability. The aim is to determine if the selected Tweet events and their associated rationales provide a plausible and understandable narrative for the model's predicted currency movements, thereby offering more than just a "black box" forecast.

This qualitative component will involve careful manual review and may benefit from expert judgment on the nuances of Persian language, Iranian culture, and economic discourse. It will be crucial for understanding *why* the model makes certain predictions and for building confidence in its outputs beyond mere statistical accuracy.

## **5. Expected Outcomes, Challenges, and Timeline**

### **5.1. Expected Outcomes and Contributions**

This research is anticipated to yield several significant outcomes and contributions:

- **A Novel Forecasting Framework for USD/IRR:** The primary outcome will be a novel, empirically validated LLM-based framework specifically adapted for forecasting the USD/IRR exchange rate, effectively integrating real-time Iranian Tweet events. This framework will be the first of its kind tailored to the unique complexities of the Iranian economic and information environment.
- **Improved Forecasting Accuracy:** It is expected that the proposed framework will demonstrate improved forecasting accuracy for the USD/IRR rate compared to traditional statistical models, simpler machine learning approaches, and LLM baselines that do not incorporate sophisticated event processing or iterative refinement.
- **Enhanced Understanding of Social Media Influence in Iran:** The research will provide a refined understanding of the types of social media narratives, sentiments, and specific Tweet-driven events that exert the most significant influence on the Iranian Rial. The qualitative analysis and the learned logic of the Reasoning Agent will offer insights into these dynamics.
- **Methodological Advancements for Event-Driven Forecasting:** The adaptation and application of the Wang et al. (2024) 5 methodology to a highly volatile, data-scarce, and politically sensitive emerging market like Iran will offer methodological contributions to the broader field of event-driven forecasting, particularly in contexts with limited data transparency.
- **Potential for Publicly Available Resources:** Subject to ethical considerations and data source permissions, the project aims to make its codebase publicly available. Furthermore, an anonymized dataset categorizing types of impactful Iranian Tweet events (rather than raw tweet data, to protect privacy) could be shared to foster further research in this domain.

### **5.2. Addressing Challenges**

The proposed research, while promising, faces several notable challenges, particularly those unique to the Iranian context.

- **Data Access and Reliability (Iran-Specific):**
- *Challenge:* Obtaining consistent, long-term, and high-frequency open market USD/IRR data can be difficult. Official exchange rates often do not reflect the true market value accessible to most.43
- *Mitigation:* Data will be triangulated from multiple available open market sources (e.g., Bonbast 46, Navasan 45). The focus will be on daily frequency. All assumptions regarding the chosen rate series will be clearly documented, and sensitivity analyses using different available rate series may be performed if significant discrepancies exist.
- *Challenge:* Access to Twitter (X) data from or about Iran can be affected by API restrictions, platform policies, or potential government censorship and internet disruptions.52
- *Mitigation:* Academic datasets of Persian tweets 84 will be explored as supplementary sources. Robust and ethical scraping methods will be employed, respecting Twitter's Terms of Service. Potential biases in the collected tweet dataset (e.g., due to self-selection of users or filtering) will be acknowledged as a limitation.
- **Persian NLP Nuances:**
- *Challenge:* Persian tweets are rich in dialectal variations, slang, coded language, satire, and sarcasm, which can be misinterpreted by LLMs not deeply attuned to Iranian culture and language.19
- *Mitigation:* Priority will be given to LLMs with strong native Persian training (e.g., PersianMind 30, Fibonacci LLM 33) or those that demonstrate excellent cross-lingual transfer for Persian. The iterative refinement capability of the Evaluation Agent is a key mitigation strategy, as it allows the model to learn from errors caused by misinterpreting such nuances. Few-shot examples in agent prompting will include instances of such complex Persian expressions.
- **Defining and Validating "Tweet Events" in the Iranian Context:**
- *Challenge:* Differentiating genuine, impactful "events" signaled in tweets from mere noise, rumors, propaganda, or potentially coordinated inauthentic behavior is a significant hurdle in the Iranian information ecosystem. Assessing the credibility of tweet sources is also difficult.73
- *Mitigation:* The Reasoning Agent's logic will incorporate heuristics for source type (e.g., verified news agency, known analyst, anonymous account). The Evaluation Agent's role is crucial here: if consistently ignoring certain types of unverified but ultimately impactful narratives leads to forecast errors, the system can learn to adjust its sensitivity. Cross-referencing tweet-driven event signals with official news (even if delayed) can provide some validation. The adaptive nature of the Wang et al. (2024) 5 framework is particularly beneficial here. Instead of needing to pre-define all possible event types, the system *learns* what constitutes an impactful event through the iterative error correction driven by the Evaluation Agent. For example, if tweets *interpreting* sanctions in a specific way prove more influential than the factual announcement itself, the Evaluation Agent can help discover this more granular event definition by linking forecast failures to these specific interpretations that were initially overlooked.
- **Geopolitical Sensitivity and Model Bias:**
- *Challenge:* The model might inadvertently learn and reflect biases present in the Twitter data or amplify politically charged narratives. Ensuring objectivity in a highly politicized context is paramount.
- *Mitigation:* A conscious effort will be made to design neutral prompts for the agents. The types of tweets identified as influential by the model will be transparently reported. Qualitative analysis (Section 4.4) will critically examine the model's interpretations for potential biases. This will be acknowledged as an inherent limitation of using social media data from such environments.
- **Computational Resources:**
- *Challenge:* Fine-tuning large LLMs and running multiple iterations of agent interactions can be computationally demanding, requiring significant GPU resources.
- *Mitigation:* Parameter-efficient fine-tuning techniques like LoRA will be employed. Agent prompting will be optimized for conciseness and efficiency where possible. Access to adequate GPU resources through academic or cloud computing platforms will be secured.

This phased approach, with clear deliverables, provides a structured path toward achieving the research objectives while allowing for iterative development and adaptation based on emerging results.

## **6. Resources**

Successful completion of this research will require access to specific computational, software, data, and human resources:

- **Computational Resources:**
- High-Performance Computing (HPC) cluster access with multiple GPUs (e.g., NVIDIA A100 40G or equivalent, as used in Wang et al. (2024) 5) will be essential for the fine-tuning of large language models and for running the iterative agent-based pipeline.
- Sufficient storage for datasets, model checkpoints, and experimental results.
- **Software and Libraries:**
- Programming Environment: Python (version 3.8 or higher).
- LLM Frameworks: Hugging Face Transformers library for model access, tokenization, and fine-tuning.
- Deep Learning Libraries: PyTorch or TensorFlow.
- Data Processing: Pandas, NumPy, Scikit-learn.
- Twitter Data Collection: Access to the Twitter API (X API) via academic research tracks if possible, or proficiency with scraping tools like snscrape 59 (adapted for Persian).
- Persian NLP Tools: Libraries for Persian text normalization and tokenization (e.g., Hazm, Parsivar).
- Version Control: Git and GitHub/GitLab for code management and collaboration.
- **Data Access:**
- Historical USD/IRR Data: Continuous access to and/or methods for regularly updating data from sources like Bonbast.com 46, Navasan.net 45, and Alanchand.com.44
- Iranian Economic Data: Access to official publications or data portals from the Statistical Center of Iran (SCI) 49, Central Bank of Iran (CBI) 43, and Ministry of Oil.47
- International Data: Access to OPEC reports, international commodity price data (e.g., via APIs or financial data providers), and sanction announcement databases (e.g., OFAC).
- Tweet Data: Ongoing capability to collect or access a substantial volume of relevant Persian-language tweets.
- **Human Resources:**
- Primary Researcher: PhD candidate responsible for all aspects of the research, including literature review, data collection and processing, model development, experimentation, analysis, and writing.
- Supervisory Support: Guidance from academic supervisors with expertise in machine learning, NLP, time series analysis, and potentially Iranian studies or economics.
- Consultation (Optional): Potential for consultation with experts on the Iranian economy or Persian linguistics for validating qualitative findings or refining the initial design of prompts and reasoning logic.

Securing these resources, particularly consistent access to high-quality data and sufficient computational power, will be critical for the project's execution.

## **7. References**

*(This section will be populated with a comprehensive list of all cited academic papers 5 and data sources 57 using a consistent academic citation style, such as APA or IEEE, based on the snippets used throughout the proposal.)*

- Ataei, T., Darvishi, A., et al. (2019). *Pars-ABSA: A New Aspect-Based Manually Annotated Persian Dataset*. 29
- Farahani, F., Gharachorloo, M., et al. (2020). *ParsBERT: Transformer-based Model for Persian Language Understanding*. arXiv:2005.12515. 23
- Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2024). *Large language models are zero-shot time series forecasters*. Advances in Neural Information Processing Systems, 36. (Cited as in 5)
- Jin, M., Wang, S., Ma, L., et al. (2023). *Time-LLM: Time series forecasting by reprogramming large language models*. arXiv preprint arXiv:2310.01728. (Cited as in 5)
- Wang, X., Feng, M., Qiu, J., Gu, J., & Zhao, J. (2024). *From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection*. arXiv:2409.17515v3 [cs.AI]. 5 *(Additional references from S_B* and S_S* snippets would be listed here in a full bibliography)*

### **Works cited**

1. Economy of Iran - Wikipedia, accessed May 25, 2025, [https://en.wikipedia.org/wiki/Economy_of_Iran](https://en.wikipedia.org/wiki/Economy_of_Iran)
2. Sanctions and Self-Sufficiency: The Evolution of Iran's Manufacturing Sector, accessed May 25, 2025, [https://www.habtoorresearch.com/programmes/sanctions-iran-manufacturing/](https://www.habtoorresearch.com/programmes/sanctions-iran-manufacturing/)
3. The Iranian Regime's Economic Crisis in a Critical Year - Iran Focus, accessed May 25, 2025, [https://iranfocus.com/economy/53855-the-iranian-regimes-economic-crisis-in-a-critical-year/](https://iranfocus.com/economy/53855-the-iranian-regimes-economic-crisis-in-a-critical-year/)
4. ۱۰ رویداد مهم اقتصاد ایران در ۱۴۰۳؛ سالی که ریال آب رفت و سرمایه‌ها دود شد - BBC, accessed May 25, 2025, [https://www.bbc.com/persian/articles/c789e19j9kxo](https://www.bbc.com/persian/articles/c789e19j9kxo)
5. 2409.17515v3.pdf
6. How Economic News Moves Markets - Federal Reserve Bank of New York, accessed May 25, 2025, [https://www.newyorkfed.org/medialibrary/media/research/current_issues/ci14-6.html](https://www.newyorkfed.org/medialibrary/media/research/current_issues/ci14-6.html)
7. Price drift before U.S. macroeconomic news: private information about public announcements? - European Central Bank, accessed May 25, 2025, [https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1901.en.pdf](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1901.en.pdf)
8. (PDF) Natural language processing (nlp) for financial text analysis - ResearchGate, accessed May 25, 2025, [https://www.researchgate.net/publication/385860012_Natural_language_processing_nlp_for_financial_text_analysis](https://www.researchgate.net/publication/385860012_Natural_language_processing_nlp_for_financial_text_analysis)
9. Economic News, Social Media Sentiments, and Stock Returns: Which Is a Bigger Driver?, accessed May 25, 2025, [https://www.mdpi.com/1911-8074/18/1/16](https://www.mdpi.com/1911-8074/18/1/16)
10. Sentiment Analysis of Financial News: Mechanics & Statistics - Dow Jones, accessed May 25, 2025, [https://www.dowjones.com/professional/risk/resources/blog/a-primer-for-sentiment-analysis-of-financial-news](https://www.dowjones.com/professional/risk/resources/blog/a-primer-for-sentiment-analysis-of-financial-news)
11. Sentiment Analysis in Financial News: Enhancing Predictive Models for Stock Market Behavior - ResearchGate, accessed May 25, 2025, [https://www.researchgate.net/publication/390056578_Sentiment_Analysis_in_Financial_News_Enhancing_Predictive_Models_for_Stock_Market_Behavior](https://www.researchgate.net/publication/390056578_Sentiment_Analysis_in_Financial_News_Enhancing_Predictive_Models_for_Stock_Market_Behavior)
12. Social media engagement and cryptocurrency performance - PMC - PubMed Central, accessed May 25, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10174546/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10174546/)
13. Currency Exchange Rate Forecasting with Social Media Sentiment Analysis - ResearchGate, accessed May 25, 2025, [https://www.researchgate.net/publication/357161227_Currency_Exchange_Rate_Forecasting_with_Social_Media_Sentiment_Analysis](https://www.researchgate.net/publication/357161227_Currency_Exchange_Rate_Forecasting_with_Social_Media_Sentiment_Analysis)
14. Evaluating the Impact of Twitter Sentiment on Stock Market Performance: A Multi-Period Analysis, accessed May 25, 2025, [https://thesis.eur.nl/pub/73766/610851.pdf](https://thesis.eur.nl/pub/73766/610851.pdf)
15. More than Words: Twitter Chatter and Financial Market Sentiment - Federal Reserve Board, accessed May 25, 2025, [https://www.federalreserve.gov/econres/feds/files/2023034pap.pdf](https://www.federalreserve.gov/econres/feds/files/2023034pap.pdf)
16. Market-Derived Financial Sentiment Analysis: Context-Aware Language Models for Crypto Forecasting - arXiv, accessed May 25, 2025, [https://arxiv.org/pdf/2502.14897](https://arxiv.org/pdf/2502.14897)
17. Investors Guide to Event Study Analysis in Markets, accessed May 25, 2025, [https://www.numberanalytics.com/blog/investors-guide-event-study-analysis-in-markets](https://www.numberanalytics.com/blog/investors-guide-event-study-analysis-in-markets)
18. Advancing Persian LLM Evaluation - ACL Anthology, accessed May 25, 2025, [https://aclanthology.org/2025.findings-naacl.147.pdf](https://aclanthology.org/2025.findings-naacl.147.pdf)
19. Multilingual Language Models in Persian NLP Tasks: A Performance ‎Comparison of Fine-Tuning Techniques - Journal of AI and Data Mining, accessed May 25, 2025, [https://jad.shahroodut.ac.ir/article_3392.html](https://jad.shahroodut.ac.ir/article_3392.html)
20. PerCul: A Story-Driven Cultural Evaluation of LLMs in Persian - arXiv, accessed May 25, 2025, [https://arxiv.org/html/2502.07459v1](https://arxiv.org/html/2502.07459v1)
21. Multi-BERT: Leveraging Adapters for Low-Resource Multi-Domain Adaptation - ACL Anthology, accessed May 25, 2025, [https://aclanthology.org/2025.wnut-1.12.pdf](https://aclanthology.org/2025.wnut-1.12.pdf)
22. Boosting Sentiment Analysis in Persian through a GAN-Based Synthetic Data Augmentation Method - ACL Anthology, accessed May 25, 2025, [https://aclanthology.org/2025.abjadnlp-1.7.pdf](https://aclanthology.org/2025.abjadnlp-1.7.pdf)
23. Bert Base Parsbert Uncased · Models - Dataloop, accessed May 25, 2025, [https://dataloop.ai/library/model/hooshvarelab_bert-base-parsbert-uncased/](https://dataloop.ai/library/model/hooshvarelab_bert-base-parsbert-uncased/)
24. Bert Base Parsbert Armanner Uncased · Models - Dataloop, accessed May 25, 2025, [https://dataloop.ai/library/model/hooshvarelab_bert-base-parsbert-armanner-uncased/](https://dataloop.ai/library/model/hooshvarelab_bert-base-parsbert-armanner-uncased/)
25. Hooshvare - GitHub, accessed May 25, 2025, [https://github.com/hooshvare](https://github.com/hooshvare)
26. Persian Sentiment Analysis via a Transformer Model concerning Banking Sector, accessed May 25, 2025, [https://www.researchgate.net/publication/373485694_Persian_Sentiment_Analysis_via_a_Transformer_Model_concerning_Banking_Sector](https://www.researchgate.net/publication/373485694_Persian_Sentiment_Analysis_via_a_Transformer_Model_concerning_Banking_Sector)
27. Persian sentiment system involving dataset balancing, data preprocessing, and ParsBERT model fine-tuning for news article sentiment analysis - GitHub, accessed May 25, 2025, [https://github.com/marzinouri/persian-sentiment-analysis](https://github.com/marzinouri/persian-sentiment-analysis)
28. DeepSentiParsBERT: A Deep Learning Model for Persian Sentiment Analysis using ParsBERT - ResearchGate, accessed May 25, 2025, [https://www.researchgate.net/profile/Fahime-Ghasemian/publication/370304945_DeepSentiParsBERT_A_Deep_Learning_Model_for_Persian_Sentiment_Analysis_Using_ParsBERT/links/64b6cf518de7ed28baaaa500/DeepSentiParsBERT-A-Deep-Learning-Model-for-Persian-Sentiment-Analysis-Using-ParsBERT.pdf](https://www.researchgate.net/profile/Fahime-Ghasemian/publication/370304945_DeepSentiParsBERT_A_Deep_Learning_Model_for_Persian_Sentiment_Analysis_Using_ParsBERT/links/64b6cf518de7ed28baaaa500/DeepSentiParsBERT-A-Deep-Learning-Model-for-Persian-Sentiment-Analysis-Using-ParsBERT.pdf)
29. exploiting bert to improve aspect-based sentiment analysis performance on persian language - arXiv, accessed May 25, 2025, [https://arxiv.org/pdf/2012.07510](https://arxiv.org/pdf/2012.07510)
30. universitytehran/PersianMind-v1.0 - Hugging Face, accessed May 25, 2025, [https://huggingface.co/universitytehran/PersianMind-v1.0](https://huggingface.co/universitytehran/PersianMind-v1.0)
31. PersianMind: A Cross-Lingual Persian-English Large Language Model - arXiv, accessed May 25, 2025, [https://arxiv.org/html/2401.06466v1](https://arxiv.org/html/2401.06466v1)
32. (PDF) FarsEval-PKBETS: A new diverse benchmark for evaluating Persian large language models - ResearchGate, accessed May 25, 2025, [https://www.researchgate.net/publication/390991743_FarsEval-PKBETS_A_new_diverse_benchmark_for_evaluating_Persian_large_language_models](https://www.researchgate.net/publication/390991743_FarsEval-PKBETS_A_new_diverse_benchmark_for_evaluating_Persian_large_language_models)
33. fibonacciai/Persian-llm-fibonacci-1-7b-chat.P1_0 - Hugging Face, accessed May 25, 2025, [https://huggingface.co/fibonacciai/Persian-llm-fibonacci-1-7b-chat.P1_0](https://huggingface.co/fibonacciai/Persian-llm-fibonacci-1-7b-chat.P1_0)
34. fibonacciai/Persian-llm-fibonacci-1-pro · Datasets at Hugging Face, accessed May 25, 2025, [https://huggingface.co/datasets/fibonacciai/Persian-llm-fibonacci-1-pro](https://huggingface.co/datasets/fibonacciai/Persian-llm-fibonacci-1-pro)
35. Persian-llm-fibonacci-1-7b-chat.P1_0 - PromptLayer, accessed May 25, 2025, [https://www.promptlayer.com/models/persian-llm-fibonacci-1-7b-chatp10](https://www.promptlayer.com/models/persian-llm-fibonacci-1-7b-chatp10)
36. What are the best open-source LLMs for highly accurate translations between English and Persian? - Reddit, accessed May 25, 2025, [https://www.reddit.com/r/LanguageTechnology/comments/1jacxz8/what_are_the_best_opensource_llms_for_highly/](https://www.reddit.com/r/LanguageTechnology/comments/1jacxz8/what_are_the_best_opensource_llms_for_highly/)
37. Khayyam Challenge (PersianMMLU): Is Your LLM Truly Wise to The Persian Language?, accessed May 25, 2025, [https://arxiv.org/html/2404.06644v1](https://arxiv.org/html/2404.06644v1)
38. FaMTEB: Massive Text Embedding Benchmark in Persian Language - arXiv, accessed May 25, 2025, [https://arxiv.org/html/2502.11571v1](https://arxiv.org/html/2502.11571v1)
39. arXiv:2502.11571v1 [cs.CL] 17 Feb 2025, accessed May 25, 2025, [https://arxiv.org/pdf/2502.11571](https://arxiv.org/pdf/2502.11571)
40. PERCUL: A Story-Driven Cultural Evaluation of LLMs in Persian - ACL Anthology, accessed May 25, 2025, [https://aclanthology.org/2025.naacl-long.631.pdf](https://aclanthology.org/2025.naacl-long.631.pdf)
41. Evaluating Parameter-Efficient Finetuning Approaches for Pre-trained Models on the Financial Domain - ACL Anthology, accessed May 25, 2025, [https://aclanthology.org/2023.findings-emnlp.1035.pdf](https://aclanthology.org/2023.findings-emnlp.1035.pdf)
42. Iran - Index of Economic Freedom - The Heritage Foundation, accessed May 25, 2025, [https://www.heritage.org/index/pages/country-pages/iran](https://www.heritage.org/index/pages/country-pages/iran)
43. Foreign Exchange Rate: Central Bank of Iran (CBI): End of Period: US Dollar - CEIC, accessed May 25, 2025, [https://www.ceicdata.com/en/iran/foreign-exchange-rates/foreign-exchange-rate-central-bank-of-iran-cbi-end-of-period-us-dollar](https://www.ceicdata.com/en/iran/foreign-exchange-rates/foreign-exchange-rate-central-bank-of-iran-cbi-end-of-period-us-dollar)
44. Exchange rate of US dollar to Iranian Rials, USD-HAV to IRR | Alanchand, accessed May 25, 2025, [https://alanchand.com/en/currencies-price/usd-hav](https://alanchand.com/en/currencies-price/usd-hav)
45. Daily exchange rates for US Dollar Sell - Navasan, accessed May 25, 2025, [https://www.navasan.net/en/dayRates.php?item=usd_sell](https://www.navasan.net/en/dayRates.php?item=usd_sell)
46. SamadiPour/rial-exchange-rates-archive: Archive of Iranian Rial currency conversion rate from https://bonbast.com - GitHub, accessed May 25, 2025, [https://github.com/SamadiPour/rial-exchange-rates-archive](https://github.com/SamadiPour/rial-exchange-rates-archive)
47. A report on Iran's oil price fluctuations last year (Report) - IranOilGas Network, accessed May 25, 2025, [https://www.iranoilgas.com/news/details.aspx?id=27477&title=A+report+on+Iran%E2%80%99s+oil+price+fluctuations+last+year+(Report)](https://www.iranoilgas.com/news/details.aspx?id=27477&title=A+report+on+Iran%E2%80%99s+oil+price+fluctuations+last+year+(Report))
48. پیش بینی قیمت دلار در سال ۱۴۰۳ - وبلاگ کیلوتن, accessed May 25, 2025, [https://blog.kilooton.com/dollar-price-forecast-in-1403/](https://blog.kilooton.com/dollar-price-forecast-in-1403/)
49. Statistical Centre of Iran, accessed May 25, 2025, [https://www.amar.org.ir/en/](https://www.amar.org.ir/en/)
50. Statistical Center of Iran - Wikipedia, accessed May 25, 2025, [https://en.wikipedia.org/wiki/Statistical_Center_of_Iran](https://en.wikipedia.org/wiki/Statistical_Center_of_Iran)
51. Basic Sources of International Economic Statistics: Exchange Rates - Research Guides, accessed May 25, 2025, [https://libguides.princeton.edu/internationalecon/xrates](https://libguides.princeton.edu/internationalecon/xrates)
52. Country policy and information note: social media, surveillance and sur place activities, Iran, April 2025 (accessible) - GOV.UK, accessed May 25, 2025, [https://www.gov.uk/government/publications/iran-country-policy-and-information-notes/country-policy-and-information-note-social-media-surveillance-and-sur-place-activities-iran-april-2025-accessible](https://www.gov.uk/government/publications/iran-country-policy-and-information-notes/country-policy-and-information-note-social-media-surveillance-and-sur-place-activities-iran-april-2025-accessible)
53. Iranian Twitter Influencer (by number of Followers) - ResearchGate, accessed May 25, 2025, [https://www.researchgate.net/figure/ranian-Twitter-Influencer-by-number-of-Followers_fig1_342889484](https://www.researchgate.net/figure/ranian-Twitter-Influencer-by-number-of-Followers_fig1_342889484)
54. Iran, Twitter, and Business: 16 People to Follow - Bourse & Bazaar Foundation, accessed May 25, 2025, [https://www.bourseandbazaar.org/articles/2015/3/8/iran-twitter-and-business-16-people-to-follow](https://www.bourseandbazaar.org/articles/2015/3/8/iran-twitter-and-business-16-people-to-follow)
55. Political Polarization Mechanics on Persian Twitter (X): A Social Network Analysis of the 2024 Iranian Presidential Election - Journal of Cyberspace Studies, accessed May 25, 2025, [https://jcss.ut.ac.ir/article_100571.html](https://jcss.ut.ac.ir/article_100571.html)
56. Exchange rate of dollar Cash to Iranian Rials, USD to IRR | Alanchand, accessed May 25, 2025, [https://alanchand.com/en/currencies-price/usd](https://alanchand.com/en/currencies-price/usd)
57. Iranian rial to US dollars Exchange Rate History | Currency Converter - Wise, accessed May 25, 2025, [https://wise.com/in/currency-converter/irr-to-usd-rate/history](https://wise.com/in/currency-converter/irr-to-usd-rate/history)
58. 1 USD to IRR - US Dollars to Iranian Rials Exchange Rate - XE.com, accessed May 25, 2025, [https://www.xe.com/en-us/currencyconverter/convert/?Amount=1&From=USD&To=IRR](https://www.xe.com/en-us/currencyconverter/convert/?Amount=1&From=USD&To=IRR)
59. obada-jaras/Arabic-Tweets-Scraper: This project is a python script that utilizes the snscrape library and multithreading to efficiently scrape Arabic tweets from Twitter. The use of multithreading improves the performance of the script, enabling the collection of a larger number of tweets. - GitHub, accessed May 25, 2025, [https://github.com/obada-jaras/Arabic-Tweets-Scraper](https://github.com/obada-jaras/Arabic-Tweets-Scraper)
60. X (Twitter) - Recent Posts Data Collector - Communalytic - A no-code computational social science research tool for studying online communities and public discourse on social media, accessed May 25, 2025, [https://communalytic.org/docs/x-recent-posts-data-collector/](https://communalytic.org/docs/x-recent-posts-data-collector/)
61. X Provides Premium Perks to Hezbollah, Other U.S.-Sanctioned Groups - TTP, accessed May 25, 2025, [https://www.techtransparencyproject.org/articles/x-provides-premium-perks-to-hezbollah-other-us-sanctioned-groups](https://www.techtransparencyproject.org/articles/x-provides-premium-perks-to-hezbollah-other-us-sanctioned-groups)
62. International Dignitaries Rally for a New Iran Policy, Support NCRI as Democratic Alternative, accessed May 25, 2025, [https://www.ncr-iran.org/en/news/iran-resistance/international-dignitaries-rally-for-a-new-iran-policy-support-ncri-as-democratic-alternative/](https://www.ncr-iran.org/en/news/iran-resistance/international-dignitaries-rally-for-a-new-iran-policy-support-ncri-as-democratic-alternative/)
63. Crude Oil - Price - Chart - Historical Data - News - Trading Economics, accessed May 25, 2025, [https://tradingeconomics.com/commodity/crude-oil](https://tradingeconomics.com/commodity/crude-oil)
64. Economic Diplomacy Deputy Ministry of Foreign Affairs of the I.R. Iran ::, accessed May 25, 2025, [https://en-economic.mfa.ir/](https://en-economic.mfa.ir/)
65. Iran Data Portal, accessed May 25, 2025, [https://irandataportal.syr.edu/](https://irandataportal.syr.edu/)
66. Is there a language detection that detects Arabic and Persian languages? - Stack Overflow, accessed May 25, 2025, [https://stackoverflow.com/questions/76408247/is-there-a-language-detection-that-detects-arabic-and-persian-languages](https://stackoverflow.com/questions/76408247/is-there-a-language-detection-that-detects-arabic-and-persian-languages)
67. Fine-Tuning Your First Large Language Model (LLM) with PyTorch and Hugging Face, accessed May 25, 2025, [https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face](https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face)
68. What is few shot prompting? - IBM, accessed May 25, 2025, [https://www.ibm.com/think/topics/few-shot-prompting](https://www.ibm.com/think/topics/few-shot-prompting)
69. The Few Shot Prompting Guide - PromptHub, accessed May 25, 2025, [https://www.prompthub.us/blog/the-few-shot-prompting-guide](https://www.prompthub.us/blog/the-few-shot-prompting-guide)
70. Can LLMs Learn Macroeconomic Narratives from Social Media? - arXiv, accessed May 25, 2025, [https://arxiv.org/html/2406.12109v2](https://arxiv.org/html/2406.12109v2)
71. arXiv:2406.10811v1 [cs.CL] 16 Jun 2024, accessed May 25, 2025, [https://arxiv.org/pdf/2406.10811?](https://arxiv.org/pdf/2406.10811)
72. What is chain of thought (CoT) prompting? - IBM, accessed May 25, 2025, [https://www.ibm.com/think/topics/chain-of-thoughts](https://www.ibm.com/think/topics/chain-of-thoughts)
73. How to Craft Prompts for Different Large Language Models Tasks - phData, accessed May 25, 2025, [https://www.phdata.io/blog/how-to-craft-prompts-for-different-large-language-models-tasks/](https://www.phdata.io/blog/how-to-craft-prompts-for-different-large-language-models-tasks/)
74. Prompt Engineering for Content Creation - PromptHub, accessed May 25, 2025, [https://www.prompthub.us/blog/prompt-engineering-for-content-creation](https://www.prompthub.us/blog/prompt-engineering-for-content-creation)
75. Unmasking Digital Falsehoods: A Comparative Analysis of LLM-Based Misinformation Detection Strategies ∗ Corresponding author: tianyihuang@berkeley.edu - arXiv, accessed May 25, 2025, [https://arxiv.org/html/2503.00724v1](https://arxiv.org/html/2503.00724v1)
76. Large language models can consistently generate high-quality content for election disinformation operations - PubMed Central, accessed May 25, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11913289/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11913289/)
77. Multi-agent systems powered by large language models: applications in swarm intelligence, accessed May 25, 2025, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1593017/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1593017/full)
78. Promptware Engineering: Software Engineering for LLM Prompt Development - arXiv, accessed May 25, 2025, [https://arxiv.org/html/2503.02400v1](https://arxiv.org/html/2503.02400v1)
79. Tutorial | Prompt engineering with LLMs - Dataiku Knowledge Base, accessed May 25, 2025, [https://knowledge.dataiku.com/latest/gen-ai/text-processing/tutorial-prompt-engineering.html](https://knowledge.dataiku.com/latest/gen-ai/text-processing/tutorial-prompt-engineering.html)
80. Understanding the Effects of Iterative Prompting on Truthfulness - arXiv, accessed May 25, 2025, [https://arxiv.org/pdf/2402.06625](https://arxiv.org/pdf/2402.06625)
81. Midyear Roundup: Nation-State Cyber Threats in 2025 - GovTech, accessed May 25, 2025, [https://www.govtech.com/blogs/lohrmann-on-cybersecurity/midyear-roundup-nation-state-cyber-threats-in-2025](https://www.govtech.com/blogs/lohrmann-on-cybersecurity/midyear-roundup-nation-state-cyber-threats-in-2025)
82. برترین کانال‌ها و پیج های اخبار اقتصادی در شبکه‌های اجتماعی | بلاگ دیتاک, accessed May 25, 2025, [https://dataak.com/blog/%D8%A8%D8%B1%D8%AA%D8%B1%DB%8C%D9%86-%D8%B1%D8%B3%D8%A7%D9%86%D9%87_%D9%87%D8%A7%DB%8C_%D8%A7%D8%AE%D8%A8%D8%A7%D8%B1_%D8%A7%D9%82%D8%AA%D8%B5%D8%A7%D8%AF%DB%8C_%DB%B9%DB%B9/](https://dataak.com/blog/%D8%A8%D8%B1%D8%AA%D8%B1%DB%8C%D9%86-%D8%B1%D8%B3%D8%A7%D9%86%D9%87_%D9%87%D8%A7%DB%8C_%D8%A7%D8%AE%D8%A8%D8%A7%D8%B1_%D8%A7%D9%82%D8%AA%D8%B5%D8%A7%D8%AF%DB%8C_%DB%B9%DB%B9/)
83. ۱۰۰ توییتری تاثیرگذار در توییتر ایرانی‌ها - IranWire, accessed May 25, 2025, [https://iranwire.com/fa/features/55390/](https://iranwire.com/fa/features/55390/)
84. Persian-Twitter-Analysis/PersianTwitterDataset - GitHub, accessed May 25, 2025, [https://github.com/Persian-Twitter-Analysis/PersianTwitterDataset](https://github.com/Persian-Twitter-Analysis/PersianTwitterDataset)
85. Prompt Engineering Best Practices You Should Know For Any LLM | Astera, accessed May 25, 2025, [https://www.astera.com/type/blog/prompt-engineering-best-practices/](https://www.astera.com/type/blog/prompt-engineering-best-practices/)