# DeepSeekMath 7B Full Technical Report

---

### DeepSeekMath 7B — Purpose & Full Technical Report

---

### 1 Why the paper exists

- **Gap the authors saw:** open-source LLMs lag far behind GPT-4/Gemini-Ultra on *mathematical* reasoning benchmarks.
- **Goal:** show that (i) carefully-mined web math data and (ii) a light-weight RL recipe can push a *7-billion-parameter* model to within striking distance of the best closed systems, all while remaining fully open-weight. citeturn0search0

---

### 2 What the authors built

| Stage | Model name | Key idea | Tokens / data |
| --- | --- | --- | --- |
| **Continual pre-training** | **DeepSeekMath-Base 7B** | Start from the code-oriented DeepSeek-Coder-Base-v1.5 7B; continue training on a *math-centric* corpus | 500 B tokens total (56 % = 120 B “DeepSeekMath Corpus” math tokens; 20 % GitHub code; 10 % arXiv; 10 % CC NL) citeturn0search0 |
| **SFT (instruction tuning)** | **DeepSeekMath-Instruct 7B** | 776 K chain-of-thought, program-of-thought & tool-integrated examples (EN + ZH) |  |
| **Reinforcement learning** | **DeepSeekMath-RL 7B** | *Group Relative Policy Optimisation (GRPO)* – a PPO variant that: ➊ samples a *group* of 64 answers per prompt, ➋ uses the **group’s mean score as the baseline** (no value head), ➌ adds a KL penalty to the loss instead of the reward. This removes a second model and almost halves memory use. citeturn0search1 |  |

---

### 3 Building the DeepSeekMath Corpus

1. **Seed** = OpenWebMath pages.
2. **fastText classifier** trained on seed vs. random CC negatives.
3. **Iterate ×4:** classify CC ⇒ pick top-scoring pages ⇒ manually label high-yield domains/URL patterns ⇒ retrain the classifier.
4. **De-contaminate** by exact 10-gram overlap filter against GSM8K, MATH, CMATH, etc.
    
    *Result*: 35.5 M webpages ≈ 120 B tokens in 40+ languages (EN & ZH dominate). citeturn0search0
    

---

### 4 Key empirical findings

| Benchmark (chain-of-thought, no tools) | DeepSeekMath-Base 7B | DeepSeekMath-Instruct 7B | DeepSeekMath-RL 7B |
| --- | --- | --- | --- |
| **GSM8K** | 64.2 % | 82.9 % | **88.2 %** |
| **MATH (comp-level)** | 36.2 % | 46.8 % | **51.7 %** |
| **CMATH (ZH)** | 71.7 % | 84.6 % | **88.8 %** |

With Python execution (PoT/Tool-integrated), DeepSeekMath-RL hits 58.8 % on MATH — the first open model above 50 %. citeturn0search1

---

### 5 Ablation & analysis highlights

*Code pre-training matters*

- Doing **400 B code tokens ➜ 150 B math** lifted both symbolic and program-aided reasoning more than “400 B general ➜ math”.

*ArXiv papers disappoint*

- Replacing web math with arXiv LaTeX (MathPile or RedPajama-arXiv) gave *no* gain and sometimes regression on GSM8K/MATH.

*Why RL helps*

- RL **doesn’t raise Pass@K** (raw solution rate) but *does* raise Maj@K: it *re-orders* the generation distribution so correct answers float to the top, fixing “preference mis-alignment.”

---

### 6 Contributions in context

| Contribution | Why it matters |
| --- | --- |
| **Largest open math-specific corpus mined from CC (120 B tokens)** | Shows web data > arXiv for math; pipeline reusable for other domains. |
| **GRPO algorithm** | Removes the value head → ½ GPU RAM vs. PPO; still beats PPO-like baselines. |
| **State-of-the-art open 7 B maths model** | 51.7 % MATH rivals GPT-3.5/Gemini-Pro class models. |
| **Unified view of SFT / RFT / DPO / PPO / GRPO** | Helps community reason about alignment methods as points in one framework. |

---

### 7 Limitations noted by the authors

- Geometry & formal proof remain weaker than GPT-4.
- Few-shot generalisation is limited (model is small).
- RL used only GSM8K + MATH questions; broader reward data could help.

---

### 8 Future directions (paper & authors)

1. Broaden the corpus beyond quantitative problems (geometry diagrams, theorem­-proof pairs).
2. Explore *weak-to-strong* RL algorithms robust to noisy rewards.
3. Merge data selection with advanced decoding (tree search) to push Pass@K, not just Maj@K.

---

### 9 Practical take-aways

- **If you need a compact, permissive, tool-ready math model, DeepSeekMath-RL 7B is currently the leader.**
- **Data quality ≫ parameter count** for niche skills: a 7 B model + 120 B focused tokens can match 540 B Minerva on MATH.
- **RLHF can be cheap:** you can ditch the critic if you use clever baselines (GRPO).

---

**Abstract**:

Mathematical reasoning in language models has been a long-standing challenge due to its inherent complexity and structured nature. The introduction of DeepSeekMath 7B represents a significant step in this area, as it leverages a meticulously curated data selection pipeline, combined with advanced optimization techniques like Group Relative Policy Optimization (GRPO), to push the limits of mathematical reasoning in open-source language models. DeepSeekMath 7B achieves an impressive performance on the MATH benchmark, approaching the capabilities of closed-source models like GPT-4 and Gemini-Ultra, without relying on external toolkits or voting methods.

---

### **1. Introduction**

Mathematical reasoning has been a critical benchmark for evaluating the effectiveness of language models (LMs). Despite advancements in natural language understanding, LMs have struggled with the inherent structure and precision required in mathematical problem-solving. While models such as GPT-4 and Gemini-Ultra have achieved state-of-the-art results in mathematical reasoning tasks, they are not publicly available, and the open-source models remain significantly behind in performance.

**DeepSeekMath 7B** is an open-source, domain-specific model designed to overcome these challenges by integrating mathematical data and optimizing reasoning capabilities. Built upon DeepSeek-Coder-Base-v1.5, DeepSeekMath 7B uses a large-scale, math-specific corpus derived from 120 billion math-related tokens sourced from Common Crawl and enhanced by human annotation. This model outperforms other open-source models and achieves competitive performance on academic benchmarks.

---

### **2. Data Selection and Pre-Training**

The foundation of DeepSeekMath’s capabilities is the **DeepSeekMath Corpus**, a large-scale dataset of math-related tokens, totalling 120 billion tokens. The data selection process uses a **fastText-based classifier**, which first trains on OpenWebMath (a math-specific dataset) to identify positive examples. Negative examples are gathered from a diverse set of web pages, ensuring the model learns to distinguish relevant math content from non-relevant content.

The curated dataset is further refined through human annotation, which improves the quality and relevance of the training data. This process results in a high-quality pre-training corpus that is multilingual, contributing to the improved performance of DeepSeekMath on both English and Chinese mathematical benchmarks.

---

### **3. Model Architecture and Optimization**

DeepSeekMath is initialized with **DeepSeek-Coder-Base-v1.5 7B**, a model originally trained for code understanding. The decision to start from a code-based model was informed by the observation that models trained on code data exhibit strong performance on structured tasks like mathematical reasoning.

Once initialized, DeepSeekMath undergoes **mathematical instruction tuning** with methods like **chain-of-thought (CoT)** reasoning, **program-of-thought (PoT)**, and **tool-integrated reasoning (TIR)**. These techniques improve the model's ability to reason about mathematical problems, helping it break down complex tasks into manageable steps.

The result is **DeepSeekMath-Instruct 7B**, a model that outperforms other 7B models and competes with 70B instruction-tuned models. This is a significant achievement, demonstrating the power of mathematical-specific training and instruction-based fine-tuning.

---

### **4. Group Relative Policy Optimization (GRPO)**

One of the key innovations in DeepSeekMath is the introduction of **Group Relative Policy Optimization (GRPO)**, a variant of **Proximal Policy Optimization (PPO)**. GRPO enhances the model’s mathematical reasoning abilities while optimizing memory usage, making it more efficient during training.

Unlike traditional PPO, which uses a critic model to estimate the baseline, GRPO estimates the baseline from group scores. This change leads to reduced training resource requirements while still yielding substantial performance improvements.

During the reinforcement learning (RL) phase, GRPO achieves impressive results, particularly on in-domain benchmarks like **GSM8K** (82.9% → 88.2%) and **MATH** (46.8% → 51.7%), as well as out-of-domain tasks like **CMATH** (84.6% → 88.8%). This demonstrates the power of GRPO in improving both mathematical reasoning and the general performance of instruction-tuned models.

---

### **5. Evaluation and Benchmarking**

DeepSeekMath 7B achieves remarkable results on various mathematical reasoning benchmarks:

- **GSM8K**: 82.9% → 88.2%
- **MATH**: 46.8% → 51.7%
- **CMATH**: 84.6% → 88.8%

These scores reflect a strong improvement in mathematical reasoning capabilities, bringing DeepSeekMath closer to closed-source models like **GPT-4** and **Gemini-Ultra**, which perform exceptionally well on similar benchmarks.

The model’s ability to achieve competitive results without relying on external toolkits or voting methods is a major achievement, further solidifying its place in the open-source AI community.

---

### **6. Results and Conclusions**

DeepSeekMath 7B has demonstrated its capability to handle complex mathematical reasoning tasks, outperforming other open-source models and approaching the performance of leading closed-source models like GPT-4. This was achieved through a combination of carefully curated training data, mathematical instruction tuning, and the novel use of GRPO.

The results on benchmarks like **MATH**, **GSM8K**, and **CMATH** show significant improvements, particularly when compared to other open-source models in the 7B category. The use of mathematical instruction tuning and reinforcement learning methods like GRPO has proven effective in enhancing the model’s reasoning abilities.

Moreover, the multilingual nature of DeepSeekMath and its ability to perform well on both English and Chinese benchmarks suggest that this model has broad applicability in mathematical reasoning tasks across different languages.

In conclusion, DeepSeekMath represents a significant step forward in the development of mathematical reasoning capabilities in language models. Its performance on academic benchmarks and the introduction of GRPO as an optimization technique provide a promising foundation for future advancements in AI-driven mathematical reasoning.

---

### **7. Future Directions**

While DeepSeekMath has made substantial progress, there is still room for improvement in mathematical reasoning models. Future work could focus on further optimizing the GRPO algorithm, exploring additional training techniques like **Rejection Sampling Fine-Tuning (RFT)** and **Direct Preference Optimization (DPO)**, and expanding the multilingual capabilities of the model.

As the research community continues to develop more advanced techniques and datasets, DeepSeekMath sets the stage for further breakthroughs in AI-driven mathematical reasoning and problem-solving.

---

**1.1. Contributions**

In this section, we outline the key contributions made by the DeepSeekMath team in the areas of **math pre-training at scale** and the **exploration and analysis of reinforcement learning** techniques.

---

### **Math Pre-Training at Scale**

1. **Building the DeepSeekMath Corpus**:
    
    One of the primary contributions of our research is the **DeepSeekMath Corpus**, a high-quality dataset consisting of 120 billion tokens of mathematical content sourced from web pages filtered for math-related data. This dataset is **nearly seven times larger** than the math-specific web pages used in the **Minerva** model (Lewkowycz et al., 2022a) and **nine times the size** of the recently released **OpenWebMath** (Paster et al., 2023). By implementing a carefully designed data selection pipeline, we were able to extract valuable mathematical information from publicly available data, primarily sourced from **Common Crawl**.
    
2. **Achieving Strong Performance with Smaller Models**:
    
    Our pre-trained model, **DeepSeekMath-Base 7B**, which contains fewer parameters than other large-scale models like **Minerva 540B**, achieves comparable performance. This demonstrates that **the number of parameters is not the only determinant** of a model's mathematical reasoning capability. With high-quality data, even smaller models can exhibit strong performance, providing valuable insight into efficient model scaling.
    
3. **Impact of Code Training**:
    
    Another significant finding from our research is that **code training prior to math training** improves a model's ability to solve mathematical problems. Our experiments show that training models on code-related tasks before tackling math-related problems enhances their reasoning abilities, both in scenarios where external tools are used and those where they are not. This partially addresses the long-standing question of whether training on code data can improve reasoning skills, at least in the context of mathematical reasoning.
    
4. **Limitations of Training on ArXiv Papers**:
    
    Although many research models incorporate data from **arXiv** papers, especially in the mathematical domain, we found that **training on arXiv papers did not yield significant improvements** on the mathematical benchmarks used in our study. This finding challenges the common assumption that data from academic papers, like those found on arXiv, is always beneficial for enhancing model performance in mathematical tasks.
    

---

### **Exploration and Analysis of Reinforcement Learning**

1. **Introduction of Group Relative Policy Optimization (GRPO)**:
    
    One of the major innovations in our research is the introduction of **Group Relative Policy Optimization (GRPO)**, a more efficient reinforcement learning algorithm. GRPO eliminates the need for a critic model, which is commonly used in **Proximal Policy Optimization (PPO)**. Instead, GRPO estimates the baseline using **group scores**, significantly reducing the computational resources required for training. This makes GRPO a cost-effective solution for reinforcement learning tasks, particularly in the context of large-scale language models.
    
2. **Performance Enhancements through GRPO**:
    
    Our experiments demonstrate that **GRPO significantly improves the performance** of our **instruction-tuned model, DeepSeekMath-Instruct**. By using only instruction-tuning data, GRPO enhances both in-domain and out-of-domain performance. This improvement is especially notable in mathematical benchmarks, where the use of GRPO leads to better outcomes compared to traditional training methods.
    
3. **Unified Paradigm for Reinforcement Learning Techniques**:
    
    We propose a **unified paradigm** for understanding and comparing different reinforcement learning methods, including **Rejection Sampling Fine-Tuning (RFT)**, **Direct Preference Optimization (DPO)**, **PPO**, and **GRPO**. Our research includes extensive experiments comparing techniques such as **online vs. offline training**, **outcome vs. process supervision**, and **single-turn vs. iterative reinforcement learning**. This helps us gain deeper insights into the essential components of reinforcement learning and its impact on model performance.
    
4. **Understanding the Effectiveness of Reinforcement Learning**:
    
    Finally, our research delves into the **reasons behind the effectiveness** of reinforcement learning in improving model performance. By using our unified paradigm, we explore potential directions for achieving more effective reinforcement learning in large language models (LLMs). This includes optimizing training procedures and exploring alternative reinforcement learning strategies to further enhance model performance in mathematical reasoning tasks.
    

---

---

### **Mathematical Reasoning in English and Chinese**

1. **English Benchmarks**:
    - **GSM8K** (Cobbe et al., 2021)
    - **MATH** (Hendrycks et al., 2021)
    - **SAT** (Azerbayev et al., 2023)
    - **OCW Courses** (Lewkowycz et al., 2022a)
    - **MMLU-STEM** (Hendrycks et al., 2020)
    
    **Chinese Benchmarks**:
    
    - **MGSM-zh** (Shi et al., 2023)
    - **CMATH** (Wei et al., 2023)
    - **Gaokao-MathCloze** (Zhong et al., 2023)
    - **Gaokao-MathQA** (Zhong et al., 2023)
    
    **Evaluation Details**:
    
    DeepSeekMath-Base was assessed on both English and Chinese mathematical reasoning tasks, which cover a range of problems from **grade-school level to college level**. The evaluation included:
    
    - **Self-contained text solutions** without tool use
    - Ability to **solve problems using Python**
    
    **Results**:
    
    - **DeepSeekMath-Base** showed competitive performance on **English benchmarks**, performing similarly to the **closed-source Minerva 540B** model (Lewkowycz et al., 2022a), and surpassing other **open-source models** (e.g., **Mistral 7B** (Jiang et al., 2023) and **Llemma-34B** (Azerbayev et al., 2023)) in most cases, regardless of whether those models had undergone math pre-training.
    - **DeepSeekMath-Base** outperformed **all open-source base models**, even when they were specifically pre-trained on math data, often by a significant margin.
    - On **Chinese benchmarks**, DeepSeekMath-Base **outperformed** models trained exclusively on English math data, likely due to the inclusion of **high-quality non-English math data** in the pre-training process. This highlights the importance of incorporating diverse linguistic resources in training.
    
    **DeepSeekMath-Instruct and DeepSeekMath-RL**:
    
    - With **mathematical instruction tuning** and **reinforcement learning (RL)**, these models achieved **over 50% accuracy** on the **competition-level MATH dataset**, setting a new benchmark within the open-source community.

---

### **Formal Mathematics Evaluation**

1. **Task**: Informal-to-formal theorem proving task
    - **Dataset**: **miniF2F** (Zheng et al., 2021)
    - **Tool**: **Isabelle** (Wenzel et al., 2008) as the proof assistant.
    
    **Results**:
    
    **DeepSeekMath-Base** demonstrated strong **few-shot autoformalization** performance, which indicates its ability to automatically convert informal problem descriptions into formal mathematical proofs. This capability is particularly important for models that aim to assist in theorem proving and formal verification tasks.
    

---

### **Natural Language Understanding, Reasoning, and Code Evaluation**

1. **Massive Multitask Language Understanding (MMLU)**:
    - The **MMLU** benchmark (Hendrycks et al., 2020) covers **57 multiple-choice tasks**, spanning a wide range of subjects. This provides an overall assessment of the model's general understanding and reasoning capabilities.
2. **BIG-Bench Hard (BBH)**:
    - **BBH** (Suzgun et al., 2022) consists of **23 challenging tasks** that typically require **multi-step reasoning**. This benchmark is useful for testing a model’s ability to perform complex reasoning tasks across various domains.
3. **Code Language Model Evaluation**:
    - **HumanEval** (Chen et al., 2021) and **MBPP** (Austin et al., 2021) are widely used to evaluate the performance of code language models, focusing on their ability to solve programming challenges.
    
    **Results**:
    
    - **Math pre-training** significantly improved the **language understanding** and **reasoning** performance of **DeepSeekMath-Base** on the **MMLU** and **BBH** benchmarks. This demonstrates that the training on mathematical tasks enhances the model’s general problem-solving abilities, not just its mathematical reasoning.
    - Additionally, math pre-training also improved the model’s performance on **code-related tasks** such as **HumanEval** and **MBPP**, showcasing the broader benefits of math-based training for models in coding and programming-related domains.

---

### **Summary of Key Results**

- **DeepSeekMath-Base**:
    - **English benchmarks**: Competitive with closed-source models like Minerva 540B, outperforming open-source models.
    - **Chinese benchmarks**: Outperformed models trained exclusively on English data, due to the inclusion of diverse linguistic data in training.
    - **Formal mathematics**: Strong performance on informal-to-formal theorem proving tasks.
    - **General understanding**: Significant improvements in natural language understanding and multi-step reasoning tasks after math pre-training.
    - **Code performance**: Enhanced ability to handle programming challenges.
- **DeepSeekMath-Instruct and DeepSeekMath-RL**:
    - Achieved **over 50% accuracy** on the MATH benchmark, setting new open-source records for competition-level math problem-solving.

These evaluations highlight DeepSeekMath’s ability to handle a wide range of tasks, from mathematical reasoning to formal theorem proving and code generation, making it a versatile and powerful model in the open-source AI landscape.

### **2. Math Pre-Training**

**2.1. Data Collection and Decontamination**

In this section, we describe the process of building the **DeepSeekMath Corpus** by collecting mathematical web data from **Common Crawl** using an iterative pipeline. This method ensures the gathering of high-quality, math-related content from a large web corpus, with particular attention to decontamination to prevent the inclusion of any benchmark data.

---

### **Data Collection Pipeline Overview**

The pipeline for constructing the **DeepSeekMath Corpus** consists of several steps, as depicted in **Figure 2**:

1. **Seed Corpus Selection**:
    
    We begin by selecting **OpenWebMath** (Paster et al., 2023), a curated collection of high-quality math-related texts from the web. This seed corpus serves as the basis for training our data-recall model.
    
2. **Training a fastText Model**:
    
    Using the OpenWebMath seed corpus, we train a **fastText model** (Joulin et al., 2016), a text classification algorithm, to identify additional math-related web pages. To train the model, we randomly select:
    
    - **500,000 data points** from OpenWebMath as **positive examples**.
    - **500,000 web pages** from Common Crawl as **negative examples**.
    We configure the fastText model with the following parameters:
    - **Vector dimension**: 256
    - **Learning rate**: 0.1
    - **Maximum n-gram length**: 3
    - **Minimum number of word occurrences**: 3
    - **Number of training epochs**: 3
3. **Recall Math-Related Webpages**:
    
    After training the fastText model, we use it to recall math-related web pages from **deduplicated Common Crawl**. The dataset is first cleaned using **URL-based deduplication** and **near-deduplication techniques**, which reduces the size of the original Common Crawl to **40 billion HTML pages**. The fastText model ranks the collected web pages by their likelihood of being math-related, and only the **top-ranking pages** are retained.
    
4. **Data Preservation and Assessment**:
    
    To determine the optimal size of the dataset, we assess different token volumes (**40B, 80B, 120B, and 160B tokens**) through pre-training experiments. In the initial iteration, we retain the top **40B tokens** for further processing.
    

---

### **Iteration Process for Data Enhancement**

The first iteration of data collection provides a significant quantity of math-related content, but several math-related web pages remain uncollected, mainly due to the lack of diversity in the positive training examples. To address this, we take the following steps:

1. **Identifying Additional Math Sources**:
    
    We expand the **seed corpus** by identifying additional math-related domains. A **domain** is defined as a set of web pages that share the same base URL. We organize the entire **Common Crawl** into disjoint domains and calculate the percentage of pages from each domain that were collected in the first iteration.
    
2. **Domain Classification**:
    
    Domains where **more than 10%** of the pages have been collected are classified as **math-related** (e.g., **mathoverflow.net**). We then manually annotate the URLs associated with math-related content (e.g., **mathoverflow.net/questions**) within these identified domains.
    
3. **Expanding the Seed Corpus**:
    
    We add uncollected web pages linked to these annotated URLs to the seed corpus. This ensures that we increase the diversity of our training examples for the fastText model.
    
4. **Subsequent Iterations**:
    
    After each iteration, we retrain the fastText model to improve its ability to recall more relevant mathematical data. By performing this iterative process, we eventually expand the math corpus to **35.5 million math-related web pages**, totaling **120 billion tokens**.
    

---

### **Data Collection Termination**

In the **fourth iteration**, we observed that nearly **98% of the desired math-related data** had already been collected in the previous iteration. Therefore, we decided to cease further data collection, as continuing would yield diminishing returns.

---

### **Decontamination Process**

To ensure that the **DeepSeekMath Corpus** remains free from benchmark contamination, we carefully filter out any web pages that contain content from well-known **English mathematical benchmarks** such as **GSM8K** (Cobbe et al., 2021) and **MATH** (Hendrycks et al., 2021), as well as **Chinese benchmarks** like **CMATH** (Wei et al., 2023) and **AGIEval** (Zhong et al., 2023). The contamination filtering steps are as follows:

1. **10-Gram String Matching**:
    
    Any text segment in the corpus that contains a **10-gram string** (a sequence of 10 consecutive words) that exactly matches a substring from any evaluation benchmark is **removed**.
    
2. **Partial Matching for Shorter Texts**:
    
    For texts shorter than 10 grams but with at least 3 grams of overlap, we perform **exact matching** to filter out contaminated web pages. This ensures that no questions or answers from the benchmark datasets are included in the training data.
    

By following these decontamination methods, we guarantee that the **DeepSeekMath Corpus** remains clean and free from any direct overlap with the benchmark datasets, ensuring the integrity of the training process.

![Screenshot 2025-04-24 at 10.34.49 PM.png](DeepSeekMath%207B%20Full%20Technical%20Report%201df9381f908c80dfa20cd8a7896b73c4/Screenshot_2025-04-24_at_10.34.49_PM.png)

---

### **2. Math Pre-Training**

**2.1. Data Collection and Decontamination**

In this section, we provide a detailed overview of how the **DeepSeekMath Corpus** was constructed using data from **Common Crawl**. This process involves a **systematic pipeline** for collecting a large-scale mathematical corpus, beginning with a seed corpus and extending through multiple iterations to gather and refine the data. The approach used here can be generalized to other domains as well, such as coding.

---

### **Pipeline Overview**

The process of building the **DeepSeekMath Corpus** consists of several stages, shown in **Figure 2**. The pipeline is iterative and involves:

1. **Seed Corpus Selection**:
We begin by choosing **OpenWebMath** (Paster et al., 2023) as our initial seed corpus. This collection includes high-quality math-related texts from the web and serves as a starting point for gathering more mathematical data.
2. **Training the fastText Model**:
To identify additional math-related web pages, we train a **fastText model** (Joulin et al., 2016) using the OpenWebMath corpus. Specifically:
    - **500,000 data points** are randomly selected from the seed corpus to serve as **positive training examples**.
    - **500,000 web pages** are chosen from **Common Crawl** as **negative examples**.
    The model is trained with the following settings:
    - **Vector dimension**: 256
    - **Learning rate**: 0.1
    - **Maximum n-gram length**: 3
    - **Minimum number of word occurrences**: 3
    - **Number of training epochs**: 3
3. **Recalling Math-Related Web Pages**:
After training the fastText model, we use it to **recall math-related web pages** from **deduplicated Common Crawl**. To reduce the size of the original Common Crawl dataset, we employ **URL-based deduplication** and **near-deduplication techniques**, resulting in a refined set of **40 billion HTML web pages**. These pages are ranked based on their likelihood of containing mathematical content, and only the **top-ranking pages** are retained for further processing.
4. **Data Volume and Pre-Training Experimentation**:
The volume of data retained is evaluated through pre-training experiments. We assess different token volumes by retaining the top **40B, 80B, 120B, and 160B tokens**. In the first iteration, we decide to keep the **top 40 billion tokens** for further refinement.
    
    ![Screenshot 2025-04-24 at 10.35.45 PM.png](DeepSeekMath%207B%20Full%20Technical%20Report%201df9381f908c80dfa20cd8a7896b73c4/Screenshot_2025-04-24_at_10.35.45_PM.png)
    

---

### **Iteration Process for Data Refinement**

Despite the initial success, many mathematical web pages remain uncollected in the first iteration, primarily due to the insufficient diversity of the positive training examples. To address this issue, we take the following steps:

1. **Identifying Additional Math Sources**:
We broaden our seed corpus by identifying additional **math-related domains**. A **domain** is defined as a set of web pages sharing the same base URL (e.g., **mathoverflow.net**). The entire **Common Crawl** dataset is segmented into disjoint domains. For each domain, we calculate the percentage of web pages that were collected in the first iteration.
2. **Classifying Math-Related Domains**:
Domains where **more than 10%** of the web pages were collected in the first iteration are classified as **math-related**. For example, **mathoverflow.net** is classified as math-related due to its focus on mathematical discussions. We then manually annotate URLs associated with mathematical content within these domains (e.g., **mathoverflow.net/questions**).
3. **Expanding the Seed Corpus**:
We add uncollected web pages linked to these newly annotated URLs to the seed corpus. This enables us to gather a more diverse set of positive examples, improving the training of the fastText model in subsequent iterations.
4. **Subsequent Iterations**:
After each iteration, we **retrain the fastText model**, which becomes better at recalling relevant mathematical content. Over four iterations, we collect **35.5 million math-related web pages**, totaling **120 billion tokens**. By the fourth iteration, nearly **98%** of the target data has already been collected, so we cease further data collection.

---

![Screenshot 2025-04-24 at 10.35.58 PM.png](DeepSeekMath%207B%20Full%20Technical%20Report%201df9381f908c80dfa20cd8a7896b73c4/Screenshot_2025-04-24_at_10.35.58_PM.png)

### **Decontamination Process**

To maintain the integrity of the **DeepSeekMath Corpus** and avoid **benchmark contamination**, we apply stringent filtering procedures to remove any web pages that contain content from widely used mathematical benchmarks. This ensures that our training data does not overlap with test data used to evaluate model performance.

The contamination filtering criteria are as follows:

1. **10-Gram Exact Matching**:
We filter out any text segment in the corpus that contains a **10-gram string** (a sequence of 10 consecutive words) that matches exactly with any part of the text from the **benchmark datasets**. This is done to ensure that no questions or answers from benchmarks like **GSM8K** (Cobbe et al., 2021) and **MATH** (Hendrycks et al., 2021) are included.
2. **Partial Matching for Shorter Texts**:
For benchmark texts that are **shorter than 10 grams**, but overlap by at least **3 grams**, we employ **exact matching** to filter out these contaminated web pages. This step is critical for removing shorter benchmark segments that may still be present in the training data.
    
    ![Screenshot 2025-04-24 at 10.36.11 PM.png](DeepSeekMath%207B%20Full%20Technical%20Report%201df9381f908c80dfa20cd8a7896b73c4/Screenshot_2025-04-24_at_10.36.11_PM.png)
    

---

![Screenshot 2025-04-24 at 10.36.24 PM.png](DeepSeekMath%207B%20Full%20Technical%20Report%201df9381f908c80dfa20cd8a7896b73c4/Screenshot_2025-04-24_at_10.36.24_PM.png)

![Screenshot 2025-04-24 at 10.36.35 PM.png](DeepSeekMath%207B%20Full%20Technical%20Report%201df9381f908c80dfa20cd8a7896b73c4/Screenshot_2025-04-24_at_10.36.35_PM.png)

### **Final Corpus**

After completing the iterative data collection and decontamination steps, the final **DeepSeekMath Corpus** contains:

- **35.5 million math-related web pages**
- **120 billion tokens**

This high-quality, diverse, and benchmark-free dataset will be used for pre-training **DeepSeekMath models** and serves as a foundation for future advancements in mathematical reasoning tasks.

---

### **3. Supervised Fine-Tuning (SFT)**

**3.1. SFT Data Curation**

In this section, we describe the process of curating a high-quality **mathematical instruction-tuning dataset** for **DeepSeekMath-Instruct 7B**, which covers a wide range of mathematical problems and solutions across **English** and **Chinese** languages. The dataset is designed to incorporate **chain-of-thought (CoT)**, **program-of-thought (PoT)**, and **tool-integrated reasoning** formats, providing a diverse set of training examples for various levels of mathematical complexity.

---

### **English Mathematical Datasets**

For the **English dataset**, we utilize the following collections:

1. **GSM8K** and **MATH**:
    
    We annotate problems from these widely used mathematical datasets with **tool-integrated solutions**, enabling the model to reason step-by-step and leverage external tools when solving problems.
    
2. **MathInstruct** (Yue et al., 2023):
    
    We adopt a subset of this dataset, which contains instruction-based mathematical problems solved using **CoT** and **PoT**.
    
3. **Lila-OOD** (Mishra et al., 2022):
    
    This dataset is used to provide further training examples that include problems solved using **CoT** and **PoT**, ensuring a broad and diverse range of problem-solving strategies.
    

The **English dataset** spans various fields of mathematics, including:

![Screenshot 2025-04-24 at 10.37.30 PM.png](DeepSeekMath%207B%20Full%20Technical%20Report%201df9381f908c80dfa20cd8a7896b73c4/Screenshot_2025-04-24_at_10.37.30_PM.png)

- Algebra
- Probability
- Number theory
- Calculus
- Geometry

The total number of training examples in the English dataset is **776,000**.

---

### **Chinese Mathematical Datasets**

For the **Chinese dataset**, we collect a diverse range of **K-12 mathematical problems**, covering **76 sub-topics** such as:

- Linear equations
- Geometry
- Probability
- Statistics
- Calculus

The solutions for the Chinese dataset are annotated using both **CoT** and **tool-integrated reasoning formats** to provide a well-rounded approach to mathematical reasoning.

---

### **3.2. Training and Evaluating DeepSeekMath-Instruct 7B**

In this section, we introduce the training procedure for **DeepSeekMath-Instruct 7B**, which undergoes **mathematical instruction tuning** based on the pre-trained **DeepSeekMath-Base**. The training process and evaluation metrics are described as follows:

---

### **Training Process**

1. **Training Examples**:
    
    Training examples are randomly concatenated until the total context length reaches **4,000 tokens** (4K tokens), ensuring that each training example is processed efficiently.
    
2. **Training Parameters**:
    - **Batch size**: 256
    - **Learning rate**: 5e-5 (constant learning rate)
    - **Training steps**: 500 steps

The model is trained for **500 steps** with the specified parameters to tune it for mathematical instruction-based problem-solving.

---

### **Evaluation on Mathematical Performance**

The **DeepSeekMath-Instruct 7B** model is evaluated on four key quantitative reasoning benchmarks in **English** and **Chinese**. We assess the model's performance both with and without tool use, comparing its results to **leading closed-source** and **open-source models**.

### **Closed-Source Models**:

These models are proprietary and have undergone extensive alignment procedures:

1. **GPT-4** (OpenAI, 2023) and **GPT-4 Code Interpreter**.
2. **Gemini Ultra** and **Pro** (Anil et al., 2023).
3. **Inflection-2** (Inflection AI, 2023).
4. **Grok-1**.
5. Chinese proprietary models such as **Baichuan-3** and **GLM-4** (Du et al., 2022).

### **Open-Source Models**:

These models are available for public use, though they vary in size and mathematical enhancement:

1. **DeepSeek-LLM-Chat 67B** (DeepSeek-AI, 2024).
2. **Qwen 72B** (Bai et al., 2023).
3. **SeaLLM-v2 7B** (Nguyen et al., 2023).
4. **ChatGLM3 6B** (ChatGLM3 Team, 2023).
5. Enhanced models like **InternLM2-Math 20B**, **Math-Shepherd-Mistral 7B**, and **MAmmoTH 70B** which apply math-focused reinforcement learning and instruction tuning.

---

### **Results on the MATH Benchmark (No Tool Use)**

In the evaluation setting where **tool use is disallowed**, **DeepSeekMath-Instruct 7B** demonstrates strong step-by-step reasoning performance. Notably:

- On the **competition-level MATH dataset**, **DeepSeekMath-Instruct 7B** surpasses **all open-source models** and the majority of **proprietary models** (e.g., **Inflection-2** and **Gemini Pro**) by **at least 9%** absolute accuracy.
- This is true even for models that are significantly larger, such as **Qwen 72B**, or models that have been specifically enhanced with **math-focused reinforcement learning**, such as **WizardMath-v1.1 7B**.
- While **DeepSeekMath-Instruct** rivals the performance of **Chinese proprietary models** like **GLM-4** and **Baichuan-3** on MATH, it still lags behind **GPT-4** and **Gemini Ultra** in performance.

---

### **Results on the MATH Benchmark (With Tool Use)**

When **tool use** is allowed, **DeepSeekMath-Instruct 7B** shows a remarkable performance improvement. The model reaches an accuracy of approximately **60%** on the **MATH** benchmark, surpassing all existing **open-source models**. On other benchmarks, it competes closely with **DeepSeek-LLM-Chat 67B**, which is **10 times larger**.

---

### **4. Reinforcement Learning**

**4.1. Group Relative Policy Optimization**

Reinforcement Learning (RL) has been shown to effectively improve the mathematical reasoning abilities of **large language models (LLMs)** after the **Supervised Fine-Tuning (SFT)** stage (Luo et al., 2023; Wang et al., 2023b). In this section, we introduce an efficient and effective RL algorithm called **Group Relative Policy Optimization (GRPO)**.

---

### **4.1.1. From PPO to GRPO**

The **Proximal Policy Optimization (PPO)** algorithm (Schulman et al., 2017) has been widely used in fine-tuning LLMs. PPO is an actor-critic RL algorithm that optimizes the following **surrogate objective** to update policy models:

$J_{PPO}(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_\theta^{old}(O|q)} \left[\min \left( \pi_\theta(o_t|q, o_<t) \frac{A_t}{\pi_\theta^{old}(o_t|q, o_<t)}, 1 + \epsilon \right) \right]$

Where:

- **πθ** and **πθ_old** represent the current and old policy models.
- **q** and **o** are the question and output pairs sampled from the question dataset and the old policy, respectively.
- **A_t** is the **advantage**, computed using **Generalized Advantage Estimation (GAE)** (Schulman et al., 2015), which helps to reduce variance in the reward calculation.

The core of PPO involves maximizing the objective, which encourages small, consistent updates to the policy to avoid large deviations in behavior. PPO also includes a **KL penalty** to control the divergence between the current and reference policies, stabilizing training and preventing over-optimization.

While PPO is effective, it has a substantial **computational and memory burden** because it requires a value function that is typically trained alongside the policy model. This value function, which serves as a **baseline** for advantage calculation, significantly increases the model's memory and computation cost.

### **Introduction to GRPO**

**Group Relative Policy Optimization (GRPO)** is a variant of PPO designed to optimize **mathematical reasoning models** while reducing the memory and computational costs associated with value functions.

In GRPO, we **eliminate the value model**, which is a substantial computational resource in PPO. Instead, **GRPO uses the average reward of multiple sampled outputs**, produced in response to the same question, as the baseline for updating the policy model. This approach aligns well with the comparative nature of reward models, which are typically trained on datasets of comparisons between outputs for the same input.

In more detail:

- For each question **q**, GRPO samples a group of **outputs** $\{o_1, o_2, \dots, o_G\}$  from the **old policy** $\pi_\theta^{old}$.
- The policy model is then updated by maximizing the objective function that compares the group of outputs for each question, rather than relying on the individual reward score for each token.

The new objective for GRPO can be written as:

$J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_1^G \sim \pi_\theta^{old}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \pi_\theta(o_i, t|q, o_i < t) \frac{\hat{A}_{i,t}}{\pi_\theta^{old}(o_i, t|q, o_i < t)}, 1 + \epsilon \right) \right]$

Where:

- **$\hat{A}_{i,t}$** is the **relative advantage** for each output in the group, calculated based on the reward comparison among the outputs within the group.
- **KL divergence** is added directly to the loss function, regularizing the model by comparing the **current policy** with the **reference policy**, which is the policy trained at the previous step.

---

### **Advantages of GRPO Over PPO**

1. **Elimination of the Value Function**:
    - **PPO** requires a **value function** for each token to compute the advantage and optimize the policy.
    - **GRPO** removes the need for a value function, significantly reducing the computational cost, especially in the context of LLMs with large models.
2. **Group-Based Advantage Calculation**:
    - In **PPO**, the advantage is calculated for individual tokens and requires the reward model to be accurate at every token. This can be challenging and inefficient.
    - **GRPO** samples **groups of outputs** from the old policy, and calculates the advantage for each group, making it more efficient and suitable for models with complex outputs, such as those used in mathematical reasoning tasks.
3. **Better Memory and Computational Efficiency**:
    - **GRPO** minimizes the memory overhead by avoiding the need to maintain large value functions and by reducing the complexity of advantage estimation, making it more scalable for large models.
4. **Regularization by KL Divergence**:
    - Unlike PPO, where the **KL penalty** is applied during the reward calculation, **GRPO** directly adds the **KL divergence** between the trained policy and the reference policy to the loss function. This simplifies the training process and avoids complications in calculating advantages.

![Screenshot 2025-04-24 at 10.41.50 PM.png](DeepSeekMath%207B%20Full%20Technical%20Report%201df9381f908c80dfa20cd8a7896b73c4/Screenshot_2025-04-24_at_10.41.50_PM.png)

---

### **Algorithm for GRPO**

![Screenshot 2025-04-24 at 10.39.59 PM.png](DeepSeekMath%207B%20Full%20Technical%20Report%201df9381f908c80dfa20cd8a7896b73c4/Screenshot_2025-04-24_at_10.39.59_PM.png)

The algorithm for **Iterative Group Relative Policy Optimization** is outlined as follows:

1. **Input**: The initial policy model $\pi_\theta^{init}$, reward models $r_\phi$, task prompts D, and hyperparameters ϵ, β, and $\mu$.
2. **Initialization**: Set the policy model πθ\pi_\theta to πθinit\pi_\theta^{init}.
3. **For each iteration**:
    - Update the reference model $\pi_\theta^{ref}$ to the current policy $\pi_\theta$.
    - **Sample outputs** for each question from the old policy $\pi_\theta^{old}$.
    - **Compute rewards** for each output using the reward model $r_\phi$.
    - **Calculate advantages** for each output in the group using group relative advantage estimation.
    - **Update the policy model** by maximizing the GRPO objective.

This process is repeated for multiple iterations to continuously improve the policy model, leading to better performance in mathematical reasoning tasks.

---

### **Summary**

- **GRPO** is an efficient variant of **PPO** designed to optimize **LLMs** for mathematical reasoning tasks.
- By **eliminating the need for a value function** and using **group-based advantage calculation**, **GRPO** significantly reduces the computational cost and memory requirements, making it more suitable for large-scale models.
- The **GRPO algorithm** is based on **group relative advantage estimation** and **KL regularization**, offering an effective approach to improve mathematical reasoning in models while maintaining efficiency.

This approach paves the way for scaling RL-based fine-tuning techniques to large LLMs, enabling them to perform better on complex reasoning tasks, including those encountered in **mathematics**.