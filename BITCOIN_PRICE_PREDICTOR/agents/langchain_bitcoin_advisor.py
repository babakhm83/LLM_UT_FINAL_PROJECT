# Bitcoin Investment Advisory Pipeline - LangChain Integration

"""
This module implements a LangChain-powered version of the Bitcoin investment advisory pipeline
with enhanced user interaction, conversational memory, and structured agent communication.

Features:
- Conversational interface with memory
- LangChain agent orchestration
- Tool-based agent interactions
- Structured output parsing
- Enhanced error handling and retries
- Interactive user experience
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
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict

# LangChain imports
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import BaseTool, tool
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Install with: pip install langchain langchain-google-genai")

# Import existing agents
try:
    from multi_agent_pipeline import (
        NewsAgent, AnalysisAgent, ForecastAgent, RecommendationAgent,
        DEFAULT_CONFIG, NewsArticle
    )
except ImportError:
    print("Error: multi_agent_pipeline.py not found. Please ensure it's in the same directory.")
    sys.exit(1)

# Standard imports
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bitcoin_langchain_agent.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("bitcoin_langchain")


# -----------------------------
# LangChain Tools
# -----------------------------

class NewsCollectionTool(BaseTool):
    """Tool for collecting Bitcoin news"""

    name: str = "collect_bitcoin_news"
    description: str = "Collect and categorize Bitcoin-related news from RSS feeds. Returns structured news data with short-term and long-term impact analysis."
    config: Dict[str, Any] = Field(default_factory=dict)
    news_agent: Any = Field(default=None)

    def __init__(self, config: Dict[str, Any]):
        # Pass agent into BaseTool initializer so Pydantic registers the field
        super().__init__(config=config, news_agent=NewsAgent(config))

    def _run(self, target_date: Optional[str] = None) -> str:
        """Run news collection"""
        try:
            if target_date is None:
                target_date = dt.datetime.now().strftime('%Y-%m-%d')

            logger.info(f"NewsCollectionTool: Collecting news for {target_date}")
            results = self.news_agent.run(target_date=target_date)

            return json.dumps({
                "status": "success",
                "data": results,
                "message": f"Successfully collected {results['total_articles']} news articles"
            }, indent=2)

        except Exception as e:
            logger.error(f"NewsCollectionTool error: {e}")
            return json.dumps({
                "status": "error",
                "error": str(e),
                "message": "Failed to collect news"
            }, indent=2)


class NewsAnalysisTool(BaseTool):
    """Tool for analyzing news effects"""

    name: str = "analyze_news_effects"
    description: str = "Analyze the effects of news on Bitcoin prices using trained models. Takes news data and returns probability distributions and market sentiment."
    config: Dict[str, Any] = Field(default_factory=dict)
    analysis_agent: Any = Field(default=None)

    def __init__(self, config: Dict[str, Any]):
        # Pass agent into BaseTool initializer so Pydantic registers the field
        super().__init__(config=config, analysis_agent=AnalysisAgent(config))

    def _run(self, news_data_json: str) -> str:
        """Run news analysis"""
        try:
            news_data = json.loads(news_data_json)
            logger.info("NewsAnalysisTool: Analyzing news effects")

            results = self.analysis_agent.run(news_data)

            return json.dumps({
                "status": "success",
                "data": results,
                "message": "Successfully analyzed news effects"
            }, indent=2)

        except Exception as e:
            logger.error(f"NewsAnalysisTool error: {e}")
            return json.dumps({
                "status": "error",
                "error": str(e),
                "message": "Failed to analyze news effects"
            }, indent=2)


class PriceForecastTool(BaseTool):
    """Tool for forecasting Bitcoin prices"""

    name: str = "forecast_bitcoin_prices"
    description: str = "Forecast Bitcoin prices for the next 10 days based on news analysis. Returns detailed price predictions with confidence intervals."
    config: Dict[str, Any] = Field(default_factory=dict)
    forecast_agent: Any = Field(default=None)

    def __init__(self, config: Dict[str, Any]):
        # Pass agent into BaseTool initializer so Pydantic registers the field
        super().__init__(config=config, forecast_agent=ForecastAgent(config))

    def _run(self, analysis_data_json: str) -> str:
        """Run price forecasting"""
        try:
            analysis_data = json.loads(analysis_data_json)
            logger.info("PriceForecastTool: Forecasting prices")

            results = self.forecast_agent.run(analysis_data)

            return json.dumps({
                "status": "success",
                "data": results,
                "message": "Successfully forecasted Bitcoin prices"
            }, indent=2)

        except Exception as e:
            logger.error(f"PriceForecastTool error: {e}")
            return json.dumps({
                "status": "error",
                "error": str(e),
                "message": "Failed to forecast prices"
            }, indent=2)


class InvestmentAdvisoryTool(BaseTool):
    """Tool for generating investment recommendations"""

    name: str = "generate_investment_advisory"
    description: str = "Generate comprehensive investment advisory based on complete analysis. Returns detailed investment recommendations with risk assessment."
    config: Dict[str, Any] = Field(default_factory=dict)
    recommendation_agent: Any = Field(default=None)

    def __init__(self, config: Dict[str, Any]):
        # Pass agent into BaseTool initializer so Pydantic registers the field
        super().__init__(config=config, recommendation_agent=RecommendationAgent(config))

    def _run(self, complete_data_json: str) -> str:
        """Run investment advisory generation"""
        try:
            complete_data = json.loads(complete_data_json)
            logger.info("InvestmentAdvisoryTool: Generating advisory")

            advisory = self.recommendation_agent.run(complete_data)

            return json.dumps({
                "status": "success",
                "advisory": advisory,
                "message": "Successfully generated investment advisory"
            }, indent=2)

        except Exception as e:
            logger.error(f"InvestmentAdvisoryTool error: {e}")
            return json.dumps({
                "status": "error",
                "error": str(e),
                "message": "Failed to generate investment advisory"
            }, indent=2)


# -----------------------------
# LangChain Agent Orchestrator
# -----------------------------

class BitcoinLangChainOrchestrator:
    """
    LangChain-powered orchestrator for the Bitcoin investment advisory pipeline
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the LangChain orchestrator

        Args:
            config_path: Path to configuration file
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required but not installed. Install with: pip install langchain langchain-openai langchain-google-genai")

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

        # Initialize tools
        self.tools = [
            NewsCollectionTool(config=self.config),
            NewsAnalysisTool(config=self.config),
            PriceForecastTool(config=self.config),
            InvestmentAdvisoryTool(config=self.config)
        ]

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Create memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 interactions
        )

        # Create agent
        self.agent_executor = self._create_agent()

        # Create output directory
        os.makedirs(self.config.get("output_dir", "outputs/langchain_results"), exist_ok=True)

        logger.info("BitcoinLangChainOrchestrator initialized")

    def _update_nested_dict(self, d: dict, u: dict) -> dict:
        """Recursively update nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

    def _initialize_llm(self):
        """Initialize Gemini LLM"""
        gemini_key = self.config["api_keys"].get("gemini")
        if gemini_key:
            try:
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=gemini_key,
                    temperature=0.1,
                    streaming=True,
                    callbacks=[StreamingStdOutCallbackHandler()]
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                raise ValueError(f"Gemini initialization failed: {e}")

        raise ValueError("Gemini API key not found. Please check your configuration.")

    def _create_agent(self):
        """Create the LangChain agent with tools"""
        # Ensure tools are initialized
        if not self.tools:
            raise ValueError("No tools initialized. Ensure tools are properly configured.")

        # Construct tool names for logging and debugging
        try:
            tool_names = ", ".join([t.name for t in self.tools])
            logger.info(f"Creating agent with tools: {tool_names}")
        except Exception as e:
            logger.error(f"Error constructing tool names: {e}")
            raise ValueError("Failed to construct tool names for the system prompt.")

        # Define the system prompt and pre-format with tool names so there are no unresolved template variables
        system_prompt = f"""You are an expert Bitcoin investment advisor with access to specialized tools for comprehensive market analysis.

        Your capabilities:
        1. **News Collection**: Gather and categorize Bitcoin news from multiple sources
        2. **News Analysis**: Analyze market impact and sentiment from news data
        3. **Price Forecasting**: Generate 10-day Bitcoin price predictions
        4. **Investment Advisory**: Create detailed investment recommendations

        WORKFLOW:
        1. Start by collecting the latest Bitcoin news
        2. Analyze the effects of that news on market sentiment
        3. Generate price forecasts based on the analysis
        4. Provide comprehensive investment advisory

        Always use the tools in sequence and provide clear explanations of your reasoning.
        Be thorough but concise in your responses.
        If any tool fails, explain the issue and provide alternative analysis.

        Available tools: {tool_names}"""

        # Pre-format the system prompt to inject tool names so ChatPromptTemplate has no required external vars
        formatted_system_prompt = system_prompt.format(tool_names=tool_names)

        # Ensure agent_scratchpad is initialized as a list of BaseMessage objects
        agent_scratchpad: List[BaseMessage] = []

        # Build prompt and partially apply required variables so the prompt doesn't expose unresolved template vars
        prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]).partial(
            tools=self.tools,
            tool_names=tool_names,
            agent_scratchpad=agent_scratchpad,
        )

        # Create agent using the explicit prompt to satisfy the API contract.
        try:
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt,
            )
        except ValueError as e:
            # Some LangChain versions validate prompt input variables differently and may
            # raise a ValueError about missing 'tool_names' or 'tools'. Try a safe fallback.
            msg = str(e)
            logger.warning(f"create_react_agent with custom prompt failed: {msg}")
            if "tool_names" in msg or "tools" in msg:
                logger.info("Falling back to create_react_agent without custom prompt to accommodate LangChain variant.")
                try:
                    # In this fallback we call create_react_agent without prompt; let LangChain build a default prompt
                    agent = create_react_agent(
                        llm=self.llm,
                        tools=self.tools,
                    )
                except Exception as e2:
                    logger.error(f"Fallback create_react_agent also failed: {e2}")
                    raise ValueError(f"Failed to create LangChain agent (fallback also failed): {e2}")
            else:
                logger.error(f"Error creating LangChain agent: {e}")
                raise ValueError("Failed to create LangChain agent.")
        except Exception as e:
            logger.error(f"Error creating LangChain agent: {e}")
            raise ValueError("Failed to create LangChain agent.")

        # Create executor
        try:
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=300,  # 5 minutes timeout
            )
        except Exception as e:
            logger.error(f"Error creating AgentExecutor: {e}")
            raise ValueError("Failed to create AgentExecutor.")

    def run_interactive_session(self):
        """Run an interactive session with the user"""
        print("\n" + "="*70)
        print("🤖 BITCOIN INVESTMENT ADVISOR - LANGCHAIN POWERED")
        print("="*70)
        print("I can help you with Bitcoin investment analysis using:")
        print("• Real-time news collection and analysis")
        print("• Market sentiment evaluation")
        print("• 10-day price forecasting")
        print("• Comprehensive investment recommendations")
        print("\nType 'quit' or 'exit' to end the session.")
        print("-" * 70)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Your Bitcoin analysis session has ended.")
                    break

                if not user_input:
                    continue

                print("\n🤖 Advisor: ", end="", flush=True)

                # Run agent
                response = self.agent_executor.invoke({"input": user_input})

                # Save conversation
                self._save_conversation(user_input, response.get("output", ""))

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Interactive session error: {e}")

    def run_automated_pipeline(self, target_date: str = None) -> Dict[str, Any]:
        """Run the complete automated pipeline"""
        if target_date is None:
            target_date = dt.datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Running automated pipeline for {target_date}")

        # Create a structured prompt for the automated run
        automated_prompt = f"""
Please run the complete Bitcoin investment advisory pipeline for {target_date}:

1. First, collect the latest Bitcoin news using the news collection tool
2. Then analyze the effects of that news on Bitcoin prices
3. Generate a 10-day price forecast based on the analysis
4. Finally, create a comprehensive investment advisory

Provide a detailed summary of each step and the final recommendation.
"""

        try:
            response = self.agent_executor.invoke({"input": automated_prompt})

            # Parse and structure the results
            result = {
                'target_date': target_date,
                'status': 'completed',
                'timestamp': dt.datetime.now().isoformat(),
                'response': response.get('output', ''),
                'conversation_history': self._get_conversation_history()
            }

            # Save results
            self._save_results(result, target_date)

            return result

        except Exception as e:
            logger.error(f"Automated pipeline failed: {e}")
            return {
                'target_date': target_date,
                'status': 'failed',
                'error': str(e),
                'timestamp': dt.datetime.now().isoformat()
            }

    def _get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        history = []
        if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
            for msg in self.memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})
        return history

    def _save_conversation(self, user_input: str, ai_response: str):
        """Save conversation to memory"""
        # This is handled automatically by LangChain memory
        pass

    def _save_results(self, results: Dict[str, Any], target_date: str):
        """Save pipeline results to file"""
        output_path = os.path.join(
            self.config.get("output_dir", "outputs/langchain_results"),
            f"bitcoin_langchain_{target_date}.json"
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"LangChain results saved to {output_path}")


# -----------------------------
# Main Functions
# -----------------------------

def interactive_mode(config_path: str = None):
    """Run in interactive mode"""
    try:
        orchestrator = BitcoinLangChainOrchestrator(config_path=config_path)
        orchestrator.run_interactive_session()
    except Exception as e:
        print(f"❌ Failed to start interactive mode: {e}")
        logger.error(f"Interactive mode error: {e}")


def automated_mode(config_path: str = None, target_date: str = None):
    """Run in automated mode"""
    try:
        orchestrator = BitcoinLangChainOrchestrator(config_path=config_path)
        results = orchestrator.run_automated_pipeline(target_date=target_date)

        if results['status'] == 'completed':
            print("\n" + "="*60)
            print("LANGCHAIN PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Date: {results['target_date']}")
            print("\nFINAL RESPONSE:")
            print("-" * 40)
            print(results['response'][:1000] + "..." if len(results['response']) > 1000 else results['response'])
            print("-" * 40)
        else:
            print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Failed to run automated mode: {e}")
        logger.error(f"Automated mode error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Bitcoin Investment Advisory LangChain Agent')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--date', type=str, help='Target date for analysis (YYYY-MM-DD)')
    parser.add_argument('--mode', choices=['interactive', 'automated'], default='interactive',
                       help='Run mode: interactive (default) or automated')

    args = parser.parse_args()

    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain is required but not installed.")
        print("Install with: pip install langchain langchain-google-genai")
        sys.exit(1)

    if args.mode == 'interactive':
        print("Starting Bitcoin Investment Advisor in INTERACTIVE mode...")
        interactive_mode(config_path=args.config)
    else:
        print("Starting Bitcoin Investment Advisor in AUTOMATED mode...")
        automated_mode(config_path=args.config, target_date=args.date)


if __name__ == "__main__":
    main()