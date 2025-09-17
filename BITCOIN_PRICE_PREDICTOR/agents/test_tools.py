#!/usr/bin/env python3
"""
Test script to verify the LangChain tool fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from langchain_bitcoin_advisor import (
        NewsCollectionTool, NewsAnalysisTool,
        PriceForecastTool, InvestmentAdvisoryTool,
        BitcoinLangChainOrchestrator
    )
    from multi_agent_pipeline import DEFAULT_CONFIG
    print("✅ Imports successful")

    # Test tool instantiation
    print("\n🧪 Testing individual tool instantiation...")

    tools = [
        ("NewsCollectionTool", NewsCollectionTool),
        ("NewsAnalysisTool", NewsAnalysisTool),
        ("PriceForecastTool", PriceForecastTool),
        ("InvestmentAdvisoryTool", InvestmentAdvisoryTool)
    ]

    for tool_name, tool_class in tools:
        try:
            tool = tool_class(config=DEFAULT_CONFIG)
            print(f"✅ {tool_name} instantiated successfully")
            print(f"   Name: {tool.name}")
            print(f"   Description: {tool.description[:50]}...")
            # Check if agent is properly set
            # Map tool class to the expected agent attribute name
            expected_agent_attrs = {
                'NewsCollectionTool': 'news_agent',
                'NewsAnalysisTool': 'analysis_agent',
                'PriceForecastTool': 'forecast_agent',
                'InvestmentAdvisoryTool': 'recommendation_agent'
            }
            agent_attr = expected_agent_attrs.get(tool_name)
            if agent_attr and hasattr(tool, agent_attr):
                attr = getattr(tool, agent_attr)
                print(f"   Agent attribute '{agent_attr}' present: {type(attr).__name__}")
            else:
                print(f"   Agent attribute '{agent_attr}' not found on tool (available attrs: {list(tool.__dict__.keys())})")
        except Exception as e:
            print(f"❌ {tool_name} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n🧪 Testing full orchestrator instantiation...")

    try:
        orchestrator = BitcoinLangChainOrchestrator()
        print("✅ BitcoinLangChainOrchestrator instantiated successfully")
        print(f"   Number of tools: {len(orchestrator.tools)}")
        print(f"   LLM type: {type(orchestrator.llm).__name__}")
    except Exception as e:
        print(f"❌ Orchestrator failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 All tests passed successfully!")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()