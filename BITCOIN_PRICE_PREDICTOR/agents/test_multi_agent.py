#!/usr/bin/env python3
"""
Simple test script for the Bitcoin Investment Advisory Multi-Agent Pipeline

This script demonstrates how to use the new multi-agent architecture
to run the complete Bitcoin investment advisory pipeline.
"""

import sys
import os
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from multi_agent_pipeline import BitcoinPipelineOrchestrator
except ImportError as e:
    print(f"Error importing multi-agent pipeline: {e}")
    print("Make sure multi_agent_pipeline.py is in the same directory")
    sys.exit(1)

def create_sample_config():
    """Create a sample configuration file for testing"""
    config = {
        "api_keys": {
            "gemini": "AIzaSyCj2Km9Agz40pVF1ZvXgDNNrhBvGfFxQ3w",  # Replace with your key
            "openai": os.environ.get("OPENAI_API_KEY", ""),
            "openai_base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        },
        "models": {
            "summarization_model": "gemini-1.5-flash",
            "analysis_model_path": "../main_models/my-awesome-model_final_bitcoin-individual-news-dataset/checkpoint-400",
            "forecast_model_path": "../main_models/qwen_bitcoin_chat_fast_more_longer_explanation_v2/checkpoint-612",
            "advisory_model_path": "../main_models/my-awesome-model_final_bitcoin-investment-advisory-dataset_v2/checkpoint-400",
            "advisory_model": "gpt-4o"
        },
        "news": {
            "max_articles": 3,  # Reduced for testing
            "short_term_count": 5,
            "long_term_count": 5,
            "sources": [
                "http://feeds.feedburner.com/CoingeckoBuzz"
            ]
        },
        "output_dir": "test_outputs",
        "prompt_style": "comprehensive"
    }

    # Save sample config
    config_path = os.path.join(os.path.dirname(__file__), "test_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Sample config saved to: {config_path}")
    return config_path

def run_test_pipeline():
    """Run a test of the multi-agent pipeline"""
    print("=" * 60)
    print("BITCOIN INVESTMENT ADVISORY MULTI-AGENT PIPELINE TEST")
    print("=" * 60)

    # Create sample config if it doesn't exist
    config_path = os.path.join(os.path.dirname(__file__), "test_config.json")
    if not os.path.exists(config_path):
        print("Creating sample configuration...")
        config_path = create_sample_config()

    print(f"Using configuration: {config_path}")

    try:
        # Initialize the orchestrator
        print("\nInitializing Bitcoin Pipeline Orchestrator...")
        orchestrator = BitcoinPipelineOrchestrator(config_path=config_path)

        # Run the pipeline
        print("\nStarting pipeline execution...")
        start_time = datetime.now()

        results = orchestrator.run_pipeline()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Display results
        print(f"\nPipeline completed in {duration:.1f} seconds")
        print(f"Status: {results['status']}")

        if results['status'] == 'completed':
            print("\n" + "="*50)
            print("PIPELINE RESULTS SUMMARY")
            print("="*50)

            # Show pipeline stages
            for stage in results['pipeline_stages']:
                status_icon = "✅" if stage['status'] == 'completed' else "❌"
                print(f"{status_icon} {stage['stage'].replace('_', ' ').title()}")

            # Show key metrics
            news_data = results['pipeline_stages'][0]['data']
            print(f"\n📊 News Articles Processed: {news_data['total_articles']}")
            print(f"📈 Short-term News: {news_data['short_term_count']}")
            print(f"📉 Long-term News: {news_data['long_term_count']}")

            # Show forecast summary
            forecast_prices = results.get('forecast_prices', [])
            if forecast_prices:
                current_price = forecast_prices[0]
                final_price = forecast_prices[-1]
                change_pct = ((final_price / current_price) - 1) * 100
                print(f"📈 Current Price: ${current_price:.2f}")
                print(f"🎯 10-Day Target: ${final_price:.2f} ({change_pct:+.2f}%)")
            # Show advisory preview
            advisory = results.get('final_advisory', '')
            if advisory:
                print("\n📋 Investment Advisory Preview:")
                print("-" * 40)
                # Show first 300 characters
                preview = advisory[:300] + "..." if len(advisory) > 300 else advisory
                print(preview)
                print("-" * 40)

            # Show output location
            output_file = f"{orchestrator.config['output_dir']}/bitcoin_pipeline_{results['target_date']}.json"
            print(f"\n💾 Full results saved to: {output_file}")

        else:
            print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("Bitcoin Multi-Agent Pipeline Test Script")
    print("This will test the new 4-agent architecture")

    # Check if user wants to proceed
    response = input("\nDo you want to run the test? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Test cancelled.")
        return

    # Run the test
    run_test_pipeline()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the generated JSON file for complete results")
    print("2. Review the investment advisory in the output")
    print("3. Modify config.json for your specific model paths and API keys")
    print("4. Run with: python multi_agent_pipeline.py --config config.json")


if __name__ == "__main__":
    main()