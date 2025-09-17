#!/usr/bin/env python3
"""
Test script for improved JSON extraction functions
"""

import re
import json

def extract_trading_recommendation_from_text(text):
    """Extract structured trading recommendation with 10-day forecast from model output"""
    
    # Clean the text first
    text = text.strip()
    
    # Method 1: Try to find complete JSON structure
    json_patterns = [
        r'\{[^{}]*"forecast_10d"[^{}]*\}',  # Simple JSON
        r'\{(?:[^{}]|{[^{}]*})*"forecast_10d"(?:[^{}]|{[^{}]*})*\}',  # Nested JSON
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                cleaned_json = match.strip()
                # Fix common JSON issues
                cleaned_json = re.sub(r',\s*}', '}', cleaned_json)  # Remove trailing commas
                cleaned_json = re.sub(r',\s*]', ']', cleaned_json)   # Remove trailing commas in arrays
                
                parsed = json.loads(cleaned_json)
                
                # Validate required fields
                if all(key in parsed for key in ['action', 'confidence', 'forecast_10d']):
                    # Ensure forecast_10d is a list of numbers
                    if isinstance(parsed['forecast_10d'], list) and len(parsed['forecast_10d']) > 0:
                        try:
                            parsed['forecast_10d'] = [float(x) for x in parsed['forecast_10d']]
                            return parsed
                        except (ValueError, TypeError):
                            continue
            except json.JSONDecodeError:
                continue
    
    # Method 2: Try to extract JSON from code blocks
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    code_matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in code_matches:
        try:
            parsed = json.loads(match.strip())
            if 'forecast_10d' in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Method 3: Try to find any JSON-like structure
    bracket_pattern = r'\{[^{}]*\}'
    bracket_matches = re.findall(bracket_pattern, text, re.DOTALL)
    for match in bracket_matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict) and 'forecast_10d' in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Method 4: Extract forecast_10d array separately
    forecast_patterns = [
        r'"forecast_10d"\s*:\s*\[([^\]]+)\]',
        r'forecast_10d["\']?\s*:\s*\[([^\]]+)\]',
        r'\[(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){9,})\]'  # Array of 10+ numbers
    ]
    
    for pattern in forecast_patterns:
        forecast_match = re.search(pattern, text, re.IGNORECASE)
        if forecast_match:
            try:
                price_str = forecast_match.group(1)
                # Handle different separators
                price_str = re.sub(r'\s+', ' ', price_str)  # Normalize whitespace
                prices = []
                
                # Try comma separation first
                if ',' in price_str:
                    prices = [float(p.strip()) for p in price_str.split(',') if p.strip()]
                # Try space separation
                elif ' ' in price_str:
                    prices = [float(p.strip()) for p in price_str.split(' ') if p.strip()]
                else:
                    prices = [float(price_str.strip())]
                
                if len(prices) >= 5:  # At least 5 prices found
                    return {
                        'action': 'UNKNOWN',
                        'confidence': 0,
                        'forecast_10d': prices[:10],  # Take first 10
                        'stop_loss': None,
                        'take_profit': None
                    }
            except (ValueError, TypeError):
                continue
    
    # Method 5: Extract any comma-separated numbers (fallback)
    number_patterns = [
        r'(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)+)',  # Comma-separated numbers
        r'(\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?)+)',      # Space-separated numbers
    ]
    
    for pattern in number_patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                # Take the longest sequence
                longest_match = max(matches, key=len)
                if ',' in longest_match:
                    prices = [float(p.strip()) for p in longest_match.split(',') if p.strip()]
                else:
                    prices = [float(p.strip()) for p in longest_match.split() if p.strip()]
                
                if len(prices) >= 3:  # At least 3 prices
                    return {
                        'action': 'UNKNOWN',
                        'confidence': 0,
                        'forecast_10d': prices[:10],
                        'stop_loss': None,
                        'take_profit': None
                    }
            except (ValueError, TypeError):
                continue
    
    # Method 6: Single number fallback
    single_number_pattern = r'(\d+(?:\.\d+)?)'
    single_matches = re.findall(single_number_pattern, text)
    if single_matches:
        try:
            # Take the first reasonable price (between 1000 and 200000)
            for match in single_matches:
                price = float(match)
                if 1000 <= price <= 200000:  # Reasonable Bitcoin price range
                    return {
                        'action': 'UNKNOWN',
                        'confidence': 0,
                        'forecast_10d': [price],
                        'stop_loss': None,
                        'take_profit': None
                    }
        except ValueError:
            pass
    
    return None

def post_process_recommendation(recommendation, current_price=None):
    """Post-process and validate trading recommendation"""
    if not recommendation:
        return None
    
    # Ensure all required fields exist
    processed = {
        'action': recommendation.get('action', 'UNKNOWN').upper(),
        'confidence': max(0, min(100, int(recommendation.get('confidence', 0)))),
        'forecast_10d': recommendation.get('forecast_10d', []),
        'stop_loss': recommendation.get('stop_loss'),
        'take_profit': recommendation.get('take_profit')
    }
    
    # Validate and clean forecast_10d
    if processed['forecast_10d']:
        # Filter out unreasonable prices
        valid_prices = []
        for price in processed['forecast_10d']:
            try:
                price_float = float(price)
                # Bitcoin price range validation (conservative range)
                if 100 <= price_float <= 1000000:
                    valid_prices.append(price_float)
            except (ValueError, TypeError):
                continue
        
        processed['forecast_10d'] = valid_prices[:10]  # Keep max 10 prices
    
    # Validate action
    valid_actions = ['BUY', 'SELL', 'HOLD', 'UNKNOWN']
    if processed['action'] not in valid_actions:
        processed['action'] = 'UNKNOWN'
    
    # Add validation flags
    processed['is_valid'] = len(processed['forecast_10d']) > 0
    processed['forecast_length'] = len(processed['forecast_10d'])
    
    return processed

# Test cases
if __name__ == "__main__":
    test_outputs = [
        '44736713.0',  # Single large number - should be filtered out
        '909090909090.0',  # Another single large number - should be filtered out
        '45000.50',  # Reasonable single number
        '{"action":"BUY","confidence":75,"forecast_10d":[48500.25,49200.10]}',  # Partial JSON
        '[45000, 46000, 47000, 48000, 49000, 50000, 51000, 52000, 53000, 54000]',  # Just array
        'Bitcoin price forecast: 45000, 46000, 47000, 48000, 49000',  # Comma-separated
        '{"action":"SELL","confidence":68,"stop_loss":9593.73,"take_profit":8370.01,"forecast_10d":[9174.91, 8277.01, 6955.27, 7754.00, 7621.30, 8265.59, 8736.98, 8621.90, 8129.97, 8926.57]}',  # Complete JSON
    ]

    print("Testing improved extraction function:")
    print("=" * 50)
    
    for i, test_output in enumerate(test_outputs):
        print(f"\nTest {i+1}: {test_output[:50]}{'...' if len(test_output) > 50 else ''}")
        
        result = extract_trading_recommendation_from_text(test_output)
        if result:
            processed = post_process_recommendation(result)
            print(f"✅ Extracted: {processed['action']}, Confidence: {processed['confidence']}%, Forecast length: {processed['forecast_length']}")
            if processed['forecast_10d']:
                print(f"   Forecast sample: {processed['forecast_10d'][:3]}...")
                print(f"   Valid: {processed['is_valid']}")
        else:
            print("❌ No extraction possible")