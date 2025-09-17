import os
import re
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# --- Global variables for worker processes ---
_worker_model = None
_worker_tokenizer = None
_worker_device = "cpu"
_test_dataset = None
_base_qwen_tokenizer = None

# --- Utility Functions (copied from notebook) ---

def extract_prices_from_text(text):
    """Extract price predictions from model output"""
    price_pattern = r'(\d+(?:\.\d+)?(?:,\s*\d+(?:\.\d+)?)*)'  
    matches = re.findall(price_pattern, text)
    
    if matches:
        prices_str = matches[0]
        try:
            prices = [float(p.strip()) for p in prices_str.split(',')]
            return prices
        except:
            return []
    return []

def calculate_price_differences(predictions, actual):
    """Calculate various price difference metrics"""
    if len(predictions) != len(actual):
        return None
    
    predictions = np.array(predictions)
    actual = np.array(actual)
    
    abs_diff = np.abs(predictions - actual)
    rel_diff = ((predictions - actual) / actual * 100) if np.all(actual != 0) else np.zeros_like(actual)
    
    actual_direction = np.diff(actual)
    pred_direction = np.diff(predictions)
    
    direction_accuracy = 0
    if len(actual_direction) > 0:
        direction_correct = np.sign(actual_direction) == np.sign(pred_direction)
        direction_accuracy = np.mean(direction_correct) * 100
    
    return {
        'absolute_differences': abs_diff,
        'relative_differences': rel_diff,
        'mean_abs_diff': np.mean(abs_diff),
        'max_abs_diff': np.max(abs_diff),
        'min_abs_diff': np.min(abs_diff),
        'std_abs_diff': np.std(abs_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'max_rel_diff': np.max(rel_diff),
        'min_rel_diff': np.min(rel_diff),
        'std_rel_diff': np.std(rel_diff),
        'direction_accuracy': direction_accuracy,
        'rmse': np.sqrt(np.mean((predictions - actual) ** 2)),
        'mae': np.mean(abs_diff),
        'mape': np.mean(np.abs(rel_diff))
    }

def format_input(example, tokenizer):
    """Format input for the enhanced model"""
    instruction = example.get('instruction', '')
    user_input = example.get('input', '')
    messages = [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': user_input}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_input_for_base_model(example, tokenizer):
    """Format input for base Qwen model with bitcoin prediction task"""
    instruction = example.get('instruction', '')
    user_input = example.get('input', '')
    
    bitcoin_instruction = """You are a Bitcoin investment advisor. Based on the provided market data and news, predict the next 10 days of Bitcoin prices. Provide your predictions as comma-separated numbers."""
    
    messages = [
        {'role': 'system', 'content': bitcoin_instruction},
        {'role': 'user', 'content': f"{instruction}\n\n{user_input}\n\nPlease provide 10 Bitcoin price predictions for the next 10 days, separated by commas."}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# --- Worker Initialization and Processing Function ---

def init_worker(model_type, device_id):
    """Initializes the model and tokenizer for each worker process."""
    global _worker_model, _worker_tokenizer, _worker_device, _test_dataset, _base_qwen_tokenizer
    
    # Set device
    if torch.cuda.is_available():
        _worker_device = f"cuda:{device_id}"
        torch.cuda.set_device(_worker_device)
    
    print(f"Initializing worker {os.getpid()} on device {_worker_device} for {model_type} model...")

    # Load dataset
    _test_dataset = load_dataset('tahamajs/bitcoin-enhanced-prediction-dataset-with-local-comprehensive-news', split='train')

    # Load tokenizers
    adapter_path = './my-awesome-model_final_bitcoin-enhanced-prediction-dataset-with-local-comprehensive-news-v2/checkpoint-400'
    _worker_tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if _worker_tokenizer.pad_token is None:
        _worker_tokenizer.pad_token = _worker_tokenizer.eos_token
    
    _base_qwen_tokenizer = AutoTokenizer.from_pretrained('./Qwen3-8B', trust_remote_code=True)
    if _base_qwen_tokenizer.pad_token is None:
        _base_qwen_tokenizer.pad_token = _base_qwen_tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        './Qwen3-8B',
        torch_dtype=torch.float16,
        device_map=_worker_device,
        trust_remote_code=True
    )
    base_model.resize_token_embeddings(len(_worker_tokenizer))

    if model_type == 'enhanced':
        _worker_model = PeftModel.from_pretrained(base_model, adapter_path)
    else: # 'base'
        _worker_model = base_model
        
    _worker_model.eval()
    print(f"Worker {os.getpid()} initialized successfully.")

def process_sample(sample_index, model_type):
    """Processes a single sample for model evaluation."""
    global _worker_model, _worker_tokenizer, _worker_device, _test_dataset, _base_qwen_tokenizer
    
    try:
        test_example = _test_dataset[sample_index]
        
        if model_type == 'enhanced':
            test_text = format_input(test_example, _worker_tokenizer)
            current_tokenizer = _worker_tokenizer
        else: # 'base'
            test_text = format_input_for_base_model(test_example, _base_qwen_tokenizer)
            current_tokenizer = _base_qwen_tokenizer

        inputs = current_tokenizer(test_text, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(_worker_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _worker_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=current_tokenizer.eos_token_id
            )
        
        generated_text = current_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        predicted_prices = extract_prices_from_text(generated_text)
        actual_prices = extract_prices_from_text(test_example.get('output', ''))
        
        if predicted_prices and actual_prices:
            min_len = min(len(predicted_prices), len(actual_prices))
            if min_len > 0:
                pred_truncated = predicted_prices[:min_len]
                actual_truncated = actual_prices[:min_len]
                price_diff_analysis = calculate_price_differences(pred_truncated, actual_truncated)
                
                if price_diff_analysis:
                    return {
                        'sample_id': sample_index,
                        'predicted': pred_truncated,
                        'actual': actual_truncated,
                        'price_analysis': price_diff_analysis
                    }
    except Exception as e:
        print(f"Error in worker {os.getpid()} processing sample {sample_index}: {e}")
    return None
