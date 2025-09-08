#!/usr/bin/env python3
"""
Bitcoin Investment Advisory Dataset - Hugging Face Upload Script

This script uploads the instruction-format Bitcoin investment advisory training dataset to Hugging Face Hub.
"""

import json
import os
from datasets import Dataset
from huggingface_hub import HfApi, create_repo, login
import pandas as pd
from datetime import datetime

# Configuration
DATASET_FILE = "bitcoin_training_datasets/bitcoin_investment_training_instruction_20250908_072751.json"
HF_USERNAME = "tahamajs"  # Change this to your Hugging Face username
HF_REPO_NAME = "bitcoin-investment-advisory-dataset"
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

def load_dataset():
    """Load the training dataset from JSON file."""
    print(f"Loading dataset from: {DATASET_FILE}")
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    return data

def analyze_dataset(data):
    """Analyze the dataset and return statistics."""
    df = pd.DataFrame(data)
    
    stats = {
        'total_samples': len(data),
        'date_range': {
            'earliest': df['date'].min(),
            'latest': df['date'].max(),
            'unique_dates': df['date'].nunique()
        },
        'length_statistics': {
            'avg_input_length': df['input'].str.len().mean(),
            'avg_output_length': df['output'].str.len().mean(),
            'max_input_length': df['input'].str.len().max(),
            'max_output_length': df['output'].str.len().max()
        },
        'quality_statistics': {
            'avg_quality_score': df['quality_score'].mean(),
            'high_quality_samples': (df['quality_score'] >= 0.8).sum(),
            'excellent_quality_samples': (df['quality_score'] >= 0.9).sum()
        }
    }
    
    return stats

def create_hf_dataset(data):
    """Convert JSON data to Hugging Face Dataset format."""
    print("Converting to Hugging Face Dataset format...")
    
    # Create the dataset
    dataset = Dataset.from_list(data)
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Dataset features: {list(dataset.features.keys())}")
    
    return dataset

def create_readme(stats):
    """Create a comprehensive README for the dataset."""
    
    readme_content = f'''---
license: mit
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- bitcoin
- investment
- finance
- advisory
- cryptocurrency
- instruction-tuning
- financial-analysis
size_categories:
- 1K<n<10K
---

# Bitcoin Investment Advisory Training Dataset

## Dataset Description

This dataset contains comprehensive Bitcoin investment advisory training data designed for fine-tuning large language models to provide institutional-grade cryptocurrency investment advice. The dataset consists of {stats['total_samples']:,} high-quality instruction-input-output triplets covering Bitcoin market analysis from {stats['date_range']['earliest']} to {stats['date_range']['latest']}.

## Dataset Features

- **Total Samples**: {stats['total_samples']:,}
- **Date Range**: {stats['date_range']['earliest']} to {stats['date_range']['latest']} ({stats['date_range']['unique_dates']:,} unique dates)
- **Average Input Length**: {stats['length_statistics']['avg_input_length']:.0f} characters
- **Average Output Length**: {stats['length_statistics']['avg_output_length']:.0f} characters
- **Average Quality Score**: {stats['quality_statistics']['avg_quality_score']:.2f}
- **High Quality Samples (≥0.8)**: {stats['quality_statistics']['high_quality_samples']:,}
- **Excellent Quality Samples (≥0.9)**: {stats['quality_statistics']['excellent_quality_samples']:,}

## Data Structure

Each sample contains:

- **instruction**: Task instruction for the AI model
- **input**: Comprehensive market intelligence including price data, news analysis, and daily market summary
- **output**: Professional, institutional-grade Bitcoin investment advisory
- **date**: Trading date for the analysis
- **quality_score**: Data quality score (0.0 to 1.0)

## Sample Data

```json
{{
  "instruction": "You are an elite institutional Bitcoin investment advisor. Provide comprehensive investment advisory based on the given market intelligence.",
  "input": "Market intelligence including price trends, news analysis, and market sentiment...",
  "output": "Comprehensive institutional investment advisory with risk assessment, price targets, and recommendations...",
  "date": "2018-01-01",
  "quality_score": 0.89
}}
```

## Use Cases

- **Financial AI Training**: Fine-tune language models for investment advisory applications
- **Cryptocurrency Analysis**: Train models to analyze Bitcoin market conditions
- **Instruction Following**: Improve model ability to follow complex financial analysis instructions
- **Risk Assessment**: Develop AI systems for financial risk evaluation
- **Portfolio Management**: Create AI advisors for institutional portfolio decisions

## Training Recommendations

### Model Types
- Large Language Models (GPT, LLaMA, Mistral, etc.)
- Instruction-tuned models
- Financial domain-specific models

### Hyperparameters
- Learning Rate: 1e-5 to 5e-5
- Batch Size: 4-16 (depending on GPU memory)
- Epochs: 3-5 for fine-tuning
- Max Sequence Length: 4096-8192 tokens

### Data Preprocessing
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{HF_REPO_ID}")

# Filter high-quality samples
high_quality = dataset.filter(lambda x: x['quality_score'] >= 0.8)

# Split chronologically
train_data = high_quality.filter(lambda x: x['date'] < '2023-01-01')
val_data = high_quality.filter(lambda x: x['date'] >= '2023-01-01')
```

## Ethical Considerations

⚠️ **Important Disclaimers**:

- This dataset is for **research and educational purposes only**
- Investment advice generated by models trained on this data should include appropriate financial disclaimers
- Users should comply with relevant financial regulations in their jurisdiction
- The dataset does not constitute actual investment advice
- Past performance does not guarantee future results

## Dataset Creation

This dataset was created using:
- Historical Bitcoin price data
- News sentiment analysis
- Market intelligence aggregation
- Professional investment advisory templates
- Quality scoring and filtering

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{{bitcoin_investment_advisory_2025,
  title={{Bitcoin Investment Advisory Training Dataset}},
  author={{{HF_USERNAME}}},
  year={{2025}},
  url={{https://huggingface.co/datasets/{HF_REPO_ID}}},
  note={{Comprehensive Bitcoin investment advisory dataset for AI training}}
}}
```

## License

This dataset is released under the MIT License. See LICENSE for details.

## Contact

For questions or issues, please open an issue in the dataset repository or contact the author.

---

**Disclaimer**: This dataset is for research purposes only. Always consult with qualified financial advisors before making investment decisions.
'''
    
    return readme_content

def upload_to_huggingface(dataset, stats):
    """Upload the dataset to Hugging Face Hub."""
    
    try:
        # Login check
        print("Checking Hugging Face authentication...")
        api = HfApi()
        
        # Create repository
        print(f"Creating repository: {HF_REPO_ID}")
        try:
            create_repo(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                exist_ok=True,
                private=False
            )
            print("✅ Repository created successfully")
        except Exception as e:
            print(f"⚠️  Repository might already exist: {e}")
        
        # Upload dataset
        print("Uploading dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            HF_REPO_ID,
            commit_message="Upload Bitcoin Investment Advisory Dataset"
        )
        print("✅ Dataset uploaded successfully")
        
        # Create and upload README
        print("Creating and uploading README...")
        readme_content = create_readme(stats)
        
        # Save README locally first
        readme_path = "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message="Add comprehensive README"
        )
        print("✅ README uploaded successfully")
        
        # Clean up local README
        if os.path.exists(readme_path):
            os.remove(readme_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Verify your Hugging Face username in the script")
        print("4. Ensure you have write permissions")
        return False

def main():
    """Main execution function."""
    
    print("🚀 Bitcoin Investment Advisory Dataset - Hugging Face Upload")
    print("=" * 60)
    
    # Check if dataset file exists
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Dataset file not found: {DATASET_FILE}")
        print("Please make sure the file path is correct.")
        return
    
    try:
        # Load and analyze dataset
        data = load_dataset()
        stats = analyze_dataset(data)
        
        # Display dataset statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total Samples: {stats['total_samples']:,}")
        print(f"   Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        print(f"   Unique Dates: {stats['date_range']['unique_dates']:,}")
        print(f"   Average Quality Score: {stats['quality_statistics']['avg_quality_score']:.2f}")
        print(f"   High Quality Samples: {stats['quality_statistics']['high_quality_samples']:,}")
        
        # Create Hugging Face dataset
        hf_dataset = create_hf_dataset(data)
        
        # Ask for confirmation
        print(f"\n🤔 Ready to upload to: https://huggingface.co/datasets/{HF_REPO_ID}")
        print("   This will make the dataset publicly available.")
        
        confirm = input("\nProceed with upload? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("Upload cancelled.")
            return
        
        # Upload to Hugging Face
        success = upload_to_huggingface(hf_dataset, stats)
        
        if success:
            print(f"\n🎉 SUCCESS! Dataset uploaded successfully!")
            print(f"🔗 View your dataset: https://huggingface.co/datasets/{HF_REPO_ID}")
            print(f"📚 Usage: datasets.load_dataset('{HF_REPO_ID}')")
        else:
            print("\n❌ Upload failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
