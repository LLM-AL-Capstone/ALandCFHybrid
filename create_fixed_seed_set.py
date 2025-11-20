#!/usr/bin/env python3
"""
Create Fixed Seed Set for Active Learning Experiments

This script generates a fixed seed set (initial labeled examples) for reproducibility.
The seed set is saved as a CSV file and should be used consistently across all experiments
on the same dataset.

Usage:
    python create_fixed_seed_set.py
"""

import pandas as pd
import os
import sys
from utils import load_config, load_dataset, get_unique_labels, shuffle_dataframe


def create_seed_set(dataset_name: str = None):
    """
    Create fixed seed set for a dataset.
    
    Args:
        dataset_name: Optional dataset name override. If None, uses config.
    """
    print("=" * 80)
    print("Fixed Seed Set Creator")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    dataset_config = config['dataset']
    processing = config['processing']
    al_config = config['active_learning']
    
    # Determine dataset file
    if dataset_name:
        train_file_name = f"{dataset_name}_train.csv"
    else:
        train_file_name = dataset_config['train_file']
    
    train_file = f"{config['directories']['input_data']}/{train_file_name}"
    
    if not os.path.exists(train_file):
        print(f"❌ Error: Training file not found: {train_file}")
        sys.exit(1)
    
    print(f"\n📁 Dataset: {train_file_name}")
    print(f" Seed: {processing['seed']}")
    print(f"📊 Samples per class: {al_config['initial_labeled_per_class']}")
    
    # Load and shuffle training data
    print(f"\nLoading training data...")
    df = load_dataset(train_file, config)
    print(f"  Total examples: {len(df)}")
    
    # Shuffle with seed for reproducibility
    df = shuffle_dataframe(df, processing['seed'])
    print(f"  ✓ Shuffled with seed={processing['seed']}")
    
    # Get column names
    col_id = dataset_config['columns']['id']
    col_text = dataset_config['columns']['text']
    col_label = dataset_config['columns']['label']
    
    # Get unique labels
    unique_labels = get_unique_labels(
        df,
        col_label,
        dataset_config.get('exclude_labels', [])
    )
    
    # Filter out null/NaN values
    unique_labels = [
        label for label in unique_labels
        if pd.notna(label) and str(label).lower() not in ['none', 'null', '', 'nan']
    ]
    
    print(f"\n📋 Labels found: {unique_labels}")
    print(f"  Number of classes: {len(unique_labels)}")
    
    # Extract seed set (stratified sampling)
    initial_per_class = al_config['initial_labeled_per_class']
    seed_examples = []
    
    print(f"\n🌱 Creating seed set...")
    for label in unique_labels:
        label_df = df[df[col_label] == label]
        num_samples = min(initial_per_class, len(label_df))
        
        if num_samples == 0:
            print(f"  ⚠️  Warning: No examples for label '{label}'")
            continue
        
        # Take first N examples after shuffling
        samples = label_df.head(num_samples)
        seed_examples.append(samples)
        
        print(f"  ✓ {label}: {num_samples} examples")
    
    # Combine all seed examples
    seed_df = pd.concat(seed_examples, ignore_index=True)
    
    # Save fixed seed set
    seed_file = train_file.replace('.csv', '_seed_set.csv')
    seed_df.to_csv(seed_file, index=False)
    
    print(f"\n✅ Fixed seed set created successfully!")
    print(f"   File: {seed_file}")
    print(f"   Total examples: {len(seed_df)}")
    print(f"\n📊 Class distribution:")
    for label, count in seed_df[col_label].value_counts().items():
        print(f"   {label}: {count}")
    
    # Show sample IDs for reference
    print(f"\n🆔 Seed example IDs:")
    seed_ids = seed_df[col_id].tolist()
    print(f"   {seed_ids[:10]}..." if len(seed_ids) > 10 else f"   {seed_ids}")
    
    print(f"\n" + "=" * 80)
    print("IMPORTANT: Use this seed set for ALL experiments on this dataset!")
    print("This ensures fair comparison across different models/methods.")
    print("=" * 80)
    
    return seed_file


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create fixed seed set for Active Learning')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (e.g., "yelp", "emotions"). If not provided, uses config.yaml')
    
    args = parser.parse_args()
    
    create_seed_set(args.dataset)


if __name__ == "__main__":
    main()


