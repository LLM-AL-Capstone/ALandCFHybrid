#!/usr/bin/env python3
"""
Full-ICL Oracle Baseline

Budget = Number of few-shot examples in ICL prompt
Uses FULL training dataset for retrieval
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
import yaml
import pandas as pd
from datetime import datetime
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.config_loader import load_config
from utils.data_loader import load_dataset, get_unique_labels, shuffle_dataframe
from utils.llm_provider import get_llm_provider
from utils.classifier import SimpleICLClassifier


def load_baseline_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'baseline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_budget(train_data, test_data, budget, config, llm, labels, run_dir, method):
    print(f"\n{'='*70}")
    print(f"  ICL Budget={budget}, Retrieval={method}")
    print(f"{'='*70}")
    
    print(f"  Full training pool: {len(train_data)} examples")
    print(f"  ICL budget: {budget} examples")
    print(f"  Test set: {len(test_data)} examples")
    
    # Initialize classifier with budget
    classifier = SimpleICLClassifier(config, llm)
    classifier.k_per_class = budget
    classifier.max_icl_examples = budget
    
    # Train on FULL dataset
    classifier.train(train_data)
    
    # Evaluate
    test_texts = [ex['text'] for ex in test_data]
    test_labels = [ex['label'] for ex in test_data]
    predictions = classifier.predict_batch(test_texts)
    
    # Metrics
    acc = accuracy_score(test_labels, predictions)
    f1m = f1_score(test_labels, predictions, average='macro', zero_division=0)
    f1w = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    
    print(f"  Accuracy: {acc:.4f}, F1 Macro: {f1m:.4f}")
    
    # Save report
    report_path = os.path.join(run_dir, f'report_{method}_budget_{budget}.txt')
    with open(report_path, 'w') as f:
        f.write(f"ICL Budget: {budget}\n")
        f.write(f"Training Pool: {len(train_data)} (FULL dataset)\n\n")
        f.write(classification_report(test_labels, predictions, digits=4))
    
    return {'icl_budget': budget, 'accuracy': float(acc), 'f1_macro': float(f1m), 'f1_weighted': float(f1w)}


def main():
    print("\n" + "="*70)
    print("  FULL-ICL ORACLE BASELINE")
    print("="*70)
    
    config = load_baseline_config()
    os.makedirs(config['directories']['output_data'], exist_ok=True)
    
    llm = get_llm_provider(config)
    
    # Build file paths
    train_path = os.path.join(config['directories']['input_data'], config['dataset']['train_file'])
    test_path = os.path.join(config['directories']['input_data'], config['dataset']['test_file'])
    
    # Load datasets
    train_df = load_dataset(train_path, config)
    test_df = load_dataset(test_path, config)
    
    # Filter excluded labels
    exclude = config['dataset'].get('exclude_labels', [])
    col_label = config['dataset']['columns']['label']
    if exclude:
        train_df = train_df[~train_df[col_label].isin(exclude)]
        test_df = test_df[~test_df[col_label].isin(exclude)]
    
    train_df = shuffle_dataframe(train_df, config['processing']['seed'])
    
    # Get column names
    col_id = config['dataset']['columns']['id']
    col_text = config['dataset']['columns']['text']
    
    # Convert to standardized dict format with 'text' and 'label' keys
    train_data = [
        {'id': row[col_id], 'text': row[col_text], 'label': row[col_label]}
        for _, row in train_df.iterrows()
    ]
    test_data = [
        {'id': row[col_id], 'text': row[col_text], 'label': row[col_label]}
        for _, row in test_df.iterrows()
    ]
    
    labels = get_unique_labels(train_df, col_label, exclude)
    
    print(f"\nFull training: {len(train_data)}, Test: {len(test_data)}, Labels: {labels}")
    
    # Build output directory name with timestamp, model, and dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['llm']['openai']['model']
    dataset_name = config['dataset']['train_file'].replace('_train.csv', '').replace('.csv', '')
    run_dir = os.path.join(config['directories']['output_data'], f'full_icl_oracle_{timestamp}_{model_name}_{dataset_name}')
    os.makedirs(run_dir, exist_ok=True)
    
    methods = config['baseline']['retrieval_methods']
    budgets = config['baseline']['icl_budgets']
    
    all_results = {m: {} for m in methods}
    
    for method in methods:
        for budget in budgets:
            if len(train_data) >= budget:
                r = evaluate_budget(train_data, test_data, budget, config, llm, labels, run_dir, method)
                all_results[method][budget] = r
    
    with open(os.path.join(run_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    for method in methods:
        print(f"{method.upper()}")
        print(f"{'Budget':<10} {'Accuracy':<12} {'F1 Macro':<12}")
        print("-"*35)
        for budget in budgets:
            if budget in all_results[method]:
                r = all_results[method][budget]
                print(f"{budget:<10} {r['accuracy']:<12.4f} {r['f1_macro']:<12.4f}")
    
    print(f"\nResults: {run_dir}\n")


if __name__ == "__main__":
    main()
