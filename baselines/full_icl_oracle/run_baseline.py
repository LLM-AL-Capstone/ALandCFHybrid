#!/usr/bin/env python3
"""
Full-ICL Oracle Baseline (Revised)

This baseline establishes the UPPER BOUND by using ALL training labels.
Annotation budget does NOT apply - Full-ICL always has access to entire labeled dataset.

Retrieval Methods: BM25, Contriever
ICL Budgets (k): 10, 20 (number of few-shot examples in prompt)

Results serve as horizontal lines for comparison with Active Learning.
"""

import os
import json
import yaml
import pandas as pd
from datetime import datetime
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import local retrieval classifier
from retrieval_icl_classifier import RetrievalICLClassifier

# Import LLM provider from parent directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.llm_provider import get_llm_provider


def load_baseline_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'baseline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_full_icl_baseline(
    dataset_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    icl_budgets: List[int],
    retrieval_method: str,
    llm_provider,
    output_dir: str
):
    """
    Run Full-ICL Oracle baseline (REVISED)
    
    Full-ICL Oracle uses ALL training labels (annotation budget N/A).
    ICL budget k = number of few-shot examples retrieved per test sample.
    
    Args:
        dataset_name: Name of dataset
        train_df: Training DataFrame (ALL LABELED)
        test_df: Test DataFrame
        icl_budgets: [10, 20] - number of examples in ICL prompt
        retrieval_method: 'bm25' or 'contriever'
        llm_provider: LLM provider instance
        output_dir: Output directory
    """
    print(f"\n{'='*80}")
    print(f"Full-ICL Oracle Baseline: {dataset_name}")
    print(f"Retrieval Method: {retrieval_method.upper()}")
    print(f"{'='*80}\n")
    
    print(f"Training samples: {len(train_df)} (ALL LABELED - annotation budget N/A)")
    print(f"Test samples: {len(test_df)}")
    print(f"ICL budgets (k): {icl_budgets}")
    
    # Get label space
    label_list = sorted(train_df['label'].unique().tolist())
    print(f"Label space: {label_list}\n")
    
    results = []
    
    for k in icl_budgets:
        print(f"\n{'-'*80}")
        print(f"Running with ICL budget k = {k}")
        print(f"{'-'*80}")
        
        # Initialize retrieval-based classifier
        clf = RetrievalICLClassifier(
            llm_provider=llm_provider,
            retrieval_method=retrieval_method,
            k=k
        )
        
        # Build retrieval index from FULL training set
        clf.fit(train_df, text_col='text', label_col='label')
        
        # Predict on test set
        print(f"  Classifying {len(test_df)} test examples...")
        test_texts = test_df['text'].tolist()
        predictions = clf.predict_batch(test_texts)
        
        # Calculate metrics
        test_labels = test_df['label'].tolist()
        accuracy = accuracy_score(test_labels, predictions)
        f1_macro = f1_score(test_labels, predictions, average='macro')
        f1_weighted = f1_score(test_labels, predictions, average='weighted')
        
        print(f"\nResults for k={k}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Macro: {f1_macro:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        
        # Save detailed report
        report_path = os.path.join(output_dir, f'report_{retrieval_method}_k{k}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Full-ICL Oracle Baseline\n")
            f.write(f"Retrieval Method: {retrieval_method}\n")
            f.write(f"ICL Budget k: {k}\n")
            f.write(f"Training Pool: {len(train_df)} (ALL LABELED)\n")
            f.write(f"Test Set: {len(test_df)}\n\n")
            f.write(classification_report(test_labels, predictions, digits=4))
        
        results.append({
            'retrieval_method': retrieval_method,
            'icl_budget_k': k,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'train_size_all_labeled': len(train_df),
            'test_size': len(test_df)
        })
    
    return results


def main():
    print("\n" + "="*80)
    print("FULL-ICL ORACLE BASELINE (REVISED)")
    print("="*80)
    
    # Load config
    config = load_baseline_config()
    
    # Build file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, 'input_data', config['dataset']['train_file'])
    test_path = os.path.join(base_dir, 'input_data', config['dataset']['test_file'])
    
    # Load data
    print(f"\nLoading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Get column mappings
    text_col = config['dataset']['columns']['text']
    label_col = config['dataset']['columns']['label']
    
    # Rename to standard columns
    train_df = train_df.rename(columns={text_col: 'text', label_col: 'label'})
    test_df = test_df.rename(columns={text_col: 'text', label_col: 'label'})
    
    # Extract dataset name
    dataset_name = config['dataset']['train_file'].replace('_train.csv', '').replace('.csv', '')
    
    # Build output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['llm']['openai']['model']
    output_dir = os.path.join(
        os.path.dirname(__file__),
        'output',
        f'full_icl_oracle_{timestamp}_{model_name}_{dataset_name}'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Initialize LLM
    llm_provider = get_llm_provider(config)
    
    # Get parameters
    icl_budgets = config['baseline']['icl_budgets']
    retrieval_methods = config['baseline']['retrieval_methods']
    
    all_results = []
    
    # Run baseline for each retrieval method
    for retrieval_method in retrieval_methods:
        results = run_full_icl_baseline(
            dataset_name=dataset_name,
            train_df=train_df,
            test_df=test_df,
            icl_budgets=icl_budgets,
            retrieval_method=retrieval_method,
            llm_provider=llm_provider,
            output_dir=output_dir
        )
        all_results.extend(results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, 'full_icl_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Save config
    config_path = os.path.join(output_dir, 'config_used.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for retrieval_method in retrieval_methods:
        print(f"{retrieval_method.upper()}")
        print(f"{'k':<10} {'Accuracy':<12} {'F1 Macro':<12}")
        print("-"*35)
        method_results = [r for r in all_results if r['retrieval_method'] == retrieval_method]
        for r in method_results:
            print(f"{r['icl_budget_k']:<10} {r['accuracy']:<12.4f} {r['f1_macro']:<12.4f}")
        print()
    
    print(f"Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
