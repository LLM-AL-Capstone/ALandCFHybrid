#!/usr/bin/env python3
"""
Post-hoc Retrieval Evaluation Script

Evaluates multiple retrieval strategies on a completed AL run without re-running
the AL + CF generation process. This allows fair comparison across retrievers
using the same labeled pool.

Usage:
    python evaluate_retrievers.py --run_folder output_data/{run_folder}
    python evaluate_retrievers.py --run_folder output_data/{run_folder} --retrievers bm25 contriever
    python evaluate_retrievers.py --run_folder output_data/{run_folder} --k_values 10 20 30

Output:
    Creates retrieval_comparison_results.csv in the run folder with results
    for all combinations of:
    - Retrievers: BM25, Contriever, BGE-Large
    - CF strategies: mixed, factual_anchored
    - Pool types: full (factuals + CFs), factuals_only
    - Budget checkpoints: Each iteration's labeled pool
"""

import argparse
import json
import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.llm_provider import get_llm_provider
from utils.retrieval_classifier import (
    BM25Retrieval,
    ContrieverRetrieval, 
    BGELargeRetrieval,
    SentenceTransformerRetrieval,
    TFIDFRetrieval
)
from utils.classifier import SimpleICLClassifier
from utils.data_loader import load_dataset


def load_config(run_folder: str) -> dict:
    """Load config from run folder or fall back to main config."""
    # Try to load config_used.yaml from run folder
    config_used_path = os.path.join(run_folder, 'config_used.yaml')
    if os.path.exists(config_used_path):
        with open(config_used_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Fall back to main config.yaml
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    raise FileNotFoundError(f"No config found in {run_folder} or current directory")


def load_test_set(config: dict) -> List[Dict]:
    """Load test set from config."""
    dataset_config = config['dataset']
    test_file = f"{config['directories']['input_data']}/{dataset_config['test_file']}"
    
    try:
        df_test = load_dataset(test_file, config)
    except FileNotFoundError:
        print(f"Warning: Test file not found: {test_file}")
        return []
    
    col_id = dataset_config['columns']['id']
    col_text = dataset_config['columns']['text']
    col_label = dataset_config['columns']['label']
    
    test_pool = []
    for _, row in df_test.iterrows():
        test_pool.append({
            'id': row[col_id],
            'text': row[col_text],
            'label': row[col_label]
        })
    
    return test_pool


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load a checkpoint file."""
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def load_final_pool(run_folder: str) -> List[Dict]:
    """Load final labeled pool from CSV."""
    pool_path = os.path.join(run_folder, 'final_labeled_pool.csv')
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Final labeled pool not found: {pool_path}")
    
    df = pd.read_csv(pool_path)
    pool = df.to_dict('records')
    return pool


def is_counterfactual(example: Dict) -> bool:
    """
    Check if an example is a counterfactual.
    
    CFs are identified by having 'original_id' field (reference to parent factual).
    Also handles 'is_counterfactual' field for backward compatibility.
    """
    # Primary check: CFs have original_id field
    if 'original_id' in example:
        orig_id = example['original_id']
        # Handle NaN from CSV loading
        if pd.notna(orig_id) and orig_id:
            return True
    # Backward compatibility: check is_counterfactual field
    is_cf = example.get('is_counterfactual', False)
    if isinstance(is_cf, str):
        return is_cf.lower() in ('true', '1', '1.0')
    return bool(is_cf)


def get_factuals_only(pool: List[Dict]) -> List[Dict]:
    """Filter pool to include only factual examples."""
    return [ex for ex in pool if not is_counterfactual(ex)]


def count_cfs(pool: List[Dict]) -> int:
    """Count counterfactuals in pool."""
    return len([ex for ex in pool if is_counterfactual(ex)])


def get_retriever_class(retriever_name: str):
    """Get retriever class by name."""
    retrievers = {
        'bm25': BM25Retrieval,
        'contriever': ContrieverRetrieval,
        'bge_large': BGELargeRetrieval,
        'sentence_transformers': SentenceTransformerRetrieval,
        'tfidf': TFIDFRetrieval
    }
    if retriever_name not in retrievers:
        raise ValueError(f"Unknown retriever: {retriever_name}. Options: {list(retrievers.keys())}")
    return retrievers[retriever_name]


def create_retriever_config(base_config: dict, retriever_name: str, cf_strategy: str, k_max: int) -> dict:
    """Create a modified config for a specific retriever."""
    config = base_config.copy()
    
    # Deep copy evaluation config
    config['evaluation'] = base_config['evaluation'].copy()
    config['evaluation']['retrieval'] = base_config['evaluation'].get('retrieval', {}).copy()
    
    # Set retriever and strategy
    config['evaluation']['classifier_type'] = 'retrieval'
    config['evaluation']['retrieval']['embedding_backend'] = retriever_name
    config['evaluation']['retrieval']['cf_inclusion_strategy'] = cf_strategy
    config['evaluation']['retrieval']['total_k_max'] = k_max
    
    return config


def evaluate_classifier(classifier, test_pool: List[Dict]) -> Dict:
    """Evaluate classifier on test set."""
    if not test_pool:
        return {}
    
    texts = [ex['text'] for ex in test_pool]
    true_labels = [ex['label'] for ex in test_pool]
    
    predictions = classifier.predict_batch(texts)
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1_macro': f1_score(true_labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(true_labels, predictions, average='weighted', zero_division=0),
        'precision_macro': precision_score(true_labels, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(true_labels, predictions, average='macro', zero_division=0)
    }
    
    return metrics


def run_evaluation(
    run_folder: str,
    retrievers: List[str] = None,
    cf_strategies: List[str] = None,
    k_values: List[int] = None,
    use_checkpoints: bool = True,
    include_static: bool = False
):
    """
    Run post-hoc evaluation on multiple retriever configurations.
    
    Args:
        run_folder: Path to completed AL run folder
        retrievers: List of retrievers to evaluate (default: all)
        cf_strategies: List of CF strategies (default: ['mixed', 'factual_anchored'])
        k_values: List of k_max values to test (default: [20])
        use_checkpoints: Whether to evaluate at each checkpoint (default: True)
        include_static: Whether to include Static ICL evaluation (default: False)
    """
    print("=" * 80)
    print("Post-hoc Retrieval Evaluation")
    print("=" * 80)
    print(f"Run folder: {run_folder}")
    
    # Default values
    if retrievers is None:
        retrievers = ['bm25', 'contriever', 'bge_large']
    if cf_strategies is None:
        cf_strategies = ['mixed', 'factual_anchored']
    if k_values is None:
        k_values = [20]
    
    print(f"Retrievers: {retrievers}")
    print(f"CF strategies: {cf_strategies}")
    print(f"K values: {k_values}")
    print(f"Include Static ICL: {include_static}")
    
    # Load config and test set
    config = load_config(run_folder)
    test_pool = load_test_set(config)
    print(f"Test set: {len(test_pool)} examples")
    
    # Initialize LLM provider
    llm_provider = get_llm_provider(config)
    
    # Find checkpoints
    checkpoint_dir = os.path.join(run_folder, 'checkpoints')
    checkpoints = []
    
    if use_checkpoints and os.path.exists(checkpoint_dir):
        checkpoint_files = sorted([
            f for f in os.listdir(checkpoint_dir) 
            if f.startswith('checkpoint_iter_') and f.endswith('.json')
        ])
        for cf in checkpoint_files:
            iter_num = int(cf.split('_')[2].split('.')[0])
            checkpoints.append({
                'iteration': iter_num,
                'path': os.path.join(checkpoint_dir, cf)
            })
        print(f"Found {len(checkpoints)} checkpoints")
    
    # Also add final pool (but avoid duplicates with checkpoints)
    try:
        final_pool = load_final_pool(run_folder)
        # Determine iteration number from al_results.csv
        results_path = os.path.join(run_folder, 'al_results.csv')
        if os.path.exists(results_path):
            df_results = pd.read_csv(results_path)
            final_iter = int(df_results['iteration'].max())
        else:
            final_iter = len(checkpoints) + 1
        
        # Only add final pool if not already covered by a checkpoint
        existing_iters = {cp['iteration'] for cp in checkpoints}
        if final_iter not in existing_iters:
            checkpoints.append({
                'iteration': final_iter,
                'pool': final_pool,
                'is_final': True
            })
            print(f"Added final pool (iteration {final_iter})")
        else:
            print(f"Final pool iteration {final_iter} already covered by checkpoint, skipping duplicate")
    except FileNotFoundError:
        print("Warning: No final_labeled_pool.csv found")
    
    # Sort checkpoints by iteration
    checkpoints.sort(key=lambda x: x['iteration'])
    print(f"Evaluating {len(checkpoints)} checkpoints: iterations {[cp['iteration'] for cp in checkpoints]}")
    
    if not checkpoints:
        print("Error: No checkpoints or final pool found!")
        return
    
    # Get batch_size from config for budget calculation
    batch_size = config['active_learning'].get('batch_size', 10)
    
    # Run evaluations
    all_results = []
    # Calculate total evaluations: retrievers * strategies * k_values * 2 (full/factuals_only) + static (if enabled) * 2
    total_evals = len(checkpoints) * len(retrievers) * len(cf_strategies) * len(k_values) * 2
    if include_static:
        total_evals += len(checkpoints) * 2  # Static ICL: full + factuals_only
    eval_count = 0
    
    for checkpoint in checkpoints:
        iteration = checkpoint['iteration']
        
        # Load pool
        if 'pool' in checkpoint:
            labeled_pool = checkpoint['pool']
        else:
            cp_data = load_checkpoint(checkpoint['path'])
            labeled_pool = cp_data['labeled_pool']
        
        # Get factuals-only pool
        factuals_only = get_factuals_only(labeled_pool)
        
        # Calculate budget used = iteration * batch_size (simpler and more accurate)
        # Budget represents factuals labeled beyond the seed set
        budget_used = iteration * batch_size
        
        total_factuals = len(factuals_only)
        total_cfs = count_cfs(labeled_pool)
        
        print(f"\n--- Iteration {iteration} (factuals: {total_factuals}, CFs: {total_cfs}) ---")
        
        # ========== Static ICL Evaluation ==========
        if include_static:
            # Evaluate Static ICL on full pool
            eval_count += 1
            print(f"  [{eval_count}/{total_evals}] static / - / - / full...", end=" ")
            
            try:
                classifier = SimpleICLClassifier(config, llm_provider)
                classifier.train(labeled_pool)
                
                metrics = evaluate_classifier(classifier, test_pool)
                
                result = {
                    'iteration': iteration,
                    'budget_used': budget_used,
                    'pool_type': 'full',
                    'retriever': 'static',
                    'cf_strategy': '-',
                    'k_max': '-',
                    'labeled_pool_size': len(labeled_pool),
                    'total_factuals': total_factuals,
                    'total_cfs': total_cfs,
                    **metrics
                }
                all_results.append(result)
                print(f"F1={metrics.get('f1_macro', 'N/A'):.4f}" if metrics else "No metrics")
            except Exception as e:
                print(f"Error: {e}")
            
            # Evaluate Static ICL on factuals-only pool
            eval_count += 1
            print(f"  [{eval_count}/{total_evals}] static / - / - / factuals_only...", end=" ")
            
            try:
                classifier = SimpleICLClassifier(config, llm_provider)
                classifier.train(factuals_only)
                
                metrics = evaluate_classifier(classifier, test_pool)
                
                result = {
                    'iteration': iteration,
                    'budget_used': budget_used,
                    'pool_type': 'factuals_only',
                    'retriever': 'static',
                    'cf_strategy': '-',
                    'k_max': '-',
                    'labeled_pool_size': len(factuals_only),
                    'total_factuals': total_factuals,
                    'total_cfs': 0,
                    **metrics
                }
                all_results.append(result)
                print(f"F1={metrics.get('f1_macro', 'N/A'):.4f}" if metrics else "No metrics")
            except Exception as e:
                print(f"Error: {e}")
        
        # ========== Retrieval-based Evaluation ==========
        for retriever_name in retrievers:
            for cf_strategy in cf_strategies:
                for k_max in k_values:
                    # Evaluate on full pool (factuals + CFs)
                    eval_count += 1
                    print(f"  [{eval_count}/{total_evals}] {retriever_name} / {cf_strategy} / k={k_max} / full...", end=" ")
                    
                    try:
                        retriever_config = create_retriever_config(config, retriever_name, cf_strategy, k_max)
                        RetrieverClass = get_retriever_class(retriever_name)
                        classifier = RetrieverClass(retriever_config, llm_provider)
                        classifier.train(labeled_pool)
                        
                        metrics = evaluate_classifier(classifier, test_pool)
                        
                        result = {
                            'iteration': iteration,
                            'budget_used': budget_used,
                            'pool_type': 'full',
                            'retriever': retriever_name,
                            'cf_strategy': cf_strategy,
                            'k_max': k_max,
                            'labeled_pool_size': len(labeled_pool),
                            'total_factuals': total_factuals,
                            'total_cfs': total_cfs,
                            **metrics
                        }
                        all_results.append(result)
                        print(f"F1={metrics.get('f1_macro', 'N/A'):.4f}" if metrics else "No metrics")
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    # Evaluate on factuals-only pool
                    eval_count += 1
                    print(f"  [{eval_count}/{total_evals}] {retriever_name} / - / k={k_max} / factuals_only...", end=" ")
                    
                    try:
                        retriever_config = create_retriever_config(config, retriever_name, 'mixed', k_max)
                        RetrieverClass = get_retriever_class(retriever_name)
                        classifier = RetrieverClass(retriever_config, llm_provider)
                        classifier.train(factuals_only)
                        
                        metrics = evaluate_classifier(classifier, test_pool)
                        
                        result = {
                            'iteration': iteration,
                            'budget_used': budget_used,
                            'pool_type': 'factuals_only',
                            'retriever': retriever_name,
                            'cf_strategy': '-',
                            'k_max': k_max,
                            'labeled_pool_size': len(factuals_only),
                            'total_factuals': total_factuals,
                            'total_cfs': 0,
                            **metrics
                        }
                        all_results.append(result)
                        print(f"F1={metrics.get('f1_macro', 'N/A'):.4f}" if metrics else "No metrics")
                    except Exception as e:
                        print(f"Error: {e}")
    
    # Save results
    if all_results:
        output_path = os.path.join(run_folder, 'retrieval_comparison_results.csv')
        df_results = pd.DataFrame(all_results)
        
        # Sort by (retriever, cf_strategy, pool_type, iteration) for easy table splitting
        sort_order = {
            'retriever': ['static', 'bm25', 'contriever', 'bge_large'],
            'pool_type': ['full', 'factuals_only'],
            'cf_strategy': ['-', 'mixed', 'factual_anchored']
        }
        
        # Create sort keys
        df_results['retriever_order'] = df_results['retriever'].map(
            {v: i for i, v in enumerate(sort_order['retriever'])}
        ).fillna(99)
        df_results['pool_order'] = df_results['pool_type'].map(
            {v: i for i, v in enumerate(sort_order['pool_type'])}
        ).fillna(99)
        df_results['cf_order'] = df_results['cf_strategy'].map(
            {v: i for i, v in enumerate(sort_order['cf_strategy'])}
        ).fillna(99)
        
        # Sort
        df_results = df_results.sort_values(
            by=['retriever_order', 'cf_order', 'pool_order', 'iteration']
        )
        
        # Drop sort columns
        df_results = df_results.drop(columns=['retriever_order', 'pool_order', 'cf_order'])
        
        df_results.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        print(f"   Total evaluations: {len(all_results)}")
        print(f"   Sorted by: retriever → cf_strategy → pool_type → iteration")
        
        # Create pivot table: strategies (rows) × budgets (columns)
        # Create a combined strategy name for rows
        df_results['strategy'] = df_results.apply(
            lambda row: f"{row['retriever']} {row['cf_strategy']} ({row['pool_type']})" 
            if row['cf_strategy'] != '-' 
            else f"{row['retriever']} ({row['pool_type']})",
            axis=1
        )
        
        # Create pivot table with budget as columns
        pivot_df = df_results.pivot_table(
            values='f1_macro',
            index='strategy',
            columns='budget_used',
            aggfunc='first'
        )
        
        # Sort the index to match our desired order
        strategy_order = []
        for retriever in ['static', 'bm25', 'contriever', 'bge_large']:
            for cf_strat in ['-', 'mixed', 'factual_anchored']:
                for pool in ['full', 'factuals_only']:
                    if cf_strat == '-':
                        strategy_order.append(f"{retriever} ({pool})")
                    else:
                        strategy_order.append(f"{retriever} {cf_strat} ({pool})")
        
        # Reindex to match order (only keep existing strategies)
        existing_strategies = [s for s in strategy_order if s in pivot_df.index]
        pivot_df = pivot_df.reindex(existing_strategies)
        
        # Sort columns (budgets) numerically
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
        
        # Save pivot table
        pivot_path = os.path.join(run_folder, 'retrieval_comparison_pivot.csv')
        pivot_df.to_csv(pivot_path)
        print(f"✅ Pivot table saved to: {pivot_path}")
        print(f"   Format: strategies (rows) × budgets (columns)")
        
        # Print summary table
        print("\n" + "=" * 80)
        print("Pivot Table (F1 Macro)")
        print("=" * 80)
        print(pivot_df.round(4).to_string())
    else:
        print("\n❌ No results generated!")


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation of multiple retrieval strategies on a completed AL run"
    )
    parser.add_argument(
        '--run_folder', 
        type=str, 
        required=True,
        help="Path to the completed AL run folder"
    )
    parser.add_argument(
        '--retrievers',
        nargs='+',
        default=['bm25', 'contriever', 'bge_large'],
        help="Retrievers to evaluate (default: bm25 contriever bge_large)"
    )
    parser.add_argument(
        '--cf_strategies',
        nargs='+',
        default=['mixed', 'factual_anchored'],
        help="CF inclusion strategies (default: mixed factual_anchored)"
    )
    parser.add_argument(
        '--k_values',
        nargs='+',
        type=int,
        default=[20],
        help="K values for retrieval budget (default: 20)"
    )
    parser.add_argument(
        '--no_checkpoints',
        action='store_true',
        help="Only evaluate final pool, skip checkpoints"
    )
    parser.add_argument(
        '--include_static',
        action='store_true',
        help="Include Static ICL evaluation for comparison"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        run_folder=args.run_folder,
        retrievers=args.retrievers,
        cf_strategies=args.cf_strategies,
        k_values=args.k_values,
        use_checkpoints=not args.no_checkpoints,
        include_static=args.include_static
    )


if __name__ == '__main__':
    main()

