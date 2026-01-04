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


def load_existing_results(run_folder: str) -> pd.DataFrame:
    """Load existing results if available."""
    results_path = os.path.join(run_folder, 'retrieval_comparison_results.csv')
    if os.path.exists(results_path):
        try:
            df = pd.read_csv(results_path)
            print(f"📂 Loaded {len(df)} existing results from previous run")
            return df
        except Exception as e:
            print(f"⚠️  Could not load existing results: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def normalize_k_max(k_max) -> str:
    """Normalize k_max to string for comparison."""
    if pd.isna(k_max) or k_max == '-' or k_max == '-':
        return '-'
    try:
        return str(int(k_max))
    except (ValueError, TypeError):
        return '-'


def get_evaluated_combinations(df: pd.DataFrame) -> set:
    """Get set of already-evaluated combinations as tuples."""
    if df.empty:
        return set()
    
    evaluated = set()
    for _, row in df.iterrows():
        # Normalize k_max for consistent comparison
        k_max_norm = normalize_k_max(row['k_max'])
        # Use budget_used for comparison (more accurate than iteration for fractional iterations)
        budget = int(row['budget_used']) if 'budget_used' in row else int(row.get('iteration', 0) * 10)
        key = (
            budget,
            str(row['retriever']),
            str(row['cf_strategy']),
            str(row['pool_type']),
            k_max_norm
        )
        evaluated.add(key)
    return evaluated


def is_already_evaluated(budget_used: int, retriever: str, cf_strategy: str, 
                         pool_type: str, k_max, evaluated_combinations: set) -> bool:
    """Check if an evaluation combination is already done."""
    k_max_norm = normalize_k_max(k_max)
    key = (int(budget_used), retriever, cf_strategy, pool_type, k_max_norm)
    return key in evaluated_combinations


def load_checkpoints_with_custom_budgets(run_folder: str, checkpoint_dir: str, batch_size: int, config: dict, custom_budgets: List[int] = None) -> List[Dict]:
    """
    Load checkpoints and create intermediate ones for custom budgets.
    
    Args:
        run_folder: Run folder path
        checkpoint_dir: Checkpoint directory
        batch_size: Batch size for budget calculation
        config: Configuration dictionary (needed for seed size calculation)
        custom_budgets: List of custom budget values (e.g., [0, 2, 5])
    
    Returns:
        List of checkpoint dictionaries with iteration, budget, and pool
    """
    checkpoints = []
    
    # Load existing checkpoints
    if os.path.exists(checkpoint_dir):
        checkpoint_files = sorted([
            f for f in os.listdir(checkpoint_dir) 
            if f.startswith('checkpoint_iter_') and f.endswith('.json')
        ])
        for cf in checkpoint_files:
            iter_num = int(cf.split('_')[2].split('.')[0])
            cp_data = load_checkpoint(os.path.join(checkpoint_dir, cf))
            checkpoints.append({
                'iteration': iter_num,
                'budget': iter_num * batch_size,
                'pool': cp_data['labeled_pool'],
                'path': os.path.join(checkpoint_dir, cf)
            })
    
    # Add final pool if exists
    try:
        final_pool = load_final_pool(run_folder)
        results_path = os.path.join(run_folder, 'al_results.csv')
        if os.path.exists(results_path):
            df_results = pd.read_csv(results_path)
            final_iter = int(df_results['iteration'].max())
            final_budget = final_iter * batch_size
        else:
            final_iter = len(checkpoints) if checkpoints else 0
            final_budget = final_iter * batch_size
        
        existing_budgets = {cp['budget'] for cp in checkpoints}
        if final_budget not in existing_budgets:
            checkpoints.append({
                'iteration': final_iter,
                'budget': final_budget,
                'pool': final_pool,
                'is_final': True
            })
    except FileNotFoundError:
        pass
    
    # Sort by budget
    checkpoints.sort(key=lambda x: x['budget'])
    
    # Create intermediate checkpoints for custom budgets
    if custom_budgets:
        # Get seed size from config for budget 0
        initial_per_class = config.get('active_learning', {}).get('initial_labeled_per_class', 5)
        
        # Get number of classes from first checkpoint or al_results
        num_classes = None
        if checkpoints:
            first_cp_factuals = get_factuals_only(checkpoints[0]['pool'])
            if first_cp_factuals:
                labels = {ex['label'] for ex in first_cp_factuals}
                num_classes = len(labels)
        
        for target_budget in custom_budgets:
            # Check if already exists
            if any(cp['budget'] == target_budget for cp in checkpoints):
                continue
            
            # Special handling for budget 0 (seed set)
            if target_budget == 0:
                if not checkpoints:
                    continue  # Can't create budget 0 without any checkpoints
                
                # Extract seed set from first checkpoint
                first_cp = checkpoints[0]
                first_cp_factuals = get_factuals_only(first_cp['pool'])
                
                # Get seed size
                if num_classes:
                    seed_size = initial_per_class * num_classes
                else:
                    # Fallback: use labeled_pool_size from al_results for iteration 0
                    try:
                        results_path = os.path.join(run_folder, 'al_results.csv')
                        if os.path.exists(results_path):
                            df_results = pd.read_csv(results_path)
                            seed_row = df_results[df_results['iteration'] == 0]
                            if not seed_row.empty:
                                seed_size = int(seed_row['labeled_pool_size'].iloc[0])
                            else:
                                seed_size = len(first_cp_factuals) - batch_size  # Approximate
                        else:
                            seed_size = len(first_cp_factuals) - batch_size  # Approximate
                    except:
                        seed_size = len(first_cp_factuals) - batch_size  # Approximate
                
                # Take first seed_size factuals (seed set)
                seed_pool = first_cp_factuals[:seed_size].copy()
                
                checkpoints.append({
                    'iteration': 0.0,
                    'budget': 0,
                    'pool': seed_pool,
                    'is_intermediate': True,
                    'is_seed': True
                })
                # Re-sort after adding budget 0 so it's in correct position for next iterations
                checkpoints.sort(key=lambda x: x['budget'])
                continue
            
            # For budgets 2, 5, etc. - find surrounding checkpoints
            # Find the largest checkpoint <= target_budget (prev_cp)
            # Find the smallest checkpoint > target_budget (next_cp)
            prev_cp = None
            next_cp = None
            for cp in checkpoints:
                if cp['budget'] <= target_budget:
                    # Keep the largest one <= target_budget
                    if prev_cp is None or cp['budget'] > prev_cp['budget']:
                        prev_cp = cp
                elif cp['budget'] > target_budget:
                    # Keep the smallest one > target_budget
                    if next_cp is None or cp['budget'] < next_cp['budget']:
                        next_cp = cp
            
            if prev_cp is None:
                continue  # Can't create checkpoint before first one
            
            # Create intermediate pool
            if next_cp is None:
                # Use previous checkpoint's pool (budget >= target)
                intermediate_pool = prev_cp['pool'].copy()
            else:
                # Interpolate: take first N factuals from next checkpoint
                factuals_only = get_factuals_only(next_cp['pool'])
                prev_factuals = get_factuals_only(prev_cp['pool'])
                
                # Calculate seed size - find budget 0 checkpoint if it exists
                seed_cp = None
                for cp in checkpoints:
                    if cp['budget'] == 0:
                        seed_cp = cp
                        break
                
                if seed_cp:
                    # Use actual seed size from budget 0 checkpoint
                    seed_factuals = get_factuals_only(seed_cp['pool'])
                    seed_size = len(seed_factuals)
                elif prev_cp['budget'] == 0:
                    # prev_cp is budget 0
                    seed_size = len(prev_factuals)
                else:
                    # Estimate seed size from previous checkpoint
                    # prev_cp has budget B, so it has seed_size + B factuals
                    # Therefore: seed_size = len(prev_factuals) - B
                    seed_size = len(prev_factuals) - prev_cp['budget']
                
                num_factuals_needed = seed_size + target_budget
                
                if num_factuals_needed <= len(factuals_only):
                    # Take first N factuals
                    intermediate_pool = factuals_only[:num_factuals_needed].copy()
                    # Add any CFs that belong to these factuals
                    factual_ids = {f['id'] for f in intermediate_pool}
                    for cf in next_cp['pool']:
                        if is_counterfactual(cf) and cf.get('original_id') in factual_ids:
                            intermediate_pool.append(cf)
                else:
                    # Fallback: use next checkpoint's pool
                    intermediate_pool = next_cp['pool'].copy()
            
            # Calculate fractional iteration
            fractional_iter = target_budget / batch_size
            
            checkpoints.append({
                'iteration': fractional_iter,
                'budget': target_budget,
                'pool': intermediate_pool,
                'is_intermediate': True
            })
            # Re-sort after adding intermediate checkpoint for next iterations
            checkpoints.sort(key=lambda x: x['budget'])
    
    # Final re-sort by budget (redundant but safe)
    checkpoints.sort(key=lambda x: x['budget'])
    return checkpoints


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
    
    # Get batch_size from config for budget calculation
    batch_size = config['active_learning'].get('batch_size', 10)
    
    # Custom budget points to evaluate (0, 2, 5 for low budgets)
    custom_budgets = [0, 2, 5]
    
    if use_checkpoints:
        checkpoints = load_checkpoints_with_custom_budgets(
            run_folder,
            checkpoint_dir,
            batch_size,
            config,  # Pass config for seed size calculation
            custom_budgets=custom_budgets
        )
        print(f"Found {len(checkpoints)} checkpoints (including custom budgets)")
        print(f"Budgets: {[cp['budget'] for cp in checkpoints]}")
        
        # Filter checkpoints by max_budget if specified (to save tokens)
        post_hoc_config = config.get('post_hoc_evaluation', {})
        max_budget = post_hoc_config.get('max_budget', None)
        
        if max_budget is not None:
            original_count = len(checkpoints)
            checkpoints = [cp for cp in checkpoints if cp['budget'] <= max_budget]
            print(f"📊 Filtered to budgets <= {max_budget}: {[cp['budget'] for cp in checkpoints]} (removed {original_count - len(checkpoints)} checkpoints)")
    else:
        # Fallback: load final pool only
        try:
            final_pool = load_final_pool(run_folder)
            results_path = os.path.join(run_folder, 'al_results.csv')
            if os.path.exists(results_path):
                df_results = pd.read_csv(results_path)
                final_iter = int(df_results['iteration'].max())
            else:
                final_iter = 0
            checkpoints = [{
                'iteration': final_iter,
                'budget': final_iter * batch_size,
                'pool': final_pool,
                'is_final': True
            }]
        except FileNotFoundError:
            print("Error: No checkpoints or final pool found!")
            return
    
    # Sort checkpoints by budget
    checkpoints.sort(key=lambda x: x['budget'])
    print(f"Evaluating {len(checkpoints)} checkpoints: budgets {[cp['budget'] for cp in checkpoints]}")
    
    if not checkpoints:
        print("Error: No checkpoints or final pool found!")
        return
    
    # Load existing results for resume functionality
    existing_df = load_existing_results(run_folder)
    evaluated_combinations = get_evaluated_combinations(existing_df)
    print(f"📊 Found {len(evaluated_combinations)} already-evaluated combinations")
    
    # Run evaluations
    all_results = []
    # Calculate total evaluations: retrievers * strategies * k_values * 2 (full/factuals_only) + static (if enabled) * 2
    total_evals = len(checkpoints) * len(retrievers) * len(cf_strategies) * len(k_values) * 2
    if include_static:
        total_evals += len(checkpoints) * 2  # Static ICL: full + factuals_only
    eval_count = 0
    skipped_count = 0
    
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
        
        # Calculate budget used - use budget from checkpoint if available, otherwise calculate
        # Budget represents factuals labeled beyond the seed set
        budget_used = checkpoint.get('budget', iteration * batch_size)
        
        total_factuals = len(factuals_only)
        total_cfs = count_cfs(labeled_pool)
        
        print(f"\n--- Iteration {iteration} (Budget: {budget_used}, factuals: {total_factuals}, CFs: {total_cfs}) ---")
        
        # ========== Static ICL Evaluation ==========
        if include_static:
            # Evaluate Static ICL on full pool
            if is_already_evaluated(budget_used, 'static', '-', 'full', '-', evaluated_combinations):
                print(f"  ⏭️  Skipping static / - / - / full (already done)")
                skipped_count += 1
            else:
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
            if is_already_evaluated(budget_used, 'static', '-', 'factuals_only', '-', evaluated_combinations):
                print(f"  ⏭️  Skipping static / - / - / factuals_only (already done)")
                skipped_count += 1
            else:
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
                    if is_already_evaluated(budget_used, retriever_name, cf_strategy, 'full', k_max, evaluated_combinations):
                        print(f"  ⏭️  Skipping {retriever_name} / {cf_strategy} / k={k_max} / full (already done)")
                        skipped_count += 1
                    else:
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
                    if is_already_evaluated(budget_used, retriever_name, '-', 'factuals_only', k_max, evaluated_combinations):
                        print(f"  ⏭️  Skipping {retriever_name} / - / k={k_max} / factuals_only (already done)")
                        skipped_count += 1
                    else:
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
    if all_results or not existing_df.empty:
        output_path = os.path.join(run_folder, 'retrieval_comparison_results.csv')
        
        # Merge existing and new results
        if not existing_df.empty and all_results:
            print(f"\n📊 Merging {len(existing_df)} existing results with {len(all_results)} new results")
            new_df = pd.DataFrame(all_results)
            
            # Combine dataframes
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates (keep the new ones if there are any)
            key_cols = ['iteration', 'retriever', 'cf_strategy', 'pool_type', 'k_max']
            # Normalize k_max for deduplication
            combined_df['k_max_norm'] = combined_df['k_max'].apply(normalize_k_max)
            combined_df = combined_df.drop_duplicates(
                subset=['iteration', 'retriever', 'cf_strategy', 'pool_type', 'k_max_norm'],
                keep='last'
            )
            combined_df = combined_df.drop(columns=['k_max_norm'])
            
            df_results = combined_df
        elif all_results:
            df_results = pd.DataFrame(all_results)
        else:
            # No new results, just use existing
            df_results = existing_df
        
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
        print(f"   New evaluations: {len(all_results)}")
        print(f"   Skipped (already done): {skipped_count}")
        print(f"   Total results: {len(df_results)}")
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

