#!/usr/bin/env python3
"""
Script 05: Active Learning Loop with Counterfactual Augmentation

Implements baseline Active Learning using:
- Uncertainty sampling for query strategy
- Simulated oracle for labeling
- Simplified counterfactual generation for data augmentation

This replaces the old pattern-based pipeline with a cleaner, more focused approach.
"""

import sys
import random
import os
import pandas as pd
import json
import numpy as np
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

from utils import (
    load_config,
    ensure_directories,
    load_dataset,
    get_unique_labels,
    shuffle_dataframe,
    get_llm_provider
)
from utils.classifier import SimpleICLClassifier
from utils.oracle import get_oracle
from utils.uncertainty import select_uncertain_examples, get_uncertainty_statistics
from utils.counterfactual_generator import generate_counterfactuals_batch


def initialize_pools(config: dict) -> tuple:
    """
    Initialize labeled and unlabeled pools from training data.
    
    Creates stratified initial labeled set (equal per class) and
    puts remaining examples in unlabeled pool.
    
    Automatically creates and reuses configuration-specific seed sets.
    Filename format: {dataset}_seed_set_s{seed}_n{per_class}.csv
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (labeled_pool, unlabeled_pool, all_labels)
    """
    print("\n=== Initializing Data Pools ===")
    
    dataset_config = config['dataset']
    al_config = config['active_learning']
    processing = config['processing']
    
    # Get parameters
    train_file = f"{config['directories']['input_data']}/{dataset_config['train_file']}"
    seed = processing['seed']
    initial_per_class = al_config['initial_labeled_per_class']
    
    # Create configuration-specific seed set filename
    seed_file = train_file.replace('.csv', f'_seed_set_s{seed}_n{initial_per_class}.csv')
    seed_filename = os.path.basename(seed_file)
    
    # Get column names
    col_id = dataset_config['columns']['id']
    col_text = dataset_config['columns']['text']
    col_label = dataset_config['columns']['label']
    
    # Check if seed set exists for this configuration
    if os.path.exists(seed_file):
        # Use existing seed set
        print(f"✅ Using existing seed set: {seed_filename}")
        print(f"   (seed={seed}, per_class={initial_per_class})")
        seed_df = pd.read_csv(seed_file)
        
    else:
        # Auto-create seed set for this configuration
        print(f"📝 No seed set found for this configuration")
        print(f"   seed={seed}, per_class={initial_per_class}")
        print(f"   Creating: {seed_filename}")
        
        # Load and shuffle training data
        df = load_dataset(train_file, config)
        df = shuffle_dataframe(df, seed)
        
        # Get unique labels
        unique_labels = get_unique_labels(
            df, col_label, dataset_config.get('exclude_labels', [])
        )
        unique_labels = [
            label for label in unique_labels
            if pd.notna(label) and str(label).lower() not in ['none', 'null', '', 'nan']
        ]
        
        print(f"   Labels: {unique_labels}")
        
        # Extract seed examples (stratified sampling)
        seed_examples = []
        for label in unique_labels:
            label_df = df[df[col_label] == label]
            num_samples = min(initial_per_class, len(label_df))
            
            if num_samples == 0:
                print(f"     Warning: No examples for label '{label}'")
                continue
            
            samples = label_df.head(num_samples)
            seed_examples.append(samples)
            print(f"     ✓ {label}: {num_samples} examples")
        
        # Create and save seed set
        seed_df = pd.concat(seed_examples, ignore_index=True)
        seed_df.to_csv(seed_file, index=False)
        
        print(f"   ✅ Created seed set: {len(seed_df)} examples")
        print(f"   Saved to: {seed_filename}")
    
    # Load full training data for splitting
    df = load_dataset(train_file, config)
    
    # Get unique labels from seed set
    unique_labels = seed_df[col_label].unique().tolist()
    unique_labels = [
        label for label in unique_labels
        if pd.notna(label) and str(label).lower() not in ['none', 'null', '', 'nan']
    ]
    
    # Get seed IDs
    seed_ids = set(seed_df[col_id].tolist())
    
    # Split into labeled (seed) and unlabeled pools
    labeled_pool = []
    unlabeled_pool = []
    
    for idx, row in df.iterrows():
        example = {
            'id': row[col_id],
            'text': row[col_text],
            'label': row[col_label]
        }
        
        if example['id'] in seed_ids:
            labeled_pool.append(example)
        else:
            unlabeled_pool.append(example)
    
    # Display summary
    print(f"\nInitial labeled pool: {len(labeled_pool)} examples")
    seed_dist = seed_df[col_label].value_counts().to_dict()
    print(f"  Class distribution: {seed_dist}")
    print(f"Unlabeled pool: {len(unlabeled_pool)} examples")
    
    return labeled_pool, unlabeled_pool, unique_labels


def load_test_set(config: dict) -> List[Dict]:
    """
    Load test dataset for evaluation.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of test examples
    """
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


def evaluate_classifier(classifier, test_pool: List[Dict], config: dict) -> Dict:
    """
    Evaluate classifier on test set.
    
    Args:
        classifier: Trained classifier
        test_pool: List of test examples
        config: Configuration dictionary
    
    Returns:
        Dictionary with evaluation metrics
    """
    if not test_pool:
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0
        }
    
    print(f"  Evaluating on {len(test_pool)} test examples...")
    
    texts = [ex['text'] for ex in test_pool]
    true_labels = [ex['label'] for ex in test_pool]
    
    # Get predictions
    predictions = classifier.predict_batch(texts)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    # Handle case where some labels might not appear in predictions
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    except Exception as e:
        print(f"    Warning: Error calculating metrics: {e}")
        precision, recall, f1, f1_weighted = 0.0, 0.0, 0.0, 0.0
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall
    }
    
    return metrics


def save_checkpoint(iteration: int, labeled_pool: List[Dict], unlabeled_pool: List[Dict],
                   results: List[Dict], checkpoint_dir: str):
    """
    Save checkpoint for recovery.
    
    Args:
        iteration: Current iteration number
        labeled_pool: Current labeled pool
        unlabeled_pool: Current unlabeled pool
        results: Results so far
        checkpoint_dir: Directory to save checkpoint
    """
    checkpoint = {
        'iteration': iteration,
        'labeled_pool': labeled_pool,
        'unlabeled_pool': unlabeled_pool,
        'results': results
    }
    
    checkpoint_file = f"{checkpoint_dir}/checkpoint_iter_{iteration}.json"
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"  Checkpoint saved: {checkpoint_file}")


def save_results(results: List[Dict], run_dir: str, classifier_type: str):
    """
    Save final results to CSV file in run directory.
    
    Args:
        results: List of iteration results
        run_dir: Run-specific directory path
        classifier_type: Classifier type (static/retrieval)
    """
    # Save results in run directory
    results_file = f"{run_dir}/al_results.csv"
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to file
    df_results.to_csv(results_file, index=False)
    print(f"\n✅ Results saved to: {results_file}")


def save_final_labeled_pool(labeled_pool: List[Dict], run_dir: str):
    """
    Save final augmented labeled pool to run directory.
    
    Args:
        labeled_pool: Final labeled pool (with counterfactuals)
        run_dir: Run-specific directory path
    """
    # Save labeled pool in run directory
    pool_file = f"{run_dir}/final_labeled_pool.csv"
    
    df = pd.DataFrame(labeled_pool)
    
    # Save to file
    df.to_csv(pool_file, index=False)
    print(f"✅ Labeled pool saved to: {pool_file}")


def active_learning_loop(config: dict):
    """
    Main Active Learning loop.
    
    Iteratively:
    1. Train classifier on labeled pool
    2. Evaluate on test set
    3. Select uncertain examples from unlabeled pool
    4. Query oracle for labels
    5. Generate counterfactuals
    6. Update pools
    7. Check stopping criteria
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*80)
    print("Active Learning Loop: Uncertainty Sampling + Counterfactual Augmentation")
    print("="*80)
    
    al_config = config['active_learning']
    eval_config = config['evaluation']
    log_config = config['logging']
    
    # Check if AL is enabled
    if not al_config['enabled']:
        print("\nERROR: Active learning is disabled in config.yaml")
        print("Set active_learning.enabled = true")
        sys.exit(1)
    
    # Initialize components
    print("\n=== Initializing Components ===")
    llm_provider = get_llm_provider(config)
    
    # Select classifier type based on config
    eval_config = config['evaluation']
    classifier_type = eval_config.get('classifier_type', 'static')
    
    if classifier_type == 'retrieval':
        from utils.retrieval_classifier import get_retrieval_classifier
        classifier = get_retrieval_classifier(config, llm_provider)
        print(f"Using Retrieval-based ICL classifier")
    else:
        classifier = SimpleICLClassifier(config, llm_provider)
        print(f"Using Static ICL classifier")
    
    oracle = get_oracle(config)
    
    print(f"LLM Provider: {config['llm']['provider']}")
    print(f"Uncertainty Method: {al_config['uncertainty_method']}")
    print(f"Counterfactuals: {'Enabled' if al_config['counterfactuals']['enabled'] else 'Disabled'}")
    
    # Initialize data pools
    labeled_pool, unlabeled_pool, all_labels = initialize_pools(config)
    test_pool = load_test_set(config)
    
    if test_pool:
        print(f"Test set: {len(test_pool)} examples")
    else:
        print("Warning: No test set available for evaluation")
    
    # Setup iteration parameters
    budget = al_config['total_budget']
    batch_size = al_config['batch_size']
    max_iterations = al_config['max_iterations']
    
    # CF budget tracking
    cf_total_budget = al_config['counterfactuals'].get('cf_total_budget', -1)
    cf_budget_remaining = cf_total_budget  # -1 means unlimited
    
    print(f"\n📊 Budget Configuration:")
    print(f"   Real labels budget: {budget}")
    print(f"   CF budget: {cf_total_budget if cf_total_budget > 0 else 'unlimited'}")
    
    # Tracking
    results = []
    best_f1_macro = 0.0  # Changed from accuracy to F1 Macro
    patience_counter = 0
    
    # Main loop
    iteration = 0
    
    # Import datetime for timestamping
    from datetime import datetime
    
    # Create run timestamp (once at start for consistent naming)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract model name and dataset for run folder
    provider = config['llm']['provider']
    if provider == 'openai':
        model_name = config['llm']['openai']['model']
    elif provider == 'gemini':
        model_name = config['llm']['gemini']['model']
    elif provider == 'ollama':
        model_name = config['llm']['ollama']['model']
    else:
        model_name = provider
    
    # Sanitize model name for filename
    model_safe = model_name.replace('/', '-').replace(':', '-').replace(' ', '_')
    
    # Extract dataset name (without extension)
    dataset_name = config['dataset']['train_file'].replace('.csv', '').replace('_train', '')
    
    # Get evaluation method (classifier type) and query strategy
    classifier_type = eval_config.get('classifier_type', 'static')
    query_strategy = config['active_learning'].get('query_strategy', 'uncertainty')
    
    # Get seed value and initial per class
    seed = config['processing']['seed']
    initial_per_class = config['active_learning']['initial_labeled_per_class']
    
    # Create run-specific directory: timestamp_model_dataset_evalmethod_querystrategy_seed_perclass
    import os
    import shutil
    run_dir = f"{config['directories']['output_data']}/{run_timestamp}_{model_safe}_{dataset_name}_{classifier_type}_{query_strategy}_s{seed}_n{initial_per_class}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save experiment configuration as a readable text report (sanitized - no API keys)
    try:
        config_report = f"{run_dir}/experiment_config.txt"
        
        with open(config_report, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Run Timestamp: {run_timestamp}\n")
            f.write(f"Run Directory: {run_dir}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("LLM CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Provider: {config['llm']['provider']}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"API Keys/Endpoints: ***REDACTED FOR SECURITY***\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("DATASET CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Train File: {config['dataset']['train_file']}\n")
            f.write(f"Test File: {config['dataset']['test_file']}\n")
            f.write(f"Text Column: {config['dataset']['columns']['text']}\n")
            f.write(f"Label Column: {config['dataset']['columns']['label']}\n")
            f.write(f"Exclude Labels: {config['dataset'].get('exclude_labels', [])}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("ACTIVE LEARNING CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Query Strategy: {config['active_learning']['query_strategy']}\n")
            f.write(f"Uncertainty Method: {config['active_learning']['uncertainty_method']}\n")
            f.write(f"Total Budget: {config['active_learning']['total_budget']}\n")
            f.write(f"Batch Size: {config['active_learning']['batch_size']}\n")
            f.write(f"Initial Labeled per Class: {config['active_learning']['initial_labeled_per_class']}\n")
            f.write(f"Random Seed: {config['processing']['seed']}\n")
            f.write(f"Max Iterations: {config['active_learning']['max_iterations']}\n")
            f.write(f"Early Stopping Patience: {config['active_learning']['early_stopping_patience']}\n")
            f.write(f"Min Improvement: {config['active_learning']['min_improvement']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("COUNTERFACTUAL CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Enabled: {config['active_learning']['counterfactuals']['enabled']}\n")
            f.write(f"Temperature: {config['active_learning']['counterfactuals']['generation_temperature']}\n")
            f.write(f"Max Tokens: {config['active_learning']['counterfactuals']['max_tokens']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("EVALUATION CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Classifier Type: {config['evaluation']['classifier_type']}\n")
            f.write(f"Max ICL Examples: {config['evaluation']['max_icl_examples']}\n")
            f.write(f"Eval Every N Iterations: {config['evaluation']['eval_every_iterations']}\n")
            
            if config['evaluation']['classifier_type'] == 'retrieval':
                f.write(f"\nRetrieval Settings:\n")
                f.write(f"  Embedding Backend: {config['evaluation']['retrieval']['embedding_backend']}\n")
                f.write(f"  K per Class: {config['evaluation']['retrieval']['k_per_class']}\n")
                f.write(f"  Total K Max: {config['evaluation']['retrieval']['total_k_max']}\n")
                f.write(f"  Fallback Strategy: {config['evaluation']['retrieval']['fallback_strategy']}\n")
                
                backend = config['evaluation']['retrieval']['embedding_backend']
                if backend == 'sentence_transformers':
                    f.write(f"  ST Model: {config['evaluation']['retrieval']['sentence_transformers']['model']}\n")
                    f.write(f"  Device: {config['evaluation']['retrieval']['sentence_transformers']['device']}\n")
                elif backend == 'openai':
                    f.write(f"  OpenAI Model: {config['evaluation']['retrieval']['openai']['model']}\n")
                elif backend == 'tfidf':
                    f.write(f"  Max Features: {config['evaluation']['retrieval']['tfidf']['max_features']}\n")
                    f.write(f"  N-gram Range: {config['evaluation']['retrieval']['tfidf']['ngram_range']}\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("LOGGING CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Checkpoint Every: {config['logging']['checkpoint_every']} iterations\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF CONFIGURATION\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Saved experiment config: {config_report}")
        
    except Exception as e:
        print(f"⚠️  Could not save config report: {e}")
    
    # Create subdirectories within run folder
    interim_dir = f"{run_dir}/interim_output"
    checkpoint_dir = f"{run_dir}/checkpoints"
    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\n📁 Run directory: {run_dir}")
    print(f"   All outputs will be saved here")
    
    # ========================================================================
    # BASELINE EVALUATION (Iteration 0) - Before any AL iterations
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"Baseline Evaluation (Iteration 0)")
    print(f"{'='*80}")
    print(f"Labeled pool: {len(labeled_pool)} examples (seed set)")
    print(f"Unlabeled pool: {len(unlabeled_pool)} examples")
    
    baseline_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train classifier on initial seed set
    print(f"\n[Baseline] Training classifier on seed set...")
    classifier.train(labeled_pool)
    
    # Save baseline training output
    baseline_step1_file = f"{interim_dir}/iter_00_{baseline_timestamp}_{model_safe}_step1_classifier_training.json"
    with open(baseline_step1_file, 'w') as f:
        json.dump({
            'iteration': 0,
            'timestamp': baseline_timestamp,
            'step': 'classifier_training',
            'labeled_pool_size': len(labeled_pool),
            'num_labels': len(classifier.labels),
            'labels': list(classifier.labels),
            'labeled_examples': labeled_pool,
            'training_summary': {
                'total_examples': len(labeled_pool),
                'labels_count': {label: len([ex for ex in labeled_pool if ex['label'] == label]) 
                                for label in classifier.labels}
            }
        }, f, indent=2)
    print(f"  ✓ Interim output saved: {baseline_step1_file}")
    
    # Evaluate baseline
    baseline_metrics = None
    if test_pool:
        print(f"\n[Baseline] Evaluating on test set...")
        baseline_metrics = evaluate_classifier(classifier, test_pool, config)
        
        print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {baseline_metrics['f1_macro']:.4f}")
        print(f"  F1 Weighted: {baseline_metrics['f1_weighted']:.4f}")
        
        # Initialize best F1 Macro from baseline
        best_f1_macro = baseline_metrics['f1_macro']
        print(f"  ✓ Baseline F1 Macro: {best_f1_macro:.4f}")
        
        # Save baseline evaluation output
        baseline_step2_file = f"{interim_dir}/iter_00_{baseline_timestamp}_{model_safe}_step2_evaluation.json"
        
        # Build evaluation config metadata
        eval_metadata = {
            'classifier_type': eval_config.get('classifier_type', 'static'),
            'max_icl_examples': eval_config.get('max_icl_examples', 100)
        }
        
        # Add retrieval settings if using retrieval
        if eval_metadata['classifier_type'] == 'retrieval':
            retrieval_config = eval_config.get('retrieval', {})
            eval_metadata['retrieval_settings'] = {
                'embedding_backend': retrieval_config.get('embedding_backend', 'unknown'),
                'k_per_class': retrieval_config.get('k_per_class', 3),
                'total_k_max': retrieval_config.get('total_k_max', 50),
                'fallback_strategy': retrieval_config.get('fallback_strategy', 'similarity')
            }
            
            # Add model-specific info
            backend = retrieval_config.get('embedding_backend', 'unknown')
            if backend in retrieval_config:
                eval_metadata['retrieval_settings']['model_config'] = retrieval_config[backend]
        
        with open(baseline_step2_file, 'w') as f:
            json.dump({
                'iteration': 0,
                'timestamp': baseline_timestamp,
                'step': 'evaluation',
                'evaluation_config': eval_metadata,
                'metrics': baseline_metrics,
                'best_f1_macro': best_f1_macro,
                'patience_counter': 0,
                'test_pool_size': len(test_pool)
            }, f, indent=2)
        print(f"  ✓ Interim output saved: {baseline_step2_file}")
    
    # Save baseline results
    query_strategy = al_config.get('query_strategy', 'uncertainty')
    baseline_result = {
        'iteration': 0,
        'classifier_type': eval_config.get('classifier_type', 'static'),
        'query_strategy': query_strategy,
        'uncertainty_method': al_config['uncertainty_method'] if query_strategy == 'uncertainty' else 'N/A',
        'labeled_pool_size': len(labeled_pool),
        'unlabeled_pool_size': len(unlabeled_pool),
        'num_real_examples': 0,  # No new examples in baseline
        'num_counterfactuals': 0,  # No CFs in baseline
        'total_counterfactuals_so_far': 0,
        'budget_remaining': budget,
        'cf_budget_remaining': cf_budget_remaining if cf_total_budget > 0 else -1
    }
    
    if baseline_metrics:
        baseline_result.update(baseline_metrics)
    
    results.append(baseline_result)
    print(f"  ✓ Baseline results recorded (Iteration 0)")
    
    # ========================================================================
    # MAIN ACTIVE LEARNING LOOP
    # ========================================================================
    try:
        while iteration < max_iterations and budget > 0 and len(unlabeled_pool) > 0:
            iteration += 1
            
            # Create timestamp for this iteration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"\n{'='*80}")
            print(f"Iteration {iteration}/{max_iterations}")
            print(f"{'='*80}")
            print(f"Budget remaining: {budget}/{al_config['total_budget']}")
            print(f"Labeled pool: {len(labeled_pool)} examples")
            print(f"Unlabeled pool: {len(unlabeled_pool)} examples")
            
            # Step 1: Train classifier on current labeled pool
            print(f"\n[Step 1/6] Training classifier...")
            classifier.train(labeled_pool)
            
            # Save Step 1 output
            step1_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_{model_safe}_step1_classifier_training.json"
            with open(step1_file, 'w') as f:
                json.dump({
                    'iteration': iteration,
                    'timestamp': timestamp,
                    'step': 'classifier_training',
                    'labeled_pool_size': len(labeled_pool),
                    'num_labels': len(classifier.labels),
                    'labels': list(classifier.labels),
                    'labeled_examples': labeled_pool,  # Full labeled pool used for training
                    'training_summary': {
                        'total_examples': len(labeled_pool),
                        'labels_count': {label: len([ex for ex in labeled_pool if ex['label'] == label]) 
                                        for label in classifier.labels}
                    }
                }, f, indent=2)
            print(f"  ✓ Interim output saved: {step1_file}")
            
            # Step 2: Select examples based on query strategy
            query_strategy = al_config.get('query_strategy', 'uncertainty')
            
            if query_strategy == "random":
                print(f"\n[Step 2/6] Selecting random examples (baseline)...")
            else:
                print(f"\n[Step 2/6] Selecting uncertain examples...")
            
            current_batch_size = min(batch_size, len(unlabeled_pool), budget)
            
            # Handle different query strategies
            if query_strategy == "random":
                selected_indices = random.sample(range(len(unlabeled_pool)), current_batch_size)
                uncertainty_details = {
                    'method': 'random',
                    'total_pool_size': len(unlabeled_pool),
                    'batch_size': current_batch_size,
                    'selected_indices': selected_indices
                }
                print(f"  Randomly selected {len(selected_indices)} examples")
                
            elif query_strategy == "uncertainty":
                # Uncertainty-based selection
                selected_indices, uncertainty_details = select_uncertain_examples(
                    unlabeled_pool,
                    classifier,
                    current_batch_size,
                    method=al_config['uncertainty_method'],
                    return_details=True  # Get full logprobs and uncertainty data
                )
            else:
                raise ValueError(f"Unknown query_strategy: {query_strategy}. Choose 'uncertainty' or 'random'")
            
            selected_examples = [unlabeled_pool[i] for i in selected_indices]
            
            print(f"  Selected {len(selected_examples)} examples for labeling")
            
            # Save Step 2 output: selected examples with query strategy details
            step_name = 'random_selection' if query_strategy == 'random' else 'uncertainty_selection'
            step2_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_{model_safe}_step2_{step_name}.json"
            with open(step2_file, 'w') as f:
                json.dump({
                    'iteration': iteration,
                    'timestamp': timestamp,
                    'step': step_name,
                    'query_strategy': query_strategy,
                    'selected_indices': selected_indices,
                    'selected_examples': selected_examples,
                    'selection_details': uncertainty_details,  # Contains method, scores (if uncertainty), or random info
                    'unlabeled_pool_size': len(unlabeled_pool),
                    'batch_size': current_batch_size
                }, f, indent=2)
            print(f"  ✓ Interim output saved: {step2_file}")
            
            # Step 3: Query oracle
            print(f"\n[Step 3/6] Querying oracle for labels...")
            labeled_examples = oracle.label_examples(selected_examples)
            
            # Save Step 4 output: oracle labels
            step4_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_{model_safe}_step4_oracle_labeling.json"
            with open(step4_file, 'w') as f:
                json.dump({
                    'iteration': iteration,
                    'timestamp': timestamp,
                    'step': 'oracle_labeling',
                    'labeled_examples': labeled_examples,
                    'num_labeled': len(labeled_examples)
                }, f, indent=2)
            print(f"  ✓ Interim output saved: {step4_file}")
            
            # Step 4: Generate counterfactuals with quality filtering
            counterfactuals = []
            num_cfs_added = 0
            if al_config['counterfactuals']['enabled']:
                print(f"\n[Step 4/6] Generating counterfactuals with quality filtering...")
                counterfactuals, num_cfs_added, cf_generation_details = generate_counterfactuals_batch(
                    labeled_examples,
                    config,
                    llm_provider,
                    all_labels,
                    labeled_pool,           # For diversity calculation
                    classifier,             # For confidence scoring
                    cf_budget_remaining,    # Budget constraint
                    return_details=True     # Get full generation metadata and prompts
                )
                
                # Update CF budget
                if cf_budget_remaining > 0:
                    cf_budget_remaining = max(0, cf_budget_remaining - num_cfs_added)
                    print(f"  CF budget: {num_cfs_added} added, {cf_budget_remaining} remaining")
                
                # Save Step 5 output: counterfactuals with FULL generation details
                step5_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_{model_safe}_step5_counterfactual_generation.json"
                
                # Extract selected CF IDs for easy reference
                selected_cf_ids = [cf['id'] for cf in counterfactuals]
                
                with open(step5_file, 'w') as f:
                    json.dump({
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'step': 'counterfactual_generation',
                        'input_examples': labeled_examples,
                        'generated_counterfactuals': counterfactuals,
                        'selected_cf_ids': selected_cf_ids,  # Clear list of IDs that were kept after filtering
                        'generation_details': cf_generation_details,  # FULL DETAILS: prompts, times, filtering scores, etc.
                        'num_generated': len(counterfactuals),
                        'summary': {
                            'num_input_examples': len(labeled_examples),
                            'num_candidates_generated': cf_generation_details.get('num_generated', 0),
                            'num_after_filtering': cf_generation_details.get('num_filtered', 0),
                            'num_final_selected': len(counterfactuals),
                            'budget_remaining_before': cf_generation_details.get('budget_remaining_before', -1),
                            'budget_remaining_after': cf_generation_details.get('budget_remaining_after', -1)
                        }
                    }, f, indent=2)
                print(f"  ✓ Interim output saved: {step5_file}")
            else:
                print(f"\n[Step 4/6] Skipping counterfactual generation (disabled)")
                
                # Still save Step 5 output (skipped)
                step5_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_{model_safe}_step5_counterfactual_generation_skipped.json"
                with open(step5_file, 'w') as f:
                    json.dump({
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'step': 'counterfactual_generation',
                        'status': 'skipped',
                        'reason': 'counterfactuals_disabled_in_config'
                    }, f, indent=2)
                print(f"  ✓ Interim output saved: {step5_file}")
            
            # Step 5: Update pools
            print(f"\n[Step 5/6] Updating data pools...")
            
            # Capture pre-update state
            pre_labeled_size = len(labeled_pool)
            pre_unlabeled_size = len(unlabeled_pool)
            pre_budget = budget
            
            # Add to labeled pool
            labeled_pool.extend(labeled_examples)
            if counterfactuals:
                labeled_pool.extend(counterfactuals)
            
            # Remove from unlabeled pool
            for idx in sorted(selected_indices, reverse=True):
                unlabeled_pool.pop(idx)
            
            # Update budget
            budget -= len(labeled_examples)
            
            print(f"  Labeled pool: {len(labeled_pool)} examples (+{len(labeled_examples)} real, +{len(counterfactuals)} CF)")
            print(f"  Unlabeled pool: {len(unlabeled_pool)} examples")
            print(f"  Budget: {budget} remaining")
            
            # Save Step 5 output: pool updates
            step5_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_{model_safe}_step5_pool_update.json"
            with open(step5_file, 'w') as f:
                json.dump({
                    'iteration': iteration,
                    'timestamp': timestamp,
                    'step': 'pool_update',
                    'before': {
                        'labeled_pool_size': pre_labeled_size,
                        'unlabeled_pool_size': pre_unlabeled_size,
                        'budget': pre_budget
                    },
                    'changes': {
                        'real_examples_added': len(labeled_examples),
                        'counterfactuals_added': len(counterfactuals),
                        'unlabeled_examples_removed': len(selected_indices),
                        'budget_consumed': len(labeled_examples)
                    },
                    'after': {
                        'labeled_pool_size': len(labeled_pool),
                        'unlabeled_pool_size': len(unlabeled_pool),
                        'budget_remaining': budget
                    }
                }, f, indent=2)
            print(f"  ✓ Interim output saved: {step5_file}")
            
            # Step 6: Evaluate AFTER pool update (so metrics reflect current state)
            metrics = None
            if test_pool and (iteration % eval_config['eval_every_iterations'] == 0):
                print(f"\n[Step 6/6] Evaluating on test set (after pool update)...")
                # Re-train classifier on updated pool to ensure evaluation reflects current state
                classifier.train(labeled_pool)
                metrics = evaluate_classifier(classifier, test_pool, config)
                
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
                print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
                
                # Early stopping check (using F1 Macro)
                if metrics['f1_macro'] > best_f1_macro + al_config['min_improvement']:
                    best_f1_macro = metrics['f1_macro']
                    patience_counter = 0
                    print(f"  ✓ New best F1 Macro: {best_f1_macro:.4f}")
                else:
                    patience_counter += 1
                    print(f"  No improvement (patience: {patience_counter}/{al_config['early_stopping_patience']})")
                
                # Save Step 6 (evaluation) output
                step6_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_{model_safe}_step6_evaluation.json"
                
                # Build evaluation config metadata
                eval_metadata = {
                    'classifier_type': eval_config.get('classifier_type', 'static'),
                    'max_icl_examples': eval_config.get('max_icl_examples', 100)
                }
                
                # Add retrieval settings if using retrieval
                if eval_metadata['classifier_type'] == 'retrieval':
                    retrieval_config = eval_config.get('retrieval', {})
                    eval_metadata['retrieval_settings'] = {
                        'embedding_backend': retrieval_config.get('embedding_backend', 'unknown'),
                        'k_per_class': retrieval_config.get('k_per_class', 3),
                        'total_k_max': retrieval_config.get('total_k_max', 50),
                        'fallback_strategy': retrieval_config.get('fallback_strategy', 'similarity')
                    }
                    
                    # Add model-specific info
                    backend = retrieval_config.get('embedding_backend', 'unknown')
                    if backend in retrieval_config:
                        eval_metadata['retrieval_settings']['model_config'] = retrieval_config[backend]
                
                with open(step6_file, 'w') as f:
                    json.dump({
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'step': 'evaluation',
                        'evaluation_config': eval_metadata,
                        'metrics': metrics,
                        'best_f1_macro': best_f1_macro,
                        'patience_counter': patience_counter,
                        'test_pool_size': len(test_pool),
                        'labeled_pool_size': len(labeled_pool)  # Pool size at evaluation time
                    }, f, indent=2)
                print(f"  ✓ Interim output saved: {step6_file}")
                
                if patience_counter >= al_config['early_stopping_patience']:
                    print(f"\n⚠ Early stopping triggered (no improvement for {patience_counter} iterations)")
                    break
            else:
                print(f"\n[Step 6/6] Skipping evaluation (eval_every={eval_config['eval_every_iterations']})")
                
                # Still save Step 6 output (skipped)
                step6_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_{model_safe}_step6_evaluation_skipped.json"
                
                # Build evaluation config metadata (even for skipped)
                eval_metadata = {
                    'classifier_type': eval_config.get('classifier_type', 'static'),
                    'max_icl_examples': eval_config.get('max_icl_examples', 100)
                }
                
                if eval_metadata['classifier_type'] == 'retrieval':
                    retrieval_config = eval_config.get('retrieval', {})
                    eval_metadata['retrieval_settings'] = {
                        'embedding_backend': retrieval_config.get('embedding_backend', 'unknown'),
                        'k_per_class': retrieval_config.get('k_per_class', 3),
                        'total_k_max': retrieval_config.get('total_k_max', 50)
                    }
                
                with open(step6_file, 'w') as f:
                    json.dump({
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'step': 'evaluation',
                        'evaluation_config': eval_metadata,
                        'status': 'skipped',
                        'reason': f'eval_every_iterations={eval_config["eval_every_iterations"]}',
                        'labeled_pool_size': len(labeled_pool)  # Pool size at this point
                    }, f, indent=2)
                print(f"  ✓ Interim output saved: {step6_file}")
            
            # Save iteration results (metrics reflect pool state AFTER update)
            iter_result = {
                'iteration': iteration,
                'classifier_type': eval_config.get('classifier_type', 'static'),
                'query_strategy': query_strategy,
                'uncertainty_method': al_config['uncertainty_method'] if query_strategy == 'uncertainty' else 'N/A',
                'labeled_pool_size': len(labeled_pool),
                'unlabeled_pool_size': len(unlabeled_pool),
                'num_real_examples': len(labeled_examples),
                'num_counterfactuals': num_cfs_added,
                'total_counterfactuals_so_far': (cf_total_budget - cf_budget_remaining) if cf_total_budget > 0 else len([ex for ex in labeled_pool if ex.get('original_id')]),
                'budget_remaining': budget,
                'cf_budget_remaining': cf_budget_remaining if cf_total_budget > 0 else -1
            }
            
            if metrics:
                iter_result.update(metrics)
            
            results.append(iter_result)
            
            # Save checkpoint
            if iteration % log_config['checkpoint_every'] == 0:
                save_checkpoint(iteration, labeled_pool, unlabeled_pool, results, checkpoint_dir)
            
            # Save labeled pool after each iteration (prevents data loss)
            save_final_labeled_pool(labeled_pool, run_dir)
            print(f"  💾 Labeled pool saved (iteration {iteration})")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user!")
        print(f"Completed {iteration} iterations")
        print(f"Saving progress...")
        save_checkpoint(iteration, labeled_pool, unlabeled_pool, results, checkpoint_dir)
        save_results(results, run_dir, eval_config.get('classifier_type', 'static'))
        save_final_labeled_pool(labeled_pool, run_dir)
        print(f"💾 All progress saved successfully!")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print(f"Completed {iteration} iterations before error")
        print(f"Saving progress...")
        try:
            save_checkpoint(iteration, labeled_pool, unlabeled_pool, results, checkpoint_dir)
            save_results(results, run_dir, eval_config.get('classifier_type', 'static'))
            save_final_labeled_pool(labeled_pool, run_dir)
            print(f"💾 Progress saved successfully!")
        except Exception as save_error:
            print(f"⚠️ Warning: Could not save progress: {save_error}")
        raise
    
    # Final evaluation (if loop exited before final evaluation or if last iteration wasn't evaluated)
    print(f"\n{'='*80}")
    print("Final Evaluation")
    print(f"{'='*80}")
    
    classifier.train(labeled_pool)
    
    if test_pool:
        final_metrics = evaluate_classifier(classifier, test_pool, config)
        
        print(f"\nFinal Results:")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {final_metrics['f1_macro']:.4f}")
        print(f"  F1 Weighted: {final_metrics['f1_weighted']:.4f}")
        print(f"  Precision Macro: {final_metrics['precision_macro']:.4f}")
        print(f"  Recall Macro: {final_metrics['recall_macro']:.4f}")
        
        # Get detailed classification report
        texts = [ex['text'] for ex in test_pool]
        true_labels = [ex['label'] for ex in test_pool]
        predictions = classifier.predict_batch(texts)
        
        print(f"\nClassification Report:")
        print(classification_report(true_labels, predictions, zero_division=0))
        
        # Check if final evaluation is needed
        # Only add if last iteration wasn't evaluated (due to eval_every_iterations)
        # If last iteration already has metrics, it means it was evaluated - don't add duplicate
        last_result = results[-1] if results else None
        should_add_final = False
        
        if last_result is None:
            should_add_final = True
        elif 'f1_macro' not in last_result:
            # Last iteration wasn't evaluated (no metrics) - add final evaluation
            should_add_final = True
        # If last_result has 'f1_macro', iteration was already evaluated - don't add final
        
        if should_add_final:
            # Add final evaluation as a separate result entry
            final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_result = {
                'iteration': iteration + 1,  # Next iteration number (or same if loop ended naturally)
                'classifier_type': eval_config.get('classifier_type', 'static'),
                'query_strategy': query_strategy,
                'uncertainty_method': al_config['uncertainty_method'] if query_strategy == 'uncertainty' else 'N/A',
                'labeled_pool_size': len(labeled_pool),
                'unlabeled_pool_size': len(unlabeled_pool),
                'num_real_examples': 0,  # No new examples in final evaluation
                'num_counterfactuals': 0,  # No new CFs in final evaluation
                'total_counterfactuals_so_far': (cf_total_budget - cf_budget_remaining) if cf_total_budget > 0 else len([ex for ex in labeled_pool if ex.get('original_id')]),
                'budget_remaining': budget,
                'cf_budget_remaining': cf_budget_remaining if cf_total_budget > 0 else -1
            }
            final_result.update(final_metrics)
            results.append(final_result)
            print(f"  ✓ Final evaluation added to results (Iteration {final_result['iteration']})")
        else:
            print(f"  ✓ Final state already evaluated in iteration {last_result['iteration']} (no duplicate needed)")
    
    # Save results
    save_results(results, run_dir, eval_config.get('classifier_type', 'static'))
    save_final_labeled_pool(labeled_pool, run_dir)
    
    print(f"\n{'='*80}")
    print("Active Learning Complete!")
    print(f"{'='*80}")
    print(f"Total iterations: {iteration}")
    print(f"Total examples labeled: {al_config['total_budget'] - budget}")
    print(f"Final labeled pool size: {len(labeled_pool)}")
    print(f"  (including {sum(1 for ex in labeled_pool if 'original_id' in ex)} counterfactuals)")


def main():
    """Main execution."""
    # Load configuration
    config = load_config()
    ensure_directories(config)
    
    # Run active learning loop (creates run-specific directories)
    active_learning_loop(config)


if __name__ == "__main__":
    main()

