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
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy types and other non-JSON-serializable types to native Python types.
    
    Args:
        obj: Object to sanitize
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, bool):
        return bool(obj)  # Ensure it's a native Python bool
    else:
        return obj

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
from utils.target_label_selector import TargetLabelSelector
from utils.probe_uncertainty import ProbeUncertaintyEstimator


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
    
    # Initialize V2 probe uncertainty estimator (if using probe_entropy)
    probe_estimator = None
    uncertainty_method = al_config.get('uncertainty_method', 'entropy')
    if uncertainty_method == 'probe_entropy':
        probe_config = al_config.get('probe_uncertainty', {})
        if probe_config.get('enabled', False):
            try:
                probe_estimator = ProbeUncertaintyEstimator(config)
                print(f"✓ V2 Probe Uncertainty Estimator initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize probe estimator: {e}")
                print(f"   Falling back to LLM-based uncertainty")
                uncertainty_method = 'entropy'  # Fallback
                probe_estimator = None
        else:
            print(f"⚠️  uncertainty_method='probe_entropy' but probe_uncertainty.enabled=false")
            print(f"   Falling back to LLM-based uncertainty")
            uncertainty_method = 'entropy'  # Fallback
    
    print(f"LLM Provider: {config['llm']['provider']}")
    print(f"Uncertainty Method: {uncertainty_method}")
    print(f"Counterfactuals: {'Enabled' if al_config['counterfactuals']['enabled'] else 'Disabled'}")
    
    # Initialize data pools
    labeled_pool, unlabeled_pool, all_labels = initialize_pools(config)
    test_pool = load_test_set(config)
    
    if test_pool:
        print(f"Test set: {len(test_pool)} examples")
    else:
        print("Warning: No test set available for evaluation")
    
    # Initialize target label selector (Version 3)
    target_label_selector = None
    if al_config['counterfactuals']['enabled']:
        try:
            seed = config['processing']['seed']
            target_label_selector = TargetLabelSelector(
                config=config,
                all_labels=all_labels,
                seed=seed
            )
            print(f"✓ Target label selector initialized (strategy: {target_label_selector.strategy})")
        except Exception as e:
            print(f"⚠️  Could not initialize target label selector: {e}")
            print(f"   Falling back to legacy distribution_strategy")
            target_label_selector = None
    
    # Setup iteration parameters
    budget = al_config['total_budget']
    batch_size = al_config['batch_size']
    max_iterations = al_config['max_iterations']
    
    # CF budget configuration (V3: per-round budget)
    cf_config = al_config['counterfactuals']
    alpha_cf = cf_config.get('alpha_cf', 1.0)  # Per-round multiplier: |C_t| <= alpha_cf * |F_t|
    # Backward compatibility: check for legacy cf_total_budget
    cf_total_budget = cf_config.get('cf_total_budget', -1)
    use_legacy_budget = (cf_total_budget > 0 and 'alpha_cf' not in cf_config)
    
    if use_legacy_budget:
        print(f"\n⚠️  Using legacy global CF budget mode (cf_total_budget={cf_total_budget})")
        print(f"   Consider switching to per-round budget (alpha_cf) for V3 behavior")
        cf_budget_remaining = cf_total_budget
    else:
        cf_budget_remaining = None  # Not used in per-round mode
        print(f"\n📊 Budget Configuration (Version 3):")
        print(f"   Real labels budget: {budget}")
        print(f"   CF per-round budget: alpha_cf={alpha_cf} (|C_t| <= {alpha_cf} * |F_t|)")
    
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
    
    # Get uncertainty method (for folder naming)
    uncertainty_method = config['active_learning'].get('uncertainty_method', 'entropy')
    if query_strategy == 'random':
        uncertainty_method = 'random'
        probe_type = ''  # No probe type for random
        uncertainty_safe = 'random'
    elif uncertainty_method == 'probe_entropy':
        probe_type = '_LRprobe'  # V2: LR probe (embedding-based)
        uncertainty_safe = 'entropy'  # Keep base name for clarity
    else:
        probe_type = '_LLMprobe'  # V1: LLM-based uncertainty (entropy, margin, least_confident)
        uncertainty_safe = uncertainty_method.replace('-', '_')
    
    # Get retrieval backend (if using retrieval)
    retrieval_backend = ''
    if classifier_type == 'retrieval':
        retrieval_backend = eval_config.get('retrieval', {}).get('embedding_backend', 'unknown')
        retrieval_backend = retrieval_backend.replace('-', '_').replace('/', '_')
        retrieval_backend = f"_{retrieval_backend}"
    
    # Get seed value and initial per class
    seed = config['processing']['seed']
    initial_per_class = config['active_learning']['initial_labeled_per_class']
    
    # Create run-specific directory: timestamp_model_dataset_evalmethod_querystrategy_uncertainty[probe_type][retrieval]_seed_perclass
    import os
    import shutil
    run_dir = f"{config['directories']['output_data']}/{run_timestamp}_{model_safe}_{dataset_name}_{classifier_type}_{query_strategy}_{uncertainty_safe}{probe_type}{retrieval_backend}_s{seed}_n{initial_per_class}"
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
            uncertainty_method = config['active_learning'].get('uncertainty_method', 'entropy')
            f.write(f"Uncertainty Method: {uncertainty_method}\n")
            
            # V2: Probe-based Entropy Uncertainty settings
            if uncertainty_method == 'probe_entropy':
                probe_config = config['active_learning'].get('probe_uncertainty', {})
                f.write("\nV2 Probe-Based Entropy Uncertainty:\n")
                f.write(f"  Enabled: {probe_config.get('enabled', False)}\n")
                f.write(f"  Embedding Model: {probe_config.get('embedding_model', 'bge-large-en-v1.5')}\n")
                f.write(f"  Device: {probe_config.get('device', 'cpu')}\n")
                f.write(f"  LR max_iter: {probe_config.get('max_iter', 1000)}\n")
                f.write(f"  LR C (regularization): {probe_config.get('C', 1.0)}\n")
            f.write("\n")
            f.write(f"Total Budget: {config['active_learning']['total_budget']}\n")
            f.write(f"Batch Size: {config['active_learning']['batch_size']}\n")
            f.write(f"Initial Labeled per Class: {config['active_learning']['initial_labeled_per_class']}\n")
            f.write(f"Random Seed: {config['processing']['seed']}\n")
            f.write(f"Max Iterations: {config['active_learning']['max_iterations']}\n")
            f.write(f"Early Stopping Patience: {config['active_learning']['early_stopping_patience']}\n")
            f.write(f"Min Improvement: {config['active_learning']['min_improvement']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("COUNTERFACTUAL CONFIGURATION (Version 3)\n")
            f.write("-" * 80 + "\n")
            cf_config = config['active_learning']['counterfactuals']
            f.write(f"Enabled: {cf_config['enabled']}\n")
            f.write(f"Max CFs per Example: {cf_config.get('max_per_example', 3)}\n")
            f.write(f"Per-Round Budget (alpha_cf): {cf_config.get('alpha_cf', 1.0)}\n")
            f.write(f"Temperature: {cf_config['generation_temperature']}\n")
            f.write(f"Max Tokens: {cf_config['max_tokens']}\n")
            f.write(f"Prompt Variation: {cf_config.get('prompt_variation', True)}\n\n")
            
            # Target Label Selection (V3)
            if 'target_label_selection' in cf_config:
                target_config = cf_config['target_label_selection']
                f.write("Target Label Selection (V3):\n")
                f.write(f"  Strategy: {target_config.get('strategy', 'uniform')}\n")
                if target_config.get('strategy') == 'hybrid':
                    f.write(f"  Lambda: {target_config.get('lambda', 0.5)}\n")
            else:
                f.write("Target Label Selection: Legacy (distribution_strategy)\n")
                f.write(f"  Distribution Strategy: {cf_config.get('distribution_strategy', 'balanced')}\n")
            f.write("\n")
            
            # Quality Filtering (V3)
            if cf_config.get('quality_filtering', {}).get('enabled', False):
                qf_config = cf_config['quality_filtering']
                f.write("Quality Filtering (V3 - Enhanced):\n")
                f.write("  Filter 1: Label-Consistency (3 conditions):\n")
                f.write(f"    tau_conf: {qf_config.get('tau_conf', 0.4)}\n")
                f.write(f"    delta: {qf_config.get('delta', 0.1)}\n")
                f.write("  Filter 2: Semantic Similarity Band:\n")
                f.write(f"    s_min: {qf_config.get('s_min', 0.7)}\n")
                f.write(f"    s_max: {qf_config.get('s_max', 0.98)}\n")
                f.write("  Filter 3: Length Ratio:\n")
                f.write(f"    r_min: {qf_config.get('r_min', 0.7)}\n")
                f.write(f"    r_max: {qf_config.get('r_max', 1.3)}\n")
                f.write("  V3 Scoring Weights:\n")
                f.write(f"    alpha: {qf_config.get('alpha', 0.3)}\n")
                f.write(f"    beta: {qf_config.get('beta', 0.5)}\n")
                f.write(f"  Embedding Model: {qf_config.get('embedding_model', 'all-MiniLM-L6-v2')}\n")
            else:
                f.write("Quality Filtering: Disabled\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("EVALUATION CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Classifier Type: {config['evaluation']['classifier_type']}\n")
            f.write(f"Max ICL Examples: {config['evaluation']['max_icl_examples']}\n")
            f.write(f"Eval Every N Iterations: {config['evaluation']['eval_every_iterations']}\n")
            
            if config['evaluation']['classifier_type'] == 'retrieval':
                retrieval_config = config['evaluation']['retrieval']
                backend = retrieval_config.get('embedding_backend', 'unknown')
                f.write(f"\nRetrieval Settings:\n")
                f.write(f"  Embedding Backend: {backend}\n")
                f.write(f"  K per Class: {retrieval_config.get('k_per_class', 3)}\n")
                f.write(f"  Total K Max: {retrieval_config.get('total_k_max', 50)}\n")
                f.write(f"  Fallback Strategy: {retrieval_config.get('fallback_strategy', 'similarity')}\n")
                
                # Backend-specific settings
                if backend == 'bm25':
                    bm25_config = retrieval_config.get('bm25', {})
                    f.write(f"\n  BM25 Configuration:\n")
                    f.write(f"    k1: {bm25_config.get('k1', 1.5)}\n")
                    f.write(f"    b: {bm25_config.get('b', 0.75)}\n")
                elif backend == 'contriever':
                    contriever_config = retrieval_config.get('contriever', {})
                    f.write(f"\n  Contriever Configuration:\n")
                    f.write(f"    Model: {contriever_config.get('model', 'facebook/contriever')}\n")
                    f.write(f"    Device: {contriever_config.get('device', 'cpu')}\n")
                elif backend == 'bge_large':
                    bge_config = retrieval_config.get('bge_large', {})
                    f.write(f"\n  BGE-Large Configuration:\n")
                    f.write(f"    Model: {bge_config.get('model', 'BAAI/bge-large-en-v1.5')}\n")
                    f.write(f"    Device: {bge_config.get('device', 'cpu')}\n")
                    f.write(f"    Normalize Embeddings: {bge_config.get('normalize_embeddings', True)}\n")
                elif backend == 'sentence_transformers':
                    st_config = retrieval_config.get('sentence_transformers', {})
                    f.write(f"\n  Sentence Transformers Configuration:\n")
                    f.write(f"    Model: {st_config.get('model', 'all-MiniLM-L6-v2')}\n")
                    f.write(f"    Device: {st_config.get('device', 'cpu')}\n")
                elif backend == 'openai':
                    openai_config = retrieval_config.get('openai', {})
                    f.write(f"\n  OpenAI Embeddings Configuration:\n")
                    f.write(f"    Model: {openai_config.get('model', 'text-embedding-3-small')}\n")
                    f.write(f"    Batch Size: {openai_config.get('batch_size', 100)}\n")
                elif backend == 'tfidf':
                    tfidf_config = retrieval_config.get('tfidf', {})
                    f.write(f"\n  TF-IDF Configuration:\n")
                    f.write(f"    Max Features: {tfidf_config.get('max_features', 1000)}\n")
                    f.write(f"    N-gram Range: {tfidf_config.get('ngram_range', [1, 2])}\n")
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
    
    # Create selected samples tracking file
    samples_file = f"{run_dir}/selected_samples.txt"
    
    # Write seed set to tracking file
    with open(samples_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SELECTED SAMPLES TRACKING\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run: {run_timestamp}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Query Strategy: {query_strategy}\n")
        f.write(f"Seed: {seed}, Initial per class: {initial_per_class}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("SEED SET (Initial Labeled Pool)\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, ex in enumerate(labeled_pool, 1):
            # Truncate text if too long
            text_display = ex['text'][:80] + "..." if len(ex['text']) > 80 else ex['text']
            f.write(f"{idx}. ID: {ex['id']}\n")
            f.write(f"   Text: {text_display}\n")
            f.write(f"   Label: {ex['label']}\n\n")
        
        f.write(f"\nTotal seed set examples: {len(labeled_pool)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"✅ Selected samples tracking initialized: selected_samples.txt")
    
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
        'alpha_cf': alpha_cf,
        'cf_budget_remaining': cf_budget_remaining if use_legacy_budget else None
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
            
            # Train V2 probe estimator (if using probe_entropy)
            if probe_estimator is not None:
                print(f"  Training V2 probe estimator on labeled pool...")
                try:
                    probe_estimator.train_probe(labeled_pool)
                    print(f"  ✓ Probe trained on {len(labeled_pool)} examples")
                except Exception as e:
                    print(f"  ⚠️  Probe training failed: {e}")
                    print(f"     Falling back to LLM-based uncertainty for this iteration")
            
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
                # Pass probe_estimator for V2 probe_entropy method
                selected_indices, uncertainty_details = select_uncertain_examples(
                    unlabeled_pool,
                    classifier,
                    current_batch_size,
                    method=uncertainty_method,  # Use the actual method (may have been adjusted)
                    return_details=True,  # Get full logprobs and uncertainty data
                    probe_estimator=probe_estimator  # V2: Pass probe estimator for probe_entropy
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
            
            # Append selected examples to tracking file
            with open(samples_file, 'a') as f:
                f.write("=" * 80 + "\n")
                f.write(f"ITERATION {iteration} - Selected Examples\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Query Strategy: {query_strategy}\n")
                if query_strategy == 'uncertainty':
                    f.write(f"Uncertainty Method: {al_config['uncertainty_method']}\n")
                f.write(f"Batch Size: {len(labeled_examples)}\n\n")
                
                for idx, ex in enumerate(labeled_examples, 1):
                    # Truncate text if too long
                    text_display = ex['text'][:80] + "..." if len(ex['text']) > 80 else ex['text']
                    f.write(f"{idx}. ID: {ex['id']}\n")
                    f.write(f"   Text: {text_display}\n")
                    f.write(f"   Label: {ex['label']}\n")
                    
                    # Add uncertainty score if available
                    if query_strategy == 'uncertainty' and 'uncertainty_details' in locals():
                        ex_idx = selected_examples.index(ex) if ex in selected_examples else -1
                        if ex_idx >= 0 and ex_idx < len(uncertainty_details.get('scores', [])):
                            score = uncertainty_details['scores'][ex_idx]
                            f.write(f"   Uncertainty Score: {score:.4f}\n")
                    f.write("\n")
                
                f.write(f"Total examples selected in iteration {iteration}: {len(labeled_examples)}\n")
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Step 4: Generate counterfactuals with quality filtering
            counterfactuals = []
            num_cfs_added = 0
            if al_config['counterfactuals']['enabled']:
                print(f"\n[Step 4/6] Generating counterfactuals with quality filtering...")
                # Get alpha_cf for this round (V3 per-round budget)
                current_alpha_cf = alpha_cf if not use_legacy_budget else None
                
                counterfactuals, num_cfs_added, cf_generation_details = generate_counterfactuals_batch(
                    labeled_examples,
                    config,
                    llm_provider,
                    all_labels,
                    labeled_pool,           # For diversity calculation
                    classifier,             # For confidence scoring (used if probe_estimator not provided)
                    alpha_cf=current_alpha_cf,  # Version 3: per-round budget multiplier
                    target_label_selector=target_label_selector,  # Version 3: target label selection
                    probe_estimator=probe_estimator,  # V2: Use probe for CF quality filtering if available
                    return_details=True     # Get full generation metadata and prompts
                )
                
                # Update CF budget (legacy mode only)
                if use_legacy_budget and cf_budget_remaining is not None and cf_budget_remaining > 0:
                    cf_budget_remaining = max(0, cf_budget_remaining - num_cfs_added)
                    print(f"  CF budget (legacy): {num_cfs_added} added, {cf_budget_remaining} remaining")
                else:
                    print(f"  CFs added this round: {num_cfs_added} (per-round budget: alpha_cf={alpha_cf})")
                
                # Save Step 5 output: counterfactuals with FULL generation details
                step5_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_{model_safe}_step5_counterfactual_generation.json"
                
                # Extract selected CF IDs for easy reference
                selected_cf_ids = [cf['id'] for cf in counterfactuals]
                
                with open(step5_file, 'w') as f:
                    # Sanitize all data for JSON serialization
                    data_to_save = {
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
                    }
                    # Sanitize all data to ensure JSON serialization works (convert numpy bools, etc.)
                    sanitized_data = sanitize_for_json(data_to_save)
                    json.dump(sanitized_data, f, indent=2)
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
                
                # Retrain V2 probe estimator on updated pool (if using probe_entropy)
                if probe_estimator is not None:
                    try:
                        probe_estimator.train_probe(labeled_pool)
                    except Exception as e:
                        print(f"  ⚠️  Probe retraining failed: {e}")
                
                metrics = evaluate_classifier(classifier, test_pool, config)
                
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
                print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
                
                # Early stopping check (using F1 Macro)
                if metrics['f1_macro'] >= best_f1_macro + al_config['min_improvement']:
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
                'total_counterfactuals_so_far': len([ex for ex in labeled_pool if ex.get('original_id')]),
                'budget_remaining': budget,
                'alpha_cf': alpha_cf,
                'cf_budget_remaining': cf_budget_remaining if use_legacy_budget else None
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
                'total_counterfactuals_so_far': len([ex for ex in labeled_pool if ex.get('original_id')]),
                'budget_remaining': budget,
                'alpha_cf': alpha_cf,
                'cf_budget_remaining': cf_budget_remaining if use_legacy_budget else None
            }
            final_result.update(final_metrics)
            results.append(final_result)
            print(f"  ✓ Final evaluation added to results (Iteration {final_result['iteration']})")
        else:
            print(f"  ✓ Final state already evaluated in iteration {last_result['iteration']} (no duplicate needed)")
    
    # Save results
    save_results(results, run_dir, eval_config.get('classifier_type', 'static'))
    save_final_labeled_pool(labeled_pool, run_dir)
    
    # Finalize selected samples tracking file with summary
    with open(samples_file, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total iterations completed: {iteration}\n")
        f.write(f"Total real examples selected: {al_config['total_budget'] - budget}\n")
        f.write(f"Total counterfactuals generated: {sum(1 for ex in labeled_pool if 'original_id' in ex)}\n")
        f.write(f"Final labeled pool size: {len(labeled_pool)}\n")
        f.write(f"Budget consumed: {al_config['total_budget'] - budget}/{al_config['total_budget']}\n")
        
        if test_pool and final_metrics:
            f.write(f"\nFinal Test Performance:\n")
            f.write(f"  Accuracy: {final_metrics['accuracy']:.4f}\n")
            f.write(f"  F1 Macro: {final_metrics['f1_macro']:.4f}\n")
            f.write(f"  F1 Weighted: {final_metrics['f1_weighted']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF SELECTION TRACKING\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Selected samples tracking finalized: selected_samples.txt")
    
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

