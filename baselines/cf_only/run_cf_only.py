#!/usr/bin/env python3
"""
CF-Only Ablation Study

Tests the effect of counterfactual generation alone:
- Random factual selection (no active learning)
- Generate counterfactuals for selected examples
- One-shot evaluation (no iterative AL loop)

Budgets: 10, 20, 30, 40, 50, 100
"""

import os
import sys
import yaml
import pandas as pd
import random
from datetime import datetime
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import (
    get_llm_provider,
    load_dataset,
    get_unique_labels,
    shuffle_dataframe
)
from utils.classifier import SimpleICLClassifier
from utils.oracle import get_oracle
from utils.counterfactual_generator import generate_counterfactuals_batch
from utils.target_label_selector import TargetLabelSelector
from utils.retrieval_classifier import get_retrieval_classifier


def load_cf_only_config():
    """Load CF-Only configuration"""
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'cf_only_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_pools_cf_only(config: dict) -> tuple:
    """Initialize pools for CF-Only (same as AL loop)"""
    dataset_config = config['dataset']
    al_config = config['active_learning']
    processing = config['processing']
    
    train_file = f"{config['directories']['input_data']}/{dataset_config['train_file']}"
    seed = processing['seed']
    initial_per_class = al_config['initial_labeled_per_class']
    
    # Load or create seed set
    seed_file = train_file.replace('.csv', f'_seed_set_s{seed}_n{initial_per_class}.csv')
    seed_filename = os.path.basename(seed_file)
    col_id = dataset_config['columns']['id']
    col_text = dataset_config['columns']['text']
    col_label = dataset_config['columns']['label']
    
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
        
        df = load_dataset(train_file, config)
        df = shuffle_dataframe(df, seed)
        unique_labels = get_unique_labels(df, col_label, dataset_config.get('exclude_labels', []))
        unique_labels = [l for l in unique_labels if pd.notna(l) and str(l).lower() not in ['none', 'null', '', 'nan']]
        
        print(f"   Labels: {unique_labels}")
        
        seed_examples = []
        for label in unique_labels:
            label_df = df[df[col_label] == label]
            num_samples = min(initial_per_class, len(label_df))
            if num_samples > 0:
                samples = label_df.head(num_samples)
                seed_examples.append(samples)
                print(f"     ✓ {label}: {num_samples} examples")
            else:
                print(f"     Warning: No examples for label '{label}'")
        
        seed_df = pd.concat(seed_examples, ignore_index=True)
        seed_df.to_csv(seed_file, index=False)
        
        print(f"   ✅ Created seed set: {len(seed_df)} examples")
        print(f"   Saved to: {seed_filename}")
    
    # Load full training data
    df = load_dataset(train_file, config)
    unique_labels = seed_df[col_label].unique().tolist()
    unique_labels = [l for l in unique_labels if pd.notna(l) and str(l).lower() not in ['none', 'null', '', 'nan']]
    seed_ids = set(seed_df[col_id].tolist())
    
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
    
    return labeled_pool, unlabeled_pool, unique_labels


def load_test_set_cf_only(config: dict) -> List[Dict]:
    """Load test set"""
    dataset_config = config['dataset']
    test_file = f"{config['directories']['input_data']}/{dataset_config['test_file']}"
    
    try:
        df_test = load_dataset(test_file, config)
        col_id = dataset_config['columns']['id']
        col_text = dataset_config['columns']['text']
        col_label = dataset_config['columns']['label']
        
        test_pool = []
        for idx, row in df_test.iterrows():
            test_pool.append({
                'id': row[col_id],
                'text': row[col_text],
                'label': row[col_label]
            })
        return test_pool
    except FileNotFoundError:
        return []


def evaluate_classifier_cf_only(classifier, test_pool: List[Dict]) -> Dict:
    """Evaluate classifier on test set"""
    if not test_pool:
        return {}
    
    texts = [ex['text'] for ex in test_pool]
    labels = [ex['label'] for ex in test_pool]
    
    predictions = []
    for text in texts:
        pred = classifier.predict(text)
        predictions.append(pred)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall
    }


def run_cf_only_ablation(
    budget: int,
    labeled_pool: List[Dict],
    unlabeled_pool: List[Dict],
    test_pool: List[Dict],
    all_labels: List[str],
    config: dict,
    llm_provider,
    classifier,
    oracle
) -> Dict:
    """
    Run CF-Only ablation for a specific budget.
    
    Steps:
    1. Baseline: Evaluate on seed set only
    2. Select random examples (budget size)
    3. Query oracle for labels
    4. Generate counterfactuals
    5. Evaluate final performance
    """
    print(f"\n{'='*80}")
    print(f"CF-Only Ablation: Budget = {budget}")
    print(f"{'='*80}\n")
    
    # Step 1: Baseline evaluation (seed set only)
    print(f"[Baseline] Evaluating on seed set only...")
    classifier.train(labeled_pool)
    baseline_metrics = evaluate_classifier_cf_only(classifier, test_pool)
    
    print(f"  Baseline F1 Macro: {baseline_metrics.get('f1_macro', 0):.4f}")
    
    # Step 2: Select random examples
    print(f"\n[Step 1] Selecting {budget} random examples...")
    if budget > len(unlabeled_pool):
        budget = len(unlabeled_pool)
        print(f"  Warning: Budget reduced to {budget} (unlabeled pool size)")
    
    # Use seed + budget to ensure different random selections for different budgets
    # but reproducible across runs
    random.seed(config['processing']['seed'] + budget)
    selected_indices = random.sample(range(len(unlabeled_pool)), budget)
    selected_examples = [unlabeled_pool[i] for i in selected_indices]
    
    print(f"  Selected {len(selected_examples)} random examples")
    
    # Step 3: Query oracle
    print(f"\n[Step 2] Querying oracle for labels...")
    labeled_factuals = oracle.label_examples(selected_examples)
    
    print(f"  Labeled {len(labeled_factuals)} examples")
    
    # Step 4: Generate counterfactuals
    print(f"\n[Step 3] Generating counterfactuals...")
    cf_config = config['active_learning']['counterfactuals']
    
    if cf_config.get('enabled', False):
        # Initialize target label selector
        target_label_selector = None
        if 'target_label_selection' in cf_config:
            seed = config['processing']['seed']
            target_label_selector = TargetLabelSelector(
                config=config,
                all_labels=all_labels,
                seed=seed
            )
        
        alpha_cf = cf_config.get('alpha_cf', 1.0)
        
        counterfactuals, num_cfs_added = generate_counterfactuals_batch(
            labeled_examples=labeled_factuals,
            config=config,
            llm_provider=llm_provider,
            all_labels=all_labels,
            labeled_pool=labeled_pool,
            classifier=classifier,
            alpha_cf=alpha_cf,
            target_label_selector=target_label_selector,
            return_details=False
        )
        
        print(f"  Generated {num_cfs_added} counterfactuals")
        
        # Add factuals and CFs to labeled pool
        labeled_pool.extend(labeled_factuals)
        labeled_pool.extend(counterfactuals)
    else:
        print("  Counterfactuals disabled - only adding factuals")
        labeled_pool.extend(labeled_factuals)
        num_cfs_added = 0
    
    # Step 5: Final evaluation
    print(f"\n[Step 4] Evaluating final performance...")
    classifier.train(labeled_pool)
    final_metrics = evaluate_classifier_cf_only(classifier, test_pool)
    
    print(f"  Final F1 Macro: {final_metrics.get('f1_macro', 0):.4f}")
    print(f"  Improvement: {final_metrics.get('f1_macro', 0) - baseline_metrics.get('f1_macro', 0):.4f}")
    
    return {
        'budget': budget,
        'baseline_f1_macro': baseline_metrics.get('f1_macro', 0),
        'final_f1_macro': final_metrics.get('f1_macro', 0),
        'improvement': final_metrics.get('f1_macro', 0) - baseline_metrics.get('f1_macro', 0),
        'num_factuals': len(labeled_factuals),
        'num_counterfactuals': num_cfs_added,
        'final_labeled_pool_size': len(labeled_pool),
        **final_metrics
    }


def main():
    print("\n" + "="*80)
    print("CF-ONLY ABLATION STUDY")
    print("="*80)
    
    # Load config
    config = load_cf_only_config()
    
    # Extract dataset name
    dataset_name = config['dataset']['train_file'].replace('_train.csv', '').replace('.csv', '')
    
    # Build output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['llm']['openai']['model']
    output_dir = os.path.join(
        os.path.dirname(__file__),
        'output',
        f'cf_only_{timestamp}_{model_name}_{dataset_name}'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nDataset: {dataset_name}")
    print(f"Output directory: {output_dir}\n")
    
    # Initialize components
    llm_provider = get_llm_provider(config)
    
    eval_config = config['evaluation']
    classifier_type = eval_config.get('classifier_type', 'static')
    
    if classifier_type == 'retrieval':
        classifier = get_retrieval_classifier(config, llm_provider)
        print(f"Using Retrieval-based ICL classifier")
    else:
        classifier = SimpleICLClassifier(config, llm_provider)
        print(f"Using Static ICL classifier")
    
    oracle = get_oracle(config)
    
    # Initialize pools
    labeled_pool, unlabeled_pool, all_labels = initialize_pools_cf_only(config)
    test_pool = load_test_set_cf_only(config)
    
    print(f"\nInitial labeled pool (seed): {len(labeled_pool)} examples")
    print(f"Unlabeled pool: {len(unlabeled_pool)} examples")
    print(f"Test pool: {len(test_pool)} examples")
    print(f"Labels: {all_labels}\n")
    
    # Get budgets from config
    budgets = config['cf_only']['budgets']
    
    # Run CF-Only ablation for each budget
    all_results = []
    
    for budget in budgets:
        # Reset pools for each budget (start fresh from seed)
        labeled_pool_reset, unlabeled_pool_reset, _ = initialize_pools_cf_only(config)
        
        # Create a fresh classifier instance for each budget to avoid state issues
        if classifier_type == 'retrieval':
            classifier_reset = get_retrieval_classifier(config, llm_provider)
        else:
            classifier_reset = SimpleICLClassifier(config, llm_provider)
        
        result = run_cf_only_ablation(
            budget=budget,
            labeled_pool=labeled_pool_reset,
            unlabeled_pool=unlabeled_pool_reset,
            test_pool=test_pool,
            all_labels=all_labels,
            config=config,
            llm_provider=llm_provider,
            classifier=classifier_reset,
            oracle=oracle
        )
        
        all_results.append(result)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, 'cf_only_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Save config
    config_path = os.path.join(output_dir, 'config_used.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Budget':<10} {'Baseline F1':<15} {'Final F1':<15} {'Improvement':<15}")
    print("-"*55)
    for r in all_results:
        print(f"{r['budget']:<10} {r['baseline_f1_macro']:<15.4f} {r['final_f1_macro']:<15.4f} {r['improvement']:<15.4f}")
    
    print(f"\nResults saved to: {results_path}\n")


if __name__ == "__main__":
    main()

