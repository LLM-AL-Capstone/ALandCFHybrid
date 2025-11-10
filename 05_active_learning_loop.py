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
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (labeled_pool, unlabeled_pool, all_labels)
    """
    print("\n=== Initializing Data Pools ===")
    
    dataset_config = config['dataset']
    al_config = config['active_learning']
    processing = config['processing']
    
    # Load training data
    train_file = f"{config['directories']['input_data']}/{dataset_config['train_file']}"
    df = load_dataset(train_file, config)
    
    # Get column names
    col_id = dataset_config['columns']['id']
    col_text = dataset_config['columns']['text']
    col_label = dataset_config['columns']['label']
    
    # Shuffle with seed for reproducibility
    df = shuffle_dataframe(df, processing['seed'])
    
    # Get unique labels (excluding specified labels)
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
    
    print(f"Unique labels: {unique_labels}")
    print(f"Number of labels: {len(unique_labels)}")
    
    # Create stratified initial labeled set
    initial_per_class = al_config['initial_labeled_per_class']
    
    labeled_indices = []
    for label in unique_labels:
        label_df = df[df[col_label] == label]
        num_samples = min(initial_per_class, len(label_df))
        
        if num_samples == 0:
            print(f"  Warning: No examples for label '{label}'")
            continue
        
        sample_indices = label_df.head(num_samples).index.tolist()
        labeled_indices.extend(sample_indices)
    
    # Create labeled and unlabeled pools
    labeled_pool = []
    unlabeled_pool = []
    
    for idx, row in df.iterrows():
        example = {
            'id': row[col_id],
            'text': row[col_text],
            'label': row[col_label]
        }
        
        if idx in labeled_indices:
            labeled_pool.append(example)
        else:
            unlabeled_pool.append(example)
    
    print(f"\nInitial labeled pool: {len(labeled_pool)} examples")
    print(f"  ({initial_per_class} per class × {len(unique_labels)} classes)")
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
                   results: List[Dict], config: dict):
    """
    Save checkpoint for recovery.
    
    Args:
        iteration: Current iteration number
        labeled_pool: Current labeled pool
        unlabeled_pool: Current unlabeled pool
        results: Results so far
        config: Configuration dictionary
    """
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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


def save_results(results: List[Dict], config: dict):
    """
    Save final results to CSV.
    
    Args:
        results: List of iteration results
        config: Configuration dictionary
    """
    results_file = config['logging']['results_file']
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    df_results.to_csv(results_file, index=False)
    
    print(f"\nResults saved to: {results_file}")


def save_final_labeled_pool(labeled_pool: List[Dict], config: dict):
    """
    Save final augmented labeled pool.
    
    Args:
        labeled_pool: Final labeled pool (with counterfactuals)
        config: Configuration dictionary
    """
    output_file = f"{config['directories']['output_data']}/final_labeled_pool.csv"
    
    df = pd.DataFrame(labeled_pool)
    df.to_csv(output_file, index=False)
    
    print(f"Final labeled pool saved to: {output_file}")


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
    classifier = SimpleICLClassifier(config, llm_provider)
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
    
    # Tracking
    results = []
    best_accuracy = 0.0
    patience_counter = 0
    
    # Main loop
    iteration = 0
    
    # Setup interim output directory
    interim_dir = f"{config['directories']['output_data']}/interim_output"
    import os
    os.makedirs(interim_dir, exist_ok=True)
    
    # Import datetime for timestamping
    from datetime import datetime
    
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
            step1_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_step1_classifier_training.json"
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
            
            # Step 2: Evaluate current model
            metrics = None
            if test_pool and (iteration % eval_config['eval_every_iterations'] == 0):
                print(f"\n[Step 2/6] Evaluating on test set...")
                metrics = evaluate_classifier(classifier, test_pool, config)
                
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
                print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
                
                # Early stopping check
                if metrics['accuracy'] > best_accuracy + al_config['min_improvement']:
                    best_accuracy = metrics['accuracy']
                    patience_counter = 0
                    print(f"  ✓ New best accuracy: {best_accuracy:.4f}")
                else:
                    patience_counter += 1
                    print(f"  No improvement (patience: {patience_counter}/{al_config['early_stopping_patience']})")
                
                # Save Step 2 output
                step2_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_step2_evaluation.json"
                with open(step2_file, 'w') as f:
                    json.dump({
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'step': 'evaluation',
                        'metrics': metrics,
                        'best_accuracy': best_accuracy,
                        'patience_counter': patience_counter,
                        'test_pool_size': len(test_pool)
                    }, f, indent=2)
                print(f"  ✓ Interim output saved: {step2_file}")
                
                if patience_counter >= al_config['early_stopping_patience']:
                    print(f"\n⚠ Early stopping triggered (no improvement for {patience_counter} iterations)")
                    break
            else:
                print(f"\n[Step 2/6] Skipping evaluation (eval_every={eval_config['eval_every_iterations']})")
                
                # Still save Step 2 output (skipped)
                step2_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_step2_evaluation_skipped.json"
                with open(step2_file, 'w') as f:
                    json.dump({
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'step': 'evaluation',
                        'status': 'skipped',
                        'reason': f'eval_every_iterations={eval_config["eval_every_iterations"]}'
                    }, f, indent=2)
                print(f"  ✓ Interim output saved: {step2_file}")
            
            # Step 3: Select uncertain examples
            print(f"\n[Step 3/6] Selecting uncertain examples...")
            
            current_batch_size = min(batch_size, len(unlabeled_pool), budget)
            
            selected_indices, uncertainty_details = select_uncertain_examples(
                unlabeled_pool,
                classifier,
                current_batch_size,
                method=al_config['uncertainty_method'],
                return_details=True  # Get full logprobs and uncertainty data
            )
            
            selected_examples = [unlabeled_pool[i] for i in selected_indices]
            
            print(f"  Selected {len(selected_examples)} examples for labeling")
            
            # Save Step 3 output: selected examples with FULL uncertainty details
            step3_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_step3_uncertainty_selection.json"
            with open(step3_file, 'w') as f:
                json.dump({
                    'iteration': iteration,
                    'timestamp': timestamp,
                    'step': 'uncertainty_selection',
                    'selected_indices': selected_indices,
                    'selected_examples': selected_examples,
                    'uncertainty_analysis': uncertainty_details,  # FULL DETAILS: logprobs, entropy, etc.
                    'unlabeled_pool_size': len(unlabeled_pool),
                    'batch_size': current_batch_size
                }, f, indent=2)
            print(f"  ✓ Interim output saved: {step3_file}")
            
            # Step 4: Query oracle
            print(f"\n[Step 4/6] Querying oracle for labels...")
            labeled_examples = oracle.label_examples(selected_examples)
            
            # Save Step 4 output: oracle labels
            step4_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_step4_oracle_labeling.json"
            with open(step4_file, 'w') as f:
                json.dump({
                    'iteration': iteration,
                    'timestamp': timestamp,
                    'step': 'oracle_labeling',
                    'labeled_examples': labeled_examples,
                    'num_labeled': len(labeled_examples)
                }, f, indent=2)
            print(f"  ✓ Interim output saved: {step4_file}")
            
            # Step 5: Generate counterfactuals
            counterfactuals = []
            if al_config['counterfactuals']['enabled']:
                print(f"\n[Step 5/6] Generating counterfactuals...")
                counterfactuals, cf_generation_details = generate_counterfactuals_batch(
                    labeled_examples,
                    config,
                    llm_provider,
                    all_labels,
                    return_details=True  # Get full generation metadata and prompts
                )
                print(f"  Generated {len(counterfactuals)} counterfactuals")
                
                # Save Step 5 output: counterfactuals with FULL generation details
                step5_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_step5_counterfactual_generation.json"
                with open(step5_file, 'w') as f:
                    json.dump({
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'step': 'counterfactual_generation',
                        'input_examples': labeled_examples,
                        'generated_counterfactuals': counterfactuals,
                        'generation_details': cf_generation_details,  # FULL DETAILS: prompts, times, etc.
                        'num_generated': len(counterfactuals)
                    }, f, indent=2)
                print(f"  ✓ Interim output saved: {step5_file}")
            else:
                print(f"\n[Step 5/6] Skipping counterfactual generation (disabled)")
                
                # Still save Step 5 output (skipped)
                step5_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_step5_counterfactual_generation_skipped.json"
                with open(step5_file, 'w') as f:
                    json.dump({
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'step': 'counterfactual_generation',
                        'status': 'skipped',
                        'reason': 'counterfactuals_disabled_in_config'
                    }, f, indent=2)
                print(f"  ✓ Interim output saved: {step5_file}")
            
            # Step 6: Update pools
            print(f"\n[Step 6/6] Updating data pools...")
            
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
            
            # Save Step 6 output: pool updates
            step6_file = f"{interim_dir}/iter_{iteration:02d}_{timestamp}_step6_pool_update.json"
            with open(step6_file, 'w') as f:
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
            print(f"  ✓ Interim output saved: {step6_file}")
            
            # Save iteration results
            iter_result = {
                'iteration': iteration,
                'labeled_pool_size': len(labeled_pool),
                'unlabeled_pool_size': len(unlabeled_pool),
                'num_real_examples': len(labeled_examples),
                'num_counterfactuals': len(counterfactuals),
                'budget_remaining': budget
            }
            
            if metrics:
                iter_result.update(metrics)
            
            results.append(iter_result)
            
            # Save checkpoint
            if iteration % log_config['checkpoint_every'] == 0:
                save_checkpoint(iteration, labeled_pool, unlabeled_pool, results, config)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user!")
        print(f"Completed {iteration} iterations")
        save_checkpoint(iteration, labeled_pool, unlabeled_pool, results, config)
        save_results(results, config)
        sys.exit(0)
    
    # Final evaluation
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
    
    # Save results
    save_results(results, config)
    save_final_labeled_pool(labeled_pool, config)
    
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
    
    # Create checkpoint directory
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Run active learning loop
    active_learning_loop(config)


if __name__ == "__main__":
    main()

