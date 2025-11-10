#!/usr/bin/env python3
"""
Script 04: Counterfactual Evaluation

Evaluate the effectiveness of counterfactual data augmentation by testing LLM 
as a few-shot classifier with and without counterfactuals.

Input: 
- Filtered counterfactuals from Script 03
- Separate test set
Output: 
- Performance metrics (Precision, Recall, F-Score) across different sample sizes
"""

import sys
import pandas as pd
import numpy as np
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import List, Dict
from utils import (
    load_config,
    ensure_directories,
    load_dataset,
    get_unique_labels,
    shuffle_dataframe,
    get_llm_provider
)


def get_initial_message(label_list: List[str]) -> List[Dict[str, str]]:
    """
    Create initial system prompt for few-shot classification.
    
    Args:
        label_list: List of masked labels (e.g., ['concept A', 'concept B', ...])
        
    Returns:
        Initial message list
    """
    labels_str = ', '.join(label_list)
    
    messages = [
        {
            "role": "system",
            "content": f"""You are a text classifier. Classify sentences into one of these labels: {labels_str}.
            Respond with only the label, nothing else."""
        }
    ]
    
    return messages


def update_example(messages: List[Dict[str, str]], text: str, label: str):
    """
    Add a training example to the context.
    
    Args:
        messages: Message list to update
        text: Example text
        label: Masked label
    """
    messages.append({"role": "user", "content": text})
    messages.append({"role": "assistant", "content": label})


def get_response(llm_provider, messages: List[Dict[str, str]], query: str) -> str:
    """
    Query LLM for label prediction.
    
    Args:
        llm_provider: LLM provider instance
        messages: Context messages
        query: Test sentence to classify
        
    Returns:
        Predicted label
    """
    # Add query to messages
    query_messages = messages + [{"role": "user", "content": query}]
    
    # Use llm_provider's chat_completion method directly
    # It already handles GPT-5-nano specific parameters (temperature=1.0, reasoning_effort="low", no stop)
    response = llm_provider.chat_completion(
        messages=query_messages,
        temperature=0,  # Provider will adjust to 1.0 for GPT-5-nano
        max_tokens=4096,
        stop=["\n"]  # Provider will remove this for GPT-5-nano
    )
    
    return response.strip()


def counterfactual_shots(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    label_map: Dict[str, str],
    config: dict,
    llm_provider,
    shuffle_seed: int
):
    """
    Enhanced evaluation with larger counterfactual dataset and higher shot counts.
    
    Args:
        df: Filtered counterfactuals DataFrame
        df_test: Test dataset DataFrame
        label_map: Mapping from original labels to masked labels
        config: Configuration dictionary
        llm_provider: LLM provider instance
        shuffle_seed: Random seed for shuffling
        
    Returns:
        Final precision, recall, f-score tuple
    """
    print(f"\n--- Evaluation with shuffle seed: {shuffle_seed} ---")
    
    # Get configuration
    dataset_config = config['dataset']
    processing = config['processing']
    dirs = config['directories']
    
    # Get model name for file naming
    model_name = config['llm']['provider']
    if config['llm']['provider'] == 'ollama':
        model_name = config['llm']['ollama']['model'].replace(':', '_').replace('/', '_').replace('.', '_')
    elif config['llm']['provider'] == 'gemini':
        model_name = config['llm']['gemini']['model'].replace('-', '_').replace('/', '_').replace('.', '_')
    elif config['llm']['provider'] == 'openai':
        model_name = config['llm']['openai']['model'].replace('-', '_').replace('/', '_').replace('.', '_')
    
    dataset_name = dataset_config['train_file'].replace('.csv', '')
    
    # Checkpoint file
    checkpoint_file = f"{dirs['output_data']}/[{shuffle_seed}][{model_name}]eval_checkpoint_{dataset_name}.csv"
    
    col_text = dataset_config['columns']['text']
    col_label = dataset_config['columns']['label']
    
    selections = processing.get('evaluation_shots', [10, 15, 30, 50, 70, 90, 120])
    max_counterfactuals = processing['max_counterfactuals_per_example']
    
    # Reverse label map for converting back
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Get unique examples (remove duplicate IDs from counterfactuals)
    df_unique = df.drop_duplicates(subset='id')
    print(f"  Available unique training examples: {len(df_unique)}")
    
    # Enhanced: Check if we have enough examples for higher shot counts
    max_possible_shots = len(df_unique)
    if max_possible_shots < max(selections):
        print(f"  INFO: Limiting shot counts to {max_possible_shots} due to available data")
        selections = [s for s in selections if s <= max_possible_shots]
    
    # Shuffle with seed
    shuffled_df = shuffle_dataframe(df_unique, shuffle_seed)
    
    results = []
    
    # Check for checkpoint
    start_selection_idx = 0
    if pd.io.common.file_exists(checkpoint_file):
        print(f"  Found checkpoint file. Loading progress...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        results = checkpoint_df.values.tolist()
        start_selection_idx = len(results)
        print(f"  Resuming from shot configuration {start_selection_idx+1}/{len(selections)}")
        if start_selection_idx >= len(selections):
            print(f"  All shot configurations complete, loading from checkpoint")
            df_results = checkpoint_df
            # Return final metrics
            final_row = df_results.iloc[-1]
            return final_row['precision'], final_row['recall'], final_row['fscore']
    
    # Test different sample sizes
    try:
        for sel_idx, selection in enumerate(selections):
            # Skip already processed shot configurations
            if sel_idx < start_selection_idx:
                continue
                
            print(f"\n  Testing with {selection} shots [{sel_idx+1}/{len(selections)}]...")
            
            # Select N examples
            selected_df = shuffled_df.head(selection)
            
            # Initialize context with system prompt
            masked_labels = list(label_map.values())
            messages = get_initial_message(masked_labels)
            
            # Enhanced: Add progress tracking for context building
            print(f"    Building context with {len(selected_df)} examples...")
            context_examples = 0
            
            # Add original examples to context
            for idx, (_, row) in enumerate(selected_df.iterrows()):
                ori_text = row['ori_text']
                ori_label = row['ori_label']
                masked_label = label_map[ori_label]
                
                update_example(messages, ori_text, masked_label)
                context_examples += 1
                
                # Enhanced: Add up to max_counterfactuals per example with better filtering
                counterfactuals = df[
                    (df['id'] == row['id']) &
                    (df['matched_pattern'] == True) &
                    (df['heuristic_filtered'] == True) &
                    (df['is_ori'] == False) &
                    (df['is_target'] == True)
                ]
                
                # Add counterfactuals to context
                added_cf = 0
                for _, cf_row in counterfactuals.head(max_counterfactuals).iterrows():
                    cf_text = cf_row['counterfactual']
                    cf_label = cf_row['target_label']
                    masked_cf_label = label_map[cf_label]
                    
                    update_example(messages, cf_text, masked_cf_label)
                    context_examples += 1
                    added_cf += 1
                
                # Progress indicator for context building
                if (idx + 1) % 20 == 0:
                    print(f"      Context progress: {idx+1}/{len(selected_df)} examples processed")
            
            print(f"    Context built: {context_examples} total examples ({len(selected_df)} originals + counterfactuals)")
            
            # Use entire test dataset for comprehensive evaluation
            test_size = len(df_test)  # Use all test examples
            test_sample = df_test.copy()  # Use entire dataset without sampling
            
            y_true = []
            y_pred = []
            
            # Checkpoint saving interval for test examples
            CHECKPOINT_INTERVAL = 25  # Save progress every 25 test examples
            
            print(f"    Classifying {len(test_sample)} test examples...")
            
            for idx, (_, test_row) in enumerate(test_sample.iterrows()):
                test_text = test_row[col_text]
                true_label = test_row[col_label]
                
                # Skip if label not in mapping
                if true_label not in label_map:
                    continue
                    
                masked_true_label = label_map[true_label]
                
                # Get prediction
                try:
                    pred_label = get_response(llm_provider, messages, test_text)
                    
                    y_true.append(masked_true_label)
                    y_pred.append(pred_label)
                    
                except Exception as e:
                    print(f"      Error at test index {idx}: {e}")
                    continue
                
                # Save checkpoint every CHECKPOINT_INTERVAL examples
                if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                    print(f"      Progress: {idx+1}/{len(test_sample)} - Saving checkpoint...")
                    
                    # Calculate intermediate metrics
                    if len(y_true) > 0:
                        prf_temp = precision_recall_fscore_support(
                            y_true, y_pred, average='macro', zero_division=0
                        )
                        temp_precision, temp_recall, temp_fscore = prf_temp[0], prf_temp[1], prf_temp[2]
                        temp_accuracy = accuracy_score(y_true, y_pred)
                        
                        # Create temporary results list for this checkpoint
                        # Include all previously completed shot configurations + current one
                        temp_results = results[:sel_idx] + [[selection, temp_precision, temp_recall, temp_fscore, temp_accuracy, len(y_true)]]
                        
                        df_checkpoint = pd.DataFrame(
                            temp_results, 
                            columns=['shots', 'precision', 'recall', 'fscore', 'accuracy', 'test_size']
                        )
                        df_checkpoint.to_csv(checkpoint_file, index=False)
                        print(f"      Checkpoint saved: {len(y_true)} examples classified so far")
            
            if len(y_true) == 0:
                print(f"    WARNING: No valid predictions for {selection}-shot")
                continue
            
            # Calculate metrics
            prf = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            
            precision, recall, fscore = prf[0], prf[1], prf[2]
            
            # Enhanced: Calculate accuracy as well
            accuracy = accuracy_score(y_true, y_pred)
            
            print(f"    Results for {selection}-shot:")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall: {recall:.4f}")
            print(f"      F-Score: {fscore:.4f}")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      Test examples: {len(y_true)}")
            
            results.append([selection, precision, recall, fscore, accuracy, len(y_true)])
            
            # Save checkpoint after each shot configuration
            df_checkpoint = pd.DataFrame(
                results, 
                columns=['shots', 'precision', 'recall', 'fscore', 'accuracy', 'test_size']
            )
            df_checkpoint.to_csv(checkpoint_file, index=False)
            print(f"    Checkpoint saved: {sel_idx+1}/{len(selections)} configurations complete")
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user!")
        print(f"Processed {len(results)}/{len(selections)} shot configurations")
        print(f"Progress saved to: {checkpoint_file}")
        df_checkpoint = pd.DataFrame(
            results, 
            columns=['shots', 'precision', 'recall', 'fscore', 'accuracy', 'test_size']
        )
        df_checkpoint.to_csv(checkpoint_file, index=False)
        print("Run script again to resume from checkpoint")
        import sys
        sys.exit(0)
    
    # Enhanced: Save results with additional metrics
    df_results = pd.DataFrame(
        results, 
        columns=['shots', 'precision', 'recall', 'fscore', 'accuracy', 'test_size']
    )
    
    dataset_name = dataset_config['train_file'].replace('.csv', '')
    output_file = f"{dirs['archive']}/gpt/[{shuffle_seed}][{model_name}]_counter_{dataset_name}_prf.csv"
    df_results.to_csv(output_file, index=False)
    
    print(f"\n  Results saved to: {output_file}")
    
    # Clean up checkpoint file on successful completion
    import os
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"  Checkpoint file removed (evaluation complete)")
    
    # Return final metrics
    final_row = df_results.iloc[-1]
    return final_row['precision'], final_row['recall'], final_row['fscore']


def evaluate_counterfactuals(config: dict, llm_provider):
    """
    Run evaluation across multiple seeds for robust results.
    
    Args:
        config: Configuration dictionary
        llm_provider: LLM provider instance
    """
    print("\n=== Starting Counterfactual Evaluation ===")
    
    # Load configuration
    dirs = config['directories']
    dataset_config = config['dataset']
    processing = config['processing']
    seed = processing['seed']
    
    # Get model name for file naming
    model_name = config['llm']['provider']
    if config['llm']['provider'] == 'ollama':
        model_name = config['llm']['ollama']['model'].replace(':', '_')
    elif config['llm']['provider'] == 'gemini':
        model_name = config['llm']['gemini']['model'].replace('-', '_')
    elif config['llm']['provider'] == 'openai':
        model_name = config['llm']['openai']['model'].replace('-', '_')
    
    # Load filtered counterfactuals
    dataset_file = dataset_config['train_file']
    filtered_file = f"{dirs['output_data']}/[{seed}][{model_name}]filtered_{dataset_file}"
    
    try:
        df = pd.read_csv(filtered_file)
        print(f"\nINFO: Loaded {len(df)} filtered counterfactuals")
    except FileNotFoundError:
        print(f"ERROR: Filtered file not found: {filtered_file}")
        print("Please run Script 03 (03_counterfactual_filtering.py) first")
        sys.exit(1)
    
    # Load test dataset
    test_file = f"{dirs['input_data']}/{dataset_config['test_file']}"
    df_test = load_dataset(test_file, config)
    
    col_label = dataset_config['columns']['label']
    
    # Create label mapping (mask labels with abstract concepts)
    # Get all unique labels from both original and target labels in counterfactuals
    ori_labels = get_unique_labels(
        df, 
        'ori_label',
        dataset_config.get('exclude_labels', [])
    )
    target_labels = get_unique_labels(
        df, 
        'target_label', 
        dataset_config.get('exclude_labels', [])
    )
    
    # Combine and deduplicate all labels
    unique_labels = sorted(list(set(ori_labels + target_labels)))
    
    label_map = {
        label: f'concept {chr(65 + i)}' 
        for i, label in enumerate(unique_labels)
    }
    
    print(f"\nINFO: Label mapping:")
    for orig, masked in label_map.items():
        print(f"  {orig} → {masked}")
    
    # Simplified: Run evaluation with just one seed for faster testing
    evaluation_seeds = [42]  # Single seed for quick evaluation
    
    all_results = []
    
    print(f"\nINFO: Running evaluation with {len(evaluation_seeds)} seed for quick testing")
    
    for shuffle_seed in evaluation_seeds:
        try:
            precision, recall, fscore = counterfactual_shots(
                df, df_test, label_map, config, llm_provider, shuffle_seed
            )
            all_results.append({
                'seed': shuffle_seed,
                'precision': precision,
                'recall': recall,
                'fscore': fscore
            })
        except Exception as e:
            print(f"ERROR: Evaluation failed for seed {shuffle_seed}: {e}")
            continue
    
    # Enhanced: Calculate comprehensive statistics
    if all_results:
        precisions = [r['precision'] for r in all_results]
        recalls = [r['recall'] for r in all_results]
        fscores = [r['fscore'] for r in all_results]
        
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_fscore = np.mean(fscores)
        
        std_precision = np.std(precisions)
        std_recall = np.std(recalls)
        std_fscore = np.std(fscores)
        
        print(f"\n{'='*60}")
        print("Enhanced Evaluation Results Summary:")
        print(f"{'='*60}")
        print(f"Seeds evaluated: {len(all_results)}/{len(evaluation_seeds)}")
        print(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"Average F-Score: {avg_fscore:.4f} ± {std_fscore:.4f}")
        print(f"{'='*60}")

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("Script 04: Counterfactual Evaluation")
    print("="*60)
    
    # Load configuration
    config = load_config()
    ensure_directories(config)
    
    # Initialize LLM provider
    print(f"\nINFO: Using LLM provider: {config['llm']['provider']}")
    llm_provider = get_llm_provider(config)
    
    # Run evaluation
    evaluate_counterfactuals(config, llm_provider)
    
    print("="*60)
    print("Script 04 Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
