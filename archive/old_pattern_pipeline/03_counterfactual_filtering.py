#!/usr/bin/env python3
"""
Script 03: Counterfactual Filtering (LLM-Based)

Filter counterfactual sentences generated in Script 02 to ensure high quality.
Uses three-stage filtering pipeline:
1. Heuristic filtering (remove meta-responses)
2. LLM semantic filtering (validate phrase usage & context)
3. Discriminator filtering (validate label transformation)

Input: output_data/[{SEED}]counterfactuals_{dataset}.csv
Output: 
- output_data/[{seed}]filtered_{dataset}.csv (complete with filter results)
- output_data/[{seed}]fine_tuneset_{dataset}.csv (high-quality subset)
"""

import sys
import pandas as pd
import ast
from utils import (
    load_config,
    ensure_directories,
    get_llm_provider
)


def heuristic_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter 1: Remove GPT meta-responses using heuristic rules.
    
    Marks as INVALID if counterfactual contains both:
    - "given the constraints" (case-insensitive)
    - "the assistant" (case-insensitive)
    
    Args:
        df: DataFrame with counterfactuals
        
    Returns:
        DataFrame with 'heuristic_filtered' column added
    """
    print("\n--- Filter 1: Heuristic Filtering ---")
    
    def is_valid(text):
        """Check if counterfactual is valid (not a meta-response)"""
        if pd.isna(text):
            return False
        text_lower = str(text).lower()
        return not ("given the constraints" in text_lower and "the assistant" in text_lower)
    
    df['heuristic_filtered'] = df['counterfactual'].apply(is_valid)
    
    passed = df['heuristic_filtered'].sum()
    total = len(df)
    pass_rate = passed / total if total > 0 else 0
    
    print(f"  Passed: {passed}/{total} ({pass_rate:.2%})")
    
    return df


def llm_semantic_filtering(df: pd.DataFrame, config: dict, llm_provider) -> pd.DataFrame:
    """
    Filter 2: LLM-based semantic validation.
    
    Verifies that counterfactual:
    1. Uses one of the candidate phrases 
    2. Maintains semantic context from original
    
    Args:
        df: DataFrame with counterfactuals
        config: Configuration dictionary
        llm_provider: LLM provider instance
        
    Returns:
        DataFrame with 'matched_pattern' column added
    """
    print("\n--- Filter 2: LLM Semantic Filtering ---")

    llm_config = config['llm']['models']['semantic_filtering']

    # Initialize matched_pattern column
    df['matched_pattern'] = None

    # Only process rows that passed heuristic filter
    valid_indices = df[df['heuristic_filtered'] == True].index

    print(f"  Processing {len(valid_indices)} counterfactuals...")

    dirs = config['directories']
    dataset_config = config['dataset']
    processing = config['processing']
    seed = processing['seed']
    dataset_file = dataset_config['train_file']
    dataset_name = dataset_file.replace('.csv', '')

    # Derive a model-friendly name for checkpoint file
    try:
        model_name = llm_provider.model.replace('/', '_').replace('-', '_').replace('.', '_')
    except Exception:
        # Fallback to provider name in config
        model_name = config['llm']['provider']

    checkpoint_file = f"{dirs['interim_output']}/[{seed}][{model_name}]semantic_checkpoint_{dataset_name}.csv"

    # Resume from checkpoint if exists
    start_idx = 0
    if pd.io.common.file_exists(checkpoint_file):
        print(f"  Found checkpoint file. Loading progress from {checkpoint_file}...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        if 'matched_pattern' in checkpoint_df.columns:
            # Update df with existing matched_pattern values
            df['matched_pattern'] = checkpoint_df.get('matched_pattern')
        # Determine where to resume
        processed = df.loc[valid_indices, 'matched_pattern'].notna().sum()
        start_idx = int(processed)
        print(f"  Resuming from row {start_idx+1}/{len(valid_indices)}")

    try:
        for idx, row_idx in enumerate(valid_indices):
            # Skip already processed rows
            if idx < start_idx:
                continue

            # Periodic progress print
            if (idx + 1) % 50 == 0:
                print(f"    {idx+1}/{len(valid_indices)}...")

            # Save checkpoint every 50 rows
            if (idx + 1) % 50 == 0:
                df.to_csv(checkpoint_file, index=False)
                print(f"  Checkpoint saved to {checkpoint_file}")

            row = df.loc[row_idx]

            # Show progress for each item
            print(f"    [{idx+1}/{len(valid_indices)}] Validating {row['id']}: {row['ori_label']} → {row['target_label']}...", end=' ', flush=True)

            ori_text = row['ori_text']
            highlight = row['highlight']
            counterfactual = row['counterfactual']

            # Parse candidate phrases
            try:
                candidate_phrases = ast.literal_eval(row['candidate_phrases'])
            except Exception:
                candidate_phrases = row['candidate_phrases']

            # Construct validation prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at validating text transformations."
                },
                {
                    "role": "user",
                    "content": f"""Validate the following transformation:

                Original sentence: "{ori_text}"
                Original phrase: "{highlight}"
                Candidate replacement phrases: {candidate_phrases}
                Modified sentence: "{counterfactual}"

                Check TWO criteria:
                1. Does the modified sentence use ONE of the candidate phrases (not rephrased)?
                2. Does the modified sentence maintain similar topic as the original?

                Answer with only 'YES' if BOTH criteria are met, otherwise 'NO'."""
                }
            ]

            # Call LLM
            try:
                response = llm_provider.chat_completion(
                    messages=messages,
                    temperature=llm_config['temperature'],
                    max_tokens=llm_config['max_tokens']
                )

                # Parse response
                response_clean = response.strip().upper()
                is_valid = 'YES' in response_clean

                df.at[row_idx, 'matched_pattern'] = is_valid

                print('✓' if is_valid else '✗')

            except Exception as e:
                print(f"Error: {e}")
                df.at[row_idx, 'matched_pattern'] = False

    except KeyboardInterrupt:
        print("\n\nScript interrupted by user!")
        print(f"Processed {idx+1}/{len(valid_indices)} examples")
        print(f"Progress saved to: {checkpoint_file}")
        df.to_csv(checkpoint_file, index=False)
        print("Run script again to resume from checkpoint")
        import sys
        sys.exit(0)

    # Calculate statistics
    passed = df['matched_pattern'].sum()
    total = len(valid_indices)
    pass_rate = passed / total if total > 0 else 0

    print(f"\n  Passed: {passed}/{total} ({pass_rate:.2%})")
    print(f"  The pattern keeping rate is: {pass_rate:.2f}")

    # Clean up checkpoint file on successful completion
    import os
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\n  Checkpoint file removed (processing complete)")

    return df


def gpt_discriminator_filtering(df: pd.DataFrame, config: dict, llm_provider) -> pd.DataFrame:
    """
    Filter 3: GPT discriminator validation.
    
    Validates that counterfactual:
    1. NOT about original label anymore
    2. IS about target label
    
    Args:
        df: DataFrame with counterfactuals
        config: Configuration dictionary
        llm_provider: LLM provider instance
        
    Returns:
        DataFrame with 'is_ori' and 'is_target' columns added
    """
    print("\n--- Filter 3: Discriminator Filtering ---")
    
    llm_config = config['llm']['models']['discriminator_filtering']
    dirs = config['directories']
    dataset_config = config['dataset']
    processing = config['processing']
    seed = processing['seed']
    # Derive dataset name from train_file
    dataset_file = dataset_config['train_file']
    dataset_name = dataset_file.replace('.csv', '')
    model_name = llm_provider.model.replace('/', '_').replace('-', '_').replace('.', '_')
    
    # Checkpoint file
    checkpoint_file = f"{dirs['output_data']}/[{seed}][{model_name}]discriminator_checkpoint_{dataset_name}.csv"
    
    # Initialize columns
    df['is_ori'] = None
    df['is_target'] = None
    
    # Check for checkpoint
    start_idx = 0
    if pd.io.common.file_exists(checkpoint_file):
        print(f"  Found checkpoint file. Loading progress...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        # Update df with checkpoint data
        for col in ['is_ori', 'is_target']:
            if col in checkpoint_df.columns:
                df[col] = checkpoint_df[col]
        # Find where to resume
        valid_indices = df[
            (df['heuristic_filtered'] == True) & 
            (df['matched_pattern'] == True)
        ].index
        # Count how many already have is_ori and is_target values
        processed = df.loc[valid_indices, 'is_ori'].notna().sum()
        start_idx = processed
        print(f"  Resuming from row {start_idx+1}/{len(valid_indices)}")
    
    # Only process rows that passed previous filters
    valid_indices = df[
        (df['heuristic_filtered'] == True) & 
        (df['matched_pattern'] == True)
    ].index
    
    print(f"  Processing {len(valid_indices)} counterfactuals...")
    
    try:
        for idx, row_idx in enumerate(valid_indices):
            # Skip already processed rows
            if idx < start_idx:
                continue
                
            if (idx + 1) % 50 == 0:
                print(f"    {idx+1}/{len(valid_indices)}...")
                # Save checkpoint every 50 rows
                df.to_csv(checkpoint_file, index=False)
                print(f"  Checkpoint saved to {checkpoint_file}")
            
            row = df.loc[row_idx]
            
            # Show progress for each item
            print(f"    [{idx+1}/{len(valid_indices)}] Classifying {row['id']}: {row['ori_label']} → {row['target_label']}...", end=' ', flush=True)
            
            counterfactual = row['counterfactual']
            ori_label = row['ori_label']
            target_label = row['target_label']
            
            # Construct classification prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at text classification."
                },
                {
                    "role": "user",
                    "content": f"""Classify this sentence for two categories:

                Sentence: "{counterfactual}"

                Question 1: Does this sentence belong to category '{ori_label}'?
                Question 2: Does this sentence belong to category '{target_label}'?

                Answer in format: ANSWER1, ANSWER2
                Where each answer is either YES or NO.

                Example: NO, YES"""
                }
            ]
            
            # Call LLM
            try:
                response = llm_provider.chat_completion(
                    messages=messages,
                    temperature=llm_config['temperature'],
                    max_tokens=llm_config['max_tokens']
                )
                
                # Parse response (format: "NO, YES" or "YES, NO")
                parts = response.strip().upper().replace(',', ' ').split()
                
                if len(parts) >= 2:
                    is_original = 'YES' in parts[0]
                    is_target = 'YES' in parts[-1]
                else:
                    is_original = True  # Conservative: mark as failed
                    is_target = False
                
                df.at[row_idx, 'is_ori'] = is_original
                df.at[row_idx, 'is_target'] = is_target
                
                # Show result
                result = "✓" if (not is_original and is_target) else ("~" if (not is_original or is_target) else "✗")
                print(result)
                
            except Exception as e:
                print(f"Error: {e}")
                df.at[row_idx, 'is_ori'] = True
                df.at[row_idx, 'is_target'] = False
    
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user!")
        print(f"Processed {idx+1}/{len(valid_indices)} examples")
        print(f"Progress saved to: {checkpoint_file}")
        df.to_csv(checkpoint_file, index=False)
        print("Run script again to resume from checkpoint")
        import sys
        sys.exit(0)
    
    # Calculate statistics
    not_ori = (df['is_ori'] == False).sum()
    is_target = (df['is_target'] == True).sum()
    both_pass = ((df['is_ori'] == False) & (df['is_target'] == True)).sum()
    total = len(valid_indices)
    
    not_ori_rate = not_ori / total if total > 0 else 0
    is_target_rate = is_target / total if total > 0 else 0
    
    print(f"\n  Not original label: {not_ori}/{total} ({not_ori_rate:.2%})")
    print(f"  Is target label: {is_target}/{total} ({is_target_rate:.2%})")
    print(f"  Passed both: {both_pass}/{total}")
    print(f"\n  The percentage of is not original label: {not_ori_rate:.2f}")
    print(f"  The percentage of is the target counterfactual label: {is_target_rate:.2f}")
    
    # Clean up checkpoint file on successful completion
    import os
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\n  Checkpoint file removed (processing complete)")
    
    return df


def create_fine_tune_dataset(df: pd.DataFrame, config: dict):
    """
    Create fine-tuning dataset from filtered counterfactuals.
    
    Only includes counterfactuals that passed all three filters.
    
    Args:
        df: DataFrame with filter results
        config: Configuration dictionary
    """
    print("\n--- Creating Fine-Tune Dataset ---")
    
    # Filter for high-quality counterfactuals
    df_finetune = df[
        (df['heuristic_filtered'] == True) &
        (df['matched_pattern'] == True) &
        (df['is_ori'] == False) &
        (df['is_target'] == True)
    ].copy()
    
    # Select and rename columns for fine-tuning
    df_finetune = df_finetune[[
        'id', 'ori_text', 'ori_label', 'pattern', 
        'counterfactual', 'target_label'
    ]].rename(columns={
        'counterfactual': 'counter_text',
        'target_label': 'counter_label'
    })
    
    print(f"  High-quality counterfactuals: {len(df_finetune)}")
    
    # Save fine-tuning dataset
    dirs = config['directories']
    seed = config['processing']['seed']
    dataset_file = config['dataset']['train_file']
    
    # Get model name for file naming
    model_name = config['llm']['provider']
    if config['llm']['provider'] == 'ollama':
        model_name = config['llm']['ollama']['model'].replace(':', '_')
    elif config['llm']['provider'] == 'gemini':
        model_name = config['llm']['gemini']['model'].replace('-', '_')
    elif config['llm']['provider'] == 'openai':
        model_name = config['llm']['openai']['model'].replace('-', '_')
    
    output_file = f"{dirs['output_data']}/[{seed}][{model_name}]fine_tuneset_{dataset_file}"
    df_finetune.to_csv(output_file, index=False)
    
    print(f"  Saved to: {output_file}")
    print(f"\n  The number of finetune dataset is: {len(df_finetune)}")


def filter_counterfactuals(config: dict, llm_provider):
    """
    Run three-stage filtering pipeline.
    
    Args:
        config: Configuration dictionary
        llm_provider: LLM provider instance
    """
    print("\n=== Starting Counterfactual Filtering ===")
    
    # Load configuration
    dirs = config['directories']
    seed = config['processing']['seed']
    dataset_file = config['dataset']['train_file']
    
    # Get model name for file naming
    model_name = config['llm']['provider']
    if config['llm']['provider'] == 'ollama':
        model_name = config['llm']['ollama']['model'].replace(':', '_')
    elif config['llm']['provider'] == 'gemini':
        model_name = config['llm']['gemini']['model'].replace('-', '_')
    elif config['llm']['provider'] == 'openai':
        model_name = config['llm']['openai']['model'].replace('-', '_')
    
    # Load counterfactuals from Script 02
    input_file = f"{dirs['output_data']}/[{seed}][{model_name}]counterfactuals_{dataset_file}"
    
    # Check if semantic filtering already complete
    semantic_interim_file = f"{dirs['interim_output']}/[{seed}][{model_name}]semantic_filtered_{dataset_file}"
    
    if pd.io.common.file_exists(semantic_interim_file):
        print(f"\nINFO: Found existing semantic filtered file: {semantic_interim_file}")
        print("INFO: Skipping Filter 1 & 2, loading from checkpoint...")
        df = pd.read_csv(semantic_interim_file)
        print(f"INFO: Loaded {len(df)} counterfactuals")
    else:
        try:
            df = pd.read_csv(input_file)
            print(f"\nINFO: Loaded {len(df)} counterfactuals from {input_file}")
        except FileNotFoundError:
            print(f"ERROR: Counterfactual file not found: {input_file}")
            print("Please run Script 02 (02_counterfactual_over_generation.py) first")
            sys.exit(1)
        
        # Run three-stage filtering
        df = heuristic_filtering(df)
        
        # Save interim output
        interim_file = f"{dirs['interim_output']}/[{seed}][{model_name}]heuristic_filtered_{dataset_file}"
        df.to_csv(interim_file, index=False)
        print(f"  Interim saved to: {interim_file}")
        
        df = llm_semantic_filtering(df, config, llm_provider)
        
        # Save interim output
        df.to_csv(semantic_interim_file, index=False)
        print(f"  Interim saved to: {semantic_interim_file}")
    
    # Always run discriminator filtering (has its own checkpoint)
    
    df = gpt_discriminator_filtering(df, config, llm_provider)
    
    # Save complete filtered dataset
    output_file = f"{dirs['output_data']}/[{seed}][{model_name}]filtered_{dataset_file}"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Complete filtered dataset saved to: {output_file}")
    
    # Create fine-tuning dataset
    create_fine_tune_dataset(df, config)


def main():
    """Main execution with error handling"""
    print("\n" + "="*60)
    print("Script 03: Counterfactual Filtering")
    print("="*60)
    
    try:
        # Load configuration
        config = load_config()
        ensure_directories(config)
        
        # Initialize LLM provider
        print(f"\nINFO: Using LLM provider: {config['llm']['provider']}")
        llm_provider = get_llm_provider(config)
        
        # Run filtering pipeline
        filter_counterfactuals(config, llm_provider)
        
        print("\n" + "="*60)
        print("Script 03 Complete!")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nERROR: Required input file not found: {e}")
        print("Make sure Script 02 completed successfully before running Script 03")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Script 03 failed: {e}")
        print("Check your configuration and input files")
        sys.exit(1)


if __name__ == "__main__":
    main()
