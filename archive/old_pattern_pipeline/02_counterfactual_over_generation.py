#!/usr/bin/env python3
"""
Script 02: Counterfactual Ov    # Output structure
    col_names = [
        "id",
        "ori_text",
        "ori_label",
        "pattern",
        "highlight",
        "candidate_phrases",
        "target_label",
        "counterfactual"
    ]
    
    # Define output file path
    output_file = f"{dirs['output_data']}/[{seed}]counterfactuals_{dataset_config['train_file']}"
    
    # Initialize data collector and error tracking
    data_collector = []
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    # Resume from checkpoint if available
    checkpoint_file = output_file.replace('.csv', '_checkpoint.csv')
    start_index = 0script generates complete counterfactual sentences using LLM.
It reads candidate phrases from Script 01 and creates full sentence transformations.

Input: output_data/[{SEED}][{MODEL}]{dataset}_candidate_phrases_annotated_data.csv
Output: output_data/[{seed}][{model}]counterfactuals_{dataset}.csv
"""

import sys
import pandas as pd
import ast
import time
import os
from utils import (
    load_config,
    ensure_directories,
    get_llm_provider
)


def generate_counterfactuals(config: dict, llm_provider):
    """
    Generate complete counterfactual sentences.
    
    Args:
        config: Configuration dictionary
        llm_provider: LLM provider instance
    """
    print("\n=== Starting Counterfactual Over-Generation ===\n")
    
    # Load configuration
    dirs = config['directories']
    dataset_config = config['dataset']
    processing = config['processing']
    llm_config = config['llm']['models']['counterfactual_generation']
    
    seed = processing['seed']
    dataset_name = dataset_config['train_file'].replace('.csv', '')
    
    # Get model name for file naming
    model_name = config['llm']['provider']
    if config['llm']['provider'] == 'ollama':
        model_name = config['llm']['ollama']['model'].replace(':', '_')
    elif config['llm']['provider'] == 'gemini':
        model_name = config['llm']['gemini']['model'].replace('-', '_')
    elif config['llm']['provider'] == 'openai':
        model_name = config['llm']['openai']['model'].replace('-', '_')
    
    # Construct input filename
    candidate_file = f"[{seed}][{model_name}]{dataset_name}_candidate_phrases_annotated_data.csv"
    input_path = f"{dirs['output_data']}/{candidate_file}"
    
    # Load candidate phrases from Script 01
    try:
        df = pd.read_csv(input_path)
        print(f"INFO: Loaded {len(df)} candidate phrase sets")
    except FileNotFoundError:
        print(f"ERROR: Candidate file not found: {input_path}")
        print("Please run Script 01 (01_data_formatting.py) first")
        sys.exit(1)
    
    # Define output schema
    col_names = [
        "id",
        "ori_text",
        "ori_label",
        "pattern",
        "highlight",
        "candidate_phrases",
        "target_label",
        "counterfactual"
    ]
    
    # Define output file path
    output_file = f"{dirs['output_data']}/[{seed}][{model_name}]counterfactuals_{dataset_config['train_file']}"
    
    # Initialize data collector and error tracking
    data_collector = []
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    # Resume from checkpoint if available
    checkpoint_file = output_file.replace('.csv', '_checkpoint.csv')
    start_index = 0
    
    if os.path.exists(checkpoint_file):
        print(f"INFO: Found checkpoint file. Loading progress...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        data_collector = checkpoint_df.values.tolist()
        start_index = len(data_collector)
        print(f"INFO: Resuming from example {start_index + 1}/{len(df)}")
    
    print(f"INFO: Generating counterfactuals for {len(df)} examples...")
    print(f"INFO: Press Ctrl+C to stop safely and save progress")
    print()

    # Process counterfactuals with rate limiting and error handling
    try:
        for index, row in df.iterrows():
            # Skip already processed items
            if index < start_index:
                continue
            
            if (index + 1) % 50 == 0:
                print(f"\nProgress: {index+1}/{len(df)} examples processed")
                # Save checkpoint every 50 items
                if data_collector:
                    checkpoint_df = pd.DataFrame(data_collector, columns=col_names)
                    checkpoint_df.to_csv(checkpoint_file, index=False)
                    print(f"Checkpoint saved to {checkpoint_file}")
            
            text = row["ori_text"]
            label = row["ori_label"]
            target_label = row["target_label"]
            highlight = row["highlight"]
            pattern = row["pattern"]
            
            # Show progress for each item
            print(f"  [{index+1}/{len(df)}] Processing {row['id']}: {label} → {target_label}...", end=' ', flush=True)
            
            # Parse candidate phrases
            try:
                generated_phrases = ast.literal_eval(row["candidate_phrases"])
            except:
                generated_phrases = row["candidate_phrases"]
            
            # Construct prompt messages
            messages = [
            {
                "role": "system",
                "content": "The assistant will create a counterfactual example close to the original sentence that contains one of the given phrases."
            },
            {
                "role": "user",
                "content": f"""Task: Transform the sentence to express a different category.

                Instructions:
                1. Use ONE phrase from the 'generated phrases' list (exactly as written, no rewording)
                2. Change the sentence from '{label}' to '{target_label}' category
                3. The modified sentence should NOT also express '{label}'
                4. Do NOT use the word '{target_label}' in the sentence (avoid label leakage)
                5. Keep the sentence grammatically correct and natural

                Data:
                - Original text: {text}
                - Original label: {label}
                - Target label: {target_label}
                - Generated phrases: {generated_phrases}

                Modified text:"""
            }
            ]
            
            # Call LLM with error handling
            try:
                response = llm_provider.chat_completion(
                    messages=messages,
                    temperature=llm_config['temperature'],
                    max_tokens=llm_config['max_tokens']
                )
                
                # Clean response (remove quotes if present)
                counterfactual = response.strip().strip('"\'')
                
                print(f"✓")  # Success indicator
                consecutive_errors = 0  # Reset error counter on success
                
                # Store result
                data_collector.append([
                    row["id"],
                    row["ori_text"],
                    row["ori_label"],
                    row["pattern"],
                    row["highlight"],
                    row["candidate_phrases"],
                    row["target_label"],
                    counterfactual
                ])
                
            except Exception as e:
                error_str = str(e)
                print(f"✗ (Error: {error_str[:50]})")
                consecutive_errors += 1
                
                # Check for specific error types
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    print(f"\nERROR: API quota exhausted!")
                    print(f"Processed {index}/{len(df)} examples before quota limit")
                    print(f"Saving progress to checkpoint file...")
                    
                    # Save current progress
                    if data_collector:
                        checkpoint_df = pd.DataFrame(data_collector, columns=col_names)
                        checkpoint_df.to_csv(checkpoint_file, index=False)
                        print(f"Progress saved to: {checkpoint_file}")
                    
                    print(f"\nTo resume later:")
                    print(f"1. Wait for quota reset (usually midnight UTC)")
                    print(f"2. Re-run this script - it will resume from checkpoint")
                    sys.exit(1)
                
                # Check for too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\nERROR: {max_consecutive_errors} consecutive failures!")
                    print(f"This might indicate a persistent issue. Stopping to prevent wasted quota.")
                    
                    # Save current progress
                    if data_collector:
                        checkpoint_df = pd.DataFrame(data_collector, columns=col_names)
                        checkpoint_df.to_csv(checkpoint_file, index=False)
                        print(f"Progress saved to: {checkpoint_file}")
                    
                    sys.exit(1)
                
                # Store with empty counterfactual on error
                data_collector.append([
                    row["id"],
                    row["ori_text"],
                    row["ori_label"],
                    row["pattern"],
                    row["highlight"],
                    row["candidate_phrases"],
                    row["target_label"],
                    ""  # Empty counterfactual
                ])
                
                print(f" (ERROR - collector now has {len(data_collector)} items)")
        
    except KeyboardInterrupt:
        print(f"\n\nScript interrupted by user!")
        print(f"Processed {len(data_collector)}/{len(df)} examples")
        
        # Save current progress
        if data_collector:
            checkpoint_df = pd.DataFrame(data_collector, columns=col_names)
            checkpoint_df.to_csv(checkpoint_file, index=False)
            print(f"Progress saved to: {checkpoint_file}")
            print(f"Run script again to resume from checkpoint")
        
        sys.exit(1)
    
    # Processing completed successfully
    print(f"\n✓ Counterfactual generation completed successfully!")
    print(f"Generated {len(data_collector)} counterfactuals")
    print(f"DEBUG: Final data_collector length: {len(data_collector)}")
    
    # Save final output
    df2 = pd.DataFrame(data_collector, columns=col_names)
    print(f"DEBUG: DataFrame shape: {df2.shape}")
    df2.to_csv(output_file, index=False)
    
    # Clean up checkpoint file on successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Removed checkpoint file (no longer needed)")
    
    print(f"\n✓ SUCCESS: Generated {len(df2)} counterfactuals")
    print(f"  Output saved to: {output_file}\n")


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("Script 02: Counterfactual Over-Generation")
    print("="*60)
    
    # Load configuration
    config = load_config()
    ensure_directories(config)
    
    # Initialize LLM provider
    print(f"\nINFO: Using LLM provider: {config['llm']['provider']}")
    llm_provider = get_llm_provider(config)
    
    # Generate counterfactuals
    generate_counterfactuals(config, llm_provider)
    
    print("="*60)
    print("Script 02 Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
