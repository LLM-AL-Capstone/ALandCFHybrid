#!/usr/bin/env python3
"""
Script 01: Data Formatting for LLM-Based Counterfactual Generation

This script:
1. Identifies key pattern phrases in labeled text using LLMs
2. Generates alternative phrases to transform emotions

Output:
- annotated_data_with_pattern_{dataset}.csv: Key phrases per example
- [{seed}]{dataset}_candidate_phrases_annotated_data.csv: Alternative phrases per target emotion
"""

import sys
import pandas as pd
import json
import ast
import time
import re
from typing import List, Dict
from utils import (
    load_config, 
    ensure_directories,
    load_dataset,
    get_unique_labels,
    shuffle_dataframe,
    get_llm_provider
)


def get_llm_patterns(config: dict, llm_provider):
    """
    Identify key emotional phrases using LLM.
    
    Args:
        config: Configuration dictionary
        llm_provider: LLM provider instance
    """
    print("\n=== Starting Pattern Identification ===\n")
    
    # Load configuration
    dirs = config['directories']
    dataset_config = config['dataset']
    processing = config['processing']
    llm_config = config['llm']['models']['pattern_identification']
    
    seed = processing['seed']
    max_examples_per_label = processing.get('max_examples_per_label', 50)  
    
    # Load dataset
    file_path = f"{dirs['input_data']}/{dataset_config['train_file']}"
    df = load_dataset(file_path, config)
    
    # Shuffle with seed
    df = shuffle_dataframe(df, seed)
    
    # Get column names
    col_id = dataset_config['columns']['id']
    col_text = dataset_config['columns']['text']
    col_label = dataset_config['columns']['label']
    
    # Only limit by max_examples_per_label for computational efficiency
    selected_df = df
    
    # Get unique labels (excluding specified labels)
    unique_labels = get_unique_labels(
        df, 
        col_label, 
        dataset_config.get('exclude_labels', [])
    )
    
    # Filter out null/NaN values and 'none' (case-insensitive)
    unique_labels = [
        label for label in unique_labels 
        if pd.notna(label) and str(label).lower() not in ['none', 'null', '', 'nan']
    ]
    
    print(f"INFO: Processing full dataset with {len(selected_df)} examples")
    print(f"INFO: Unique labels: {unique_labels}\n")
    
    # Output structure
    col_names = ["id", "ori_text", "ori_label", "pattern", "highlight"]
    data_collector = []
    
    # Process each label
    for label in unique_labels:
        print(f"Processing label: {label}")
        
        # Filter examples for this label
        label_df = selected_df[selected_df[col_label] == label]
        
        # Enhanced: Take up to max_examples_per_label (default 50) examples per label
        num_samples = min(max_examples_per_label, len(label_df))
        sample_examples = label_df.head(num_samples)
        
        if len(sample_examples) == 0:
            print(f"  ⚠ WARNING: No examples found for label '{label}', skipping")
            continue
        
        print(f"  INFO: Processing {len(sample_examples)} examples for '{label}'")
        
        # Process in batches for better API efficiency
        batch_size = 20  
        
        for batch_start in range(0, len(sample_examples), batch_size):
            batch_end = min(batch_start + batch_size, len(sample_examples))
            batch_examples = sample_examples.iloc[batch_start:batch_end]
            
            print(f"    Processing batch {batch_start//batch_size + 1}: examples {batch_start+1}-{batch_end}")
            
            # Create examples text for prompt
            examples_text = "\n".join([
                f"- {row[col_text]}" 
                for _, row in batch_examples.iterrows()
            ])
        
            # Construct prompt messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at identifying key phrases in text that strongly express specific categories or labels. You must follow the exact output format specified."
                },
                {
                    "role": "user",
                    "content": f"""Task: Analyze these sentences labeled as '{label}' and identify key phrases that clearly express this category.

                    Sentences:
                    {examples_text}

                    Instructions:
                    1. For EACH sentence above, identify 1-3 key phrases that strongly express the '{label}' category
                    2. Copy the EXACT original sentence without any modifications
                    3. List the key phrases separated by commas

                    Required Output Format (use exactly this format):
                    SENTENCE: [exact original sentence]
                    PHRASES: [phrase1, phrase2, phrase3]
                    ---

                    Example:
                    Input sentence: "the food was delicious and affordable prices"
                    Required output:
                    SENTENCE: the food was delicious and affordable prices
                    PHRASES: delicious, affordable prices
                    ---

                    Important: Use the exact format shown above. Each sentence must be followed by its phrases and then "---" separator."""
                }
            ]
            
            # Call LLM
            try:
                print(f" Processing batch {batch_start//batch_size + 1} for '{label}'")
                
                result = llm_provider.chat_completion(
                    messages=messages,
                    temperature=llm_config['temperature'],
                    max_tokens=llm_config['max_tokens']
                )
                
                # Parse model response with robust model-agnostic parsing
                result_clean = result.strip()
                
                # Remove markdown code blocks if present
                if result_clean.startswith('```'):
                    lines = result_clean.split('\n')
                    result_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else result_clean
                    result_clean = result_clean.replace('```', '').strip()
                
                # Model-agnostic parsing with multiple strategies
                analyses = []
                
                def parse_structured_format(text):
                    """Parse expected SENTENCE:/PHRASES: format"""
                    parsed_analyses = []
                    sections = text.split('---')
                    
                    for section in sections:
                        section = section.strip()
                        if not section:
                            continue
                        
                        lines = [line.strip() for line in section.split('\n') if line.strip()]
                        sentence = None
                        phrases = []
                        
                        for line in lines:
                            if line.startswith('SENTENCE:'):
                                sentence = line[9:].strip()
                            elif line.startswith('PHRASES:'):
                                phrase_text = line[8:].strip()
                                phrases = [p.strip() for p in phrase_text.split(',') if p.strip()]
                        
                        if sentence and phrases:
                            parsed_analyses.append({
                                'sentence': sentence,
                                'key_phrases': phrases
                            })
                    
                    return parsed_analyses
                
                def parse_natural_format(text):
                    """Parse natural language responses (common with Qwen)"""
                    parsed_analyses = []
                    
                    # Look for patterns like "1. Sentence: ... Phrases: ..." or numbered lists
                    patterns = [
                        r'(?:(?:\d+\.?\s*)?(?:sentence|text):\s*["\']*([^"\'\n]+)["\']*.*?(?:phrases?|words?|terms?):\s*([^\n]+))',
                        r'(?:(?:\d+\.?\s*)?["\']*([^"\'\n]+)["\']*.*?(?:key phrases?|phrases?|words?):\s*([^\n]+))',
                        r'(?:(?:\d+\.?\s*)?)["\']*([^"\'\n,]+)["\']*\s*[-–—]\s*([^\n]+)',
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                        for match in matches:
                            if len(match) == 2:
                                sentence = match[0].strip()
                                phrase_text = match[1].strip()
                                phrases = [p.strip() for p in re.split(r'[,;]', phrase_text) if p.strip()]
                                
                                if sentence and phrases:
                                    parsed_analyses.append({
                                        'sentence': sentence,
                                        'key_phrases': phrases
                                    })
                    
                    return parsed_analyses
                
                def parse_bullet_format(text):
                    """Parse bullet point or list format responses"""
                    parsed_analyses = []
                    lines = text.split('\n')
                    current_sentence = None
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Check if line contains one of our original sentences
                        for _, row in batch_examples.iterrows():
                            original_text = row[col_text]
                            if original_text.lower() in line.lower() or line.lower() in original_text.lower():
                                current_sentence = original_text
                                # Extract phrases from the same line or look for following phrases
                                phrase_part = line.split(':')[-1] if ':' in line else ''
                                phrases = [p.strip() for p in re.split(r'[,;]', phrase_part) if p.strip()]
                                
                                if phrases:
                                    parsed_analyses.append({
                                        'sentence': current_sentence,
                                        'key_phrases': phrases
                                    })
                                break
                    
                    return parsed_analyses
                
                # Try parsing strategies in order of preference
                analyses = parse_structured_format(result_clean)
                
                if not analyses:
                    print(f"   Structured format failed, trying natural language parsing...")
                    analyses = parse_natural_format(result_clean)
                
                if not analyses:
                    print(f"    Natural format failed, trying bullet point parsing...")
                    analyses = parse_bullet_format(result_clean)
                
                if not analyses:
                    # Final fallback: regex extraction
                    print(f"    All parsing strategies failed, trying regex fallback...")
                    sentence_matches = re.findall(r'SENTENCE:\s*(.+)', result_clean)
                    phrase_matches = re.findall(r'PHRASES:\s*(.+)', result_clean)
                    
                    for i in range(min(len(sentence_matches), len(phrase_matches))):
                        sentence = sentence_matches[i].strip()
                        phrases = [p.strip() for p in phrase_matches[i].split(',') if p.strip()]
                        if sentence and phrases:
                            analyses.append({
                                'sentence': sentence,
                                'key_phrases': phrases
                            })
                
                if not analyses:
                    print(f"    ERROR: All parsing strategies failed. Raw response:")
                    print(f"    '{result_clean[:1000]}'")  # Show more of the response
                    print(f"    ERROR: Failed to process batch for '{label}': No valid sentences/phrases found")
                    continue
                
                print(f"    ✓ Parsed {len(analyses)} sentence-phrase pairs")
                
                # Helper function for enhanced text normalization
                def normalize_text(text):
                    """Enhanced text normalization for better matching"""
                    # Remove common prefixes (bullets, dashes, numbers with periods)
                    text = re.sub(r'^[-•*\d\.]+\s*', '', text.strip())
                    
                    # Remove truncation markers and ellipsis
                    text = text.replace('...', '').replace('…', '').strip()
                    
                    # Lowercase and normalize whitespace
                    text = text.lower().strip()
                    
                    # Remove quotes
                    text = text.replace('"', '').replace("'", '')
                    
                    # Normalize multiple spaces to single space
                    text = re.sub(r'\s+', ' ', text)
                    
                    # Remove trailing periods and spaces
                    text = text.rstrip('. ')
                    
                    return text
                
                # Process each analysis
                matched_count = 0
                for analysis in analyses:
                    sentence = analysis.get('sentence', '')
                    key_phrases = analysis.get('key_phrases', [])
                    
                    # Find matching row in dataframe - try exact match first
                    matching_rows = batch_examples[batch_examples[col_text] == sentence]
                    
                    # If no exact match, try normalized fuzzy matching
                    if len(matching_rows) == 0:
                        norm_sentence = normalize_text(sentence)
                        
                        for batch_idx, (_, row) in enumerate(batch_examples.iterrows()):
                            norm_original = normalize_text(row[col_text])
                            
                            # Try exact normalized match first
                            if norm_sentence == norm_original:
                                matching_rows = batch_examples.iloc[[batch_idx]]
                                break
                            
                            # Try partial match for truncated text (at least 80% similarity)
                            # Check if normalized sentence is a prefix of original (for truncation)
                            if len(norm_sentence) >= 20:  # Only for reasonably long sentences
                                if norm_original.startswith(norm_sentence) or norm_sentence.startswith(norm_original):
                                    matching_rows = batch_examples.iloc[[batch_idx]]
                                    break
                    
                    if len(matching_rows) > 0:
                        row = matching_rows.iloc[0]
                        matched_count += 1
                        
                        # Format pattern and highlight
                        pattern = ", ".join(key_phrases)
                        highlight = str([[[phrase]] for phrase in key_phrases])
                        
                        data_collector.append([
                            row[col_id],
                            row[col_text],
                            row[col_label],
                            pattern,
                            highlight
                        ])
                    else:
                        # Debug output for unmatched sentences
                        print(f"      ⚠ No match found for: '{sentence[:50]}...'")
                        if len(batch_examples) > 0:
                            print(f"        Sample original: '{batch_examples.iloc[0][col_text][:50]}...'")
                
                print(f"    ✓ Processed {len(analyses)} analyses, matched {matched_count} to original sentences")
                
                print(f"    ✓ Processed {len(analyses)} examples in batch")
                
            except Exception as e:
                print(f"    ERROR: Failed to process batch for '{label}': {e}")
                continue
    
    # Create DataFrame and save
    df_patterns = pd.DataFrame(data_collector, columns=col_names)
    
    dataset_name = dataset_config['train_file'].replace('.csv', '')
    model_name = config['llm']['provider']
    if config['llm']['provider'] == 'ollama':
        model_name = config['llm']['ollama']['model'].replace(':', '_')
    elif config['llm']['provider'] == 'gemini':
        model_name = config['llm']['gemini']['model'].replace('-', '_')
    elif config['llm']['provider'] == 'openai':
        model_name = config['llm']['openai']['model'].replace('-', '_')
    
    output_file = f"{dirs['output_data']}/[{seed}][{model_name}]annotated_data_with_pattern_{dataset_name}.csv"
    df_patterns.to_csv(output_file, index=False)
    
    print(f"\n✓ SUCCESS: Saved {len(df_patterns)} annotated examples to:")
    print(f"  {output_file}\n")


def get_candidate_phrases(config: dict, llm_provider):
    """
    Generate alternative phrases for label transformation.
    
    Args:
        config: Configuration dictionary
        llm_provider: LLM provider instance
    """
    print("\n=== Starting Candidate Phrase Generation ===\n")
    
    # Load configuration
    dirs = config['directories']
    dataset_config = config['dataset']
    processing = config['processing']
    llm_config = config['llm']['models']['candidate_generation']
    
    seed = processing['seed']
    token_limit = processing['token_limit']
    
    # Load original dataset for unique labels
    file_path = f"{dirs['input_data']}/{dataset_config['train_file']}"
    df_original = load_dataset(file_path, config)
    
    col_label = dataset_config['columns']['label']
    unique_labels = get_unique_labels(
        df_original,
        col_label,
        dataset_config.get('exclude_labels', [])
    )
    
    # Filter out null/NaN values and 'none' (case-insensitive)
    unique_labels = [
        label for label in unique_labels 
        if pd.notna(label) and str(label).lower() not in ['none', 'null', '', 'nan']
    ]
    
    # Load annotated patterns from previous step
    dataset_name = dataset_config['train_file'].replace('.csv', '')
    model_name = config['llm']['provider']
    if config['llm']['provider'] == 'ollama':
        model_name = config['llm']['ollama']['model'].replace(':', '_')
    elif config['llm']['provider'] == 'gemini':
        model_name = config['llm']['gemini']['model'].replace('-', '_')
    elif config['llm']['provider'] == 'openai':
        model_name = config['llm']['openai']['model'].replace('-', '_')
    
    pattern_file = f"{dirs['output_data']}/[{seed}][{model_name}]annotated_data_with_pattern_{dataset_name}.csv"
    
    try:
        df = pd.read_csv(pattern_file)
        print(f"INFO: Loaded {len(df)} annotated patterns")
    except FileNotFoundError:
        print(f"ERROR: Pattern file not found: {pattern_file}")
        print("Please run get_llm_patterns() first")
        sys.exit(1)
    
    # Initialize token counter
    num_tokens = 0
    processed_examples = 0
    
    # Initialize output structure
    col_names_2 = [
        "id", "ori_text", "ori_label", "pattern", 
        "highlight", "target_label", "candidate_phrases"
    ]
    data_collector_2 = []
    
    print(f"INFO: Generating candidates for {len(unique_labels)} target labels")
    print(f"INFO: Processing {len(df)} annotated examples\n")
    
    # Checkpoint file for candidate phrase generation
    checkpoint_file = f"{dirs['output_data']}/[{seed}][{model_name}]{dataset_name}_candidate_phrases_checkpoint.csv"
    
    # Check for existing checkpoint
    start_index = 0
    if pd.io.common.file_exists(checkpoint_file):
        print(f"INFO: Found checkpoint file, loading progress...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        data_collector_2 = checkpoint_df.values.tolist()
        
        # Find last processed example ID
        if len(data_collector_2) > 0:
            last_id = data_collector_2[-1][0]
            # Find index of last processed row
            last_row_idx = df[df['id'] == last_id].index
            if len(last_row_idx) > 0:
                start_index = last_row_idx[0] + 1
        
        print(f"INFO: Resuming from example {start_index+1}/{len(df)}")
        print(f"INFO: Already have {len(data_collector_2)} candidate phrase sets\n")
    
    # Iterate through annotated examples
    for i, row in df.iterrows():
        # Skip already processed examples
        if i < start_index:
            continue
        # Check token budget
        if num_tokens > token_limit:
            print(f"WARNING: Token limit ({token_limit:,}) reached. Stopping.")
            print(f"Processed {processed_examples} examples out of {len(df)}")
            break
        
        # Progress logging for large datasets
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(df)} examples processed")
            print(f"  Tokens used so far: {num_tokens:,}")
        
        print(f"Processing example {i+1}/{len(df)}: {row['id']}")
        
        sentence = row['ori_text']
        label = row['ori_label']
        
        # Parse highlight to get individual phrases
        try:
            highlight = ast.literal_eval(row['highlight'])
            marked_phrases = [phrase[0][0] for phrase in highlight]
        except:
            print(f"  ⚠ Warning: Could not parse highlight for {row['id']}")
            continue
        
        # For each matched phrase
        for matched_phrase in marked_phrases:
            # For each target label (excluding original)
            for target_label in unique_labels:
                if target_label == label:
                    continue
                
                # Construct prompt
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert at generating alternative phrases for label transformation in text. You must provide exactly the requested format."
                    },
                    {
                        "role": "user",
                        "content": f"""Task: Generate alternative phrases to change the label in a sentence.

                        Original sentence: "{sentence}"
                        Current phrase: "{matched_phrase}" (expresses '{label}')
                        Target label: '{target_label}'

                        Generate 5-7 alternative phrases that could replace "{matched_phrase}" to express '{target_label}' instead of '{label}'.

                        Requirements:
                        1. Each alternative should fit naturally in the sentence context
                        2. Maintain grammatical structure and meaning flow
                        3. Only change the highlight tone to reflect '{target_label}'
                        4. Make alternatives diverse but appropriate
                        5. Return ONLY the alternative phrases, separated by commas

                        Expected output format: phrase1, phrase2, phrase3, phrase4, phrase5

                        Example:
                        If replacing "terrible" (negative) with positive alternatives:
                        excellent, wonderful, amazing, fantastic, great"""
                    }
                ]
                
                # Call LLM
                try:
                    print(f"  Generating alternatives for '{matched_phrase}' -> '{target_label}'")
                    
                    result = llm_provider.chat_completion(
                        messages=messages,
                        temperature=llm_config['temperature'],
                        max_tokens=llm_config['max_tokens']
                    )
                    
                    # Count tokens (approximate)
                    prompt_text = " ".join([m['content'] for m in messages])
                    num_tokens += llm_provider.count_tokens(prompt_text + result)
                    
                    # Add delay to avoid TPM rate limit (200K tokens/minute for GPT-5-nano)
                    # Each call uses ~1500-2000 tokens, so wait to stay under limit
                    time.sleep(1.2)  # ~50 calls/minute = ~100K tokens/minute (safe margin)
                    
                    # Parse candidate phrases with robust parsing
                    def parse_candidates(text):
                        """Parse candidate phrases from various model response formats"""
                        text = text.strip()
                        
                        # Remove common prefixes/suffixes
                        prefixes_to_remove = [
                            'here are', 'the alternatives are', 'alternative phrases:', 'alternatives:',
                            'possible replacements:', 'suggestions:', 'here are the alternatives:',
                            'the alternative phrases are:', 'replacements:'
                        ]
                        
                        text_lower = text.lower()
                        for prefix in prefixes_to_remove:
                            if text_lower.startswith(prefix):
                                text = text[len(prefix):].strip()
                                break
                        
                        # Remove numbered lists (1., 2., etc.)
                        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
                        
                        # Remove bullet points
                        text = re.sub(r'^[-•*]\s*', '', text, flags=re.MULTILINE)
                        
                        # Split by various separators
                        separators = [',', ';', '\n', '|']
                        candidates = []
                        
                        # Try comma separation first (most common)
                        if ',' in text:
                            candidates = [p.strip() for p in text.split(',')]
                        elif ';' in text:
                            candidates = [p.strip() for p in text.split(';')]
                        elif '\n' in text:
                            candidates = [p.strip() for p in text.split('\n')]
                        elif '|' in text:
                            candidates = [p.strip() for p in text.split('|')]
                        else:
                            # Single phrase or space-separated
                            candidates = [text.strip()]
                        
                        # Clean up candidates
                        cleaned_candidates = []
                        for candidate in candidates:
                            candidate = candidate.strip()
                            # Remove quotes
                            candidate = candidate.strip('"\'')
                            # Remove extra whitespace
                            candidate = re.sub(r'\s+', ' ', candidate)
                            # Filter out empty or very short candidates
                            if candidate and len(candidate) > 1:
                                cleaned_candidates.append(candidate)
                        
                        return cleaned_candidates
                    
                    candidates = parse_candidates(result)
                    
                    # Store result
                    data_collector_2.append([
                        row['id'],
                        row['ori_text'],
                        row['ori_label'],
                        row['pattern'],
                        matched_phrase,
                        target_label,
                        str(candidates)
                    ])
                    
                    processed_examples += 1
                    
                except Exception as e:
                    print(f"  ⚠ Error generating candidates: {e}")
                    continue
        
        # Save checkpoint every 10 examples
        if (i + 1) % 10 == 0:
            print(f"  Tokens used so far: {num_tokens:,}")
            print(f"  Saving checkpoint...")
            df_checkpoint = pd.DataFrame(data_collector_2, columns=col_names_2)
            df_checkpoint.to_csv(checkpoint_file, index=False)
            print(f"  Checkpoint saved: {i+1}/{len(df)} examples processed")
    
    # Create DataFrame and save
    df2 = pd.DataFrame(data_collector_2, columns=col_names_2)
    output_file = f"{dirs['output_data']}/[{seed}][{model_name}]{dataset_name}_candidate_phrases_annotated_data.csv"
    df2.to_csv(output_file, index=False)
    
    print(f"\n✓ SUCCESS: Saved {len(df2)} candidate phrase sets to:")
    print(f"  {output_file}")
    print(f"  Total tokens used: {num_tokens:,}")
    print(f"  Processed examples: {processed_examples}")
    print(f"  Expected improvement: {len(df2)} vs ~800 candidate sets in original approach\n")
    
    # Clean up checkpoint file on successful completion
    import os
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"  Checkpoint file removed (generation complete)")


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("Script 01: Data Formatting")
    print("="*60)
    
    # Load configuration
    config = load_config()
    ensure_directories(config)
    
    # Initialize LLM provider
    print(f"\nINFO: Using LLM provider: {config['llm']['provider']}")
    llm_provider = get_llm_provider(config)
    
    # Run both steps (comment out pattern identification if already completed)
    get_llm_patterns(config, llm_provider)  
    get_candidate_phrases(config, llm_provider) 
    
    print("="*60)
    print("Script 01 Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
