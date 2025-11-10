"""
Simplified Counterfactual Generator for Active Learning

Generates counterfactuals using direct LLM prompting without pattern matching
or complex filtering. Relies on strong prompt engineering for quality.
"""

from typing import List, Dict
import time


def generate_counterfactuals_batch(
    labeled_examples: List[Dict],
    config: dict,
    llm_provider,
    all_labels: List[str],
    return_details: bool = False
) -> tuple:
    """
    Generate counterfactuals for a batch of newly labeled examples.
    
    For each example, generates counterfactuals for ALL remaining target labels
    (excluding the original label).
    
    Args:
        labeled_examples: List of dicts with 'id', 'text', 'label'
        config: Configuration dictionary
        llm_provider: LLM provider instance
        all_labels: List of all possible labels in the dataset
        return_details: If True, returns (counterfactuals, details_list)
    
    Returns:
        If return_details=False: List of counterfactual examples (dicts with 'id', 'text', 'label')
        If return_details=True: Tuple of (counterfactuals, details_list) where details contains
                               generation metadata and API responses
    """
    cf_config = config['active_learning']['counterfactuals']
    
    if not cf_config['enabled']:
        print("  Counterfactual generation disabled in config")
        return []
    
    temperature = cf_config['generation_temperature']
    max_tokens = cf_config['max_tokens']
    
    # Generate for ALL remaining labels (not limited by per_example anymore)
    print(f"  Generating counterfactuals for ALL remaining labels per example")
    print(f"  Processing {len(labeled_examples)} examples")
    
    counterfactuals = []
    generation_details = []
    total_generated = 0
    
    for i, example in enumerate(labeled_examples):
        original_text = example['text']
        original_label = example['label']
        example_id = example['id']
        
        # Get target labels (ALL except original)
        target_labels = [label for label in all_labels if label != original_label]
        
        print(f"    [{i+1}/{len(labeled_examples)}] Example {example_id}: '{original_text[:40]}...' ({original_label})")
        print(f"                        Generating {len(target_labels)} CFs → {target_labels}")
        
        # Generate one counterfactual per target label (NO LIMIT)
        for target_label in target_labels:
            try:
                cf_text, cf_detail = generate_single_counterfactual(
                    original_text,
                    original_label,
                    target_label,
                    llm_provider,
                    temperature,
                    max_tokens,
                    return_details=return_details
                )
                
                if cf_text and cf_text.strip():
                    cf_id = f"{example_id}_cf_{target_label}"
                    counterfactuals.append({
                        'id': cf_id,
                        'text': cf_text,
                        'label': target_label,
                        'original_id': example_id,
                        'original_label': original_label
                    })
                    total_generated += 1
                    print(f"                          ✓ {target_label}: '{cf_text[:50]}...'")
                    
                    if return_details and cf_detail:
                        generation_details.append({
                            'cf_id': cf_id,
                            'original_example': {
                                'id': example_id,
                                'text': original_text,
                                'label': original_label
                            },
                            'target_label': target_label,
                            'generated_text': cf_text,
                            'generation_metadata': cf_detail
                        })
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"      Warning: Failed to generate CF for {original_label} -> {target_label}: {e}")
                continue
    
    print(f"  Generated {total_generated} counterfactuals from {len(labeled_examples)} examples")
    
    if return_details:
        return counterfactuals, generation_details
    return counterfactuals


def generate_single_counterfactual(
    original_text: str,
    original_label: str,
    target_label: str,
    llm_provider,
    temperature: float = 0.7,
    max_tokens: int = 256,
    return_details: bool = False
) -> tuple:
    """
    Generate a single counterfactual by rewriting text to express target label.
    
    Uses direct prompting without pattern identification. The prompt is designed
    to be dataset-agnostic and emphasize minimal, natural changes.
    
    Args:
        original_text: Original text to transform
        original_label: Current label of the text
        target_label: Desired label for counterfactual
        llm_provider: LLM provider instance
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        return_details: If True, returns (cf_text, details_dict)
    
    Returns:
        If return_details=False: Counterfactual text
        If return_details=True: Tuple of (cf_text, details_dict) with generation metadata
    """
    # Construct strong prompt emphasizing quality and minimal changes
    messages = [
        {
            "role": "system",
            "content": "You are an expert at rewriting text to express different categories while maintaining the original structure and theme."
        },
        {
            "role": "user",
            "content": f"""Task: Rewrite the following text to express '{target_label}' instead of '{original_label}'.

Original text: "{original_text}"
Current label: {original_label}
Target label: {target_label}

Instructions:
1. Rewrite the text to clearly express '{target_label}' 
2. Keep the sentence structure, length, and theme as similar as possible
3. Make only the MINIMAL changes necessary to shift the meaning
4. Ensure the output is natural and grammatically correct
5. Do NOT explicitly use the word '{target_label}' in your rewrite (avoid label leakage)
6. The rewritten text should NOT also express '{original_label}'

Output only the rewritten text, nothing else."""
        }
    ]
    
    # Generate counterfactual
    import time as time_module
    start_time = time_module.time()
    
    response = llm_provider.chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    end_time = time_module.time()
    generation_time = end_time - start_time
    
    # Clean response
    cf_text = response.strip().strip('"\'')
    
    # Basic quality check: ensure it's different from original
    if cf_text.lower() == original_text.lower():
        print(f"      Warning: Generated CF identical to original")
    
    # Prepare detailed information if requested
    if return_details:
        details = {
            'original_text': original_text,
            'original_label': original_label,
            'target_label': target_label,
            'generated_text': cf_text,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'generation_time_seconds': generation_time,
            'prompt_messages': messages,
            'is_identical_to_original': cf_text.lower() == original_text.lower()
        }
        return cf_text, details
    
    return cf_text, None


def generate_counterfactuals_for_evaluation(
    examples: List[Dict],
    config: dict,
    llm_provider,
    all_labels: List[str],
    num_per_target: int = 1
) -> List[Dict]:
    """
    Generate counterfactuals for evaluation purposes.
    
    Similar to generate_counterfactuals_batch but with configurable
    number of CFs per target label.
    
    Args:
        examples: List of examples to generate CFs for
        config: Configuration dictionary
        llm_provider: LLM provider instance
        all_labels: List of all possible labels
        num_per_target: Number of CFs to generate per target label
    
    Returns:
        List of counterfactual examples
    """
    cf_config = config['active_learning']['counterfactuals']
    temperature = cf_config['generation_temperature']
    max_tokens = cf_config['max_tokens']
    
    counterfactuals = []
    
    for example in examples:
        original_text = example['text']
        original_label = example['label']
        
        # Get all target labels except original
        target_labels = [label for label in all_labels if label != original_label]
        
        for target_label in target_labels:
            for _ in range(num_per_target):
                try:
                    cf_text, _ = generate_single_counterfactual(
                        original_text,
                        original_label,
                        target_label,
                        llm_provider,
                        temperature,
                        max_tokens,
                        return_details=False
                    )
                    
                    if cf_text and cf_text.strip():
                        counterfactuals.append({
                            'id': f"{example['id']}_cf_{target_label}",
                            'text': cf_text,
                            'label': target_label,
                            'original_id': example['id'],
                            'original_label': original_label
                        })
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Warning: CF generation failed: {e}")
                    continue
    
    return counterfactuals

