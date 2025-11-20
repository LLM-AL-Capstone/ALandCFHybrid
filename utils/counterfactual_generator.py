"""
Counterfactual Generator with Quality Filtering for Active Learning

Generates counterfactuals using direct LLM prompting with quality-based filtering.
Supports over-generation, multiple quality metrics, and budget constraints.
"""

from typing import List, Dict, Tuple
import time
import random
from utils.cf_quality_scorer import CFQualityScorer


def generate_counterfactuals_batch(
    labeled_examples: List[Dict],
    config: dict,
    llm_provider,
    all_labels: List[str],
    labeled_pool: List[Dict],
    classifier,
    cf_budget_remaining: int,
    return_details: bool = False
) -> tuple:
    """
    Generate and filter counterfactuals for a batch of newly labeled examples.
    
    Supports over-generation with quality filtering to keep only best CFs.
    
    Args:
        labeled_examples: List of dicts with 'id', 'text', 'label'
        config: Configuration dictionary
        llm_provider: LLM provider instance
        all_labels: List of all possible labels in the dataset
        labeled_pool: Current labeled pool (for diversity calculation)
        classifier: Trained classifier (for confidence scoring)
        cf_budget_remaining: Remaining CF budget (-1 for unlimited)
        return_details: If True, returns (counterfactuals, details_list)
    
    Returns:
        Tuple of (counterfactuals, num_added, details_list)
        - counterfactuals: List of CF dicts
        - num_added: Number of CFs added (may be less than generated due to budget)
        - details_list: Generation and filtering details (if return_details=True)
    """
    cf_config = config['active_learning']['counterfactuals']
    
    if not cf_config['enabled']:
        print("  Counterfactual generation disabled in config")
        if return_details:
            return [], 0, []
        return [], 0
    
    # Check if budget exhausted
    if cf_budget_remaining == 0:
        print("  CF budget exhausted - skipping generation")
        if return_details:
            return [], 0, []
        return [], 0
    
    # Get configuration
    max_per_example = cf_config.get('max_per_example', 5)
    generation_multiplier = cf_config.get('generation_multiplier', 2.0)
    quality_filtering_enabled = cf_config.get('quality_filtering', {}).get('enabled', False)
    distribution_strategy = cf_config.get('distribution_strategy', 'balanced')
    
    print(f"  CF Generation Settings:")
    print(f"    Max per example: {max_per_example}")
    print(f"    Generation multiplier: {generation_multiplier}x")
    print(f"    Quality filtering: {'enabled' if quality_filtering_enabled else 'disabled'}")
    print(f"    Budget remaining: {cf_budget_remaining if cf_budget_remaining > 0 else 'unlimited'}")
    
    # Initialize quality scorer if filtering enabled
    quality_scorer = None
    if quality_filtering_enabled:
        quality_scorer = CFQualityScorer(config)
    
    # Generate CFs for each example
    all_cf_candidates = []
    generation_details = []
    
    for i, example in enumerate(labeled_examples):
        print(f"    [{i+1}/{len(labeled_examples)}] Processing: '{example['text'][:40]}...' ({example['label']})")
        
        # Generate CF candidates for this example
        candidates = generate_cf_candidates_for_example(
            example,
            all_labels,
            config,
            llm_provider,
            max_per_example,
            generation_multiplier,
            distribution_strategy,
            return_details
        )
        
        # Add to candidates list with source example info
        for cf_dict, cf_detail in candidates:
            all_cf_candidates.append({
                'cf': cf_dict,
                'original_example': example,
                'generation_detail': cf_detail
            })
        
        print(f"                Generated {len(candidates)} CF candidates")
    
    print(f"\n  Total CF candidates generated: {len(all_cf_candidates)}")
    
    # Filter candidates by quality if enabled
    if quality_filtering_enabled and quality_scorer and len(all_cf_candidates) > 0:
        print(f"  Applying quality filtering...")
        filtered_cfs, filtering_details = filter_by_quality(
            all_cf_candidates,
            labeled_examples,
            labeled_pool,
            classifier,
            quality_scorer,
            max_per_example
        )
    else:
        # No filtering - keep all candidates
        filtered_cfs = [item['cf'] for item in all_cf_candidates]
        filtering_details = []
    
    print(f"  After quality filtering: {len(filtered_cfs)} CFs")
    
    # Apply budget constraint
    if cf_budget_remaining > 0 and len(filtered_cfs) > cf_budget_remaining:
        print(f"  Budget constraint: keeping best {cf_budget_remaining} of {len(filtered_cfs)} CFs")
        # Already sorted by quality, so just take first cf_budget_remaining
        filtered_cfs = filtered_cfs[:cf_budget_remaining]
    
    num_added = len(filtered_cfs)
    
    print(f"  ✓ Final: {num_added} counterfactuals added to pool\n")
    
    # Compile details if requested
    if return_details:
        combined_details = {
            'num_examples': len(labeled_examples),
            'num_generated': len(all_cf_candidates),
            'num_filtered': len(filtered_cfs),
            'num_added': num_added,
            'budget_remaining_before': cf_budget_remaining,
            'budget_remaining_after': max(0, cf_budget_remaining - num_added) if cf_budget_remaining > 0 else -1,
            'generation_details': [item['generation_detail'] for item in all_cf_candidates if item['generation_detail']],
            'filtering_details': filtering_details
        }
        return filtered_cfs, num_added, combined_details
    
    return filtered_cfs, num_added


def generate_cf_candidates_for_example(
    example: Dict,
    all_labels: List[str],
    config: dict,
    llm_provider,
    max_per_example: int,
    generation_multiplier: float,
    distribution_strategy: str,
    return_details: bool
) -> List[Tuple[Dict, Dict]]:
    """
    Generate CF candidates for a single example.
    
    Returns:
        List of (cf_dict, detail_dict) tuples
    """
    cf_config = config['active_learning']['counterfactuals']
    original_text = example['text']
    original_label = example['label']
    example_id = example['id']
    
    # Get target labels (exclude original)
    target_labels = [label for label in all_labels if label != original_label]
    
    if len(target_labels) == 0:
        return []
    
    # Determine how many CFs to generate
    num_to_generate = int(max_per_example * generation_multiplier)
    
    # Distribute generations across target labels
    label_distribution = distribute_generations(
        target_labels,
        num_to_generate,
        distribution_strategy
    )
    
    print(f"                Generating {num_to_generate} CFs across {len(target_labels)} labels")
    
    # Generate CFs
    candidates = []
    temperature = cf_config.get('generation_temperature', 0.7)
    max_tokens = cf_config.get('max_tokens', 256)
    prompt_variation = cf_config.get('prompt_variation', True)
    
    for target_label, count in label_distribution.items():
        for variation_idx in range(count):
            try:
                # Use prompt variation if generating multiple CFs for same label
                use_variation = prompt_variation and count > 1
                
                cf_text, cf_detail = generate_single_counterfactual(
                    original_text,
                    original_label,
                    target_label,
                    llm_provider,
                    temperature,
                    max_tokens,
                    variation_idx if use_variation else 0,
                    return_details
                )
                
                if cf_text and cf_text.strip():
                    cf_id = f"{example_id}_cf_{target_label}_{variation_idx}"
                    cf_dict = {
                        'id': cf_id,
                        'text': cf_text,
                        'label': target_label,
                        'original_id': example_id,
                        'original_label': original_label
                    }
                    candidates.append((cf_dict, cf_detail))
                
                # Rate limiting
                time.sleep(0.3)
                
            except Exception as e:
                print(f"      Warning: Failed to generate CF for {target_label}: {e}")
                continue
    
    return candidates


def distribute_generations(
    target_labels: List[str],
    num_to_generate: int,
    strategy: str
) -> Dict[str, int]:
    """
    Distribute CF generations across target labels.
    
    Args:
        target_labels: List of target labels
        num_to_generate: Total number of CFs to generate
        strategy: Distribution strategy
    
    Returns:
        Dict mapping label to number of CFs to generate
    """
    num_labels = len(target_labels)
    
    if num_labels == 0:
        return {}
    
    if strategy == 'balanced':
        # Distribute evenly, remainder randomly assigned
        base_count = num_to_generate // num_labels
        remainder = num_to_generate % num_labels
        
        distribution = {label: base_count for label in target_labels}
        
        # Assign remainder randomly
        for label in random.sample(target_labels, remainder):
            distribution[label] += 1
        
        return distribution
    
    elif strategy == 'random':
        # Random distribution
        distribution = {label: 0 for label in target_labels}
        for _ in range(num_to_generate):
            label = random.choice(target_labels)
            distribution[label] += 1
        return distribution
    
    elif strategy in ['priority', 'quality_first']:
        # For now, same as balanced (priority needs labeled pool info)
        # TODO: Implement priority based on label counts
        base_count = num_to_generate // num_labels
        remainder = num_to_generate % num_labels
        distribution = {label: base_count for label in target_labels}
        for label in random.sample(target_labels, remainder):
            distribution[label] += 1
        return distribution
    
    else:
        # Default to balanced
        base_count = num_to_generate // num_labels
        distribution = {label: base_count for label in target_labels}
        remainder = num_to_generate - (base_count * num_labels)
        for label in random.sample(target_labels, remainder):
            distribution[label] += 1
        return distribution


def filter_by_quality(
    cf_candidates: List[Dict],
    labeled_examples: List[Dict],
    labeled_pool: List[Dict],
    classifier,
    quality_scorer: CFQualityScorer,
    max_per_example: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter CF candidates by quality scores.
    
    Args:
        cf_candidates: List of candidate dicts with 'cf', 'original_example', etc.
        labeled_examples: Newly labeled examples
        labeled_pool: Current labeled pool
        classifier: Trained classifier
        quality_scorer: Quality scorer instance
        max_per_example: Max CFs to keep per original example
    
    Returns:
        Tuple of (filtered_cfs, filtering_details)
    """
    # Group candidates by original example
    candidates_by_example = {}
    for item in cf_candidates:
        orig_id = item['original_example']['id']
        if orig_id not in candidates_by_example:
            candidates_by_example[orig_id] = []
        candidates_by_example[orig_id].append(item)
    
    # Extract labeled pool texts for diversity calculation
    labeled_pool_texts = [ex['text'] for ex in labeled_pool]
    
    # Score and filter candidates for each example
    all_filtered_cfs = []
    filtering_details = []
    
    for orig_id, candidates in candidates_by_example.items():
        original_example = candidates[0]['original_example']
        
        # Score each candidate
        scored_candidates = []
        for item in candidates:
            cf = item['cf']
            
            try:
                combined_score, score_dict, passes_filters = quality_scorer.compute_combined_score(
                    cf['text'],
                    original_example['text'],
                    cf['label'],
                    labeled_pool_texts,
                    classifier
                )
                
                if passes_filters:
                    # Convert NumPy types to Python native types for JSON serialization
                    combined_score_float = float(combined_score)
                    diversity_float = float(score_dict['diversity'])
                    confidence_float = float(score_dict['confidence'])
                    validity_float = float(score_dict['validity'])
                    
                    scored_candidates.append({
                        'cf': cf,
                        'combined_score': combined_score_float,
                        'score_details': score_dict
                    })
                    
                    filtering_details.append({
                        'cf_id': cf['id'],
                        'original_id': orig_id,
                        'target_label': cf['label'],
                        'combined_score': combined_score_float,
                        'diversity': diversity_float,
                        'confidence': confidence_float,
                        'validity': validity_float,
                        'passed_filters': True,
                        'kept': False  # Will update later
                    })
                else:
                    filtering_details.append({
                        'cf_id': cf['id'],
                        'original_id': orig_id,
                        'target_label': cf['label'],
                        'passed_filters': False,
                        'kept': False
                    })
            
            except Exception as e:
                print(f"      Warning: Error scoring CF {cf['id']}: {e}")
                continue
        
        # Sort by quality and keep top max_per_example
        scored_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        kept_candidates = scored_candidates[:max_per_example]
        
        # Mark as kept in details
        kept_ids = {item['cf']['id'] for item in kept_candidates}
        for detail in filtering_details:
            if detail['cf_id'] in kept_ids:
                detail['kept'] = True
        
        # Add to final list
        all_filtered_cfs.extend([item['cf'] for item in kept_candidates])
    
    # Sort final list by quality (for budget constraint application)
    # Recompute ordering based on all kept CFs
    all_filtered_cfs_scored = []
    for cf in all_filtered_cfs:
        # Find its score in filtering_details
        for detail in filtering_details:
            if detail.get('cf_id') == cf['id'] and detail.get('kept'):
                all_filtered_cfs_scored.append({
                    'cf': cf,
                    'score': detail.get('combined_score', 0.5)
                })
                break
    
    all_filtered_cfs_scored.sort(key=lambda x: x['score'], reverse=True)
    all_filtered_cfs = [item['cf'] for item in all_filtered_cfs_scored]
    
    return all_filtered_cfs, filtering_details


def generate_single_counterfactual(
    original_text: str,
    original_label: str,
    target_label: str,
    llm_provider,
    temperature: float = 0.7,
    max_tokens: int = 256,
    variation_idx: int = 0,
    return_details: bool = False
) -> tuple:
    """
    Generate a single counterfactual by rewriting text to express target label.
    
    Args:
        original_text: Original text to transform
        original_label: Current label of the text
        target_label: Desired label for counterfactual
        llm_provider: LLM provider instance
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        variation_idx: Prompt variation index (0, 1, 2 for different prompts)
        return_details: If True, returns (cf_text, details_dict)
    
    Returns:
        Tuple of (cf_text, details_dict)
    """
    # Select prompt variation
    prompt_variations = [
        # Variation 0: Standard
        f"""Task: Rewrite the following text to express '{target_label}' instead of '{original_label}'.

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

Output only the rewritten text, nothing else.""",
        
        # Variation 1: Transform emphasis
        f"""Transform the following text to convey '{target_label}' sentiment instead of '{original_label}'.

Original: "{original_text}"
Current sentiment: {original_label}
Target sentiment: {target_label}

Requirements:
- Modify the text to express {target_label}
- Maintain similar length and structure
- Change only what's necessary
- Keep it natural and fluent
- Avoid using the word '{target_label}' directly

Provide only the transformed text.""",
        
        # Variation 2: Reframe emphasis
        f"""Reframe this text to reflect '{target_label}' rather than '{original_label}'.

Text: "{original_text}"
From: {original_label}
To: {target_label}

Guidelines:
- Adapt the text to show {target_label}
- Stay close to the original structure
- Make minimal but effective changes
- Ensure natural language
- Don't mention '{target_label}' explicitly

Output the reframed text only."""
    ]
    
    # Select prompt (cycle through variations)
    user_content = prompt_variations[variation_idx % len(prompt_variations)]
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert at rewriting text to express different categories while maintaining the original structure and theme."
        },
        {
            "role": "user",
            "content": user_content
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
    
    # Prepare detailed information if requested
    if return_details:
        details = {
            'original_text': original_text,
            'original_label': original_label,
            'target_label': target_label,
            'generated_text': cf_text,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'variation_idx': variation_idx,
            'generation_time_seconds': generation_time,
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
    Generate counterfactuals for evaluation purposes (legacy function).
    
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
    temperature = cf_config.get('generation_temperature', 0.7)
    max_tokens = cf_config.get('max_tokens', 256)
    
    counterfactuals = []
    
    for example in examples:
        original_text = example['text']
        original_label = example['label']
        
        # Get all target labels except original
        target_labels = [label for label in all_labels if label != original_label]
        
        for target_label in target_labels:
            for idx in range(num_per_target):
                try:
                    cf_text, _ = generate_single_counterfactual(
                        original_text,
                        original_label,
                        target_label,
                        llm_provider,
                        temperature,
                        max_tokens,
                        idx,
                        return_details=False
                    )
                    
                    if cf_text and cf_text.strip():
                        counterfactuals.append({
                            'id': f"{example['id']}_cf_{target_label}_{idx}",
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
