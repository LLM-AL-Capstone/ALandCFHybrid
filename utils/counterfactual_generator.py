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
    alpha_cf: float = None,
    target_label_selector=None,
    return_details: bool = False
) -> tuple:
    """
    Generate and filter counterfactuals for a batch of newly labeled examples (Version 3).
    
    Supports over-generation with quality filtering and per-round budget.
    
    Args:
        labeled_examples: List of dicts with 'id', 'text', 'label'
        config: Configuration dictionary
        llm_provider: LLM provider instance
        all_labels: List of all possible labels in the dataset
        labeled_pool: Current labeled pool (for diversity calculation)
        classifier: Trained classifier (for confidence scoring)
        alpha_cf: Per-round budget multiplier (|C_t| <= alpha_cf * |F_t|). If None, uses config.
        target_label_selector: Target label selector instance (V3)
        return_details: If True, returns (counterfactuals, details_list)
    
    Returns:
        Tuple of (counterfactuals, num_added, details_list)
        - counterfactuals: List of CF dicts
        - num_added: Number of CFs added (may be less than generated due to per-round budget)
        - details_list: Generation and filtering details (if return_details=True)
    """
    cf_config = config['active_learning']['counterfactuals']
    
    if not cf_config['enabled']:
        print("  Counterfactual generation disabled in config")
        if return_details:
            return [], 0, []
        return [], 0
    
    # Get per-round budget multiplier (V3) with backward compatibility
    if alpha_cf is None:
        alpha_cf = cf_config.get('alpha_cf', 1.0)
        # Backward compatibility: if alpha_cf not set but cf_total_budget is, use legacy mode
        if 'alpha_cf' not in cf_config and cf_config.get('cf_total_budget', -1) > 0:
            print("  Warning: Using legacy global budget mode (cf_total_budget). Consider switching to alpha_cf.")
            # Legacy mode will be handled by caller
            alpha_cf = None
    
    # Calculate per-round budget: |C_t| <= alpha_cf * |F_t|
    if alpha_cf is not None:
        max_cfs_this_round = int(alpha_cf * len(labeled_examples))
        print(f"  Per-round budget: max {max_cfs_this_round} CFs (alpha_cf={alpha_cf}, |F_t|={len(labeled_examples)})")
    else:
        max_cfs_this_round = None  # Unlimited (legacy mode)
    
    # Get configuration
    max_per_example = cf_config.get('max_per_example', 3)
    quality_filtering_enabled = cf_config.get('quality_filtering', {}).get('enabled', False)
    
    # Check if using V3 target label selection
    use_v3_selection = target_label_selector is not None
    if not use_v3_selection:
        # Fallback to old distribution_strategy for backward compatibility
        distribution_strategy = cf_config.get('distribution_strategy', 'balanced')
    else:
        distribution_strategy = None  # Not used with V3 selector
    
    print(f"  CF Generation Settings (Version 3):")
    print(f"    CFs to generate per example: {max_per_example}")
    print(f"    Target label selection: {'V3 (one label per factual)' if use_v3_selection else f'Legacy ({distribution_strategy})'}")
    print(f"    Quality filtering: {'enabled' if quality_filtering_enabled else 'disabled'}")
    if max_cfs_this_round is not None:
        print(f"    Per-round budget: {max_cfs_this_round} CFs")
    else:
        print(f"    Budget: unlimited (legacy mode)")
    
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
            distribution_strategy,
            target_label_selector,
            classifier,  # Needed for confusion-based strategy
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
    
    # Apply per-round budget constraint (V3): |C_t| <= alpha_cf * |F_t|
    if max_cfs_this_round is not None and len(filtered_cfs) > max_cfs_this_round:
        print(f"  Per-round budget constraint: keeping top {max_cfs_this_round} of {len(filtered_cfs)} CFs (already ranked by score)")
        # Already sorted by V3 score, so just take first max_cfs_this_round
        filtered_cfs = filtered_cfs[:max_cfs_this_round]
    
    num_added = len(filtered_cfs)
    
    print(f"  ✓ Final: {num_added} counterfactuals added to pool\n")
    
    # Compile details if requested
    if return_details:
        combined_details = {
            'num_examples': len(labeled_examples),
            'num_generated': len(all_cf_candidates),
            'num_filtered': len(filtered_cfs),
            'num_added': num_added,
            'alpha_cf': alpha_cf,
            'max_cfs_this_round': max_cfs_this_round,
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
    num_per_example: int,
    distribution_strategy: str,
    target_label_selector=None,
    classifier=None,
    return_details: bool = False
) -> List[Tuple[Dict, Dict]]:
    """
    Generate CF candidates for a single example (Version 3: one target label per factual).
    
    Version 3 behavior: Selects ONE target label per factual, then generates
    all num_per_example CFs toward that selected label.
    
    Legacy behavior: If target_label_selector is None, uses old distribution_strategy
    to distribute CFs across multiple target labels.
    
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
    
    # Version 3: Use target label selector (if available)
    if target_label_selector is not None:
        # Select ONE target label using strategy
        try:
            selected_target_label = target_label_selector.select_target_label(
                original_label=original_label,
                original_text=original_text,
                classifier=classifier
            )
            print(f"                Selected target label: {selected_target_label} (strategy: {target_label_selector.strategy})")
            print(f"                Generating {num_per_example} CFs toward {selected_target_label}")
            
            # Generate all CFs toward the selected target label
            target_labels_to_generate = [selected_target_label]
            counts_per_label = {selected_target_label: num_per_example}
            
        except Exception as e:
            print(f"      Warning: Target label selection failed: {e}, falling back to uniform")
            # Fallback to uniform selection
            import random
            selected_target_label = random.choice(target_labels)
            target_labels_to_generate = [selected_target_label]
            counts_per_label = {selected_target_label: num_per_example}
    
    else:
        # Legacy: Distribute generations across target labels (old behavior)
        label_distribution = distribute_generations(
            target_labels,
            num_per_example,
            distribution_strategy
        )
        target_labels_to_generate = list(label_distribution.keys())
        counts_per_label = label_distribution
        print(f"                Generating {num_per_example} CFs across {len(target_labels)} labels (legacy mode)")
    
    # Generate CFs
    candidates = []
    temperature = cf_config.get('generation_temperature', 0.7)
    max_tokens = cf_config.get('max_tokens', 256)
    prompt_variation = cf_config.get('prompt_variation', True)
    
    for target_label in target_labels_to_generate:
        count = counts_per_label[target_label]
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
        # Distribute evenly using round-robin for deterministic balance
        base_count = num_to_generate // num_labels
        remainder = num_to_generate % num_labels
        
        distribution = {label: base_count for label in target_labels}
        
        # Assign remainder in round-robin fashion (deterministic)
        for i in range(remainder):
            distribution[target_labels[i]] += 1
        
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
        # Assign remainder in round-robin fashion (deterministic)
        for i in range(remainder):
            distribution[target_labels[i]] += 1
        return distribution
    
    else:
        # Default to balanced (round-robin)
        base_count = num_to_generate // num_labels
        distribution = {label: base_count for label in target_labels}
        remainder = num_to_generate - (base_count * num_labels)
        # Assign remainder in round-robin fashion (deterministic)
        for i in range(remainder):
            distribution[target_labels[i]] += 1
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
    Filter CF candidates using V3 enhanced filtering (3 filters) + scoring + ranking.
    
    Process:
    1. Apply all 3 filters (label-consistency, semantic band, length ratio)
    2. Score passing CFs using V3 formula
    3. Rank by score (descending)
    4. Keep top-k per example (if max_per_example is set)
    
    Args:
        cf_candidates: List of candidate dicts with 'cf', 'original_example', etc.
        labeled_examples: Newly labeled examples
        labeled_pool: Current labeled pool (not used)
        classifier: Trained classifier
        quality_scorer: Quality scorer instance
        max_per_example: Max CFs to keep per original example (after ranking)
    
    Returns:
        Tuple of (filtered_cfs, filtering_details)
    """
    all_filtered_cfs_with_scores = []  # List of (cf, score, details) tuples
    filtering_details = []
    
    total_generated = len(cf_candidates)
    label_consistency_rejected = 0
    semantic_similarity_rejected = 0
    length_ratio_rejected = 0
    total_passed_filters = 0
    
    # Step 1: Apply all 3 filters
    for item in cf_candidates:
        cf = item['cf']
        original_example = item['original_example']
        
        try:
            # Apply V3 filters (all 3)
            passes_all, filter_details = quality_scorer.filter_counterfactual(
                cf_text=cf['text'],
                original_text=original_example['text'],
                target_label=cf['label'],
                original_label=original_example['label'],
                classifier=classifier
            )
            
            if passes_all:
                # Step 2: Score passing CFs
                score, score_details = quality_scorer.compute_v3_score(
                    cf_text=cf['text'],
                    original_text=original_example['text'],
                    target_label=cf['label'],
                    original_label=original_example['label'],
                    classifier=classifier
                )
                
                all_filtered_cfs_with_scores.append((cf, score, filter_details, score_details))
                total_passed_filters += 1
                
                # Record filtering details with score
                filtering_details.append({
                    'cf_id': cf['id'],
                    'original_id': original_example['id'],
                    'target_label': cf['label'],
                    'passed': True,
                    'rejection_stage': 'none',
                    'label_consistency': filter_details.get('label_consistency', {}),
                    'semantic_similarity': filter_details.get('semantic_similarity', {}),
                    'length_ratio': filter_details.get('length_ratio', {}),
                    'v3_score': score,
                    'v3_score_details': score_details
                })
            else:
                # Track rejection reason
                rejection_stage = filter_details.get('rejection_stage', 'unknown')
                if rejection_stage == 'label_consistency':
                    label_consistency_rejected += 1
                elif rejection_stage == 'semantic_similarity':
                    semantic_similarity_rejected += 1
                elif rejection_stage == 'length_ratio':
                    length_ratio_rejected += 1
                
                # Record filtering details without score
                filtering_details.append({
                    'cf_id': cf['id'],
                    'original_id': original_example['id'],
                    'target_label': cf['label'],
                    'passed': False,
                    'rejection_stage': rejection_stage,
                    'label_consistency': filter_details.get('label_consistency', {}),
                    'semantic_similarity': filter_details.get('semantic_similarity', {}),
                    'length_ratio': filter_details.get('length_ratio', {}),
                    'v3_score': None
                })
            
        except Exception as e:
            print(f"      Warning: Error filtering CF {cf['id']}: {e}")
            filtering_details.append({
                'cf_id': cf['id'],
                'original_id': original_example['id'],
                'target_label': cf['label'],
                'passed': False,
                'error': str(e)
            })
            continue
    
    # Step 3: Rank by score (descending)
    all_filtered_cfs_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Step 4: Keep top-k per example (if max_per_example is set)
    if max_per_example > 0:
        # Group by original example ID
        from collections import defaultdict
        cf_by_original = defaultdict(list)
        for cf, score, filter_details, score_details in all_filtered_cfs_with_scores:
            original_id = cf.get('original_id', 'unknown')
            cf_by_original[original_id].append((cf, score, filter_details, score_details))
        
        # Keep top max_per_example per original
        final_cfs = []
        for original_id, cfs_list in cf_by_original.items():
            top_cfs = cfs_list[:max_per_example]
            final_cfs.extend([(cf, score, filter_details, score_details) for cf, score, filter_details, score_details in top_cfs])
        
        # Re-sort by score (global ranking)
        final_cfs.sort(key=lambda x: x[1], reverse=True)
        all_filtered_cfs_with_scores = final_cfs
    
    # Extract CFs with scores and ranking information for transparency
    all_filtered_cfs = []
    for rank, (cf, score, filter_details, score_details) in enumerate(all_filtered_cfs_with_scores, start=1):
        # Add V3 score and ranking to each CF for transparency
        cf_with_metadata = cf.copy()
        cf_with_metadata['v3_score'] = float(score)
        cf_with_metadata['v3_rank'] = rank  # Rank after filtering (1 = highest score)
        cf_with_metadata['v3_score_breakdown'] = {
            'term1': float(score_details.get('term1', 0.0)),  # (1 - p(y_label | u'))
            'term2': float(score_details.get('term2', 0.0)),  # β * p(y_target | u')
            'term3': float(score_details.get('term3', 0.0)),  # α * cos(E(u), E(u'))
            'total': float(score),
            'p_orig': float(score_details.get('p_orig', 0.0)),
            'p_target': float(score_details.get('p_target', 0.0)),
            'similarity': float(score_details.get('similarity', 0.0))
        }
        all_filtered_cfs.append(cf_with_metadata)
    
    # Print summary statistics
    print(f"  Filtering Summary:")
    print(f"    Total generated: {total_generated}")
    print(f"    Rejected by label-consistency: {label_consistency_rejected}")
    print(f"    Rejected by semantic similarity: {semantic_similarity_rejected}")
    print(f"    Rejected by length ratio: {length_ratio_rejected}")
    print(f"    Passed all filters: {total_passed_filters}")
    print(f"    After ranking: {len(all_filtered_cfs)} CFs")
    print(f"    Acceptance rate: {100 * len(all_filtered_cfs) / total_generated if total_generated > 0 else 0:.1f}%")
    
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
