# Paper-Compliant Counterfactual Filtering Implementation

## Overview

This document describes the implementation of counterfactual (CF) filtering following the paper's Algorithm 1. The implementation uses a **fixed-budget generation** approach with **two sequential quality filters**.

## Implementation Approach

### Fixed-Budget Generation

Instead of generating many CFs and selecting the best ones, we:

1. **Generate exactly `max_per_example` CFs** per labeled example (default: 3)
2. **Apply both quality filters** to all generated CFs
3. **Accept only CFs that pass both filters**

This approach provides clearer budget control and aligns with the paper's methodology.

**Config Setting:**
```yaml
counterfactuals:
  max_per_example: 3  # Generate exactly 3 CFs per example
```

## Quality Filters (Algorithm 1, Line 14)

The paper uses two sequential filters. A CF must pass **BOTH** to be accepted:

### Filter 1: Label Correctness

**Criterion:** Does the CF successfully flip the model's prediction?

**Mathematical Condition:**
```
p(y_target | CF) > p(y_orig | CF) + δ
```

Where:
- `p(y_target | CF)` = Probability of target label given CF text
- `p(y_orig | CF)` = Probability of original label given CF text  
- `δ` = Minimum margin threshold (default: 0.1)

**Additional Constraint:**
```
p(y_target | CF) ≥ min_target_confidence
```

Where:
- `min_target_confidence` = Minimum confidence in target label (default: 0.3)

**Config Settings:**
```yaml
quality_filtering:
  min_margin: 0.1              # δ threshold
  min_target_confidence: 0.3   # Minimum confidence
```

**Implementation:** `CFQualityScorer.compute_label_correctness()`

### Filter 2: Semantic Similarity

**Criterion:** Does the CF preserve the semantic context of the original?

**Mathematical Condition:**
```
cosine_similarity(embedding(original), embedding(CF)) ≥ threshold
```

Where:
- `embedding()` = SentenceTransformer embedding function
- `threshold` = Minimum similarity (default: 0.6)

**Config Settings:**
```yaml
quality_filtering:
  min_semantic_similarity: 0.6
  embedding_model: "all-MiniLM-L6-v2"
```

**Implementation:** `CFQualityScorer.compute_semantic_similarity()`

## Workflow

```
For each labeled example:
  1. Generate exactly 3 CFs (distributed across target labels)
  2. For each generated CF:
     a. Apply Filter 1 (Label Correctness)
        - If FAIL → Reject CF (reason: label_correctness)
     b. Apply Filter 2 (Semantic Similarity)  
        - If FAIL → Reject CF (reason: semantic_similarity)
     c. If both PASS → Accept CF
  3. Add all accepted CFs to pool
```

## What We Removed

The following features from the previous implementation were **NOT** in the paper and have been removed:

- ❌ **Diversity scoring** - Measuring how different CF is from labeled pool
- ❌ **Combined scoring** - Weighted combination of multiple metrics
- ❌ **Ranking-based selection** - Sorting by quality score and keeping top N
- ❌ **Generation multiplier** - Generate N×multiplier CFs then filter to N

These were replaced with the paper's simple two-filter approach.

## Files Modified

### 1. `utils/cf_quality_scorer.py` (Complete Rewrite)

**Old Implementation:**
- `compute_diversity_score()` - Removed
- `compute_confidence_score()` - Removed  
- `compute_validity_score()` - Removed
- `compute_combined_score()` - Removed

**New Implementation:**
- `compute_label_correctness()` - Filter 1
- `compute_semantic_similarity()` - Filter 2
- `filter_counterfactual()` - Apply both filters sequentially

### 2. `utils/counterfactual_generator.py`

**Changed Functions:**

- `generate_counterfactuals_batch()`:
  - Removed `generation_multiplier` 
  - Now generates exactly `max_per_example` CFs
  
- `generate_cf_candidates_for_example()`:
  - Removed `generation_multiplier` parameter
  - Renamed `max_per_example` → `num_per_example` for clarity
  
- `filter_by_quality()`:
  - Removed grouping by example
  - Removed ranking and top-N selection
  - Now applies both filters to all CFs
  - Prints detailed rejection statistics

### 3. `config.yaml.example`

**Removed Settings:**
```yaml
generation_multiplier: 2.0  # Removed
metric: "combined"          # Removed
diversity_weight: 0.5       # Removed  
confidence_weight: 0.3      # Removed
validity_weight: 0.2        # Removed
```

**New Settings:**
```yaml
quality_filtering:
  min_margin: 0.1                    # Filter 1 threshold
  min_target_confidence: 0.3         # Filter 1 confidence
  min_semantic_similarity: 0.6       # Filter 2 threshold
  embedding_model: "all-MiniLM-L6-v2"  # For semantic similarity
```

## Example Output

```
CF Generation Settings (Paper's Fixed-Budget Approach):
  CFs to generate per example: 3
  Quality filtering: enabled
  Budget remaining: 200

  [1/10] Processing: 'This movie was terrible and boring...' (negative)
              Generating 3 CFs across 2 labels

Total CF candidates generated: 30

Applying quality filtering...
Filtering Summary:
  Total generated: 30
  Rejected by label correctness: 8
  Rejected by semantic similarity: 5
  Total accepted: 17
  Acceptance rate: 56.7%

After quality filtering: 17 CFs
✓ Final: 17 counterfactuals added to pool
```

## Budget Tracking

The system tracks two metrics:

1. **Total CFs Generated** - How many CFs were generated (always = `num_examples × max_per_example`)
2. **Total CFs Accepted** - How many CFs passed both filters

Acceptance rate provides insight into filter effectiveness:
```
Acceptance Rate = (Total Accepted / Total Generated) × 100%
```

## Configuration Example

```yaml
active_learning:
  counterfactuals:
    enabled: true
    max_per_example: 3           # Generate exactly 3 CFs per example
    cf_total_budget: 200          # Stop when 200 total CFs accepted
    
    quality_filtering:
      enabled: true
      # Filter 1: Label Correctness
      min_margin: 0.1             # p(target) - p(orig) > 0.1
      min_target_confidence: 0.3  # p(target) >= 0.3
      # Filter 2: Semantic Similarity  
      min_semantic_similarity: 0.6  # cosine_sim >= 0.6
      embedding_model: "all-MiniLM-L6-v2"
    
    distribution_strategy: "balanced"
    generation_temperature: 0.7
    max_tokens: 256
    prompt_variation: true
```

## Testing the Implementation

1. Copy `config.yaml.example` to `config.yaml`
2. Adjust settings as needed
3. Run the active learning loop:
   ```bash
   python 05_active_learning_loop.py
   ```
4. Monitor the output for filtering statistics
5. Check interim outputs in `output_data/<run_name>/interim_output/` for detailed filtering results

## Key Differences from Previous Implementation

| Aspect | Previous | Paper-Compliant |
|--------|----------|-----------------|
| Generation | Generate N×2, select N best | Generate N, filter |
| Filtering | Multi-metric ranking | Two sequential filters |
| Diversity | Calculated and weighted | Not used |
| Budget | Soft (try to keep N) | Hard (generate exactly N) |
| Acceptance | Top N by score | All passing both filters |
| Complexity | High (multiple scores) | Low (two binary checks) |

## References

- **Algorithm 1, Line 14** - Paper's counterfactual filtering criterion
- **Log Probability Calculation** - See `docs/LOGPROBS_CALCULATION.md`
- **Configuration Guide** - See `config.yaml.example`
