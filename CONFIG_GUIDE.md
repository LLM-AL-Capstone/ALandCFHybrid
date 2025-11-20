# Counterfactual Generation with Quality Filtering - Configuration Guide

## Overview

This guide explains how to configure the counterfactual generation system with quality filtering. The system generates synthetic examples through LLM prompting and filters them based on quality metrics to ensure only high-value counterfactuals are added to the training pool.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Parameters](#configuration-parameters)
3. [Quality Metrics Explained](#quality-metrics-explained)
4. [Distribution Strategies](#distribution-strategies)
5. [Recommended Configurations](#recommended-configurations)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Configuration (Recommended for Beginners)

```yaml
counterfactuals:
  enabled: true
  max_per_example: 5
  generation_multiplier: 2.0
  cf_total_budget: 200
  quality_filtering:
    enabled: true
    metric: "combined"
    diversity_weight: 0.5
    confidence_weight: 0.3
    validity_weight: 0.2
    confidence:
      min_confidence: 0.5
  distribution_strategy: "balanced"
  generation_temperature: 0.7
  prompt_variation: true
  max_tokens: 256
```

---

## Configuration Parameters

### Generation Budget

#### `max_per_example`

**What it does**: Maximum number of counterfactuals to **keep** per newly labeled example.

**Type**: Integer  
**Default**: 5  
**Range**: 1-20 (typical)

**Examples**:
- `max_per_example: 3` → Keep 3 best CFs per example
- `max_per_example: 5` → Keep 5 best CFs per example (recommended)
- `max_per_example: 10` → Keep 10 best CFs per example (aggressive augmentation)

**How it works**:
- If dataset has 6 classes and example has label "joy", there are 5 target labels (anger, sadness, love, fear, surprise)
- With `max_per_example: 5`, system generates CFs for all 5 target labels
- With `max_per_example: 3`, system generates CFs for only 3 target labels (randomly selected or by priority)

**When to adjust**:
- Small datasets (100-500 examples): Use 3-5
- Medium datasets (500-2000): Use 5-8
- Large datasets (2000+): Use 2-3

---

#### `generation_multiplier`

**What it does**: Generate `multiplier × max_per_example` candidates, then filter to keep best `max_per_example`.

**Type**: Float  
**Default**: 2.0  
**Range**: 1.0-3.0

**Examples**:
- `1.0` → No over-generation (generate exactly `max_per_example`)
- `1.5` → Generate 50% more, filter to best
- `2.0` → Generate 2x more, filter to best (recommended)
- `3.0` → Generate 3x more, filter to best (highest quality, most expensive)

**Impact**:
- **Quality**: Higher multiplier = better quality (more candidates to choose from)
- **Cost**: Higher multiplier = more API calls (2x multiplier = 2x cost for CF generation)
- **Time**: Higher multiplier = longer generation time

**Formula**:
```
Total CFs generated = max_per_example × generation_multiplier
Total CFs kept = max_per_example
```

**Example**:
```yaml
max_per_example: 5
generation_multiplier: 2.0

For each example:
- Generate: 5 × 2.0 = 10 CF candidates
- Score all 10 candidates for quality
- Keep: top 5 best candidates
```

---

#### `cf_total_budget`

**What it does**: Hard limit on total counterfactuals across **entire AL run**.

**Type**: Integer  
**Default**: 200  
**Special**: -1 for unlimited

**Examples**:
- `cf_total_budget: 100` → Maximum 100 CFs total (2x ratio with 50 real labels)
- `cf_total_budget: 200` → Maximum 200 CFs total (4x ratio with 50 real labels)
- `cf_total_budget: 400` → Maximum 400 CFs total (8x ratio with 50 real labels)
- `cf_total_budget: -1` → Unlimited CFs

**How it works**:
```
Iteration 1: Generate 50 CFs → Budget: 200 - 50 = 150 remaining
Iteration 2: Generate 50 CFs → Budget: 150 - 50 = 100 remaining
Iteration 3: Generate 50 CFs → Budget: 100 - 50 = 50 remaining
Iteration 4: Generate 50 CFs → Budget: 50 - 50 = 0 remaining
Iteration 5+: CF generation stops (AL continues with real labels only)
```

**When budget is tight**:
- If iteration wants to add 50 CFs but only 30 budget remaining
- System keeps only top 30 CFs (ranked by quality score)
- Ensures budget is never exceeded

**Choosing a budget**:
```python
# Conservative: 2x ratio
cf_total_budget = total_budget * 2

# Balanced: 4x ratio (recommended)
cf_total_budget = total_budget * 4

# Aggressive: 8x ratio
cf_total_budget = total_budget * 8
```

---

### Quality Filtering

#### `quality_filtering.enabled`

**What it does**: Enable/disable quality-based filtering.

**Type**: Boolean  
**Default**: true

**Options**:
- `true` → Apply quality filtering (recommended)
- `false` → Keep all generated CFs (no filtering)

**When to disable**:
- Quick experiments or testing
- When LLM quality is already very high
- When you want to compare filtered vs unfiltered

---

#### `quality_filtering.metric`

**What it does**: Chooses which quality metric to use for filtering.

**Type**: String  
**Default**: "combined"  
**Options**: "diversity", "confidence", "validity", "combined"

**Option Details**:

##### 1. `"diversity"` - Novelty-Focused

**What it measures**: How different is the CF from existing examples in the labeled pool?

**When to use**:
- ✅ Large labeled pool (50+ examples)
- ✅ Want to avoid redundant examples
- ✅ Maximize coverage of feature space
- ✅ Prevent near-duplicates

**How it works**:
```python
# For each CF:
cf_embedding = embed(cf_text)
pool_embeddings = embed(all_labeled_pool_texts)

# Find most similar example in pool
max_similarity = max(cosine_similarity(cf_embedding, pool_emb) for pool_emb in pool_embeddings)

# Diversity = 1 - similarity (higher = more unique)
diversity_score = 1 - max_similarity
```

**Example**:
```
Labeled pool includes: "This product makes me angry"
CF candidate: "I'm furious about this product"
Similarity: 0.85 (very similar)
Diversity: 1 - 0.85 = 0.15 (LOW - likely rejected)

CF candidate: "The customer service was frustrating"
Similarity: 0.45 (moderately similar)
Diversity: 1 - 0.45 = 0.55 (GOOD - likely kept)
```

---

##### 2. `"confidence"` - Clarity-Focused

**What it measures**: How confident is the classifier that the CF belongs to its target label?

**When to use**:
- ✅ Small labeled pool (< 50 examples)
- ✅ Want clear, unambiguous examples
- ✅ Prioritize training signal strength
- ✅ Avoid confusing examples

**How it works**:
```python
# For each CF:
predictions = classifier.predict_proba([cf_text])
target_label_prob = predictions[target_label]

# Confidence = probability of intended label
confidence_score = target_label_prob
```

**Example**:
```
CF: "This movie made me angry"
Target label: "anger"

Classifier predictions:
- anger: 0.92 ← target label
- sadness: 0.05
- joy: 0.02
- fear: 0.01

Confidence score: 0.92 (HIGH - very clear example)

CF: "This situation is complex"
Target label: "anger"

Classifier predictions:
- anger: 0.38 ← target label
- sadness: 0.35
- confusion: 0.27

Confidence score: 0.38 (LOW - ambiguous, likely rejected)
```

---

##### 3. `"validity"` - Coherence-Focused

**What it measures**: Is the CF a reasonable transformation of the original text?

**When to use**:
- ✅ Ensure CFs maintain context
- ✅ Avoid completely unrelated transformations
- ✅ Check minimal editing constraint

**How it works**:
```python
# For each CF:
original_embedding = embed(original_text)
cf_embedding = embed(cf_text)
similarity = cosine_similarity(original_embedding, cf_embedding)

# Target similarity: 0.5 (sweet spot)
# Too similar (0.9+): Barely changed
# Too different (0.1-): Unrelated
validity_score = 1 - abs(similarity - 0.5) * 2
```

**Example**:
```
Original: "I love this product"

CF: "I really love this product"
Similarity: 0.95 (too similar)
Validity: 1 - abs(0.95 - 0.5)*2 = 0.10 (LOW - minimal change)

CF: "Quantum mechanics is fascinating"
Similarity: 0.05 (too different)
Validity: 1 - abs(0.05 - 0.5)*2 = 0.10 (LOW - unrelated)

CF: "This product disappoints me"
Similarity: 0.52 (just right)
Validity: 1 - abs(0.52 - 0.5)*2 = 0.96 (HIGH - good transformation)
```

---

##### 4. `"combined"` - Balanced (Recommended)

**What it measures**: Weighted combination of diversity, confidence, and validity.

**When to use**:
- ✅ Default choice for most research
- ✅ Want both novelty AND clarity
- ✅ Balanced approach

**How it works**:
```python
combined_score = (diversity_weight * diversity +
                  confidence_weight * confidence +
                  validity_weight * validity)
```

**Example**:
```yaml
diversity_weight: 0.5
confidence_weight: 0.3
validity_weight: 0.2

For a CF:
- Diversity: 0.75
- Confidence: 0.88
- Validity: 0.82

Combined: 0.5*0.75 + 0.3*0.88 + 0.2*0.82 = 0.803
```

---

#### `quality_filtering.diversity_weight`, `confidence_weight`, `validity_weight`

**What they do**: Control importance of each quality dimension when `metric="combined"`.

**Type**: Float  
**Range**: 0.0-1.0  
**Constraint**: Must sum to 1.0

**Preset Configurations**:

**Novelty-Focused** (Exploration):
```yaml
diversity_weight: 0.7
confidence_weight: 0.2
validity_weight: 0.1
```
Use when: Large labeled pool, want unique examples

**Clarity-Focused** (Few examples):
```yaml
diversity_weight: 0.2
confidence_weight: 0.7
validity_weight: 0.1
```
Use when: Small labeled pool, want clear training signal

**Balanced** (Recommended):
```yaml
diversity_weight: 0.5
confidence_weight: 0.3
validity_weight: 0.2
```
Use when: Default choice, medium-sized pool

**Transformation Quality**:
```yaml
diversity_weight: 0.3
confidence_weight: 0.3
validity_weight: 0.4
```
Use when: Want to ensure CFs are good transformations

---

#### `quality_filtering.confidence.min_confidence`

**What it does**: Minimum classifier confidence threshold to keep a CF.

**Type**: Float  
**Range**: 0.0-1.0  
**Default**: 0.5

**How it works**:
```python
# Hard filter: CF rejected if below threshold
if confidence_score < min_confidence:
    reject_cf()
```

**Guidelines**:
- **0.3-0.4**: Very lenient (keep ambiguous CFs)
- **0.5-0.6**: Balanced (recommended)
- **0.7-0.8**: Strict (only very clear CFs)
- **0.9+**: Extremely strict (may reject too many)

**Example**:
```yaml
min_confidence: 0.6

CF1: confidence = 0.85 → PASS ✓
CF2: confidence = 0.62 → PASS ✓
CF3: confidence = 0.58 → FAIL ✗ (below threshold)
CF4: confidence = 0.45 → FAIL ✗ (below threshold)
```

---

### Distribution Strategy

#### `distribution_strategy`

**What it does**: Controls how to distribute CF generations across target labels when `max_per_example > num_target_labels`.

**Type**: String  
**Default**: "balanced"  
**Options**: "balanced", "priority", "random", "quality_first"

**Scenario**: 4 target labels, want to generate 6 CFs

**Option Details**:

##### 1. `"balanced"` (Recommended)

**Behavior**: Distribute evenly across all target labels.

**Example**:
```
4 target labels: [anger, sadness, fear, love]
Need: 6 CFs total

Distribution: [2, 2, 1, 1] or [2, 1, 2, 1]
- Some labels get 2 CFs
- Some labels get 1 CF
- Fair coverage
```

##### 2. `"priority"`

**Behavior**: Give more CFs to underrepresented labels in labeled pool.

**Example**:
```
Labeled pool counts:
- anger: 50 examples
- sadness: 45 examples
- fear: 20 examples ← underrepresented
- love: 30 examples

Distribution: [1, 1, 3, 1]
- fear gets 3 CFs (needs more examples)
- Others get 1 CF each
```

##### 3. `"random"`

**Behavior**: Randomly assign CFs to target labels.

**Example**:
```
Distribution: [3, 0, 2, 1] (varies each run)
- Natural variation
- Some labels may get 0 CFs
```

##### 4. `"quality_first"`

**Behavior**: Don't pre-assign, generate many candidates and let quality filter decide.

**Example**:
```
Generate 2-3 CFs per label = 8-12 total
Score all for quality
Keep top 6 by quality score
Result: Distribution determined by quality, may be imbalanced
```

---

### Generation Settings

#### `generation_temperature`

**What it does**: Controls LLM creativity/randomness when generating CFs.

**Type**: Float  
**Range**: 0.0-1.0  
**Default**: 0.7

**Values**:
- **0.0**: Deterministic (same input → same output)
- **0.3-0.5**: Low creativity (consistent, conservative)
- **0.6-0.8**: Moderate creativity (recommended)
- **0.9-1.0**: High creativity (diverse, unpredictable)

**Impact on multiple CFs per label**:

```yaml
temperature: 0.0  # Deterministic

Label: "anger", Generate 3 CFs:
CF1: "This product makes me angry"
CF2: "This product makes me angry"  # IDENTICAL!
CF3: "This product makes me angry"  # IDENTICAL!
```

```yaml
temperature: 0.7  # Creative

Label: "anger", Generate 3 CFs:
CF1: "This product makes me angry"
CF2: "I'm furious about this product"  # Different!
CF3: "This infuriates me"  # Different!
```

**Recommendation**: Use 0.7-0.8 when `generation_multiplier > 1.0`

---

#### `prompt_variation`

**What it does**: Use different prompt templates when generating multiple CFs for the same label.

**Type**: Boolean  
**Default**: true

**How it works**:

With `prompt_variation: false`:
```
All CFs for "anger" use same prompt:
"Rewrite the text to express 'anger'..."
```

With `prompt_variation: true`:
```
CF1: "Rewrite the text to express 'anger'..."
CF2: "Transform the text to convey 'anger' sentiment..."
CF3: "Reframe this text to reflect 'anger'..."

Result: More diverse phrasings even with same target label
```

**Recommendation**: Set to `true` when `generation_multiplier > 1.0`

---

#### `max_tokens`

**What it does**: Maximum tokens per CF generation.

**Type**: Integer  
**Default**: 256  
**Range**: 50-512

**Guidelines**:
- Short texts (tweets): 100-150
- Medium texts (reviews): 200-300
- Long texts (articles): 400-512

---

## Quality Metrics Explained

### How Quality Scoring Works

For each CF candidate, the system computes:

```python
# Step 1: Compute individual scores
diversity_score = compute_diversity(cf, labeled_pool)     # 0.0-1.0
confidence_score = compute_confidence(cf, classifier)     # 0.0-1.0
validity_score = compute_validity(cf, original)           # 0.0-1.0

# Step 2: Check minimum thresholds
if confidence_score < min_confidence:
    reject_cf()  # Hard filter

# Step 3: Compute combined score
combined_score = (diversity_weight * diversity_score +
                  confidence_weight * confidence_score +
                  validity_weight * validity_score)

# Step 4: Rank all CFs by combined score
ranked_cfs = sort(cfs, by=combined_score, descending=True)

# Step 5: Keep top K
kept_cfs = ranked_cfs[:max_per_example]
```

### Example Scoring

```
Original: "The food was great"
Target label: "anger"

CF Candidate: "The food made me angry"

Diversity Score:
- Most similar in pool: "The service made me angry" (0.82 similar)
- Diversity: 1 - 0.82 = 0.18 (LOW - similar exists)

Confidence Score:
- Classifier predicts "anger": 0.88
- Confidence: 0.88 (HIGH - clear example)

Validity Score:
- Similarity to original: 0.55
- Validity: 1 - abs(0.55 - 0.5)*2 = 0.90 (HIGH - good transformation)

Combined Score (weights: 0.5, 0.3, 0.2):
0.5*0.18 + 0.3*0.88 + 0.2*0.90 = 0.534

Decision: KEEP (if above threshold) or REJECT (if below)
```

---

## Recommended Configurations

### Conservative (Low Cost, Good Quality)

Best for: Budget-constrained experiments, testing

```yaml
counterfactuals:
  enabled: true
  max_per_example: 3
  generation_multiplier: 1.5
  cf_total_budget: 150
  quality_filtering:
    enabled: true
    metric: "combined"
    diversity_weight: 0.5
    confidence_weight: 0.3
    validity_weight: 0.2
    confidence:
      min_confidence: 0.6
  distribution_strategy: "balanced"
  generation_temperature: 0.7
  prompt_variation: true
  max_tokens: 256
```

**Expected**:
- 50 real labels → ~150 CFs (3x ratio)
- Good quality (1.5x over-generation)
- Moderate API cost

---

### Balanced (Recommended)

Best for: Most research papers, default choice

```yaml
counterfactuals:
  enabled: true
  max_per_example: 5
  generation_multiplier: 2.0
  cf_total_budget: 200
  quality_filtering:
    enabled: true
    metric: "combined"
    diversity_weight: 0.5
    confidence_weight: 0.3
    validity_weight: 0.2
    confidence:
      min_confidence: 0.5
  distribution_strategy: "balanced"
  generation_temperature: 0.7
  prompt_variation: true
  max_tokens: 256
```

**Expected**:
- 50 real labels → ~200 CFs (4x ratio)
- High quality (2x over-generation)
- Reasonable API cost

---

### Aggressive (High Quality, Higher Cost)

Best for: Final experiments, publication-ready results

```yaml
counterfactuals:
  enabled: true
  max_per_example: 8
  generation_multiplier: 3.0
  cf_total_budget: 400
  quality_filtering:
    enabled: true
    metric: "combined"
    diversity_weight: 0.6
    confidence_weight: 0.3
    validity_weight: 0.1
  distribution_strategy: "quality_first"
  generation_temperature: 0.8
  prompt_variation: true
  max_tokens: 256
```

**Expected**:
- 50 real labels → ~400 CFs (8x ratio)
- Highest quality (3x over-generation)
- Higher API cost

---

## Troubleshooting

### Problem: Too few CFs generated

**Symptoms**: System generates 100 CFs but only keeps 20

**Possible causes**:
1. `min_confidence` threshold too high
2. Labeled pool too small for diversity scoring
3. Generation multiplier too low

**Solutions**:
```yaml
# Lower confidence threshold
confidence:
  min_confidence: 0.4  # Was 0.7

# Increase generation multiplier
generation_multiplier: 3.0  # Was 2.0

# Adjust diversity weight
diversity_weight: 0.3  # Was 0.6 (less emphasis on novelty)
```

---

### Problem: CFs are too similar to each other

**Symptoms**: Many CFs have very similar text

**Solutions**:
```yaml
# Increase diversity weight
diversity_weight: 0.7  # Was 0.5

# Increase temperature
generation_temperature: 0.8  # Was 0.7

# Enable prompt variation
prompt_variation: true
```

---

### Problem: CFs are low quality

**Symptoms**: Generated CFs don't clearly express target label

**Solutions**:
```yaml
# Increase confidence weight
confidence_weight: 0.5  # Was 0.3

# Raise min confidence
confidence:
  min_confidence: 0.7  # Was 0.5

# Increase generation multiplier
generation_multiplier: 3.0  # Was 2.0
```

---

### Problem: Budget exhausted too quickly

**Symptoms**: CF generation stops after 2-3 iterations

**Solutions**:
```yaml
# Increase total budget
cf_total_budget: 400  # Was 200

# Reduce per-example
max_per_example: 3  # Was 5

# Lower generation multiplier
generation_multiplier: 1.5  # Was 2.0 (fewer candidates = faster budget use)
```

---

### Problem: Slow generation

**Symptoms**: Each iteration takes very long

**Solutions**:
```yaml
# Reduce generation multiplier
generation_multiplier: 1.5  # Was 3.0

# Reduce max per example
max_per_example: 3  # Was 8

# Disable quality filtering (for testing)
quality_filtering:
  enabled: false
```

---

## For Research Papers

### Reporting Configuration

Example methodology text:

> "We employ quality-filtered counterfactual generation with a total budget of 200 synthetic examples across the Active Learning run (4:1 ratio with real labels). For each newly labeled example, we generate 10 candidate counterfactuals (generation_multiplier=2.0) and filter to retain the top 5 based on a combined quality score weighing diversity (50%), classifier confidence (30%), and transformation validity (20%). A minimum confidence threshold of 0.5 ensures only clear examples are retained. This approach yielded an average of X.X high-quality counterfactuals per real labeled example."

### Key Metrics to Report

1. **CF Generation Ratio**: `cf_total_budget / total_budget`
2. **Generation Efficiency**: `num_cfs_kept / num_cfs_generated`
3. **Average Quality Scores**: Mean diversity, confidence, validity
4. **Final Pool Composition**: X real + Y CF = Z total examples

---

## Summary

**Essential Parameters** (Must configure):
- `max_per_example`: How many CFs to keep per example
- `cf_total_budget`: Total CF budget
- `metric`: Which quality metric to use

**Important Parameters** (Affects quality/cost):
- `generation_multiplier`: Over-generation ratio
- `diversity_weight`, `confidence_weight`, `validity_weight`: Quality priorities
- `min_confidence`: Quality threshold

**Fine-Tuning Parameters** (Optional):
- `distribution_strategy`: How to distribute across labels
- `generation_temperature`: LLM creativity
- `prompt_variation`: Prompt diversity

**Start with recommended balanced configuration and adjust based on your needs!**

