# Enhancements Implemented

## Summary

Three major improvements have been added to the Active Learning system based on user requirements:

1. **Interim Outputs After Each Step** - Save detailed inputs/outputs for inspection
2. **Logprobs Visualization** - Print OpenAI probability distributions on terminal
3. **Generate CFs for ALL Labels** - Not limited to 3, generates for all remaining labels

---

## 1. Interim Outputs After Each Step

### What Was Added

After each step in the Active Learning loop, intermediate results are saved to JSON files:

**Location**: `output_data/interim_output/`

**Files Created Per Iteration**:
- `iter_XX_step3_selected_examples.json` - Which examples were selected by uncertainty sampling
- `iter_XX_step4_oracle_labels.json` - Labels provided by oracle
- `iter_XX_step5_counterfactuals.json` - Generated counterfactuals with inputs

### Example Output Structure

```json
// iter_02_step3_selected_examples.json
{
  "iteration": 2,
  "selected_indices": [142, 567, 891, ...],
  "selected_examples": [
    {
      "id": 142,
      "text": "not sure how I feel about this",
      "label": "neutral"
    },
    ...
  ]
}

// iter_02_step5_counterfactuals.json
{
  "iteration": 2,
  "input_examples": [...],
  "generated_counterfactuals": [
    {
      "id": "142_cf_joy",
      "text": "I'm so excited about how I feel about this!",
      "label": "joy",
      "original_id": 142,
      "original_label": "neutral"
    },
    ...
  ]
}
```

### How to Use

```bash
# View selected examples from iteration 2
cat output_data/interim_output/iter_02_step3_selected_examples.json

# View all counterfactuals generated
cat output_data/interim_output/iter_02_step5_counterfactuals.json

# See what oracle labeled
cat output_data/interim_output/iter_02_step4_oracle_labels.json
```

---

## 2. Logprobs Visualization on Terminal

### What Was Added

When computing uncertainty scores (Step 3), the system now prints the first 3 examples with their full probability distributions from OpenAI's logprobs API.

### Example Terminal Output

```
[Step 3/6] Selecting uncertain examples...
  Computing uncertainty scores for 470 examples...

      --- LOGPROBS RESPONSE (Example 1) ---
      Text snippet: not sure how I feel about this...
      Prediction: neutral
      Probabilities from OpenAI:
        anger       : 0.0200 ████
        fear        : 0.0100 ██
        joy         : 0.2500 ███████
        love        : 0.0000 
        neutral     : 0.4000 ████████████
        sadness     : 0.3600 ██████████
      Entropy (uncertainty): 1.2834
      ---

      --- LOGPROBS RESPONSE (Example 2) ---
      Text snippet: I love this so much!...
      Prediction: joy
      Probabilities from OpenAI:
        anger       : 0.0100 ██
        fear        : 0.0000 
        joy         : 0.9500 ████████████████████████████
        love        : 0.0300 ███
        neutral     : 0.0100 ██
        sadness     : 0.0000 
      Entropy (uncertainty): 0.2342
      ---

      --- LOGPROBS RESPONSE (Example 3) ---
      Text snippet: this is somewhat okay I guess...
      Prediction: joy
      Probabilities from OpenAI:
        anger       : 0.0500 ████
        fear        : 0.0200 ██
        joy         : 0.6500 ███████████████████
        love        : 0.0300 ███
        neutral     : 0.2000 ██████
        sadness     : 0.0500 ████
      Entropy (uncertainty): 0.8921
      ---

  Selected 10 most uncertain examples
  Uncertainty scores range: [0.156, 1.385]
```

### What You Can See

- **Actual probabilities** from OpenAI (not heuristic 0.9/0.1)
- **Visual bars** showing probability magnitude
- **Entropy** (uncertainty measure) - higher = more uncertain
- **First 3 examples only** to avoid terminal spam

### Implementation Details

- Located in: `utils/classifier.py` lines 202-221
- Prints during `predict_proba_with_logprobs()`
- Shows first 3 examples per iteration
- Includes text snippet, prediction, all label probabilities, and entropy

---

## 3. Generate Counterfactuals for ALL Labels

### What Changed

**Before**: Generated only 3 counterfactuals per example (configurable via `per_example: 3`)

**After**: Generates counterfactuals for **ALL remaining labels** (all except the original label)

### Example

If you have **6 emotion labels**: `joy, sadness, anger, fear, surprise, love`

**Original example:**
```
Text: "I'm not sure how I feel"
Label: neutral
```

**Old behavior** (`per_example: 3`):
```
Generate 3 CFs:
- CF1: joy → "I'm so excited about how I feel!"
- CF2: sadness → "I'm devastated about how I feel"
- CF3: anger → "I'm furious about how I feel"
(Skip: fear, surprise, love)
```

**New behavior** (ALL labels):
```
Generate 5 CFs (all except 'neutral'):
- CF1: joy → "I'm so excited about how I feel!"
- CF2: sadness → "I'm devastated about how I feel"
- CF3: anger → "I'm furious about how I feel"
- CF4: fear → "I'm terrified about how I feel"
- CF5: surprise → "I'm shocked about how I feel"
(No CF for 'love' since original is 'neutral')
```

Wait, if original is 'neutral' and we have 6 labels including 'neutral', then we generate 5 CFs (all except neutral).

Actually, looking at the emotions dataset, the labels are:
`['anger', 'joy', 'sadness', 'love', 'fear', 'surprise']` - 6 labels

So if original is 'joy', we generate 5 CFs (all except 'joy').

### Terminal Output

```
[Step 5/6] Generating counterfactuals...
  Generating counterfactuals for ALL remaining labels per example
  Processing 10 examples

    [1/10] Example 142: 'not sure how I feel about this...' (neutral)
                        Generating 5 CFs → ['anger', 'fear', 'joy', 'love', 'sadness']
                          ✓ anger: 'I'm furious about how I feel about this...'
                          ✓ fear: 'I'm terrified about how I feel about this...'
                          ✓ joy: 'I'm so excited about how I feel about this...'
                          ✓ love: 'I adore how I feel about this...'
                          ✓ sadness: 'I'm devastated about how I feel about this...'
    
    [2/10] Example 567: 'this is okay I guess...' (joy)
                        Generating 5 CFs → ['anger', 'fear', 'love', 'neutral', 'sadness']
                          ✓ anger: 'this is infuriating I guess...'
                          ✓ fear: 'this is terrifying I guess...'
                          ✓ love: 'I love this so much...'
                          ✓ neutral: 'this is okay I guess...'
                          ✓ sadness: 'this is sad I guess...'
    
    ...

  Generated 50 counterfactuals from 10 examples
```

### Impact on Data Augmentation

**Example with 120 label budget, 10 batch size, 6 labels:**

**Old (3 CFs per example):**
- 12 iterations × 10 labels = 120 human labels
- 120 × 3 = 360 counterfactuals
- **Total training data**: 30 initial + 120 + 360 = **510 examples**

**New (ALL labels, ~5 CFs per example):**
- 12 iterations × 10 labels = 120 human labels  
- 120 × 5 = 600 counterfactuals
- **Total training data**: 30 initial + 120 + 600 = **750 examples**

**→ 47% more training data!**

### Configuration Note

The `per_example: 3` setting in config.yaml is now **ignored** - the system always generates for all remaining labels.

If you want to limit it again, you would need to modify `utils/counterfactual_generator.py` line 55:

```python
# Current: ALL labels
target_labels = [label for label in all_labels if label != original_label]

# To limit to 3:
target_labels = [label for label in all_labels if label != original_label][:3]
```

---

## How to Run with New Features

```bash
cd "/Users/adityamisra/Documents/git repos/CISCO AI /LLM-VT-AL"
venv/bin/python 05_active_learning_loop.py
```

### What You'll See

1. **During Step 3 (Uncertainty Scoring)**:
   - First 3 examples with full logprobs visualization
   - See actual probability distributions
   - Understand why examples are selected

2. **During Step 5 (Counterfactual Generation)**:
   - Detailed progress per example
   - All target labels being generated
   - Each generated CF shown with snippet

3. **After Each Iteration**:
   - Interim JSON files saved to `output_data/interim_output/`
   - Inspect inputs/outputs of each step
   - Debug or analyze the AL process

### Monitoring Files

```bash
# Watch interim outputs being created
watch -n 2 "ls -lh output_data/interim_output/"

# View latest step 3 outputs
cat output_data/interim_output/iter_*_step3_selected_examples.json | jq .

# View all counterfactuals generated
cat output_data/interim_output/iter_*_step5_counterfactuals.json | jq .
```

---

## Benefits

### 1. Better Debugging
- See exactly what's happening at each step
- Verify uncertainty sampling is working correctly
- Inspect counterfactual quality

### 2. Better Understanding
- Visualize how logprobs actually work
- See probability distributions in real-time
- Understand model confidence levels

### 3. More Training Data
- Generate 50-70% more counterfactuals
- Better label coverage
- Stronger data augmentation

### 4. Easier Analysis
- JSON files for programmatic analysis
- Can build visualizations from interim outputs
- Track which examples were most uncertain

---

## Files Modified

1. **`05_active_learning_loop.py`**
   - Added interim output saving after steps 3, 4, 5
   - Creates JSON files in `output_data/interim_output/`

2. **`utils/classifier.py`**
   - Added verbose logprobs printing (first 3 examples)
   - Shows probability distributions with visual bars
   - Displays entropy for uncertainty measure

3. **`utils/counterfactual_generator.py`**
   - Changed to generate for ALL remaining labels
   - Removed `per_example` limit
   - Added detailed progress logging
   - Shows each CF generated with snippet

---

## Summary

All three requested enhancements have been successfully implemented:

✅ **Interim outputs** - Save inputs/outputs after each step
✅ **Logprobs visualization** - Print first 3 examples with probabilities
✅ **ALL labels CFs** - Generate counterfactuals for all remaining labels

The system is now **more transparent**, **more verbose**, and **generates more training data**!

