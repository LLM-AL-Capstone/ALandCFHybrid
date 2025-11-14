# Evaluation Metadata Tracking Enhancement

## Overview

Added comprehensive metadata tracking to evaluation outputs, making it easy to identify which evaluation strategy (static vs retrieval) was used for each experiment.

## Changes Implemented

### 1. Step 2 Evaluation JSON - Added evaluation_config

**Before:**
```json
{
  "iteration": 1,
  "timestamp": "20251110_213743",
  "step": "evaluation",
  "metrics": {
    "accuracy": 0.6224,
    "f1_macro": 0.3329
  }
}
```

**After:**
```json
{
  "iteration": 1,
  "timestamp": "20251110_213743",
  "step": "evaluation",
  "evaluation_config": {
    "classifier_type": "retrieval",
    "max_icl_examples": 100,
    "retrieval_settings": {
      "embedding_backend": "sentence_transformers",
      "k_per_class": 3,
      "total_k_max": 50,
      "fallback_strategy": "similarity",
      "model_config": {
        "model": "all-MiniLM-L6-v2",
        "device": "cpu"
      }
    }
  },
  "metrics": {
    "accuracy": 0.6224,
    "f1_macro": 0.3329
  }
}
```

### 2. al_results.csv - Added classifier_type Column

**Before:**
```csv
iteration,labeled_pool_size,accuracy,f1_macro
1,90,0.6224,0.3329
2,150,0.6071,0.3916
```

**After:**
```csv
iteration,classifier_type,labeled_pool_size,accuracy,f1_macro
1,static,90,0.6224,0.3329
2,static,150,0.6071,0.3916
---
1,retrieval,90,0.6450,0.4120
2,retrieval,150,0.6680,0.4580
```

## Benefits

### 1. Easy Experiment Comparison

**Load and Compare:**
```python
import pandas as pd

results = pd.read_csv('output_data/al_results.csv')

# Group by classifier type
static = results[results['classifier_type'] == 'static']
retrieval = results[results['classifier_type'] == 'retrieval']

print("Static ICL:")
print(f"  Max Accuracy: {static['accuracy'].max():.4f}")
print(f"  Max F1 Macro: {static['f1_macro'].max():.4f}")

print("\nRetrieval ICL:")
print(f"  Max Accuracy: {retrieval['accuracy'].max():.4f}")
print(f"  Max F1 Macro: {retrieval['f1_macro'].max():.4f}")
```

### 2. Identify Configuration from Files

**Check Any Evaluation File:**
```python
import json

with open('output_data/interim_output/iter_01_..._step2_evaluation.json') as f:
    data = json.load(f)
    
    config = data['evaluation_config']
    print(f"Classifier Type: {config['classifier_type']}")
    
    if config['classifier_type'] == 'retrieval':
        settings = config['retrieval_settings']
        print(f"Backend: {settings['embedding_backend']}")
        print(f"k per class: {settings['k_per_class']}")
        print(f"Total k max: {settings['total_k_max']}")
```

### 3. Reproducibility

Every evaluation file now contains:
- Exact classifier configuration used
- All retrieval parameters (if applicable)
- Model-specific settings
- Complete reproduction recipe

### 4. Analysis & Visualization

**Compare Static vs Retrieval Over Iterations:**
```python
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('output_data/al_results.csv')

# Plot by classifier type
for classifier_type in ['static', 'retrieval']:
    data = results[results['classifier_type'] == classifier_type]
    plt.plot(data['iteration'], data['accuracy'], 
             label=f'{classifier_type} - accuracy')
    plt.plot(data['iteration'], data['f1_macro'], 
             label=f'{classifier_type} - f1_macro', linestyle='--')

plt.xlabel('Iteration')
plt.ylabel('Score')
plt.legend()
plt.title('Static vs Retrieval ICL Comparison')
plt.show()
```

## Code Changes

### Modified Sections in 05_active_learning_loop.py

#### 1. Step 2 Evaluation (Lines 427-459)
Added `eval_metadata` dictionary with:
- `classifier_type`
- `max_icl_examples`
- `retrieval_settings` (if retrieval is used)
  - `embedding_backend`
  - `k_per_class`
  - `total_k_max`
  - `fallback_strategy`
  - `model_config` (backend-specific settings)

#### 2. Step 2 Skipped (Lines 471-493)
Same metadata added to skipped evaluation files

#### 3. Results CSV (Line 637)
Added `classifier_type` field to iteration results dictionary

## Usage Examples

### Example 1: Quick Check

```bash
# Check classifier type from JSON
cat output_data/interim_output/iter_01_*_step2_evaluation.json | grep classifier_type
```

### Example 2: Filter CSV by Type

```python
import pandas as pd

df = pd.read_csv('output_data/al_results.csv')

# Get only retrieval runs
retrieval_runs = df[df['classifier_type'] == 'retrieval']
print(retrieval_runs[['iteration', 'accuracy', 'f1_macro']])
```

### Example 3: Validate Experiment

```python
import json
import glob

# Check all evaluations used correct config
eval_files = glob.glob('output_data/interim_output/*_step2_evaluation.json')

for file in eval_files:
    with open(file) as f:
        data = json.load(f)
        config = data.get('evaluation_config', {})
        classifier = config.get('classifier_type', 'unknown')
        print(f"{file.split('/')[-1]}: {classifier}")
```

## Backward Compatibility

✅ **Fully backward compatible:**
- New fields added, none removed
- Defaults to 'static' if not specified
- Old results files still work
- No breaking changes

## Files Modified

- `05_active_learning_loop.py` - Added metadata to:
  - Step 2 evaluation JSON (lines 427-459)
  - Step 2 skipped JSON (lines 471-493)
  - Results CSV dictionary (line 637)

## Testing

Run the Active Learning loop with both configurations:

**Test 1: Static ICL**
```yaml
# config.yaml
evaluation:
  classifier_type: "static"
```
```bash
python 05_active_learning_loop.py
# Check: output_data/interim_output/iter_01_*_step2_evaluation.json
# Should have: "classifier_type": "static"
```

**Test 2: Retrieval ICL**
```yaml
# config.yaml
evaluation:
  classifier_type: "retrieval"
```
```bash
python 05_active_learning_loop.py
# Check: output_data/interim_output/iter_01_*_step2_evaluation.json
# Should have: "classifier_type": "retrieval" + retrieval_settings
```

## Summary

✅ **Implemented**: Comprehensive evaluation metadata tracking  
✅ **Location**: Step 2 JSON files + al_results.csv  
✅ **Benefit**: Easy experiment comparison and reproducibility  
✅ **Compatible**: Works with both static and retrieval approaches  

Now you can easily:
- Identify which evaluation strategy was used
- Compare static vs retrieval results
- Reproduce exact configurations
- Analyze experiments systematically

