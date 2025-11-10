# Interim Outputs - Complete Guide

## Overview

The Active Learning loop now saves **interim outputs after EVERY step** of each iteration. Each file is timestamped and contains complete information about what happened in that step.

## File Naming Convention

```
iter_{XX}_{YYYYMMDD_HHMMSS}_step{N}_{step_name}.json
```

Where:
- `XX`: Iteration number (01, 02, 03, ...)
- `YYYYMMDD_HHMMSS`: Timestamp when the step completed
- `N`: Step number (1-6)
- `step_name`: Descriptive name of the step

## Output Files Per Iteration

For each iteration, you get **6 JSON files** (one per step):

### Step 1: Classifier Training
**Filename**: `iter_XX_YYYYMMDD_HHMMSS_step1_classifier_training.json`

**Contains**:
```json
{
  "iteration": 1,
  "timestamp": "20241109_203045",
  "step": "classifier_training",
  "labeled_pool_size": 30,
  "num_labels": 6,
  "labels": ["anger", "fear", "joy", "love", "sadness", "surprise"],
  "labeled_examples": [
    {"id": 1, "text": "I am happy", "label": "joy"},
    ...
  ],
  "training_summary": {
    "total_examples": 30,
    "labels_count": {
      "anger": 5,
      "fear": 5,
      "joy": 5,
      "love": 5,
      "sadness": 5,
      "surprise": 5
    }
  }
}
```

### Step 2: Evaluation
**Filename**: `iter_XX_YYYYMMDD_HHMMSS_step2_evaluation.json`

**Contains**:
```json
{
  "iteration": 1,
  "timestamp": "20241109_203145",
  "step": "evaluation",
  "metrics": {
    "accuracy": 0.7551,
    "f1_macro": 0.7234,
    "f1_weighted": 0.7456,
    "precision_macro": 0.7123,
    "recall_macro": 0.7345,
    "confusion_matrix": [[...], ...]
  },
  "best_accuracy": 0.7551,
  "patience_counter": 0,
  "test_pool_size": 196
}
```

If evaluation is skipped:
```json
{
  "iteration": 2,
  "timestamp": "20241109_203245",
  "step": "evaluation",
  "status": "skipped",
  "reason": "eval_every_iterations=1"
}
```

### Step 3: Uncertainty Selection
**Filename**: `iter_XX_YYYYMMDD_HHMMSS_step3_uncertainty_selection.json`

**Contains** (with full logprobs and uncertainty analysis):
```json
{
  "iteration": 1,
  "timestamp": "20241109_203300",
  "step": "uncertainty_selection",
  "selected_indices": [142, 567, 891],
  "selected_examples": [
    {"id": 142, "text": "not sure how I feel", "label": "neutral"},
    ...
  ],
  "uncertainty_analysis": {
    "method": "entropy",
    "total_pool_size": 470,
    "batch_size": 10,
    "all_uncertainty_scores": [0.234, 1.567, ...],  // All 470 scores
    "selected_indices": [142, 567, 891],
    "selected_scores": [1.567, 1.489, 1.445],
    "score_statistics": {
      "min": 0.123,
      "max": 1.567,
      "mean": 0.789,
      "std": 0.234
    },
    "prediction_details": [
      {
        "method": "logprobs",
        "text": "not sure how I feel",
        "prediction": "neutral",
        "raw_openai_result": {
          "model": "gpt-4o-2024-11-20",
          "choices": [{
            "message": {"role": "assistant", "content": "neutral"},
            "finish_reason": "stop",
            "logprobs": {
              "content": [{
                "token": "neutral",
                "logprob": -0.916,
                "top_logprobs": [
                  {"token": "neutral", "logprob": -0.916},
                  {"token": "sadness", "logprob": -1.022},
                  {"token": "joy", "logprob": -1.386}
                ]
              }]
            }
          }],
          "usage": {
            "prompt_tokens": 245,
            "completion_tokens": 1,
            "total_tokens": 246
          }
        },
        "computed_probabilities": {
          "anger": 0.02,
          "joy": 0.25,
          "neutral": 0.40,
          "sadness": 0.36,
          "fear": 0.05,
          "surprise": 0.01
        },
        "probability_distribution": [0.02, 0.25, 0.40, 0.36, 0.05, 0.01],
        "entropy": 1.567,
        "labels": ["anger", "fear", "joy", "love", "sadness", "surprise"]
      }
      // ... details for all pool examples
    ]
  },
  "unlabeled_pool_size": 470,
  "batch_size": 10
}
```

### Step 4: Oracle Labeling
**Filename**: `iter_XX_YYYYMMDD_HHMMSS_step4_oracle_labeling.json`

**Contains**:
```json
{
  "iteration": 1,
  "timestamp": "20241109_203320",
  "step": "oracle_labeling",
  "labeled_examples": [
    {"id": 142, "text": "not sure how I feel", "label": "neutral"},
    {"id": 567, "text": "this is confusing", "label": "surprise"},
    ...
  ],
  "num_labeled": 10
}
```

### Step 5: Counterfactual Generation
**Filename**: `iter_XX_YYYYMMDD_HHMMSS_step5_counterfactual_generation.json`

**Contains** (with full generation details):
```json
{
  "iteration": 1,
  "timestamp": "20241109_203420",
  "step": "counterfactual_generation",
  "input_examples": [
    {"id": 142, "text": "I am so happy today!", "label": "joy"},
    ...
  ],
  "generated_counterfactuals": [
    {
      "id": "142_cf_sadness",
      "text": "I am so disappointed today.",
      "label": "sadness",
      "original_id": 142,
      "original_label": "joy"
    },
    ...
  ],
  "generation_details": [
    {
      "cf_id": "142_cf_sadness",
      "original_example": {
        "id": 142,
        "text": "I am so happy today!",
        "label": "joy"
      },
      "target_label": "sadness",
      "generated_text": "I am so disappointed today.",
      "generation_metadata": {
        "original_text": "I am so happy today!",
        "original_label": "joy",
        "target_label": "sadness",
        "generated_text": "I am so disappointed today.",
        "temperature": 0,
        "max_tokens": 256,
        "generation_time_seconds": 1.234,
        "prompt_messages": [
          {
            "role": "system",
            "content": "You are an expert at rewriting text..."
          },
          {
            "role": "user",
            "content": "Task: Rewrite the following text to express 'sadness'..."
          }
        ],
        "is_identical_to_original": false
      }
    }
    // ... details for all generated CFs
  ],
  "num_generated": 50
}
```

If counterfactuals are disabled:
```json
{
  "iteration": 1,
  "timestamp": "20241109_203420",
  "step": "counterfactual_generation",
  "status": "skipped",
  "reason": "counterfactuals_disabled_in_config"
}
```

### Step 6: Pool Update
**Filename**: `iter_XX_YYYYMMDD_HHMMSS_step6_pool_update.json`

**Contains**:
```json
{
  "iteration": 1,
  "timestamp": "20241109_203425",
  "step": "pool_update",
  "before": {
    "labeled_pool_size": 30,
    "unlabeled_pool_size": 470,
    "budget": 120
  },
  "changes": {
    "real_examples_added": 10,
    "counterfactuals_added": 50,
    "unlabeled_examples_removed": 10,
    "budget_consumed": 10
  },
  "after": {
    "labeled_pool_size": 90,
    "unlabeled_pool_size": 460,
    "budget_remaining": 110
  }
}
```

## Example: Iteration 1 Files

After iteration 1 completes, you'll have:

```
output_data/interim_output/
├── iter_01_20241109_203045_step1_classifier_training.json
├── iter_01_20241109_203145_step2_evaluation.json
├── iter_01_20241109_203300_step3_uncertainty_selection.json
├── iter_01_20241109_203320_step4_oracle_labeling.json
├── iter_01_20241109_203420_step5_counterfactual_generation.json
└── iter_01_20241109_203425_step6_pool_update.json
```

## What You Can Do With These Files

### 1. Track Progress
```bash
# See all iterations completed
ls -lh output_data/interim_output/ | grep step1

# See latest iteration
ls -lht output_data/interim_output/ | head -6
```

### 2. Analyze Uncertainty
```python
import json

# Load Step 3 from iteration 5
with open('output_data/interim_output/iter_05_*_step3_uncertainty_selection.json') as f:
    data = json.load(f)

# Analyze why examples were selected
for detail in data['uncertainty_analysis']['prediction_details'][:5]:
    print(f"Text: {detail['text']}")
    print(f"Entropy: {detail['entropy']:.4f}")
    print(f"Probabilities: {detail['computed_probabilities']}")
    print()
```

### 3. Debug Counterfactual Generation
```python
# Load Step 5
with open('output_data/interim_output/iter_05_*_step5_counterfactual_generation.json') as f:
    data = json.load(f)

# See exact prompts used
for cf in data['generation_details']:
    print(f"Original: {cf['original_example']['text']} ({cf['original_example']['label']})")
    print(f"Target: {cf['target_label']}")
    print(f"Generated: {cf['generated_text']}")
    print(f"Time: {cf['generation_metadata']['generation_time_seconds']:.2f}s")
    print(f"Prompt: {cf['generation_metadata']['prompt_messages']}")
    print("-" * 80)
```

### 4. Monitor API Costs
```python
# Sum up token usage from Step 3 files
total_tokens = 0
for file in glob.glob('output_data/interim_output/*_step3_*.json'):
    with open(file) as f:
        data = json.load(f)
    for detail in data['uncertainty_analysis']['prediction_details']:
        if 'raw_openai_result' in detail:
            usage = detail['raw_openai_result']['usage']
            total_tokens += usage['total_tokens']

print(f"Total tokens used for uncertainty scoring: {total_tokens}")
```

### 5. Reproduce Any Iteration
Each file contains complete information, so you can:
- See exact labeled pool at any point (Step 1)
- Reproduce evaluation results (Step 2)
- Understand selection criteria (Step 3)
- Verify oracle labels (Step 4)
- Audit CF generation (Step 5)
- Track pool evolution (Step 6)

## File Sizes

Typical file sizes per iteration:
- **Step 1**: ~50-500 KB (depends on labeled pool size)
- **Step 2**: ~5-20 KB
- **Step 3**: ~500 KB - 5 MB (depends on unlabeled pool size and logprobs)
- **Step 4**: ~10-50 KB
- **Step 5**: ~100 KB - 1 MB (depends on CFs generated)
- **Step 6**: ~2-5 KB

For 50 iterations, expect ~50-250 MB total.

## Tips

1. **Monitor in Real-time**:
   ```bash
   watch -n 5 'ls -lht output_data/interim_output/ | head -20'
   ```

2. **Find Latest Iteration**:
   ```bash
   ls output_data/interim_output/ | tail -6
   ```

3. **Search Specific Step**:
   ```bash
   ls output_data/interim_output/*step3* | tail -1
   ```

4. **Archive Old Runs**:
   ```bash
   tar -czf interim_outputs_run1.tar.gz output_data/interim_output/
   rm -rf output_data/interim_output/*
   ```

## Summary

✅ **Complete Transparency**: Every step logged with full details  
✅ **Timestamped**: Track exact timing of each step  
✅ **Comprehensive**: OpenAI logprobs, prompts, metadata, everything  
✅ **Debuggable**: Reproduce and understand any iteration  
✅ **Analyzable**: JSON format for easy programmatic analysis  

Every single operation in the Active Learning loop is now fully documented! 🎉

