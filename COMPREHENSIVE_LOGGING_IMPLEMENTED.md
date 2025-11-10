# Comprehensive Interim Logging - Implementation Summary

## Overview
This document describes the comprehensive logging system implemented to capture detailed information at every step of the Active Learning loop, including full OpenAI API responses, logprobs data, and generation metadata.

## What Was Implemented

### 1. Enhanced Classifier (`utils/classifier.py`)

**Changes:**
- Modified `predict_proba()` to accept `return_details` parameter
- Returns full logprobs data structure from OpenAI API
- Captures:
  - Prediction for each example
  - Raw OpenAI response (model, usage, finish_reason)
  - Complete logprobs structure with top_logprobs for each token
  - Computed probability distribution across all labels
  - Entropy (uncertainty) score

**New Return Structure:**
```python
{
    'method': 'logprobs',
    'text': 'example text...',
    'prediction': 'joy',
    'raw_openai_result': {
        'model': 'gpt-4o-2024-11-20',
        'choices': [{
            'message': {'role': 'assistant', 'content': 'joy'},
            'finish_reason': 'stop',
            'logprobs': {
                'content': [{
                    'token': 'joy',
                    'logprob': -0.234,
                    'top_logprobs': [
                        {'token': 'joy', 'logprob': -0.234},
                        {'token': 'sadness', 'logprob': -1.456},
                        ...
                    ]
                }]
            }
        }],
        'usage': {
            'prompt_tokens': 245,
            'completion_tokens': 1,
            'total_tokens': 246
        }
    },
    'computed_probabilities': {
        'anger': 0.02,
        'joy': 0.67,
        'sadness': 0.23,
        'fear': 0.05,
        'love': 0.02,
        'surprise': 0.01
    },
    'probability_distribution': [0.02, 0.67, 0.23, 0.05, 0.02, 0.01],
    'entropy': 1.234,
    'labels': ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
}
```

### 2. Enhanced LLM Provider (`utils/llm_provider.py`)

**Changes:**
- Modified `chat_completion_with_logprobs()` to return `raw_response`
- Captures serializable OpenAI response object
- Includes:
  - Model name
  - Full message content
  - Finish reason
  - First 5 tokens of logprobs (for brevity)
  - Token usage statistics

### 3. Enhanced Uncertainty Selection (`utils/uncertainty.py`)

**Changes:**
- Modified `select_uncertain_examples()` to accept `return_details` parameter
- Returns comprehensive uncertainty analysis
- Captures:
  - Uncertainty scores for ALL examples in pool
  - Selected indices and their scores
  - Score statistics (min, max, mean, std)
  - Full prediction details from classifier (including logprobs)

**New Return Structure:**
```python
{
    'method': 'entropy',
    'total_pool_size': 1850,
    'batch_size': 10,
    'all_uncertainty_scores': [0.234, 1.567, 0.891, ...],  # All 1850 scores
    'selected_indices': [142, 567, 891, ...],  # Top 10 indices
    'selected_scores': [1.567, 1.489, 1.445, ...],  # Their scores
    'score_statistics': {
        'min': 0.123,
        'max': 1.567,
        'mean': 0.789,
        'std': 0.234
    },
    'prediction_details': [
        # Full logprobs details for each example (see classifier structure above)
        {...}, {...}, ...
    ]
}
```

### 4. Enhanced Counterfactual Generator (`utils/counterfactual_generator.py`)

**Changes:**
- Modified `generate_counterfactuals_batch()` to accept `return_details` parameter
- Modified `generate_single_counterfactual()` to capture generation metadata
- Returns detailed generation information
- Captures:
  - Original text, original label, target label
  - Generated text
  - Full prompt messages sent to LLM
  - Generation parameters (temperature, max_tokens)
  - Generation time in seconds
  - Quality check results

**New Return Structure:**
```python
[
    {
        'cf_id': '42_cf_sadness',
        'original_example': {
            'id': 42,
            'text': 'I am so happy today!',
            'label': 'joy'
        },
        'target_label': 'sadness',
        'generated_text': 'I am so disappointed today.',
        'generation_metadata': {
            'original_text': 'I am so happy today!',
            'original_label': 'joy',
            'target_label': 'sadness',
            'generated_text': 'I am so disappointed today.',
            'temperature': 0,
            'max_tokens': 256,
            'generation_time_seconds': 1.234,
            'prompt_messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert at rewriting text...'
                },
                {
                    'role': 'user',
                    'content': 'Task: Rewrite the following text...'
                }
            ],
            'is_identical_to_original': False
        }
    },
    ...
]
```

### 5. Updated Main AL Loop (`05_active_learning_loop.py`)

**Changes:**
- All key steps now use `return_details=True`
- Interim outputs include comprehensive data:
  - **Step 3**: Full uncertainty analysis with logprobs for ALL examples
  - **Step 4**: Oracle-labeled examples (unchanged)
  - **Step 5**: Full counterfactual generation details with prompts and metadata

## Interim Output Files

After each iteration, three comprehensive JSON files are saved to `output_data/interim_output/`:

### 1. `iter_XX_step3_selected_examples.json`
Contains:
- Selected example indices
- Selected examples
- **NEW**: Complete uncertainty analysis
  - Uncertainty scores for all pool examples
  - OpenAI logprobs responses
  - Probability distributions
  - Entropy values

### 2. `iter_XX_step4_oracle_labels.json`
Contains:
- Oracle-labeled examples (unchanged)

### 3. `iter_XX_step5_counterfactuals.json`
Contains:
- Input examples
- Generated counterfactuals
- **NEW**: Complete generation details
  - Prompts sent to LLM
  - Generation parameters
  - Generation times
  - Quality checks

## Example Usage

### Analyzing Uncertainty Selection

```python
import json

# Load Step 3 output
with open('output_data/interim_output/iter_01_step3_selected_examples.json') as f:
    data = json.load(f)

# See why example was selected
uncertainty = data['uncertainty_analysis']
selected_idx = data['selected_indices'][0]  # First selected example

# Get its logprobs
pred_details = uncertainty['prediction_details'][selected_idx]
print(f"Text: {pred_details['text']}")
print(f"Prediction: {pred_details['prediction']}")
print(f"Probabilities: {pred_details['computed_probabilities']}")
print(f"Entropy: {pred_details['entropy']}")

# See OpenAI's raw logprobs
print(f"Raw logprobs: {pred_details['raw_openai_result']['choices'][0]['logprobs']}")
```

### Analyzing Counterfactual Generation

```python
# Load Step 5 output
with open('output_data/interim_output/iter_01_step5_counterfactuals.json') as f:
    data = json.load(f)

# See generation details for first CF
cf_detail = data['generation_details'][0]
print(f"Original: {cf_detail['original_example']['text']} ({cf_detail['original_example']['label']})")
print(f"Target: {cf_detail['target_label']}")
print(f"Generated: {cf_detail['generated_text']}")
print(f"Generation time: {cf_detail['generation_metadata']['generation_time_seconds']:.2f}s")
print(f"Prompt used:\n{cf_detail['generation_metadata']['prompt_messages']}")
```

## Benefits

### 1. **Full Transparency**
- Every API call is logged with complete request/response
- Can reproduce/debug any step of the process

### 2. **Research Analysis**
- Analyze uncertainty patterns
- Study logprobs distributions
- Understand what makes examples "uncertain"
- Evaluate counterfactual quality

### 3. **Cost Tracking**
- Token usage for every API call
- Identify expensive operations
- Optimize prompts and parameters

### 4. **Debugging**
- See exact prompts sent to LLM
- Verify probability calculations
- Identify generation issues

### 5. **Reproducibility**
- Complete record of all inputs/outputs
- Can replay any iteration
- Verify experimental results

## File Size Considerations

⚠️ **Note**: These comprehensive logs can be large:
- Each iteration's Step 3 file: ~500KB - 5MB (depends on pool size)
- Each iteration's Step 5 file: ~100KB - 1MB (depends on CFs generated)

For long runs (50+ iterations), monitor disk space and consider:
1. Periodic archival of old iterations
2. Selective logging (only first N iterations)
3. Compression of JSON files

## Terminal Output

The logprobs responses are still printed to terminal for the first 3 examples per iteration (for real-time monitoring), as previously implemented.

## Configuration

No configuration changes needed - logging is automatically enabled when the system runs. To disable comprehensive logging:

```python
# In 05_active_learning_loop.py, change:
selected_indices, uncertainty_details = select_uncertain_examples(..., return_details=True)
# To:
selected_indices = select_uncertain_examples(..., return_details=False)
```

## Summary

This implementation provides complete visibility into the Active Learning process, capturing:
- ✅ Full OpenAI API responses with logprobs
- ✅ Token-level probability distributions
- ✅ Uncertainty scores and analysis
- ✅ Counterfactual generation prompts and metadata
- ✅ Generation times and quality checks
- ✅ Token usage statistics

All data is saved in human-readable JSON format for easy exploration and analysis.

