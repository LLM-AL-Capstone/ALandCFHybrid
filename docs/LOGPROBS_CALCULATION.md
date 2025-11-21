# Log Probability Calculation Per Label

## Overview

This document explains how the system extracts **per-label probabilities** from the LLM using OpenAI's `logprobs` API. This enables accurate uncertainty estimation for Active Learning query strategies.

---

## The Complete Pipeline

### 1. Request Log Probabilities from LLM

In [`utils/classifier.py`](../utils/classifier.py), the `_predict_proba_with_logprobs()` method calls:

```python
result = self.llm_provider.chat_completion_with_logprobs(
    messages=messages,
    possible_labels=label_list,  # Pass valid labels: ['joy', 'sadness', 'anger', ...]
    temperature=0,
    max_tokens=512
)
```

### 2. OpenAI API Returns Token-Level Log Probabilities

The OpenAI API response includes:

```python
response.choices[0].logprobs.content[0] = {
    'token': 'joy',           # The predicted token
    'logprob': -0.234,        # Log probability of 'joy'
    'top_logprobs': [         # Alternative tokens with their logprobs
        {'token': 'joy', 'logprob': -0.234},
        {'token': 'sadness', 'logprob': -2.456},
        {'token': 'anger', 'logprob': -3.123},
        {'token': 'fear', 'logprob': -4.567},
        {'token': 'surprise', 'logprob': -5.123},
        # ... up to 20 alternatives
    ]
}
```

### 3. Match Tokens to Valid Labels

In [`utils/llm_provider.py`](../utils/llm_provider.py), the `chat_completion_with_logprobs()` method:

```python
# Extract logprobs for each possible label
for label in possible_labels:
    label_lower = label.lower()
    
    # Check if label appears in top_logprobs
    found = False
    
    # Check the actual predicted token
    if first_token_logprobs.token.lower().strip() == label_lower:
        label_logprobs[label] = first_token_logprobs.logprob
        found = True
    
    # Check top alternatives
    if not found and first_token_logprobs.top_logprobs:
        for alt in first_token_logprobs.top_logprobs:
            if alt.token.lower().strip() == label_lower:
                label_logprobs[label] = alt.logprob
                found = True
                break
    
    # If not found in top 20, assign very low probability
    if not found:
        label_logprobs[label] = -20.0  # exp(-20) ≈ 2e-9
```

**Result:**
```python
label_logprobs = {
    'joy': -0.234,
    'sadness': -2.456,
    'anger': -3.123,
    'fear': -4.567,
    'surprise': -5.123,
    'love': -20.0      # Not in top_logprobs
}
```

### 4. Convert Log Probabilities to Probabilities (Softmax Normalization)

```python
# Calculate log-sum-exp for numerical stability
log_sum_exp = np.log(sum(np.exp(lp) for lp in label_logprobs.values()))

# Normalize to get probabilities
for label, logprob in label_logprobs.items():
    # p(label) = exp(logprob) / sum(exp(all_logprobs))
    #          = exp(logprob - log_sum_exp)
    label_probs[label] = float(np.exp(logprob - log_sum_exp))
```

**Mathematical Formula:**

$$
p(y_i) = \frac{\exp(\text{logprob}_i)}{\sum_{j=1}^{N} \exp(\text{logprob}_j)}
$$

**Example Calculation:**

```python
# Raw logprobs
joy:      -0.234  →  exp(-0.234) = 0.791
sadness:  -2.456  →  exp(-2.456) = 0.086
anger:    -3.123  →  exp(-3.123) = 0.044
fear:     -4.567  →  exp(-4.567) = 0.010
surprise: -5.123  →  exp(-5.123) = 0.006
love:     -20.0   →  exp(-20.0)  = 0.000000002

# Sum of all = 0.937

# Normalized probabilities (sum = 1.0)
joy:      0.791 / 0.937 = 0.844
sadness:  0.086 / 0.937 = 0.092
anger:    0.044 / 0.937 = 0.047
fear:     0.010 / 0.937 = 0.011
surprise: 0.006 / 0.937 = 0.006
love:     0.000000002 / 0.937 = 0.000000002
```

### 5. Return Probability Distribution

```python
return {
    'prediction': 'joy',
    'probabilities': {
        'joy': 0.844,
        'sadness': 0.092,
        'anger': 0.047,
        'fear': 0.011,
        'surprise': 0.006,
        'love': 0.000000002
    },
    'label_logprobs': {
        'joy': -0.234,
        'sadness': -2.456,
        # ...
    },
    'raw_response': <OpenAI response object>
}
```

---

## From Probabilities to Uncertainty Scores

Once we have per-label probabilities, [`utils/uncertainty.py`](../utils/uncertainty.py) calculates uncertainty:

### Entropy (Default Method)

$$
H(p) = -\sum_{i=1}^{N} p(y_i) \log p(y_i)
$$

```python
entropy = -np.sum(probs * np.log(probs + eps), axis=1)
```

**Example:**
```python
probs = [0.844, 0.092, 0.047, 0.011, 0.006, 0.000]
entropy = -(0.844*log(0.844) + 0.092*log(0.092) + ...)
        = 0.512
```

**Interpretation:**
- **Low entropy (e.g., 0.1)**: Model is confident (one label dominates)
- **High entropy (e.g., 2.5)**: Model is uncertain (probabilities spread across labels)

### Margin Sampling

```python
margin = top1_prob - top2_prob
uncertainty = -margin  # Negate so smaller margin = higher uncertainty
```

**Example:**
```python
probs = [0.844, 0.092, 0.047, ...]
margin = 0.844 - 0.092 = 0.752
uncertainty = -0.752
```

### Least Confident

```python
uncertainty = 1 - max(probs)
```

**Example:**
```python
probs = [0.844, 0.092, 0.047, ...]
uncertainty = 1 - 0.844 = 0.156
```

---

## Verbose Output Example

When running the Active Learning loop, you'll see detailed probability breakdowns:

```
--- LOGPROBS RESPONSE (Example 1) ---
Text snippet: I absolutely love this product! It makes me so happy...
Prediction: joy
Probabilities from OpenAI:
  joy         : 0.8440 █████████████████████████████
  sadness     : 0.0920 ███
  anger       : 0.0470 █
  fear        : 0.0110 
  surprise    : 0.0060 
  love        : 0.0000 
Entropy (uncertainty): 0.5120
---

--- LOGPROBS RESPONSE (Example 2) ---
Text snippet: I'm not sure how I feel about this...
Prediction: sadness
Probabilities from OpenAI:
  sadness     : 0.3500 ███████████
  joy         : 0.2800 ████████
  anger       : 0.2200 ███████
  fear        : 0.1100 ███
  surprise    : 0.0300 █
  love        : 0.0100 
Entropy (uncertainty): 1.6234
---
```

The second example has **higher entropy** (1.62 vs 0.51), indicating more uncertainty → more likely to be selected for labeling.

---

## Key Implementation Files

| File | Purpose |
|------|---------|
| [`utils/llm_provider.py`](../utils/llm_provider.py) | Calls OpenAI API with `logprobs=True`, extracts token logprobs, converts to label probabilities |
| [`utils/classifier.py`](../utils/classifier.py) | Builds ICL prompts, calls `chat_completion_with_logprobs()`, converts to NumPy arrays |
| [`utils/uncertainty.py`](../utils/uncertainty.py) | Calculates entropy/margin/least-confident from probability distributions |
| [`05_active_learning_loop.py`](../05_active_learning_loop.py) | Orchestrates: train → predict_proba → calculate uncertainty → select examples |

---

## Advantages of Using Log Probabilities

1. **Accurate Uncertainty**: Real model confidence, not heuristics
2. **Better Sample Selection**: Identifies truly uncertain examples
3. **Data Efficiency**: Active Learning works better with accurate uncertainty
4. **Interpretability**: See exactly why the model is uncertain

---

## Fallback Behavior

If `logprobs` are **not available** (e.g., Ollama, Gemini), the system falls back to a heuristic:

```python
def _predict_proba_heuristic(self, text: str, label_list: List[str], ...):
    # Get prediction without logprobs
    prediction = self.predict(text)
    
    # Create distribution: 0.9 for predicted, 0.1 distributed among others
    prob_dist = np.zeros(n_labels)
    pred_idx = label_to_idx[prediction]
    prob_dist[pred_idx] = 0.9
    
    # Distribute remaining 0.1 among other labels
    remaining = 0.1 / (n_labels - 1)
    for i in range(n_labels):
        if i != pred_idx:
            prob_dist[i] = remaining
    
    return prob_dist
```

**Example heuristic distribution:**
```python
# If prediction = 'joy' and 6 labels
[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
#  ^   sadness anger fear surprise love
# joy
```

This is **less accurate** but allows the system to work with any LLM provider.

---

## Configuration

Enable/disable logprobs in [`config.yaml`](../config.yaml):

```yaml
evaluation:
  classifier_type: "static"     # or "retrieval"
  classifier_max_tokens: 512    # Enough tokens for response
```

For OpenAI models, logprobs are **automatically used** if available.

---

## Related Documentation

- [LOGPROBS_ENHANCEMENT.md](LOGPROBS_ENHANCEMENT.md) - Implementation details
- [RETRIEVAL_ICL_IMPLEMENTATION.md](RETRIEVAL_ICL_IMPLEMENTATION.md) - Retrieval-based classifier
- [INTERIM_OUTPUTS_GUIDE.md](INTERIM_OUTPUTS_GUIDE.md) - Output format including uncertainty scores

---

**Last Updated:** November 17, 2025
