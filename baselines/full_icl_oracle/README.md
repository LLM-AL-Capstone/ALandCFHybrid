# Full-ICL Oracle Baseline

This baseline establishes the **upper bound** performance for the Active Learning system by using **100% supervision** (full access to all training labels).

## Overview

The Full-ICL Oracle baseline evaluates how well a simple In-Context Learning (ICL) classifier performs when given access to the entire labeled training dataset. This serves as a performance ceiling to compare against Active Learning approaches.

## Key Concept: Budget

**Budget = Number of few-shot examples in the ICL prompt**

For each test example, the classifier:
1. Retrieves the top-k most similar examples from the FULL training dataset
2. Uses these k examples as few-shot demonstrations in the prompt
3. Asks the LLM to classify the test example

## Methodology

- **Training Pool**: Entire training dataset (495 examples for Yelp)
- **Retrieval Method**: Sentence Transformers (all-MiniLM-L6-v2) with cosine similarity
- **ICL Budgets Tested**: [10, 20, 30, 40, 50, 100]
- **Evaluation**: Same test set as Active Learning experiments (147 examples for Yelp)

## Configuration

Copy the example config and add your API key:

```bash
cp configs/baseline_config.yaml.example configs/baseline_config.yaml
# Edit baseline_config.yaml to add your API credentials
```

### Key Settings

```yaml
baseline:
  # Number of few-shot examples retrieved per test case
  icl_budgets: [10, 20, 30, 40, 50, 100]
  
  # Retrieval method: sentence_transformers or bm25
  retrieval_methods:
    - sentence_transformers
```

## Running the Baseline

```bash
cd baselines/full_icl_oracle
source ../../.venv/bin/activate
python run_baseline.py
```

## Output Structure

Results are saved to: `output/full_icl_oracle_{timestamp}_{model}_{dataset}/`

**Files generated:**
- `all_results.json` - Complete results for all budgets
- `summary.txt` - Performance summary table
- `report_{method}_budget_{k}.txt` - Detailed classification report per budget

**Example output directory:**
```
full_icl_oracle_20251121_195404_gpt-4o-2024-11-20_yelp/
├── all_results.json
├── summary.txt
├── report_sentence_transformers_budget_10.txt
├── report_sentence_transformers_budget_20.txt
├── report_sentence_transformers_budget_30.txt
├── report_sentence_transformers_budget_40.txt
├── report_sentence_transformers_budget_50.txt
└── report_sentence_transformers_budget_100.txt
```

## Results Interpretation

The baseline helps answer:

1. **Upper Bound**: What's the best performance achievable with full supervision?
2. **AL Efficiency**: How close does Active Learning get to this upper bound with limited labels?
3. **Budget Analysis**: How does performance scale with number of ICL examples?

### Expected Performance Hierarchy

```
Random Sampling < Active Learning < Active Learning + CF ≤ Full-ICL Oracle
```

## Comparison with Active Learning

| Approach | Labels Used | Label Source |
|----------|-------------|--------------|
| **Full-ICL Oracle** | 495 (100%) | Full training dataset |
| **Active Learning** | 20-70 (4-14%) | Seed set + selected samples |
| **AL + CF** | 20-70 + synthetic | Seed + selected + counterfactuals |

The baseline uses ~7-25x more labeled data, establishing the performance ceiling.

## Technical Details

### Retrieval Process

For each test example with budget k:
1. Encode test text using sentence transformer model
2. Compute cosine similarity with all 495 training examples
3. Select top-k most similar examples
4. Build ICL prompt with these k examples
5. Query LLM for classification

### ICL Prompt Format

```
Classify the sentiment of the following review.

Available labels: products, price, service, environment

Here are some examples:

Example 1:
Text: [Most similar example 1]
Label: [Label 1]

Example 2:
Text: [Most similar example 2]
Label: [Label 2]

...

Example k:
Text: [Most similar example k]
Label: [Label k]

Now classify this review:
Text: [Test example]
Label:
```

## Troubleshooting

**Issue**: `FileNotFoundError: baseline_config.yaml`
- **Solution**: Copy from `baseline_config.yaml.example` and add API key

**Issue**: Slow execution
- **Solution**: Normal - evaluating 6 budgets × 147 test examples = ~882 LLM calls

**Issue**: Out of memory
- **Solution**: Reduce `icl_budgets` or use smaller model

## Future Extensions

Potential improvements:
- [ ] Add BM25 retrieval method
- [ ] Test with different embedding models
- [ ] Evaluate on multiple datasets (MASSIVE, Emotions)
- [ ] Compare with fine-tuned models
- [ ] Add diversity-based retrieval

## Citation

If you use this baseline in your research, please cite the original Active Learning with Counterfactuals paper.
