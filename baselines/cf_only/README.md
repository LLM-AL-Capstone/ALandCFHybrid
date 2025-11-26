# CF-Only Ablation Study

Tests the effect of counterfactual generation alone, without active learning.

## Method

This ablation study tests the impact of counterfactual generation in isolation:

1. **Baseline**: Evaluate on seed set only
2. **Random Selection**: Select N random examples from unlabeled pool (N = budget)
3. **Query Oracle**: Get labels for selected examples
4. **Generate CFs**: Generate counterfactuals for labeled examples using quality filtering
5. **Evaluate**: Final performance with seed + factuals + CFs

**Key Difference from ACT-ICL**: No active learning iterations. This is a one-shot process that tests whether CF generation alone improves performance.

## Usage

```bash
cd baselines/cf_only
python run_cf_only.py
```

## Configuration

Edit `configs/cf_only_config.yaml`:

### Dataset Settings
- `dataset.train_file`: Training dataset filename
- `dataset.test_file`: Test dataset filename
- `dataset.columns`: Column name mappings

### Seed Set
- `active_learning.initial_labeled_per_class`: Number of examples per class in seed set (default: 5)

### Counterfactual Generation
- `active_learning.counterfactuals.enabled`: Enable/disable CF generation
- `active_learning.counterfactuals.max_per_example`: Max CFs per factual
- `active_learning.counterfactuals.alpha_cf`: Per-round budget multiplier
- `active_learning.counterfactuals.quality_filtering`: Quality filtering settings
- `active_learning.counterfactuals.target_label_selection`: Target label selection strategy

### Evaluation
- `evaluation.classifier_type`: "static" or "retrieval"
- `evaluation.retrieval`: Retrieval settings (if using retrieval)

### Budgets
- `cf_only.budgets`: List of budgets to test (default: [10, 20, 30, 40, 50, 100])

## Output

Results saved to `output/cf_only_<timestamp>_<model>_<dataset>/`:

- **`cf_only_results.csv`**: Results for all budgets with columns:
  - `budget`: Budget size (number of random factuals)
  - `baseline_f1_macro`: F1 macro on seed set only
  - `final_f1_macro`: F1 macro after adding factuals + CFs
  - `improvement`: Difference between final and baseline
  - `num_factuals`: Number of factual examples added
  - `num_counterfactuals`: Number of CFs generated
  - `final_labeled_pool_size`: Total labeled pool size
  - Additional metrics: `accuracy`, `f1_weighted`, `precision_macro`, `recall_macro`

- **`config_used.yaml`**: Snapshot of configuration used for this run

## Interpretation

The "CF-Only" row in your results table should use the `final_f1_macro` values from this ablation study.

This shows the isolated effect of:
- Random factual selection (no uncertainty-based selection)
- Counterfactual generation and quality filtering
- No iterative active learning

Compare with:
- **No-CF baselines**: Same setup but `counterfactuals.enabled: false`
- **ACT-ICL v1/v2/v3**: Full active learning with uncertainty-based selection + CFs

