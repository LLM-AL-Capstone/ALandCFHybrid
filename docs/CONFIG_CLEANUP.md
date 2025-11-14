# Config File Cleanup Summary

## What Was Removed

The `config.yaml` file has been cleaned up to remove all parameters from the old pattern-based counterfactual pipeline that are no longer used in the Active Learning system.

### Removed Sections

#### 1. `llm.models.*` (Entire Section)
**Reason:** These were model-specific settings for the old pattern-based pipeline (Scripts 01-04).

Removed:
- `pattern_identification` - For Script 01 (archived)
- `candidate_generation` - For Script 01 (archived)
- `counterfactual_generation` - For Script 02 (archived)
- `semantic_filtering` - For Script 03 (archived)
- `discriminator_filtering` - For Script 03 (archived)

#### 2. `processing.*` Parameters
**Reason:** These were specific to the old pipeline's pattern identification and evaluation approach.

Removed:
- `max_examples_per_label` - Used for pattern identification (not needed in AL)
- `token_limit` - Budget tracking for old pipeline
- `max_counterfactuals_per_example` - Old CF generation limit
- `evaluation_shots` - Old evaluation configuration
- `evaluation_seeds` - Old evaluation configuration

**Kept:**
- `seed: 42` - Still used for reproducibility

#### 3. `directories.*` Parameters
**Reason:** Simplified directory structure.

Removed:
- `interim_output` - Created dynamically by AL loop
- `archive` - Not used in AL system

**Kept:**
- `input_data`
- `output_data`

#### 4. `counterfactuals.per_example`
**Reason:** Parameter was not being used by the code. The current implementation generates 1 CF per remaining label (e.g., 5 CFs for a 6-class problem).

## What Remains

### Active Sections

1. **`llm`** - LLM provider configuration (openai, gemini, ollama)
2. **`dataset`** - Train/test files and column mappings
3. **`processing`** - Just the random seed
4. **`directories`** - Input and output directories
5. **`active_learning`** - All AL parameters (budget, batch size, uncertainty method, counterfactuals, stopping criteria)
6. **`evaluation`** - Classifier type, ICL settings, retrieval configuration
7. **`logging`** - Checkpointing and results file settings

## Changes Made

### Updated Comments
- Clarified that `train_file` is for "AL pool" not just training
- Updated counterfactual comment to explain actual behavior (1 CF per remaining label)
- Changed `min_improvement` comment from "accuracy" to "F1 Macro"

### Updated Values
- `train_file`: Changed from `yelp_train.csv` to `emotions_train.csv`
- `test_file`: Changed from `yelp_test.csv` to `emotions_test.csv`
- `total_budget`: Set to `120` (was empty)

## File Size Reduction

- **Before:** 172 lines
- **After:** 133 lines
- **Reduction:** 39 lines (~23% smaller)

## Verification

✅ Config file loads successfully
✅ All sections present: llm, dataset, processing, directories, active_learning, evaluation, logging
✅ No breaking changes to existing code

## Notes

The old pattern-based pipeline configuration can still be found in:
- `config.yaml.example` (original example file)
- `archive/old_pattern_pipeline/` (archived scripts with their own documentation)

If you need to run the old pipeline, refer to those files for the required configuration structure.

