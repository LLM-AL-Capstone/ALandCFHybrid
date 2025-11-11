# Enhancement: Model Name in Interim Output Filenames

## Summary

Interim output filenames now include the LLM model name, making it easier to track which model generated which results, especially when comparing different LLM providers or models.

## Changes Made

### File Modified
- `05_active_learning_loop.py`

### What Changed

#### 1. Model Name Extraction (Lines 349-361)
Added logic to extract and sanitize the model name from the config:

```python
# Extract model name from config (sanitize for filename)
provider = config['llm']['provider']
if provider == 'openai':
    model_name = config['llm']['openai']['model']
elif provider == 'gemini':
    model_name = config['llm']['gemini']['model']
elif provider == 'ollama':
    model_name = config['llm']['ollama']['model']
else:
    model_name = provider

# Sanitize model name for filename (replace invalid characters)
model_safe = model_name.replace('/', '-').replace(':', '-').replace(' ', '_')
```

#### 2. Filename Pattern Updates
Updated all 8 interim output filename patterns to include `{model_safe}`:

**Before:**
```
iter_01_20251109_203527_step1_classifier_training.json
```

**After:**
```
iter_01_20251109_203527_gpt-4o-2024-11-20_step1_classifier_training.json
```

## New Filename Format

```
iter_{iteration:02d}_{timestamp}_{model_safe}_step{N}_{step_name}.json
```

### Components:
- `iter_01` - Iteration number (zero-padded)
- `20251111_143022` - Timestamp (YYYYMMDD_HHMMSS)
- `gpt-4o-2024-11-20` - **NEW: Model name (sanitized)**
- `step1` - Step number
- `classifier_training` - Step description

## Examples by Provider

### OpenAI (Azure OpenAI)
Config model: `gpt-4o-2024-11-20`

```
iter_01_20251111_143022_gpt-4o-2024-11-20_step1_classifier_training.json
iter_01_20251111_143022_gpt-4o-2024-11-20_step2_evaluation.json
iter_01_20251111_143022_gpt-4o-2024-11-20_step3_uncertainty_selection.json
iter_01_20251111_143022_gpt-4o-2024-11-20_step4_oracle_labeling.json
iter_01_20251111_143022_gpt-4o-2024-11-20_step5_counterfactual_generation.json
iter_01_20251111_143022_gpt-4o-2024-11-20_step6_pool_update.json
```

### Google Gemini
Config model: `gemini-2.5-flash`

```
iter_01_20251111_143022_gemini-2.5-flash_step1_classifier_training.json
iter_01_20251111_143022_gemini-2.5-flash_step2_evaluation.json
...
```

### Ollama (Local)
Config model: `qwen2.5:7b`

Sanitized to: `qwen2.5-7b` (`:` replaced with `-`)

```
iter_01_20251111_143022_qwen2.5-7b_step1_classifier_training.json
iter_01_20251111_143022_qwen2.5-7b_step2_evaluation.json
...
```

## Character Sanitization

Model names are sanitized to be filesystem-safe:

| Original Character | Replacement | Reason |
|-------------------|-------------|---------|
| `/` | `-` | Invalid in filenames |
| `:` | `-` | Invalid on Windows |
| ` ` (space) | `_` | Better for shell/scripts |

### Examples:
- `gpt-4o-2024-11-20` → `gpt-4o-2024-11-20` (no change)
- `qwen2.5:7b` → `qwen2.5-7b`
- `llama3 8b` → `llama3_8b`
- `models/gemini-pro` → `models-gemini-pro`

## Benefits

### 1. **Easy Model Identification**
Instantly see which model generated each output file:
```bash
$ ls output_data/interim_output/
iter_01_..._gpt-4o-2024-11-20_step1_*.json
iter_01_..._gemini-2.5-flash_step1_*.json
iter_01_..._qwen2.5-7b_step1_*.json
```

### 2. **Compare Experiments**
Run the same AL experiment with different models side-by-side:
```bash
# Compare uncertainty selection across models
cat iter_01_*_gpt-4o-*_step3_uncertainty_selection.json
cat iter_01_*_gemini-*_step3_uncertainty_selection.json
cat iter_01_*_qwen-*_step3_uncertainty_selection.json
```

### 3. **Better Organization**
Files are now grouped by:
- Iteration number
- Timestamp (run time)
- **Model used** ← NEW
- Step number

### 4. **Research & Debugging**
- Track model-specific issues (e.g., "GPT-4o fails at step 3")
- Compare counterfactual quality across models
- Analyze which model provides better uncertainty estimates
- Archive results by model for papers/reports

### 5. **Multi-Model Experiments**
Easy to run experiments comparing:
- GPT-4o vs GPT-4o-mini
- Cloud (OpenAI/Gemini) vs Local (Ollama)
- Different model versions over time

## Usage

### Running Experiments

No config changes needed! Just run:

```bash
python 05_active_learning_loop.py
```

The model name is automatically extracted from your `config.yaml` and included in all filenames.

### Searching Files by Model

```bash
# Find all GPT-4o outputs
ls output_data/interim_output/*gpt-4o*

# Find all Gemini outputs
ls output_data/interim_output/*gemini*

# Compare Step 3 across all models
ls output_data/interim_output/*step3*
```

### Organizing Results

Create model-specific directories:
```bash
cd output_data/interim_output

# Organize by model
mkdir -p by_model/{gpt-4o,gemini,ollama}
mv *gpt-4o* by_model/gpt-4o/
mv *gemini* by_model/gemini/
mv *qwen* by_model/ollama/
```

## Example: Multi-Model Comparison Workflow

### 1. Run with GPT-4o
```yaml
# config.yaml
llm:
  provider: openai
  openai:
    model: gpt-4o-2024-11-20
```

```bash
python 05_active_learning_loop.py
# Creates: iter_*_gpt-4o-2024-11-20_*.json
```

### 2. Run with Gemini
```yaml
# config.yaml
llm:
  provider: gemini
  gemini:
    model: gemini-2.5-flash
```

```bash
python 05_active_learning_loop.py
# Creates: iter_*_gemini-2.5-flash_*.json
```

### 3. Compare Results
```bash
# Compare uncertainty scores
jq '.uncertainty_analysis' output_data/interim_output/*gpt-4o*step3*.json
jq '.uncertainty_analysis' output_data/interim_output/*gemini*step3*.json

# Compare counterfactual quality
jq '.generated_counterfactuals' output_data/interim_output/*gpt-4o*step5*.json
jq '.generated_counterfactuals' output_data/interim_output/*gemini*step5*.json
```

## Backward Compatibility

✅ **Fully backward compatible**
- No config changes required
- Works with existing configs
- Only changes output filenames (not content)
- Old scripts/notebooks may need filename pattern updates

## Testing

Verify the feature works:

```bash
# 1. Run AL loop
python 05_active_learning_loop.py

# 2. Check filenames
ls -la output_data/interim_output/

# 3. Verify model name in filename
# Should see something like:
# iter_01_20251111_143022_gpt-4o-2024-11-20_step1_classifier_training.json
```

Expected output:
```
output_data/interim_output/
├── iter_01_20251111_143022_gpt-4o-2024-11-20_step1_classifier_training.json
├── iter_01_20251111_143022_gpt-4o-2024-11-20_step2_evaluation.json
├── iter_01_20251111_143022_gpt-4o-2024-11-20_step3_uncertainty_selection.json
├── iter_01_20251111_143022_gpt-4o-2024-11-20_step4_oracle_labeling.json
├── iter_01_20251111_143022_gpt-4o-2024-11-20_step5_counterfactual_generation.json
├── iter_01_20251111_143022_gpt-4o-2024-11-20_step6_pool_update.json
└── ...
```

## Implementation Details

### Lines Modified
- **Lines 349-361**: Model extraction and sanitization logic
- **Line 375**: Step 1 filename
- **Line 413**: Step 2 filename
- **Line 433**: Step 2 skipped filename
- **Line 462**: Step 3 filename
- **Line 481**: Step 4 filename
- **Line 506**: Step 5 filename
- **Line 522**: Step 5 skipped filename
- **Line 558**: Step 6 filename

### Files Updated
- `05_active_learning_loop.py` - Main AL loop
- `INTERIM_OUTPUT_MODEL_NAMING.md` - This documentation (NEW)

### No Changes To
- Config files
- Other utility files
- Output CSV files (al_results.csv, final_labeled_pool.csv)
- Checkpoint files

## Status

✅ **IMPLEMENTED** - Model names now included in all interim output filenames

---

**Date**: November 11, 2025  
**Feature**: Model name in interim output filenames  
**Benefit**: Better experiment tracking and multi-model comparison

