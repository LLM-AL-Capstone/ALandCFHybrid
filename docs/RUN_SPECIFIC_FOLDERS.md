# Run-Specific Folder Organization

## Overview

Each Active Learning run now creates its own dedicated folder with all outputs organized together. This makes it easy to compare experiments and prevents file overwrites.

## Folder Structure

### Run Folder Naming
Format: `{timestamp}_{model}_{dataset}_{evalmethod}/`

Example: `20251113_215734_gpt-4o-2024-11-20_yelp_retrieval/`

Components:
- **Timestamp**: `YYYYMMDD_HHMMSS` - When the run started
- **Model**: Sanitized LLM model name (e.g., `gpt-4o-2024-11-20`, `llama3-2`)
- **Dataset**: Dataset name without `_train.csv` suffix (e.g., `yelp`, `emotions`)
- **Evalmethod**: Classifier type - `static` or `retrieval`

### Contents of Each Run Folder

```
output_data/
├── 20251113_215734_gpt-4o-2024-11-20_yelp_retrieval/
│   ├── al_results.csv                          # Final results summary
│   ├── final_labeled_pool.csv                  # All labeled data (real + CFs)
│   ├── interim_output/                         # Step-by-step iteration logs
│   │   ├── iter_01_20251113_215735_gpt-4o-2024-11-20_step1_classifier_training.json
│   │   ├── iter_01_20251113_215735_gpt-4o-2024-11-20_step2_evaluation.json
│   │   ├── iter_01_20251113_215735_gpt-4o-2024-11-20_step3_uncertainty_selection.json
│   │   ├── iter_01_20251113_215735_gpt-4o-2024-11-20_step4_oracle_labeling.json
│   │   ├── iter_01_20251113_215735_gpt-4o-2024-11-20_step5_counterfactual_generation.json
│   │   ├── iter_01_20251113_215735_gpt-4o-2024-11-20_step6_pool_update.json
│   │   ├── iter_02_...
│   │   └── ...
│   └── checkpoints/                            # AL state checkpoints
│       ├── checkpoint_iter_5.json
│       ├── checkpoint_iter_10.json
│       └── ...
├── 20251113_220015_gpt-4o-2024-11-20_emotions_static/  # Different dataset & eval method
│   └── ...
└── 20251114_093045_llama3-2_yelp_retrieval/            # Different model
    └── ...
```

## Benefits

### 1. **No File Overwrites**
Each run gets its own folder - no more lost experiments!

### 2. **Easy Comparison**
Compare different runs side-by-side:
```bash
# Compare results from two runs
diff output_data/20251113_215734_*/al_results.csv
```

### 3. **Self-Documenting**
Folder name tells you:
- When it ran
- Which model was used
- Which dataset was used
- No need to open files to identify experiments

### 4. **Clean Organization**
All outputs for a single run are in one place:
- Results
- Labeled pool
- Interim logs
- Checkpoints

## Examples

### Run with Different Models
```bash
# Run 1: GPT-4o with retrieval
# Creates: output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval/

# Run 2: Llama 3.2 with retrieval
# Creates: output_data/20251113_220540_llama3-2_yelp_retrieval/
```

### Run with Different Datasets
```bash
# Run 1: Yelp dataset with retrieval
# Creates: output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval/

# Run 2: Emotions dataset with retrieval
# Creates: output_data/20251113_221045_gpt-4o-2024-11-20_emotions_retrieval/
```

### Run with Different Evaluation Strategies
```bash
# Run 1: Retrieval ICL (k_per_class=3)
# Creates: output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval/

# Run 2: Static ICL
# Creates: output_data/20251113_221530_gpt-4o-2024-11-20_yelp_static/
```

Now you can clearly see the evaluation strategy directly from the folder name!

## Finding Your Results

### Latest Run
The most recent run will have the latest timestamp:
```bash
ls -t output_data/ | head -1
```

### All Runs for a Model
```bash
ls output_data/ | grep "gpt-4o"
```

### All Runs for a Dataset
```bash
ls output_data/ | grep "_yelp_"
```

### All Runs with Specific Eval Method
```bash
# All retrieval-based runs
ls output_data/ | grep "_retrieval"

# All static runs
ls output_data/ | grep "_static"
```

### Specific Date
```bash
ls output_data/ | grep "20251113"
```

### Compare Different Eval Methods (Same Setup)
```bash
# Compare retrieval vs static on same model/dataset
ls output_data/ | grep "gpt-4o.*_yelp"
```

## Cleanup

### Remove Old Runs
```bash
# Remove runs older than 7 days
find output_data/ -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \;
```

### Archive Important Runs
```bash
# Move successful runs to archive
mkdir -p archive_runs/
mv output_data/20251113_215734_gpt-4o-2024-11-20_yelp/ archive_runs/
```

## Migration from Old System

### Old Structure (Before)
```
output_data/
├── al_results.csv          # Overwritten each run!
├── final_labeled_pool.csv  # Overwritten each run!
├── interim_output/         # Mixed from all runs
│   ├── iter_01_20251110_..._step1_...json
│   ├── iter_01_20251111_..._step1_...json  # Different runs mixed!
│   └── ...
└── al_checkpoints/         # Mixed from all runs
    └── ...
```

### New Structure (Now)
```
output_data/
├── 20251113_215734_gpt-4o-2024-11-20_yelp_retrieval/  # Run 1: All files together
│   ├── al_results.csv
│   ├── final_labeled_pool.csv
│   ├── interim_output/
│   └── checkpoints/
└── 20251113_220015_gpt-4o-2024-11-20_emotions_static/  # Run 2: Different dataset & eval
    ├── al_results.csv
    ├── final_labeled_pool.csv
    ├── interim_output/
    └── checkpoints/
```

## Configuration

No configuration needed! The system automatically:
1. Extracts model name from `config.yaml` (`llm.provider.model`)
2. Extracts dataset name from `train_file`
3. Extracts evaluation method from `evaluation.classifier_type`
4. Creates timestamped folder with all info
5. Organizes all outputs inside

Just run:
```bash
python 05_active_learning_loop.py
```

And everything is automatically organized!

