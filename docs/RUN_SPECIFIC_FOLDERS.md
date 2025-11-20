# Run-Specific Folder Organization

## Overview

Each Active Learning run now creates its own dedicated folder with all outputs organized together. This makes it easy to compare experiments and prevents file overwrites.

## Folder Structure

### Run Folder Naming
Format: `{timestamp}_{model}_{dataset}_{evalmethod}_s{seed}_n{per_class}/`

Example: `20251113_215734_gpt-4o-2024-11-20_yelp_retrieval_s42_n5/`

Components:
- **Timestamp**: `YYYYMMDD_HHMMSS` - When the run started
- **Model**: Sanitized LLM model name (e.g., `gpt-4o-2024-11-20`, `llama3-2`)
- **Dataset**: Dataset name without `_train.csv` suffix (e.g., `yelp`, `emotions`)
- **Evalmethod**: Classifier type - `static` or `retrieval`
- **Seed**: Random seed value (e.g., `s42`, `s123`)
- **Per_class**: Initial examples per class (e.g., `n5`, `n10`) - matches seed set used

### Contents of Each Run Folder

```
output_data/
├── 20251113_215734_gpt-4o-2024-11-20_yelp_retrieval_s42_n5/
│   ├── config.yaml                             # Config snapshot for this run
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
├── 20251113_220015_gpt-4o-2024-11-20_emotions_static_s42_n5/  # Different dataset
│   └── ...
└── 20251114_093045_llama3-2_yelp_retrieval_s123_n5/           # Different model & seed
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
- Config snapshot (exact settings used)
- Results
- Labeled pool
- Interim logs
- Checkpoints

### 5. **Perfect Reproducibility**
Each run folder contains:
- `config.yaml` - Exact configuration snapshot
- Folder name encodes key parameters (seed, per_class, etc.)
- All outputs generated with that exact config

You can reproduce any run by using its config.yaml!

## Examples

### Run with Different Models
```bash
# Run 1: GPT-4o with retrieval, seed=42, 5 per class
# Creates: output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval_s42_n5/

# Run 2: Llama 3.2 with retrieval, seed=42, 5 per class
# Creates: output_data/20251113_220540_llama3-2_yelp_retrieval_s42_n5/
```

### Run with Different Datasets
```bash
# Run 1: Yelp dataset with retrieval, seed=42, 5 per class
# Creates: output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval_s42_n5/

# Run 2: Emotions dataset with retrieval, seed=42, 5 per class
# Creates: output_data/20251113_221045_gpt-4o-2024-11-20_emotions_retrieval_s42_n5/
```

### Run with Different Evaluation Strategies
```bash
# Run 1: Retrieval ICL, seed=42, 5 per class
# Creates: output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval_s42_n5/

# Run 2: Static ICL, seed=42, 5 per class
# Creates: output_data/20251113_221530_gpt-4o-2024-11-20_yelp_static_s42_n5/
```

### Run with Different Seeds
```bash
# Run 1: Retrieval ICL, seed=42, 5 per class
# Creates: output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval_s42_n5/

# Run 2: Retrieval ICL, seed=123, 5 per class
# Creates: output_data/20251113_222015_gpt-4o-2024-11-20_yelp_retrieval_s123_n5/
```

### Run with Different Initial Sizes
```bash
# Run 1: Retrieval ICL, seed=42, 5 per class
# Creates: output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval_s42_n5/

# Run 2: Retrieval ICL, seed=42, 10 per class
# Creates: output_data/20251113_222015_gpt-4o-2024-11-20_yelp_retrieval_s42_n10/
```

Now you can clearly see all experimental parameters directly from the folder name!

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

### All Runs with Specific Seed
```bash
# All runs with seed=42
ls output_data/ | grep "_s42"

# All runs with seed=123
ls output_data/ | grep "_s123"
```

### All Runs with Specific Initial Size
```bash
# All runs with 5 examples per class
ls output_data/ | grep "_n5"

# All runs with 10 examples per class
ls output_data/ | grep "_n10"
```

### Specific Date
```bash
ls output_data/ | grep "20251113"
```

### Compare Different Configurations
```bash
# Compare retrieval vs static on same model/dataset/seed/size
ls output_data/ | grep "gpt-4o.*_yelp.*_s42_n5"

# Compare different seeds on same model/dataset/eval/size
ls output_data/ | grep "gpt-4o.*_yelp_static.*_n5"

# Compare different initial sizes on same model/dataset/eval/seed
ls output_data/ | grep "gpt-4o.*_yelp_retrieval_s42"
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
mv output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval_s42_n5/ archive_runs/
```

## Reproducing a Run

### Check Configuration Used
```bash
# View the exact config used for a specific run
cat output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval_s42_n5/config.yaml
```

### Reproduce the Exact Run
```bash
# Copy the config from the run folder
cp output_data/20251113_215734_gpt-4o-2024-11-20_yelp_retrieval_s42_n5/config.yaml config.yaml

# Run again with identical settings
python 05_active_learning_loop.py

# Will use same seed set (s42_n5) and all other settings!
```

### Compare Configurations
```bash
# Compare configs from two different runs
diff output_data/run1/config.yaml output_data/run2/config.yaml

# See exactly what changed between experiments
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
4. Extracts seed value from `processing.seed`
5. Extracts initial per class from `active_learning.initial_labeled_per_class`
6. Creates timestamped folder with all info
7. Organizes all outputs inside

Just run:
```bash
python 05_active_learning_loop.py
```

And everything is automatically organized!

