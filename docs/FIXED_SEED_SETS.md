# Fixed Seed Sets for Reproducible Active Learning

## Overview

Fixed seed sets ensure that all experiments on the same dataset start with **identical initial labeled examples**. This is critical for:
- **Reproducibility**: Same results across runs
- **Fair comparison**: Different models/methods start from the same baseline
- **Scientific rigor**: Controlled experiments with consistent starting conditions

## How It Works

### Automatic Creation & Detection
The system automatically manages seed sets based on your `config.yaml` settings:

**Filename Format**: `{dataset}_seed_set_s{seed}_n{per_class}.csv`

Examples:
```
input_data/
├── yelp_train.csv                           ← Training data
├── yelp_train_seed_set_s42_n5.csv          ← Seed=42, 5 per class
├── yelp_train_seed_set_s123_n5.csv         ← Seed=123, 5 per class (different!)
├── emotions_train.csv
├── emotions_train_seed_set_s42_n5.csv      ← Seed=42, 5 per class
└── emotions_train_seed_set_s42_n10.csv     ← Seed=42, 10 per class (different!)
```

**Behavior**:
1. On startup, checks for seed set matching current config (seed + per_class)
2. If found: Uses existing seed set ✅
3. If not found: **Automatically creates it** and saves for future runs ✨
4. All subsequent runs with same config reuse the same seed set

## Creating Fixed Seed Sets

### Automatic (Recommended)
**Just run your experiment!** The system creates seed sets automatically:

```bash
python 05_active_learning_loop.py
```

If no seed set exists for your current configuration (dataset + seed + per_class), it will:
1. Create the seed set
2. Save it with configuration-specific filename
3. Continue with the experiment

### Manual (Optional)
If you want to create seed sets in advance:

```bash
# For current dataset in config
python create_fixed_seed_set.py

# For specific dataset
python create_fixed_seed_set.py --dataset yelp
python create_fixed_seed_set.py --dataset emotions
```

### Output Example
```
================================================================================
Fixed Seed Set Creator
================================================================================

📁 Dataset: yelp_train.csv
🎲 Seed: 42
📊 Samples per class: 5

Loading training data...
  Total examples: 500
  ✓ Shuffled with seed=42

📋 Labels found: ['products', 'price', 'service', 'environment']
  Number of classes: 4

🌱 Creating seed set...
  ✓ products: 5 examples
  ✓ price: 5 examples
  ✓ service: 5 examples
  ✓ environment: 5 examples

✅ Fixed seed set created successfully!
   File: input_data/yelp_train_seed_set.csv
   Total examples: 20

📊 Class distribution:
   products: 5
   price: 5
   service: 5
   environment: 5

🆔 Seed example IDs:
   ['ss361', 'ss374', 'ss155', 'ss377', 'ss124', ...]
```

## Current Seed Sets

### Yelp Dataset
- **File**: `input_data/yelp_train_seed_set.csv`
- **Total examples**: 20
- **Classes**: 4 (products, price, service, environment)
- **Per class**: 5 examples
- **Seed**: 42

**Seed IDs:**
```
ss361, ss374, ss155, ss377, ss124,        # products
ss73, ss104, ss406, ss356, ss475,         # price
ss326, ss385, ss337, ss46, ss373,         # service
ss82, ss158, ss319, ss350, ss394          # environment
```

### Emotions Dataset
- **File**: `input_data/emotions_train_seed_set.csv`
- **Total examples**: 30
- **Classes**: 6 (anger, joy, sadness, love, fear, surprise)
- **Per class**: 5 examples
- **Seed**: 42

**Seed IDs:**
```
ss139429, ss98213, ss675, ss220851, ss85132,           # anger
ss65357, ss168383, ss370767, ss58769, ss136532,       # joy
ss82673, ss104440, ss381044, ss390559, ss320659,      # sadness
ss372673, ss221348, ss230420, ss167870, ss337862,     # love
ss188518, ss188220, ss115449, ss95361, ss207906,      # fear
ss335629, ss283411, ss157062, ss185445, ss148905      # surprise
```

## Usage in Active Learning

### When Running Experiments
```bash
python 05_active_learning_loop.py
```

The system automatically manages seed sets based on your config:

**First run with a configuration:**
```
=== Initializing Data Pools ===
📝 No seed set found for this configuration
   seed=42, per_class=5
   Creating: yelp_train_seed_set_s42_n5.csv
   Labels: ['products', 'price', 'service', 'environment']
     ✓ products: 5 examples
     ✓ price: 5 examples
     ✓ service: 5 examples
     ✓ environment: 5 examples
   ✅ Created seed set: 20 examples
   Saved to: yelp_train_seed_set_s42_n5.csv

Initial labeled pool: 20 examples
  Class distribution: {'products': 5, 'price': 5, 'service': 5, 'environment': 5}
Unlabeled pool: 480 examples
```

**Subsequent runs with same configuration:**
```
=== Initializing Data Pools ===
✅ Using existing seed set: yelp_train_seed_set_s42_n5.csv
   (seed=42, per_class=5)

Initial labeled pool: 20 examples
  Class distribution: {'products': 5, 'price': 5, 'service': 5, 'environment': 5}
Unlabeled pool: 480 examples
```

**Run with different configuration:**
```
=== Initializing Data Pools ===
📝 No seed set found for this configuration
   seed=123, per_class=5
   Creating: yelp_train_seed_set_s123_n5.csv
   ...
```

## Comparison Across Experiments

With fixed seed sets, you can now fairly compare:

### Same Model, Different Eval Methods
```bash
# Experiment 1: Static ICL
# Seed set: ss361, ss374, ss155, ...  (from fixed file)

# Experiment 2: Retrieval ICL
# Seed set: ss361, ss374, ss155, ...  (SAME fixed file)
```
✅ Fair comparison - only difference is eval method

### Different Models, Same Dataset
```bash
# Experiment 1: GPT-4o
# Seed set: ss361, ss374, ss155, ...  (from fixed file)

# Experiment 2: Llama 3.2
# Seed set: ss361, ss374, ss155, ...  (SAME fixed file)
```
✅ Fair comparison - only difference is LLM model

## Best Practices

### 1. Let the System Create Seed Sets
**No manual creation needed!** Just run your experiments:
```bash
python 05_active_learning_loop.py
```

The system automatically creates and reuses seed sets based on your config.

### 2. Change Config for Different Seeds
Want to try a different seed or different number of examples per class?

```yaml
# config.yaml
processing:
  seed: 123  # Changed from 42

active_learning:
  initial_labeled_per_class: 10  # Changed from 5
```

Next run will automatically:
- Detect no seed set exists for this config
- Create `yelp_train_seed_set_s123_n10.csv`
- Use it for all future runs with same config

### 3. Version Control Seed Sets
```bash
# Add to git (multiple configs preserved!)
git add input_data/*_seed_set*.csv
git commit -m "Add seed sets: s42_n5, s42_n10, s123_n5"
```

### 4. Document Configuration in Papers
In your paper/report, specify the configuration:
- Seed value (e.g., seed=42)
- Examples per class (e.g., 5)
- Total seed set size (e.g., 20 for 4-class problem)

Example:
> "All experiments used a fixed seed set of 20 examples (5 per class) 
> selected via stratified sampling with random seed 42, ensuring 
> identical starting conditions across all runs."

### 5. Never Modify Seed Set Files
Once created, **do not manually edit** seed set files. 
If you need different seeds, change `config.yaml` and run again - 
a new seed set file will be automatically created!

### 6. Share Seed Sets with Collaborators
Include seed set files when sharing code:
```
project/
├── input_data/
│   ├── yelp_train.csv
│   ├── yelp_train_seed_set_s42_n5.csv     ← Share these!
│   ├── yelp_train_seed_set_s123_n5.csv    ← All configs!
│   └── ...
```

## Verification

### Check Which Seed Set is Being Used
Look for this in the output:
```
✅ Using existing seed set: yelp_train_seed_set_s42_n5.csv
   (seed=42, per_class=5)
```

Or if creating a new one:
```
📝 No seed set found for this configuration
   seed=42, per_class=5
   Creating: yelp_train_seed_set_s42_n5.csv
```

### List All Seed Sets
```bash
ls -lh input_data/*_seed_set*.csv
```

Example output:
```
input_data/yelp_train_seed_set_s42_n5.csv    # Seed=42, 5/class
input_data/yelp_train_seed_set_s123_n5.csv   # Seed=123, 5/class
input_data/emotions_train_seed_set_s42_n5.csv
```

### Verify Seed IDs Match
```python
import pandas as pd

seed_df = pd.read_csv('input_data/yelp_train_seed_set_s42_n5.csv')
print("Seed IDs:", seed_df['id'].tolist())
```

### Compare Across Runs
```bash
# Run 1 with seed=42
python 05_active_learning_loop.py
# Check: Uses yelp_train_seed_set_s42_n5.csv

# Run 2 with same config (seed=42)
python 05_active_learning_loop.py
# Verify: Uses SAME yelp_train_seed_set_s42_n5.csv

# Run 3 with different seed (seed=123)
# (change config.yaml first)
python 05_active_learning_loop.py
# Check: Creates/uses yelp_train_seed_set_s123_n5.csv (different file!)
```

## Technical Details

### Seed Set File Format
Standard CSV with same columns as training data:
```csv
id,example,Label
ss361,"fresh and good quality seafood...",products
ss374,"the black eyes peas...",products
...
```

### Selection Algorithm
1. Load training data
2. Shuffle with configured random seed (e.g., 42)
3. For each class:
   - Take first N examples (stratified sampling)
4. Save to `*_seed_set.csv`

### Automatic Creation Behavior
When no seed set exists for current config:
- System creates it automatically
- Saves with configuration-encoded filename
- Shows creation progress
- Continues with experiment seamlessly

## Troubleshooting

### Different Seed IDs Across Runs
Check:
1. Is `processing.seed` the same in config.yaml?
2. Is `initial_labeled_per_class` the same?
3. Check which seed set file is being loaded (filename in output)

Example:
```bash
# Run 1: seed=42, per_class=5
# Uses: yelp_train_seed_set_s42_n5.csv

# Run 2: seed=123, per_class=5 (DIFFERENT!)
# Uses: yelp_train_seed_set_s123_n5.csv (different file!)
```

### Seed Set Has Wrong Classes
- Ensure `dataset.exclude_labels` is set correctly in config.yaml
- Delete the incorrect seed set file
- Run again - it will be recreated correctly

```bash
# Delete incorrect seed set
rm input_data/yelp_train_seed_set_s42_n5.csv

# Run again to recreate
python 05_active_learning_loop.py
```

### Want to Force Regenerate Seed Set
Simply delete the seed set file and run again:
```bash
rm input_data/yelp_train_seed_set_s42_n5.csv
python 05_active_learning_loop.py  # Will recreate
```

## References

This implementation follows best practices from:
- Settles, B. (2009). "Active Learning Literature Survey"
- Consistent experimental design in ML research
- Reproducibility guidelines for scientific computing

---

*Last updated: November 18, 2024*

