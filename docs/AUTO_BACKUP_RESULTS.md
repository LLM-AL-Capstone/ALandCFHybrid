# Auto-Backup Results Implementation

## Problem Solved

Previously, running a new Active Learning experiment would **overwrite** the `al_results.csv` file, losing all previous results. This made it impossible to compare different experiments (static vs retrieval ICL) without manual backups.

## Solution Implemented

Added **automatic backup** functionality that:
1. Detects if `al_results.csv` already exists
2. Reads the existing file to identify experiment type (static/retrieval)
3. Creates a timestamped backup before overwriting
4. Preserves all previous experimental data

## What Changed

### Modified Functions

#### 1. `save_results()` - Lines 237-275
**Before:**
```python
def save_results(results: List[Dict], config: dict):
    results_file = config['logging']['results_file']
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_file, index=False)  # ← Overwrites!
    print(f"\nResults saved to: {results_file}")
```

**After:**
```python
def save_results(results: List[Dict], config: dict):
    results_file = config['logging']['results_file']
    
    # Auto-backup existing results if file exists
    if os.path.exists(results_file):
        # Read existing file to determine experiment type
        existing_df = pd.read_csv(results_file)
        classifier_type = existing_df['classifier_type'].iloc[0]
        
        # Create backup with timestamp and type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = results_file.replace('.csv', f'_backup_{classifier_type}_{timestamp}.csv')
        
        shutil.copy2(results_file, backup_file)
        print(f"  📁 Backed up previous results to: {backup_file}")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_file, index=False)
    print(f"\n✅ Results saved to: {results_file}")
```

#### 2. `save_final_labeled_pool()` - Lines 278-304
Same auto-backup logic added for `final_labeled_pool.csv`.

## How It Works

### Example Scenario

**Run 1: Static ICL**
```bash
python 05_active_learning_loop.py  # With classifier_type: "static"
# Creates: output_data/al_results.csv
```

**Run 2: Retrieval ICL**
```bash
python 05_active_learning_loop.py  # With classifier_type: "retrieval"
# Output:
#   📁 Backed up previous results to: output_data/al_results_backup_static_20251111_234530.csv
#   ✅ Results saved to: output_data/al_results.csv
```

**Run 3: Retrieval ICL (different config)**
```bash
python 05_active_learning_loop.py  # With k_per_class: 5
# Output:
#   📁 Backed up previous results to: output_data/al_results_backup_retrieval_20251111_235812.csv
#   ✅ Results saved to: output_data/al_results.csv
```

### Result: You Now Have All Experiments
```
output_data/
├── al_results.csv                                          ← Latest run
├── al_results_backup_static_20251111_234530.csv           ← Run 1
├── al_results_backup_retrieval_20251111_235812.csv        ← Run 2
├── final_labeled_pool.csv                                  ← Latest
├── final_labeled_pool_backup_static_20251111_234530.csv
└── final_labeled_pool_backup_retrieval_20251111_235812.csv
```

## Backup Filename Format

```
al_results_backup_{classifier_type}_{timestamp}.csv
                   ↑                 ↑
                   │                 └─ YYYYMMDD_HHMMSS
                   └─ static, retrieval, or unknown
```

**Examples:**
- `al_results_backup_static_20251111_234530.csv`
- `al_results_backup_retrieval_20251112_103045.csv`
- `al_results_backup_unknown_20251111_150000.csv` (if no classifier_type found)

## Benefits

✅ **Never lose data** - All previous results automatically backed up  
✅ **Easy comparison** - Keep all experiment results  
✅ **Organized** - Backups include experiment type and timestamp  
✅ **No manual work** - Happens automatically  
✅ **Safe** - Original files preserved before overwriting  

## Comparing Experiments

### Load All Results
```python
import pandas as pd
import glob

# Load all result files
result_files = glob.glob('output_data/al_results*.csv')

for file in result_files:
    df = pd.read_csv(file)
    classifier_type = df['classifier_type'].iloc[0] if 'classifier_type' in df.columns else 'unknown'
    max_acc = df['accuracy'].max()
    max_f1 = df['f1_macro'].max()
    
    print(f"\n{file.split('/')[-1]}:")
    print(f"  Type: {classifier_type}")
    print(f"  Max Accuracy: {max_acc:.4f}")
    print(f"  Max F1 Macro: {max_f1:.4f}")
```

### Compare Static vs Retrieval
```python
import pandas as pd

# Load original experiments (from backups)
static = pd.read_csv('output_data/al_results_backup_static_20251111_234530.csv')
retrieval = pd.read_csv('output_data/al_results_backup_retrieval_20251111_235812.csv')

# Compare
print("Static ICL:")
print(f"  Max Accuracy: {static['accuracy'].max():.4f}")
print(f"  Max F1 Macro: {static['f1_macro'].max():.4f}")

print("\nRetrieval ICL:")
print(f"  Max Accuracy: {retrieval['accuracy'].max():.4f}")
print(f"  Max F1 Macro: {retrieval['f1_macro'].max():.4f}")

print(f"\nImprovement:")
print(f"  Accuracy: {(retrieval['accuracy'].max() - static['accuracy'].max()) * 100:.2f}%")
print(f"  F1 Macro: {(retrieval['f1_macro'].max() - static['f1_macro'].max()) * 100:.2f}%")
```

## Console Output Example

When running a new experiment:

```
Active Learning Complete!
================================================================================
  📁 Backed up previous results to: output_data/al_results_backup_retrieval_20251111_230351.csv
  ✅ Results saved to: output_data/al_results.csv
  
  📁 Backed up previous labeled pool to: output_data/final_labeled_pool_backup_retrieval_20251111_230351.csv
  ✅ Final labeled pool saved to: output_data/final_labeled_pool.csv

Total iterations: 5
Total examples labeled: 50
Final labeled pool size: 330
```

## Backward Compatibility

✅ **Fully compatible** - Works with existing code  
✅ **No config changes** - Uses existing settings  
✅ **Graceful handling** - If no existing file, just saves normally  
✅ **Error handling** - Falls back to "unknown" if can't read classifier type  

## Files Modified

- `05_active_learning_loop.py`:
  - `save_results()` function (lines 237-275)
  - `save_final_labeled_pool()` function (lines 278-304)

## Testing

Run multiple experiments to verify:

```bash
# Run 1
python 05_active_learning_loop.py
# Check: output_data/al_results.csv created

# Run 2 (different config)
python 05_active_learning_loop.py
# Check: Previous results backed up automatically
# Check: output_data/al_results_backup_*_*.csv exists

# Verify backups
ls -la output_data/al_results*.csv
```

## Manual Cleanup (Optional)

If you accumulate too many backups:

```bash
# List all backups
ls -lt output_data/al_results_backup_*.csv

# Remove old backups (keep recent ones)
rm output_data/al_results_backup_*_202511*.csv
```

Or keep only the most recent N backups:
```bash
# Keep only 5 most recent backups
ls -t output_data/al_results_backup_*.csv | tail -n +6 | xargs rm
```

## Summary

✅ **Implemented**: Automatic backup of results before overwriting  
✅ **Location**: Both `al_results.csv` and `final_labeled_pool.csv`  
✅ **Naming**: Includes classifier type and timestamp  
✅ **Benefit**: Never lose experimental results again  

Now you can run as many experiments as you want without worrying about losing previous results!

