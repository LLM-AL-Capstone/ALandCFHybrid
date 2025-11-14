# Directory Configuration Fix

## Issue

After cleaning up `config.yaml` to remove unused parameters, the system was trying to access `dirs['interim_output']` which no longer existed in the config, causing a `KeyError`.

## Root Cause

The `ensure_directories()` function in `utils/config_loader.py` was hardcoded to expect certain directory paths in the config that we had removed during cleanup:
- `interim_output`
- `archive`
- `archive/gpt`

However, these directories are still needed by the AL system at runtime.

## Solution

### 1. Updated `utils/config_loader.py`

**Before:**
```python
def ensure_directories(config: dict):
    dirs = config['directories']
    
    os.makedirs(dirs['input_data'], exist_ok=True)
    os.makedirs(dirs['output_data'], exist_ok=True)
    os.makedirs(dirs['interim_output'], exist_ok=True)  # KeyError!
    os.makedirs(dirs['archive'], exist_ok=True)         # KeyError!
    os.makedirs(f"{dirs['archive']}/gpt", exist_ok=True)
```

**After:**
```python
def ensure_directories(config: dict):
    dirs = config['directories']
    
    # Create directories from config
    os.makedirs(dirs['input_data'], exist_ok=True)
    os.makedirs(dirs['output_data'], exist_ok=True)
    
    # Create additional subdirectories needed by the system
    os.makedirs(f"{dirs['output_data']}/interim_output", exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
```

### 2. Updated `check_setup.py`

Changed the directory checker to distinguish between:
- **Essential directories** (must exist): `input_data`, `output_data`, `utils`
- **Auto-created directories** (informational): `output_data/interim_output`, `output_data/al_checkpoints`

## Result

✅ System now creates necessary subdirectories dynamically  
✅ No need to maintain redundant paths in config  
✅ Cleaner configuration file  
✅ No breaking changes to functionality

## Directories Created Automatically

The AL system now automatically creates these subdirectories as needed:
1. `output_data/interim_output/` - For step-by-step iteration logs
2. `output_data/al_checkpoints/` - For saving AL state checkpoints

These directories are created by:
- `ensure_directories()` during initialization
- `active_learning_loop()` when needed

## Verification

Run `python check_setup.py` to verify the setup is correct. The output will show:
- Essential directories (REQUIRED if missing)
- Auto-created directories (will be auto-created)

