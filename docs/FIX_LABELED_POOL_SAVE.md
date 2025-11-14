# Fix Applied: Labeled Pool Saving Issue

## Problem Identified

The `final_labeled_pool.csv` file was incomplete (only 91 examples instead of expected 150+) because:

1. **KeyboardInterrupt handler didn't save labeled pool** - When user interrupted (Ctrl+C) or script crashed, the labeled pool CSV was never saved
2. **No general exception handler** - API errors, crashes, or other exceptions would lose all accumulated data
3. **Only saved at script completion** - If the script didn't reach the end, no CSV was written

## Root Cause

Looking at the data:
- `al_results.csv` showed 2 completed iterations (150 examples expected)
- `final_labeled_pool.csv` only had 91 examples (from iteration 1)
- Iteration 3 started but didn't complete (interim outputs show only steps 1-2)
- The script crashed/was interrupted before reaching the final save

## Changes Made

### 1. Added Incremental Saving (Line 588-590)
```python
# Save labeled pool after each iteration (prevents data loss)
save_final_labeled_pool(labeled_pool, config)
print(f"  💾 Labeled pool saved (iteration {iteration})")
```

**Impact**: Now saves the CSV after EVERY iteration, ensuring you never lose progress.

### 2. Enhanced KeyboardInterrupt Handler (Lines 592-600)
```python
except KeyboardInterrupt:
    print("\n\n⚠ Interrupted by user!")
    print(f"Completed {iteration} iterations")
    print(f"Saving progress...")
    save_checkpoint(iteration, labeled_pool, unlabeled_pool, results, config)
    save_results(results, config)
    save_final_labeled_pool(labeled_pool, config)  # ✅ NOW SAVES LABELED POOL
    print(f"💾 All progress saved successfully!")
    sys.exit(0)
```

**Impact**: When you press Ctrl+C, all data is now properly saved.

### 3. Added General Exception Handler (Lines 602-613)
```python
except Exception as e:
    print(f"\n\n❌ Error occurred: {e}")
    print(f"Completed {iteration} iterations before error")
    print(f"Saving progress...")
    try:
        save_checkpoint(iteration, labeled_pool, unlabeled_pool, results, config)
        save_results(results, config)
        save_final_labeled_pool(labeled_pool, config)  # ✅ SAVES ON ANY ERROR
        print(f"💾 Progress saved successfully!")
    except Exception as save_error:
        print(f"⚠️ Warning: Could not save progress: {save_error}")
    raise
```

**Impact**: API errors, rate limits, crashes, etc. will now save progress before failing.

## Benefits

✅ **No more data loss** - CSV updated after every iteration  
✅ **Safe interruption** - Ctrl+C now saves everything properly  
✅ **Crash recovery** - Any exception saves progress before exiting  
✅ **Better feedback** - Clear messages about what's being saved  
✅ **Debugging friendly** - Errors are re-raised after saving for investigation  

## Testing

To verify the fix works:

1. **Normal run**: 
   ```bash
   python 05_active_learning_loop.py
   ```
   Watch for "💾 Labeled pool saved (iteration N)" after each iteration

2. **Interrupt test**: 
   - Start the script
   - Press Ctrl+C during an iteration
   - Check that `final_labeled_pool.csv` has data from all completed iterations

3. **Verify incremental updates**:
   ```bash
   # In another terminal, watch the file size grow
   watch -n 2 "wc -l output_data/final_labeled_pool.csv"
   ```

## Expected Behavior Now

### During Iteration 1:
```
Iteration 1/50
...
[Step 6/6] Updating data pools...
  💾 Labeled pool saved (iteration 1)  ← NEW
```
File will have: ~90 examples (30 initial + 10 real + 50 CF)

### During Iteration 2:
```
Iteration 2/50
...
[Step 6/6] Updating data pools...
  💾 Labeled pool saved (iteration 2)  ← NEW
```
File will have: ~150 examples (90 + 10 real + 50 CF)

### On Interrupt:
```
^C
⚠ Interrupted by user!
Completed 2 iterations
Saving progress...
  Checkpoint saved: output_data/al_checkpoints/checkpoint_iter_2.json
Results saved to: output_data/al_results.csv
Final labeled pool saved to: output_data/final_labeled_pool.csv
💾 All progress saved successfully!
```

## File Location

Modified file: `05_active_learning_loop.py`
- Lines 588-613: Enhanced saving logic
- No other changes needed

## Backward Compatibility

✅ Fully backward compatible - no config changes needed  
✅ Existing checkpoints still work  
✅ No breaking changes to API or outputs  
✅ Only adds safety features  

## Status

🎉 **FIXED** - The labeled pool will now be saved correctly, even if the script is interrupted or crashes.

---

**Date**: November 11, 2025  
**Issue**: Incomplete final_labeled_pool.csv due to missing save on interruption  
**Resolution**: Added incremental saving + enhanced exception handlers

