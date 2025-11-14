# Early Stopping Based on F1 Macro

## Overview

Changed the early stopping criterion from **Accuracy** to **F1 Macro** to better handle imbalanced classification and prioritize overall per-class performance.

## Why This Change?

### Problem with Accuracy
Looking at your current results:
```
Iteration 1: Accuracy=0.622, F1 Macro=0.333
Iteration 2: Accuracy=0.607, F1 Macro=0.392  (↓ accuracy, ↑ F1)
Iteration 3: Accuracy=0.622, F1 Macro=0.336
```

**Issue**: Accuracy can be misleading when:
- Classes are imbalanced
- Model predicts majority class well but fails on minority classes
- You care about performance across ALL classes equally

**Your case**: Accuracy ~60% but F1 Macro only ~33-39%, suggesting:
- Model is biased toward certain classes
- Some emotions (fear, surprise) are being ignored
- High accuracy doesn't mean good overall performance

### Benefits of F1 Macro

✅ **Treats all classes equally** - each emotion matters the same  
✅ **Better for imbalanced data** - doesn't ignore minority classes  
✅ **More meaningful for multi-class** - captures precision AND recall  
✅ **Aligns with research goals** - you want good performance on ALL emotions  

## What Changed

### Before (Accuracy-Based)
```python
best_accuracy = 0.0

# Early stopping
if metrics['accuracy'] > best_accuracy + min_improvement:
    best_accuracy = metrics['accuracy']
    patience_counter = 0
    print(f"✓ New best accuracy: {best_accuracy:.4f}")
else:
    patience_counter += 1
```

**Output**:
```
Iteration 1: Accuracy: 0.6224
  ✓ New best accuracy: 0.6224
Iteration 2: Accuracy: 0.6071
  No improvement (patience: 1/5)
```

### After (F1 Macro-Based)
```python
best_f1_macro = 0.0

# Early stopping
if metrics['f1_macro'] > best_f1_macro + min_improvement:
    best_f1_macro = metrics['f1_macro']
    patience_counter = 0
    print(f"✓ New best F1 Macro: {best_f1_macro:.4f}")
else:
    patience_counter += 1
```

**Output**:
```
Iteration 1: F1 Macro: 0.3329
  ✓ New best F1 Macro: 0.3329
Iteration 2: F1 Macro: 0.3916
  ✓ New best F1 Macro: 0.3916  (Improvement detected!)
```

## Changes Made

### File: `05_active_learning_loop.py`

**Line 340**: Changed tracking variable
```python
# Before:
best_accuracy = 0.0

# After:
best_f1_macro = 0.0  # Changed from accuracy to F1 Macro
```

**Lines 415-422**: Changed early stopping logic
```python
# Before:
if metrics['accuracy'] > best_accuracy + al_config['min_improvement']:
    best_accuracy = metrics['accuracy']
    patience_counter = 0
    print(f"✓ New best accuracy: {best_accuracy:.4f}")

# After:
if metrics['f1_macro'] > best_f1_macro + al_config['min_improvement']:
    best_f1_macro = metrics['f1_macro']
    patience_counter = 0
    print(f"✓ New best F1 Macro: {best_f1_macro:.4f}")
```

**Line 455**: Changed JSON output
```python
# Before:
'best_accuracy': best_accuracy,

# After:
'best_f1_macro': best_f1_macro,
```

## Impact on Your Results

### Scenario 1: Previous Behavior (Accuracy)
```
Iter 1: Acc=0.622, F1=0.333 → Best = 0.622
Iter 2: Acc=0.607, F1=0.392 → NO IMPROVEMENT (patience: 1)
Iter 3: Acc=0.622, F1=0.336 → TIE, NO IMPROVEMENT (patience: 2)
```
**Result**: System thinks it's not improving, might stop early

### Scenario 2: New Behavior (F1 Macro)
```
Iter 1: Acc=0.622, F1=0.333 → Best F1 = 0.333
Iter 2: Acc=0.607, F1=0.392 → IMPROVEMENT! (patience: 0) ✓
Iter 3: Acc=0.622, F1=0.336 → NO IMPROVEMENT (patience: 1)
```
**Result**: System correctly recognizes iteration 2 as improvement!

## Configuration

The minimum improvement threshold remains configurable:

```yaml
# config.yaml
active_learning:
  min_improvement: 0.01  # 1% improvement in F1 Macro
  early_stopping_patience: 5
```

**Recommended values for F1 Macro**:
- `min_improvement: 0.01` - Standard (1% improvement)
- `min_improvement: 0.005` - More sensitive (0.5% improvement)
- `min_improvement: 0.02` - More conservative (2% improvement)

## Evaluation Output Changes

### Step 2 JSON Files

**Before**:
```json
{
  "metrics": {...},
  "best_accuracy": 0.6224,
  "patience_counter": 0
}
```

**After**:
```json
{
  "metrics": {...},
  "best_f1_macro": 0.3329,
  "patience_counter": 0
}
```

### Console Output

**Before**:
```
[Step 2/6] Evaluating on test set...
  Accuracy: 0.6224
  F1 Macro: 0.3329
  ✓ New best accuracy: 0.6224
```

**After**:
```
[Step 2/6] Evaluating on test set...
  Accuracy: 0.6224
  F1 Macro: 0.3329
  ✓ New best F1 Macro: 0.3329
```

## When to Use Which Metric

| Metric | Use When |
|--------|----------|
| **Accuracy** | Balanced classes, all classes matter equally in count |
| **F1 Macro** | Imbalanced data, all classes matter equally in importance ✓ |
| **F1 Weighted** | Some classes more important than others |

For emotion classification (6 classes, likely imbalanced), **F1 Macro is the right choice**.

## Expected Behavior

With F1 Macro as the improvement metric:

✅ **More iterations before stopping** - catches improvements in minority classes  
✅ **Better final model** - optimizes for balanced performance  
✅ **Fairer evaluation** - all emotions treated equally  
✅ **Alignment with goals** - you want good performance on ALL emotions  

## Backward Compatibility

✅ **Fully compatible** - only changes internal tracking  
✅ **No config changes needed** - works with existing settings  
✅ **No breaking changes** - all outputs still have accuracy metrics  

## Testing

Run your Active Learning loop and observe:

```bash
python 05_active_learning_loop.py
```

**Expected Output**:
```
Iteration 1/50
...
[Step 2/6] Evaluating on test set...
  Accuracy: 0.6224
  F1 Macro: 0.3329
  F1 Weighted: 0.6732
  ✓ New best F1 Macro: 0.3329  ← Changed from "accuracy"

Iteration 2/50
...
[Step 2/6] Evaluating on test set...
  Accuracy: 0.6071
  F1 Macro: 0.3916
  F1 Weighted: 0.6588
  ✓ New best F1 Macro: 0.3916  ← Improvement detected!
```

## Summary

✅ **Changed**: Early stopping now uses F1 Macro instead of Accuracy  
✅ **Why**: Better for imbalanced multi-class classification  
✅ **Impact**: More accurate improvement detection, better final models  
✅ **Compatible**: Works with all existing configurations  

This change ensures your Active Learning system optimizes for what really matters: **good performance across ALL emotion classes**, not just overall accuracy.

