# Documentation Index

This directory contains all documentation for the Active Learning with Counterfactual Augmentation system.

## 📖 Table of Contents

### 🚀 Getting Started
- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - Overview of the complete Active Learning implementation
- **[INTERIM_OUTPUTS_GUIDE.md](INTERIM_OUTPUTS_GUIDE.md)** - Guide to understanding interim output files

### 🔧 System Features

#### Core Enhancements
- **[ENHANCEMENTS_IMPLEMENTED.md](ENHANCEMENTS_IMPLEMENTED.md)** - Summary of all implemented enhancements
- **[ENHANCEMENT_SUMMARY.md](ENHANCEMENT_SUMMARY.md)** - High-level feature summary
- **[COMPREHENSIVE_LOGGING_IMPLEMENTED.md](COMPREHENSIVE_LOGGING_IMPLEMENTED.md)** - Detailed logging system

#### Evaluation & Classification
- **[RETRIEVAL_ICL_IMPLEMENTATION.md](RETRIEVAL_ICL_IMPLEMENTATION.md)** - Retrieval-based In-Context Learning classifier
- **[EVALUATION_METADATA_TRACKING.md](EVALUATION_METADATA_TRACKING.md)** - Evaluation configuration tracking in outputs
- **[LOGPROBS_ENHANCEMENT.md](LOGPROBS_ENHANCEMENT.md)** - OpenAI logprobs for accurate uncertainty estimation

#### Active Learning Improvements
- **[EARLY_STOPPING_F1_MACRO.md](EARLY_STOPPING_F1_MACRO.md)** - Changed early stopping metric from accuracy to F1 Macro

### 🐛 Bug Fixes & Improvements
- **[FIX_LABELED_POOL_SAVE.md](FIX_LABELED_POOL_SAVE.md)** - Fixed labeled pool saving on interruption
- **[INTERIM_OUTPUT_MODEL_NAMING.md](INTERIM_OUTPUT_MODEL_NAMING.md)** - Added LLM model name to interim filenames
- **[AUTO_BACKUP_RESULTS.md](AUTO_BACKUP_RESULTS.md)** - Automatic backup system for results files (deprecated - replaced by run folders)
- **[DIRECTORY_FIX.md](DIRECTORY_FIX.md)** - Fixed directory configuration after cleanup

### ⚙️ Configuration
- **[CONFIG_CLEANUP.md](CONFIG_CLEANUP.md)** - Cleaned up unused config parameters
- **[RUN_SPECIFIC_FOLDERS.md](RUN_SPECIFIC_FOLDERS.md)** - Run-specific folder organization
- **[FIXED_SEED_SETS.md](FIXED_SEED_SETS.md)** - Fixed seed sets for reproducibility (NEW!)

## 📁 File Organization

### Root Directory
- `README.md` - Main project README (kept in root for visibility)

### docs/ (This Directory)
All documentation files organized here for easy reference.

### Other Locations
- `archive/old_pattern_pipeline/README_OLD.md` - Documentation for archived pattern-based pipeline
- `input_data/README.md` - Guide for input data format

## 🔍 Quick Reference

### For New Users
1. Start with **IMPLEMENTATION_COMPLETE.md**
2. Read **INTERIM_OUTPUTS_GUIDE.md** to understand outputs
3. Check **RUN_SPECIFIC_FOLDERS.md** to understand folder structure

### For Development
- **ENHANCEMENTS_IMPLEMENTED.md** - Feature reference
- **RETRIEVAL_ICL_IMPLEMENTATION.md** - Retrieval classifier details
- **CONFIG_CLEANUP.md** - Current config parameters

### For Troubleshooting
- **DIRECTORY_FIX.md** - Directory configuration issues
- **FIX_LABELED_POOL_SAVE.md** - Data persistence issues
- **COMPREHENSIVE_LOGGING_IMPLEMENTED.md** - Logging system details

## 📝 Adding New Documentation

When creating new documentation:
1. Save it in this `docs/` folder
2. Use descriptive, uppercase filenames with underscores (e.g., `NEW_FEATURE_NAME.md`)
3. Update this INDEX.md file with a link and description
4. Keep README.md in the root directory only

## 🗂️ Documentation Categories

- **Implementation** (4 files) - How features are implemented
- **Enhancements** (7 files) - Feature improvements and additions
- **Fixes** (4 files) - Bug fixes and corrections

**Total:** 15 documentation files + this index

---

*Last updated: November 13, 2024*

