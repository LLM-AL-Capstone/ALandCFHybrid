# Old Pattern-Based Pipeline (Archived)

This directory contains the original pattern-based counterfactual generation pipeline that has been replaced with a simplified Active Learning approach.

## Archived Scripts

- **01_data_formatting.py** - Pattern identification and candidate phrase generation
- **02_counterfactual_over_generation.py** - Counterfactual generation using candidate phrases
- **03_counterfactual_filtering.py** - Three-stage filtering (heuristic, semantic, discriminator)
- **04_counterfactual_evaluation.py** - Evaluation of counterfactuals on test set

## Why Archived?

The original pipeline used a complex multi-stage approach:
1. LLM identifies key phrases in examples
2. LLM generates alternative phrases for target labels
3. LLM generates counterfactuals using specific phrases
4. Multi-stage filtering to ensure quality

**New approach:** The current Active Learning implementation uses direct LLM prompting for counterfactual generation without pattern identification or complex filtering. This simplifies the pipeline while maintaining quality through strong prompt engineering.

## Date Archived

November 2025

## Original Approach

For reference on the original pattern-based methodology, see these archived scripts. The approach was effective but computationally expensive and complex to maintain.

