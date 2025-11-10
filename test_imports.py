#!/usr/bin/env python3
"""
Simple test to verify all components can be imported.
Run this before running the full Active Learning loop.
"""

print("Testing imports...")

try:
    from utils import (
        load_config,
        ensure_directories,
        load_dataset,
        get_unique_labels,
        shuffle_dataframe,
        get_llm_provider,
        SimpleICLClassifier,
        get_oracle,
        select_uncertain_examples,
        generate_counterfactuals_batch
    )
    print("✓ All utility imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score
    print("✓ All dependency imports successful")
except ImportError as e:
    print(f"✗ Dependency error: {e}")
    print("Run: pip install -r requirements.txt")
    exit(1)

print("\n=== Testing Configuration ===")
try:
    config = load_config()
    print(f"✓ Config loaded")
    print(f"  LLM Provider: {config['llm']['provider']}")
    print(f"  Dataset: {config['dataset']['train_file']}")
    print(f"  AL Enabled: {config['active_learning']['enabled']}")
except Exception as e:
    print(f"✗ Config error: {e}")
    exit(1)

print("\n=== Testing Directory Structure ===")
try:
    ensure_directories(config)
    print("✓ Directories verified/created")
except Exception as e:
    print(f"✗ Directory error: {e}")
    exit(1)

print("\n=== All Tests Passed! ===")
print("\nYou can now run the Active Learning loop:")
print("  python 05_active_learning_loop.py")

