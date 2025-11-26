#!/bin/bash

# Run Full-ICL Oracle Baseline for All Datasets
# This script runs the baseline for multiple datasets sequentially

set -e  # Exit on error

# Activate virtual environment
source ../../.venv/bin/activate

# Datasets to process
# Format: "train_file:test_file:dataset_name:status"
DATASETS=(
    "amazon_polarity_train.csv:amazon_polarity_test.csv:amazon_polarity:done"
    "anli_train.csv:anli_test.csv:anli:pending"
    "sa_cad_train.csv:sa_cad_test.csv:sa_cad:pending"
    "yelp_train.csv:yelp_test.csv:yelp:pending"
    "nli_cad_train.csv:nli_cad_test.csv:nli_cad:pending"
)

echo "================================================================================"
echo "FULL-ICL ORACLE BASELINE - BATCH PROCESSING"
echo "================================================================================"
echo ""
echo "Datasets to process:"
for dataset in "${DATASETS[@]}"; do
    IFS=':' read -r train test name status <<< "$dataset"
    echo "  - $name: $status"
done
echo ""

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    IFS=':' read -r train_file test_file dataset_name status <<< "$dataset"
    
    if [ "$status" == "done" ]; then
        echo "⏭️  Skipping $dataset_name (already completed)"
        echo ""
        continue
    fi
    
    echo "================================================================================"
    echo "Processing: $dataset_name"
    echo "================================================================================"
    echo "Train file: $train_file"
    echo "Test file: $test_file"
    echo ""
    
    # Update config with dataset
    python -c "
import yaml

config_path = 'configs/baseline_config.yaml'

# Read config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update dataset
config['dataset']['train_file'] = '$train_file'
config['dataset']['test_file'] = '$test_file'

# Write back
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f'✓ Config updated for $dataset_name')
"
    
    # Run baseline
    echo "Running baseline..."
    python run_baseline.py
    
    if [ $? -eq 0 ]; then
        echo "✅ $dataset_name completed successfully"
    else
        echo "❌ $dataset_name failed"
        exit 1
    fi
    
    echo ""
    echo "Waiting 5 seconds before next dataset..."
    sleep 5
    echo ""
done

echo "================================================================================"
echo "✅ ALL DATASETS PROCESSED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Results are saved in: output/"
echo ""
echo "To view results:"
echo "  ls -lht output/"
echo ""
