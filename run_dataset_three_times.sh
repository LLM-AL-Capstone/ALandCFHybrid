#!/bin/bash

# Script to run AL experiment 3 times for a given dataset
# Usage: ./run_dataset_three_times.sh <dataset_name>
# Example: ./run_dataset_three_times.sh anli
#          ./run_dataset_three_times.sh yelp

set -e  # Exit on error

# Check if dataset name is provided
if [ -z "$1" ]; then
    echo "Error: Dataset name required"
    echo "Usage: ./run_dataset_three_times.sh <dataset_name>"
    echo "Example: ./run_dataset_three_times.sh anli"
    exit 1
fi

DATASET_NAME="$1"
CONFIG_FILE="config.yaml"
SCRIPT_FILE="05_active_learning_loop.py"
NUM_RUNS=3

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found!"
    exit 1
fi

# Check if script file exists
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: $SCRIPT_FILE not found!"
    exit 1
fi

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "WARNING: Virtual environment not activated!"
    echo "Attempting to activate..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "ERROR: Virtual environment not found!"
        echo "Please create it with: python3 -m venv venv"
        echo "Then activate it: source venv/bin/activate"
        exit 1
    fi
fi

# Backup original config
CONFIG_BACKUP="${CONFIG_FILE}.backup"
cp "$CONFIG_FILE" "$CONFIG_BACKUP"
echo "✓ Backed up config to $CONFIG_BACKUP"

# Update config with dataset name
TRAIN_FILE="${DATASET_NAME}_train.csv"
TEST_FILE="${DATASET_NAME}_test.csv"

# Update train_file and test_file in config.yaml using sed
sed -i.bak "s|train_file:.*|train_file: $TRAIN_FILE  # filename in input_data/ folder|" "$CONFIG_FILE"
sed -i.bak "s|test_file:.*|test_file: $TEST_FILE   # filename in input_data/ folder|" "$CONFIG_FILE"

# Remove sed backup file
rm -f "${CONFIG_FILE}.bak"

echo ""
echo "=========================================="
echo "Running $DATASET_NAME dataset - 3 times"
echo "=========================================="
echo "Train file: $TRAIN_FILE"
echo "Test file: $TEST_FILE"
echo ""

# Run the experiment 3 times
for RUN in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "----------------------------------------"
    echo "Run $RUN of $NUM_RUNS"
    echo "----------------------------------------"
    echo ""
    
    # Run the script (each run will create its own timestamped folder)
    python "$SCRIPT_FILE"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Run $RUN completed successfully"
    else
        echo ""
        echo "✗ Run $RUN failed!"
        # Restore original config
        mv "$CONFIG_BACKUP" "$CONFIG_FILE"
        exit 1
    fi
done

# Restore original config
mv "$CONFIG_BACKUP" "$CONFIG_FILE"
echo ""
echo "=========================================="
echo "All 3 runs completed successfully!"
echo "=========================================="
echo "Each run created its own timestamped folder in output_data/"
echo "Original config has been restored."


