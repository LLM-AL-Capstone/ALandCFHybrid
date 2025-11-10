#!/bin/bash

# run_all.sh - Run the entire LLM-VT-AL counterfactual generation pipeline
# Version: 2.0

set -e  # Exit on error
trap 'echo ""; echo "Pipeline interrupted. Progress has been saved."; exit 130' INT TERM

echo "=========================================="
echo "LLM-VT-AL Pipeline Execution"
echo "=========================================="
echo ""

# Function to display usage
usage() {
    echo "Usage: ./run_all.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-01    Skip Script 01 (pattern identification)"
    echo "  --skip-02    Skip Script 02 (counterfactual generation)"
    echo "  --skip-03    Skip Script 03 (filtering)"
    echo "  --skip-04    Skip Script 04 (evaluation)"
    echo "  --help       Display this help message"
    echo ""
    exit 0
}

# Parse command line arguments
SKIP_01=false
SKIP_02=false
SKIP_03=false
SKIP_04=false

for arg in "$@"; do
    case $arg in
        --skip-01) SKIP_01=true ;;
        --skip-02) SKIP_02=true ;;
        --skip-03) SKIP_03=true ;;
        --skip-04) SKIP_04=true ;;
        --help) usage ;;
        *) echo "Unknown option: $arg"; usage ;;
    esac
done

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

echo "✓ Virtual environment: $VIRTUAL_ENV"
echo ""

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "ERROR: config.yaml not found!"
    echo "Please create config.yaml from config.yaml.example"
    echo ""
    echo "Example:"
    echo "  cp config.yaml.example config.yaml"
    echo "  # Then edit config.yaml with your API keys"
    exit 1
fi

echo "✓ Configuration file found"

# Extract dataset information from config
TRAIN_FILE=$(grep "train_file:" config.yaml | awk '{print $2}')
TEST_FILE=$(grep "test_file:" config.yaml | awk '{print $2}')
DATASET_NAME=$(echo "$TRAIN_FILE" | sed 's/_train.csv//' | sed 's/.csv//')

echo "✓ Dataset: $DATASET_NAME"
echo "  - Train: $TRAIN_FILE"
echo "  - Test: $TEST_FILE"
echo ""

# Validate input files
if [ ! -f "input_data/$TRAIN_FILE" ]; then
    echo "ERROR: Training file not found: input_data/$TRAIN_FILE"
    echo "Please add your training data to the input_data/ directory"
    exit 1
fi

if [ ! -f "input_data/$TEST_FILE" ]; then
    echo "ERROR: Test file not found: input_data/$TEST_FILE"
    echo "Please add your test data to the input_data/ directory"
    exit 1
fi

echo "✓ Input files validated"

# Check LLM provider configuration
PROVIDER=$(grep "provider:" config.yaml | head -1 | awk '{print $2}')
echo "✓ LLM Provider: $PROVIDER"

# Provider-specific validation
if [ "$PROVIDER" == "ollama" ]; then
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "ERROR: Ollama is not running!"
        echo "Please start Ollama in another terminal:"
        echo "  ollama serve"
        exit 1
    fi
    echo "✓ Ollama is running"
elif [ "$PROVIDER" == "openai" ]; then
    API_KEY=$(grep -A 5 "openai:" config.yaml | grep "api_key:" | awk '{print $2}')
    if [ -z "$API_KEY" ] || [ "$API_KEY" == "YOUR_AZURE_OPENAI_API_KEY" ]; then
        echo "ERROR: Azure OpenAI API key not configured!"
        echo "Please set your API key in config.yaml"
        exit 1
    fi
    echo "✓ Azure OpenAI configured"
elif [ "$PROVIDER" == "gemini" ]; then
    API_KEY=$(grep -A 3 "gemini:" config.yaml | grep "api_key:" | awk '{print $2}')
    if [ -z "$API_KEY" ] || [ "$API_KEY" == "YOUR_GEMINI_API_KEY" ]; then
        echo "ERROR: Gemini API key not configured!"
        echo "Please set your API key in config.yaml"
        exit 1
    fi
    echo "✓ Gemini configured"
fi

echo ""
echo "=========================================="
echo "Starting Pipeline for: $DATASET_NAME"
echo "=========================================="
echo ""

# Function to run a script with error handling
run_script() {
    local script_num=$1
    local script_name=$2
    local description=$3
    
    echo "=========================================="
    echo "Step $script_num: $description"
    echo "=========================================="
    echo ""
    
    START_TIME=$(date +%s)
    
    python "$script_name"
    EXIT_CODE=$?
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "❌ ERROR: Step $script_num failed with exit code $EXIT_CODE!"
        echo ""
        echo "Possible causes:"
        echo "  - API quota exhausted (wait for quota reset)"
        echo "  - Invalid API credentials"
        echo "  - Network connectivity issues"
        echo "  - Input file format errors"
        echo ""
        echo "Duration: ${DURATION}s"
        exit $EXIT_CODE
    fi
    
    echo ""
    echo "✅ Step $script_num completed successfully! (${DURATION}s)"
    echo ""
}

# Run the pipeline
if [ "$SKIP_01" = false ]; then
    run_script 1 "01_data_formatting.py" "Pattern Identification & Candidate Generation"
else
    echo "=========================================="
    echo "Step 1: SKIPPED (--skip-01 flag)"
    echo "=========================================="
    echo ""
fi

if [ "$SKIP_02" = false ]; then
    run_script 2 "02_counterfactual_over_generation.py" "Counterfactual Over-Generation"
else
    echo "=========================================="
    echo "Step 2: SKIPPED (--skip-02 flag)"
    echo "=========================================="
    echo ""
fi

if [ "$SKIP_03" = false ]; then
    run_script 3 "03_counterfactual_filtering.py" "Three-Stage Quality Filtering"
else
    echo "=========================================="
    echo "Step 3: SKIPPED (--skip-03 flag)"
    echo "=========================================="
    echo ""
fi

if [ "$SKIP_04" = false ]; then
    run_script 4 "04_counterfactual_evaluation.py" "Few-Shot Evaluation"
else
    echo "=========================================="
    echo "Step 4: SKIPPED (--skip-04 flag)"
    echo "=========================================="
    echo ""
fi

# Pipeline completion summary
echo ""
echo "=========================================="
echo "🎉 Pipeline Complete!"
echo "=========================================="
echo ""
echo "Dataset: $DATASET_NAME"
echo ""
echo "📁 Output Files:"
echo "  Pattern annotations:     output_data/[*]annotated_data_with_pattern_${DATASET_NAME}.csv"
echo "  Candidate phrases:       output_data/[*]${DATASET_NAME}_candidate_phrases_annotated_data.csv"
echo "  Raw counterfactuals:     output_data/[*]counterfactuals_${TRAIN_FILE}"
echo "  Filtered results:        output_data/[*]filtered_${TRAIN_FILE}"
echo "  Fine-tune dataset:       output_data/[*]fine_tuneset_${TRAIN_FILE}"
echo "  Evaluation metrics:      output_data/archive/gpt/[*]_counter_${DATASET_NAME}_prf.csv"
echo ""
echo "💡 Next Steps:"
echo "  1. Review evaluation metrics in output_data/archive/gpt/"
echo "  2. Analyze fine-tune dataset for model training"
echo "  3. Compare results across different shot configurations"
echo "  4. Run on additional datasets by updating config.yaml"
echo ""
echo "To run specific steps, use flags:"
echo "  ./run_all.sh --skip-01      # Skip pattern identification"
echo "  ./run_all.sh --skip-02      # Skip counterfactual generation"
echo ""
