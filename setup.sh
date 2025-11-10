#!/bin/bash

echo "=== LLM-VT-AL Project Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p input_data
mkdir -p output_data/interim_output
mkdir -p output_data/archive/gpt
mkdir -p utils

# Create .gitkeep files to preserve directory structure
touch output_data/.gitkeep
touch output_data/interim_output/.gitkeep
touch output_data/archive/.gitkeep

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Place your dataset CSV files in the input_data/ folder"
echo ""
echo "3. Configure config.yaml with your LLM provider settings"
echo ""
echo "4. If using Ollama, ensure it's running:"
echo "   ollama serve"
echo "   ollama pull qwen2.5:7b"
echo ""
echo "5. If using Gemini, add your API key to config.yaml"
echo ""
echo "6. Run the scripts in order:"
echo "   python 01_data_formatting.py"
echo "   python 02_counterfactual_over_generation.py"
echo "   python 03_counterfactual_filtering.py"
echo "   python 05_counterfactual_evaluation.py"
