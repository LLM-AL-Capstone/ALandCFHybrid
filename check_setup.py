#!/usr/bin/env python3
"""
Check project dependencies and configuration
Run this before starting the pipeline
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f" Python {version.major}.{version.minor} detected")
        print("  Required: Python 3.8+")
        return False
    print(f" Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_virtual_env():
    """Check if virtual environment is activated"""
    print("\nChecking virtual environment...")
    if sys.prefix == sys.base_prefix:
        print("  Virtual environment NOT activated")
        print("  Run: source venv/bin/activate")
        return False
    print(f" Virtual environment active: {sys.prefix}")
    return True


def check_packages():
    """Check required packages"""
    print("\nChecking required packages...")
    required = ['pandas', 'numpy', 'yaml', 'sklearn']
    missing = []
    
    for package in required:
        try:
            if package == 'yaml':
                __import__('yaml')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  {package} installed")
        except ImportError:
            print(f"  {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def check_llm_provider():
    """Check LLM provider availability"""
    print("\nChecking LLM provider...")
    
    # Check config file
    if not os.path.exists('config.yaml'):
        print(" config.yaml not found")
        return False
    
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    provider = config['llm']['provider']
    print(f"  Provider: {provider}")
    
    if provider == 'ollama':
        try:
            import urllib.request
            base_url = config['llm']['ollama'].get('base_url', 'http://localhost:11434')
            urllib.request.urlopen(f"{base_url}/api/tags", timeout=2)
            print(f" Ollama is running at {base_url}")
            
            # Check model
            model = config['llm']['ollama']['model']
            print(f"  Model configured: {model}")
            print(f"  Make sure model is pulled: ollama pull {model}")
            return True
        except:
            print(f" Ollama not running at {base_url}")
            print("  Start Ollama: ollama serve")
            return False
    
    elif provider == 'gemini':
        try:
            import google.generativeai
            api_key = config['llm']['gemini']['api_key']
            if api_key == 'YOUR_GEMINI_API_KEY_HERE':
                print("  Gemini API key not configured")
                print("  Add your key to config.yaml")
                return False
            print("  Gemini package installed")
            print("  API key configured")
            return True
        except ImportError:
            print("  Gemini package not installed")
            print("  Install: pip install google-generativeai")
            return False
    
    return True


def check_data():
    """Check input data"""
    print("\nChecking input data...")
    
    if not os.path.exists('config.yaml'):
        print("  Cannot check - config.yaml not found")
        return False
    
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_file = config['dataset']['train_file']
    test_file = config['dataset']['test_file']
    
    train_path = f"input_data/{train_file}"
    test_path = f"input_data/{test_file}"
    
    train_exists = os.path.exists(train_path)
    test_exists = os.path.exists(test_path)
    
    if train_exists:
        print(f"  Training data: {train_path}")
    else:
        print(f"  Training data not found: {train_path}")
    
    if test_exists:
        print(f"  Test data: {test_path}")
    else:
        print(f"  Test data not found: {test_path}")
        print("   (Optional for scripts 01-03)")
    
    return train_exists


def check_directories():
    """Check directory structure"""
    print("\nChecking directory structure...")
    
    dirs = [
        'input_data',
        'output_data',
        'output_data/interim_output',
        'output_data/archive',
        'output_data/archive/gpt',
        'utils'
    ]
    
    all_exist = True
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"  {dir_path}/ found")
        else:
            print(f"  {dir_path}/ not found")
            all_exist = False
    
    if not all_exist:
        print("\n  Create directories: mkdir -p " + " ".join(dirs))
    
    return all_exist


def main():
    """Main check"""
    print("=" * 60)
    print("LLM-VT-AL Dependency Checker")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Virtual Environment", check_virtual_env()),
        ("Required Packages", check_packages()),
        ("Directory Structure", check_directories()),
        ("LLM Provider", check_llm_provider()),
        ("Input Data", check_data()),
    ]
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(passed for _, passed in checks)
    
    print("=" * 60)
    if all_passed:
        print("All checks passed! Ready to run pipeline.")
        print("\nNext steps:")
        print("  python 01_data_formatting.py")
        print("  python 02_counterfactual_over_generation.py")
        print("  python 03_counterfactual_filtering.py")
    else:
        print("Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        print("  1. Activate venv: source venv/bin/activate")
        print("  2. Install packages: pip install -r requirements.txt")
        print("  3. Start Ollama: ollama serve (in another terminal)")
        print("  4. Pull model: ollama pull qwen2.5:7b")
        print("  5. Add data to input_data/")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
