"""
Configuration loading utilities
"""

import yaml
import os


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create config.yaml based on config.yaml.example"
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_directories(config: dict):
    """
    Create necessary directories if they don't exist.
    
    Args:
        config: Configuration dictionary
    """
    dirs = config['directories']
    
    os.makedirs(dirs['input_data'], exist_ok=True)
    os.makedirs(dirs['output_data'], exist_ok=True)
    os.makedirs(dirs['interim_output'], exist_ok=True)
    os.makedirs(dirs['archive'], exist_ok=True)
    os.makedirs(f"{dirs['archive']}/gpt", exist_ok=True)
    
    print("INFO: Directory structure verified")
