"""
Data loading and validation utilities
"""

import pandas as pd
import sys
from typing import Tuple, List


def load_dataset(file_path: str, config: dict) -> pd.DataFrame:
    """
    Load dataset from CSV file with validation.
    
    Args:
        file_path: Path to CSV file
        config: Configuration dict with column mappings
        
    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"INFO: Successfully loaded {len(df)} examples from {file_path}")
        
        # Validate required columns
        col_config = config['dataset']['columns']
        required_cols = [col_config['id'], col_config['text'], col_config['label']]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
        
    except FileNotFoundError:
        print(f"ERROR: Cannot read file {file_path}")
        print(f"Please ensure the file exists in the input_data/ directory")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)


def get_unique_labels(df: pd.DataFrame, label_col: str, exclude_labels: List[str] = None) -> List[str]:
    """
    Get unique labels from dataset, excluding specified labels.
    
    Args:
        df: DataFrame
        label_col: Name of label column
        exclude_labels: Labels to exclude (e.g., ['none', 'neutral'])
        
    Returns:
        List of unique labels
    """
    exclude_labels = exclude_labels or []
    unique_labels = [
        label for label in df[label_col].unique() 
        if label not in exclude_labels
    ]
    return unique_labels


def shuffle_dataframe(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Shuffle DataFrame with seed for reproducibility.
    
    Args:
        df: DataFrame to shuffle
        seed: Random seed
        
    Returns:
        Shuffled DataFrame
    """
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
