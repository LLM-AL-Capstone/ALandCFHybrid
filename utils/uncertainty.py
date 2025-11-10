"""
Uncertainty-based Query Strategies for Active Learning

Implements various uncertainty sampling methods:
- Entropy sampling
- Margin sampling  
- Least confident sampling
"""

import numpy as np
from typing import List, Dict


def select_uncertain_examples(
    unlabeled_pool: List[Dict],
    classifier,
    batch_size: int,
    method: str = "entropy",
    return_details: bool = False
):
    """
    Select most uncertain examples from unlabeled pool.
    
    This is the core of the Active Learning query strategy.
    
    Args:
        unlabeled_pool: List of unlabeled examples (dicts with 'text' key)
        classifier: Trained classifier with predict_proba method
        batch_size: Number of examples to select
        method: Uncertainty method ('entropy', 'margin', 'least_confident')
        return_details: If True, returns (indices, details_dict) with uncertainty details
    
    Returns:
        If return_details=False: List of indices of selected examples from unlabeled_pool
        If return_details=True: Tuple of (indices, details_dict) with uncertainty scores and logprobs
    """
    if len(unlabeled_pool) == 0:
        if return_details:
            return [], {}
        return []
    
    if batch_size > len(unlabeled_pool):
        batch_size = len(unlabeled_pool)
        print(f"  Warning: batch_size reduced to {batch_size} (size of unlabeled pool)")
    
    # Extract texts
    texts = [ex['text'] for ex in unlabeled_pool]
    
    print(f"  Computing uncertainty scores for {len(texts)} examples...")
    
    # Get probability predictions (with details if requested)
    if return_details:
        probs, prediction_details = classifier.predict_proba(texts, return_details=True)
    else:
        probs = classifier.predict_proba(texts)
        prediction_details = None
    
    # Calculate uncertainty scores based on method
    if method == "entropy":
        scores = calculate_entropy(probs)
    elif method == "margin":
        scores = calculate_margin(probs)
    elif method == "least_confident":
        scores = calculate_least_confident(probs)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")
    
    # Select top-k most uncertain (highest scores)
    top_indices = np.argsort(scores)[-batch_size:][::-1]
    
    print(f"  Selected {len(top_indices)} most uncertain examples")
    print(f"  Uncertainty scores range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    if return_details:
        # Compile detailed information for all examples
        details = {
            'method': method,
            'total_pool_size': len(unlabeled_pool),
            'batch_size': batch_size,
            'all_uncertainty_scores': scores.tolist(),
            'selected_indices': top_indices.tolist(),
            'selected_scores': [float(scores[i]) for i in top_indices],
            'score_statistics': {
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'std': float(scores.std())
            }
        }
        
        # Add prediction details if available
        if prediction_details:
            details['prediction_details'] = prediction_details
        
        return top_indices.tolist(), details
    
    return top_indices.tolist()


def calculate_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Calculate entropy uncertainty.
    
    Entropy measures the overall uncertainty in the prediction distribution.
    Higher entropy = more uncertain.
    
    H(p) = -sum(p(y) * log(p(y)))
    
    Args:
        probs: Array of shape (n_examples, n_classes) with probabilities
    
    Returns:
        Array of shape (n_examples,) with entropy scores
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    return entropy


def calculate_margin(probs: np.ndarray) -> np.ndarray:
    """
    Calculate margin uncertainty.
    
    Margin is the difference between the top two predictions.
    Smaller margin = more uncertain (so we negate it).
    
    Args:
        probs: Array of shape (n_examples, n_classes) with probabilities
    
    Returns:
        Array of shape (n_examples,) with margin scores (negated)
    """
    # Sort probabilities in descending order
    sorted_probs = np.sort(probs, axis=1)
    
    # Get top two predictions
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
    
    # Calculate margin (difference between top 2)
    margin = top1 - top2
    
    # Negate so that smaller margins = higher scores (more uncertain)
    return -margin


def calculate_least_confident(probs: np.ndarray) -> np.ndarray:
    """
    Calculate least confident uncertainty.
    
    Measures uncertainty based on the most confident prediction.
    Lower confidence = more uncertain.
    
    Score = 1 - max(p(y))
    
    Args:
        probs: Array of shape (n_examples, n_classes) with probabilities
    
    Returns:
        Array of shape (n_examples,) with least confident scores
    """
    # Get maximum probability for each example
    max_probs = np.max(probs, axis=1)
    
    # Calculate uncertainty (1 - confidence)
    uncertainty = 1 - max_probs
    
    return uncertainty


def get_uncertainty_statistics(
    unlabeled_pool: List[Dict],
    classifier,
    method: str = "entropy"
) -> Dict:
    """
    Get statistics about uncertainty in the unlabeled pool.
    
    Useful for analysis and debugging.
    
    Args:
        unlabeled_pool: List of unlabeled examples
        classifier: Trained classifier
        method: Uncertainty method
    
    Returns:
        Dictionary with statistics
    """
    if len(unlabeled_pool) == 0:
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'median': 0
        }
    
    texts = [ex['text'] for ex in unlabeled_pool]
    probs = classifier.predict_proba(texts)
    
    if method == "entropy":
        scores = calculate_entropy(probs)
    elif method == "margin":
        scores = calculate_margin(probs)
    elif method == "least_confident":
        scores = calculate_least_confident(probs)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")
    
    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'median': float(np.median(scores)),
        'method': method
    }

