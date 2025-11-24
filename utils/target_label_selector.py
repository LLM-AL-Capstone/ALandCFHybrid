"""
Target Label Selection Strategies for Counterfactual Generation (Version 3)

Implements 4 strategies for selecting target labels for CF generation:
- Uniform: Random selection from non-oracle labels
- Confusion: Most probable wrong label (model confusion)
- Round-Robin: Least-used label transition pairs
- Hybrid: Combines confusion with round-robin balancing
"""

import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class TargetLabelSelector:
    """
    Selects target labels for counterfactual generation using various strategies.
    
    Version 3: Selects ONE target label per factual example, then generates
    all CFs toward that selected label.
    """
    
    def __init__(self, config: dict, all_labels: List[str], seed: Optional[int] = None):
        """
        Initialize target label selector.
        
        Args:
            config: Configuration dictionary
            all_labels: List of all possible labels in the dataset
            seed: Random seed for reproducibility (for uniform strategy)
        """
        self.config = config
        self.all_labels = all_labels
        self.seed = seed
        
        # Get strategy from config
        cf_config = config.get('active_learning', {}).get('counterfactuals', {})
        target_config = cf_config.get('target_label_selection', {})
        
        # Support backward compatibility with old 'distribution_strategy'
        if 'target_label_selection' not in cf_config and 'distribution_strategy' in cf_config:
            old_strategy = cf_config['distribution_strategy']
            if old_strategy == 'balanced':
                self.strategy = 'round_robin'
            elif old_strategy == 'random':
                self.strategy = 'uniform'
            else:
                self.strategy = 'uniform'  # Default
            print(f"  Note: Migrated 'distribution_strategy: {old_strategy}' to 'target_label_selection.strategy: {self.strategy}'")
        else:
            self.strategy = target_config.get('strategy', 'uniform')
        
        # Get lambda for hybrid strategy
        self.lambda_param = target_config.get('lambda', 0.5)
        
        # Initialize transition counters for round-robin and hybrid
        # c(y_label, y_target) tracks how many times we've generated CFs
        # for the transition from y_label to y_target
        self.transition_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        
        print(f"  Initialized Target Label Selector (V3):")
        print(f"    Strategy: {self.strategy}")
        if self.strategy == 'hybrid':
            print(f"    Lambda (λ): {self.lambda_param}")
    
    def select_target_label(
        self,
        original_label: str,
        original_text: str,
        classifier=None
    ) -> str:
        """
        Select ONE target label for counterfactual generation.
        
        Version 3: Selects one target label per factual, then all CFs
        are generated toward this label.
        
        Args:
            original_label: Original label of the factual example
            original_text: Original text (for confusion-based strategy)
            classifier: Trained classifier (required for confusion-based strategy)
        
        Returns:
            Selected target label (must be different from original_label)
        """
        # Get non-oracle labels (all labels except original)
        non_oracle_labels = [label for label in self.all_labels if label != original_label]
        
        if len(non_oracle_labels) == 0:
            raise ValueError(f"No target labels available (only one label: {original_label})")
        
        # If only one option, return it
        if len(non_oracle_labels) == 1:
            return non_oracle_labels[0]
        
        # Select based on strategy
        if self.strategy == 'uniform':
            return self._select_uniform(non_oracle_labels)
        
        elif self.strategy == 'confusion':
            if classifier is None:
                raise ValueError("Classifier required for confusion-based target label selection")
            return self._select_confusion(non_oracle_labels, original_text, original_label, classifier)
        
        elif self.strategy == 'round_robin':
            return self._select_round_robin(non_oracle_labels, original_label)
        
        elif self.strategy == 'hybrid':
            if classifier is None:
                raise ValueError("Classifier required for hybrid target label selection")
            return self._select_hybrid(non_oracle_labels, original_text, original_label, classifier)
        
        else:
            raise ValueError(f"Unknown target label selection strategy: {self.strategy}")
    
    def _select_uniform(self, non_oracle_labels: List[str]) -> str:
        """
        Strategy 1: Uniform over Non-Oracle Labels
        
        Samples y_target uniformly from Y_≠ (non-oracle labels).
        Ignores model predictions, aims for balanced distribution.
        
        Args:
            non_oracle_labels: List of labels excluding original
        
        Returns:
            Randomly selected target label
        """
        # Use global random state (seed already set at program start)
        # This ensures reproducibility while maintaining randomness across calls
        return random.choice(non_oracle_labels)
    
    def _select_confusion(
        self,
        non_oracle_labels: List[str],
        original_text: str,
        original_label: str,
        classifier
    ) -> str:
        """
        Strategy 2: Model-Confusion (Most Probable Wrong Label)
        
        Chooses the non-oracle label with highest p(y | u).
        Focuses CF generation on the label the model finds most confusing.
        
        Args:
            non_oracle_labels: List of labels excluding original
            original_text: Original text
            original_label: Original label
            classifier: Trained classifier
        
        Returns:
            Target label with highest probability
        """
        # Get probability distribution for original text
        probs = classifier.predict_proba([original_text])[0]
        labels = classifier.get_labels()
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        # Find non-oracle label with highest probability
        best_label = None
        best_prob = -1.0
        
        for label in non_oracle_labels:
            if label in label_to_idx:
                idx = label_to_idx[label]
                prob = probs[idx]
                if prob > best_prob:
                    best_prob = prob
                    best_label = label
        
        if best_label is None:
            # Fallback to uniform if classifier doesn't have all labels
            return self._select_uniform(non_oracle_labels)
        
        return best_label
    
    def _select_round_robin(
        self,
        non_oracle_labels: List[str],
        original_label: str
    ) -> str:
        """
        Strategy 3: Round-Robin (Balanced Coverage)
        
        Selects target label with least-used transition (y_label, y_target).
        Ensures global coverage of all label transition pairs.
        
        Args:
            non_oracle_labels: List of labels excluding original
            original_label: Original label
        
        Returns:
            Target label with minimum transition count
        """
        # Find label with minimum transition count
        min_count = float('inf')
        candidates = []
        
        for target_label in non_oracle_labels:
            transition = (original_label, target_label)
            count = self.transition_counts[transition]
            
            if count < min_count:
                min_count = count
                candidates = [target_label]
            elif count == min_count:
                candidates.append(target_label)
        
        # If tie, choose first (deterministic) or random
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            # Deterministic tie-breaking: use first alphabetically
            selected = sorted(candidates)[0]
        
        # Update transition count
        transition = (original_label, selected)
        self.transition_counts[transition] += 1
        
        return selected
    
    def _select_hybrid(
        self,
        non_oracle_labels: List[str],
        original_text: str,
        original_label: str,
        classifier
    ) -> str:
        """
        Strategy 4: Hybrid Confusion-Balanced
        
        Combines local model confusion with global pair balancing.
        Score: s(y) = p(y | u) - λ * NormCount(c(y_label, y))
        
        Args:
            non_oracle_labels: List of labels excluding original
            original_text: Original text
            original_label: Original label
            classifier: Trained classifier
        
        Returns:
            Target label with highest hybrid score
        """
        # Get probability distribution
        probs = classifier.predict_proba([original_text])[0]
        labels = classifier.get_labels()
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        # Normalize transition counts (for fair comparison)
        if self.transition_counts:
            max_count = max(self.transition_counts.values())
            min_count = min(self.transition_counts.values())
            count_range = max_count - min_count if max_count > min_count else 1
        else:
            count_range = 1
        
        # Score each candidate label
        best_label = None
        best_score = float('-inf')
        
        for label in non_oracle_labels:
            if label not in label_to_idx:
                continue
            
            # Get model probability
            idx = label_to_idx[label]
            p_y = probs[idx]
            
            # Get normalized transition count
            transition = (original_label, label)
            count = self.transition_counts[transition]
            normalized_count = (count - min_count) / count_range if count_range > 0 else 0
            
            # Hybrid score: p(y|u) - λ * normalized_count
            score = p_y - self.lambda_param * normalized_count
            
            if score > best_score:
                best_score = score
                best_label = label
        
        if best_label is None:
            # Fallback to uniform
            return self._select_uniform(non_oracle_labels)
        
        # Update transition count
        transition = (original_label, best_label)
        self.transition_counts[transition] += 1
        
        return best_label
    
    def get_transition_counts(self) -> Dict[Tuple[str, str], int]:
        """
        Get current transition counts (for debugging/analysis).
        
        Returns:
            Dictionary mapping (y_orig, y_target) tuples to counts
        """
        return dict(self.transition_counts)
    
    def reset_transition_counts(self):
        """Reset transition counts (useful for new runs)."""
        self.transition_counts.clear()

