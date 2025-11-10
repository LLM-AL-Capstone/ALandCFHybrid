"""
Oracle for Active Learning

Provides labels for selected examples. In simulated mode, uses ground truth
labels from the dataset. Can be extended for interactive human labeling.
"""

from typing import List, Dict


class SimulatedOracle:
    """
    Simulated oracle that uses ground truth labels from the dataset.
    
    This is useful for experiments and debugging where we want to test
    the active learning loop without requiring human input.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the oracle.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.total_queries = 0
    
    def label_examples(self, examples: List[Dict]) -> List[Dict]:
        """
        Label selected examples by revealing their ground truth labels.
        
        Args:
            examples: List of examples with 'text', 'id', and hidden 'label'
        
        Returns:
            Same examples with labels revealed (already in the dict)
        """
        self.total_queries += len(examples)
        
        # In simulated mode, examples already have 'label' field (ground truth)
        # We just need to ensure they're properly formatted
        labeled_examples = []
        
        for ex in examples:
            if 'label' not in ex:
                raise ValueError(f"Example {ex.get('id', 'unknown')} missing 'label' field for simulated oracle")
            
            labeled_examples.append({
                'id': ex['id'],
                'text': ex['text'],
                'label': ex['label']
            })
        
        print(f"  Oracle labeled {len(examples)} examples (total queries: {self.total_queries})")
        
        return labeled_examples
    
    def get_total_queries(self) -> int:
        """Get total number of queries made to oracle."""
        return self.total_queries


class InteractiveOracle:
    """
    Interactive oracle that asks human for labels.
    
    This can be used for real active learning experiments where
    a human annotator provides labels.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the oracle.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.total_queries = 0
        
        # Get available labels from config
        dataset_config = config['dataset']
        self.available_labels = None  # Will be set dynamically
    
    def label_examples(self, examples: List[Dict]) -> List[Dict]:
        """
        Ask human to label selected examples.
        
        Args:
            examples: List of examples with 'text' and 'id'
        
        Returns:
            Examples with human-provided labels
        """
        print(f"\n{'='*70}")
        print(f"HUMAN LABELING REQUIRED: {len(examples)} examples")
        print(f"{'='*70}\n")
        
        if self.available_labels:
            print(f"Available labels: {', '.join(self.available_labels)}\n")
        
        labeled_examples = []
        
        for i, ex in enumerate(examples):
            print(f"Example {i+1}/{len(examples)}:")
            print(f"ID: {ex['id']}")
            print(f"Text: {ex['text']}")
            print()
            
            # Get label from human
            while True:
                label = input("Enter label: ").strip()
                
                if label:
                    # Optionally validate against known labels
                    if self.available_labels and label not in self.available_labels:
                        print(f"Warning: '{label}' not in known labels: {self.available_labels}")
                        confirm = input("Use anyway? (y/n): ").strip().lower()
                        if confirm != 'y':
                            continue
                    break
                else:
                    print("Label cannot be empty. Please try again.")
            
            labeled_examples.append({
                'id': ex['id'],
                'text': ex['text'],
                'label': label
            })
            
            print()
        
        self.total_queries += len(examples)
        
        print(f"{'='*70}")
        print(f"Labeling complete. Total queries: {self.total_queries}")
        print(f"{'='*70}\n")
        
        return labeled_examples
    
    def set_available_labels(self, labels: List[str]):
        """Set the list of available labels for validation."""
        self.available_labels = labels
    
    def get_total_queries(self) -> int:
        """Get total number of queries made to oracle."""
        return self.total_queries


def get_oracle(config: dict):
    """
    Factory function to create appropriate oracle based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Oracle instance (Simulated or Interactive)
    """
    # For now, always use simulated oracle as specified in the plan
    return SimulatedOracle(config)

