"""
Simple In-Context Learning (ICL) Classifier for Active Learning

Uses LLM with few-shot examples for classification without training.
"""

import numpy as np
from typing import List, Dict, Optional
import time


class SimpleICLClassifier:
    """
    Simple ICL-based classifier using LLM with few-shot examples.
    
    This classifier doesn't train weights - it stores labeled examples
    and uses them as in-context examples for the LLM.
    """
    
    def __init__(self, config: dict, llm_provider):
        """
        Initialize the classifier.
        
        Args:
            config: Configuration dictionary
            llm_provider: LLM provider instance
        """
        self.config = config
        self.llm_provider = llm_provider
        self.labeled_examples = []
        self.labels = set()
        self.max_icl_examples = config['evaluation']['max_icl_examples']
    
    def train(self, labeled_pool: List[Dict]):
        """
        'Train' the classifier by storing labeled examples for ICL.
        
        Args:
            labeled_pool: List of dicts with 'text' and 'label' keys
        """
        self.labeled_examples = labeled_pool.copy()
        self.labels = set(ex['label'] for ex in labeled_pool)
        print(f"  Classifier 'trained' with {len(labeled_pool)} examples across {len(self.labels)} labels")
    
    def predict(self, text: str) -> str:
        """
        Predict label for a single text using ICL.
        
        Args:
            text: Text to classify
            
        Returns:
            Predicted label
        """
        if not self.labeled_examples:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Use up to max_icl_examples
        icl_examples = self.labeled_examples[:self.max_icl_examples]
        
        # Build ICL prompt
        labels_str = ', '.join(sorted(self.labels))
        
        # System message
        messages = [
            {
                "role": "system",
                "content": f"You are a text classifier. Classify sentences into one of these labels: {labels_str}. Respond with only the label, nothing else."
            }
        ]
        
        # Add ICL examples as user/assistant pairs
        for ex in icl_examples:
            messages.append({"role": "user", "content": ex['text']})
            messages.append({"role": "assistant", "content": ex['label']})
        
        # Add query
        messages.append({"role": "user", "content": text})
        
        # Get prediction
        response = self.llm_provider.chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=50
        )
        
        # Clean response
        predicted_label = response.strip()
        
        # If prediction not in known labels, return closest match or first label
        if predicted_label not in self.labels:
            # Simple heuristic: check if any label is substring
            for label in self.labels:
                if label.lower() in predicted_label.lower():
                    predicted_label = label
                    break
            else:
                # Default to first label if no match
                predicted_label = sorted(list(self.labels))[0]
        
        return predicted_label
    
    def predict_proba(self, texts: List[str], return_details: bool = False):
        """
        Predict probabilities for uncertainty estimation.
        
        Uses OpenAI's logprobs API if available for accurate probability distributions.
        Falls back to simplified heuristic (0.9 for predicted, 0.1 distributed) if not.
        
        Args:
            texts: List of texts to classify
            return_details: If True, returns (probs, details_list) where details contains
                           full OpenAI responses, logprobs, etc.
            
        Returns:
            If return_details=False: Array of shape (n_texts, n_labels) with probability estimates
            If return_details=True: Tuple of (probs_array, details_list)
        """
        if not self.labeled_examples:
            raise ValueError("Classifier not trained. Call train() first.")
        
        label_list = sorted(list(self.labels))
        n_labels = len(label_list)
        label_to_idx = {label: i for i, label in enumerate(label_list)}
        
        # Check if provider supports logprobs
        supports_logprobs = hasattr(self.llm_provider, 'chat_completion_with_logprobs')
        
        probs = []
        details_list = []
        
        for i, text in enumerate(texts):
            if supports_logprobs:
                # Try to get real probabilities using logprobs
                prob_dist, details = self._predict_proba_with_logprobs(
                    text, label_list, label_to_idx, n_labels, return_details=True
                )
            else:
                # Fallback to heuristic
                prob_dist = self._predict_proba_heuristic(text, label_list, label_to_idx, n_labels)
                details = {'method': 'heuristic', 'text': text[:100]}
            
            probs.append(prob_dist)
            if return_details:
                details_list.append(details)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"      Uncertainty scoring: {i+1}/{len(texts)} examples...")
            
            # Add small delay to avoid rate limits
            time.sleep(0.1)
        
        if return_details:
            return np.array(probs), details_list
        return np.array(probs)
    
    def _predict_proba_with_logprobs(self, text: str, label_list: List[str], 
                                     label_to_idx: Dict[str, int], n_labels: int,
                                     return_details: bool = False):
        """
        Get probability distribution using LLM logprobs (OpenAI-specific).
        
        Args:
            text: Text to classify
            label_list: Sorted list of labels
            label_to_idx: Mapping from label to index
            n_labels: Number of labels
            return_details: If True, returns (prob_dist, details_dict)
            
        Returns:
            If return_details=False: Probability distribution array
            If return_details=True: Tuple of (prob_dist, details_dict)
        """
        # Build ICL prompt (same as predict())
        icl_examples = self.labeled_examples[:self.max_icl_examples]
        labels_str = ', '.join(label_list)
        
        messages = [
            {
                "role": "system",
                "content": f"You are a text classifier. Classify sentences into one of these labels: {labels_str}. Respond with only the label, nothing else."
            }
        ]
        
        # Add ICL examples
        for ex in icl_examples:
            messages.append({"role": "user", "content": ex['text']})
            messages.append({"role": "assistant", "content": ex['label']})
        
        # Add query
        messages.append({"role": "user", "content": text})
        
        try:
            # Get prediction with logprobs
            result = self.llm_provider.chat_completion_with_logprobs(
                messages=messages,
                possible_labels=label_list,
                temperature=0,
                max_tokens=50
            )
            
            # Extract probabilities
            if result['probabilities'] is not None:
                # Convert label probabilities to array
                prob_dist = np.zeros(n_labels)
                for label, prob in result['probabilities'].items():
                    if label in label_to_idx:
                        prob_dist[label_to_idx[label]] = prob
                
                # Ensure probabilities sum to 1
                total = prob_dist.sum()
                if total > 0:
                    prob_dist = prob_dist / total
                
                # Calculate entropy
                entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
                
                # VERBOSE: Print logprobs for first few examples
                if hasattr(self, '_logprobs_printed'):
                    self._logprobs_printed += 1
                else:
                    self._logprobs_printed = 1
                
                if self._logprobs_printed <= 3:  # Print first 3 examples
                    print(f"\n      --- LOGPROBS RESPONSE (Example {self._logprobs_printed}) ---")
                    print(f"      Text snippet: {text[:60]}...")
                    print(f"      Prediction: {result['prediction']}")
                    print(f"      Probabilities from OpenAI:")
                    for label in label_list:
                        prob = result['probabilities'].get(label, 0.0)
                        bar = "█" * int(prob * 30)
                        print(f"        {label:12s}: {prob:.4f} {bar}")
                    print(f"      Entropy (uncertainty): {entropy:.4f}")
                    print(f"      ---")
                
                # Prepare detailed information if requested
                if return_details:
                    details = {
                        'method': 'logprobs',
                        'text': text,
                        'prediction': result['prediction'],
                        'raw_openai_result': result.get('raw_response', None),  # Full API response if available
                        'computed_probabilities': {label: float(result['probabilities'].get(label, 0.0)) 
                                                   for label in label_list},
                        'probability_distribution': prob_dist.tolist(),
                        'entropy': float(entropy),
                        'labels': label_list
                    }
                    return prob_dist, details
                
                return prob_dist
            else:
                # Logprobs not available, use heuristic
                pred_label = result['prediction']
                heuristic_dist = self._create_heuristic_distribution(pred_label, label_to_idx, n_labels)
                
                if return_details:
                    details = {
                        'method': 'heuristic_fallback',
                        'text': text,
                        'prediction': pred_label,
                        'reason': 'logprobs_not_available'
                    }
                    return heuristic_dist, details
                return heuristic_dist
        
        except Exception as e:
            print(f"      Warning: logprobs failed ({e}), using heuristic")
            # Fallback to heuristic
            pred_label = self.predict(text)
            heuristic_dist = self._create_heuristic_distribution(pred_label, label_to_idx, n_labels)
            
            if return_details:
                details = {
                    'method': 'heuristic_fallback',
                    'text': text,
                    'prediction': pred_label,
                    'reason': f'exception: {str(e)}'
                }
                return heuristic_dist, details
            return heuristic_dist
    
    def _predict_proba_heuristic(self, text: str, label_list: List[str],
                                 label_to_idx: Dict[str, int], n_labels: int) -> np.ndarray:
        """
        Get probability distribution using simple heuristic.
        
        Args:
            text: Text to classify
            label_list: Sorted list of labels
            label_to_idx: Mapping from label to index
            n_labels: Number of labels
            
        Returns:
            Probability distribution array
        """
        pred_label = self.predict(text)
        return self._create_heuristic_distribution(pred_label, label_to_idx, n_labels)
    
    def _create_heuristic_distribution(self, pred_label: str, label_to_idx: Dict[str, int], 
                                      n_labels: int) -> np.ndarray:
        """
        Create heuristic probability distribution (0.9 for predicted, 0.1 distributed).
        
        Args:
            pred_label: Predicted label
            label_to_idx: Mapping from label to index
            n_labels: Number of labels
            
        Returns:
            Probability distribution array
        """
        prob_dist = np.ones(n_labels) * (0.1 / (n_labels - 1)) if n_labels > 1 else np.ones(n_labels)
        
        if pred_label in label_to_idx:
            pred_idx = label_to_idx[pred_label]
            prob_dist[pred_idx] = 0.9
        
        return prob_dist
    
    def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Predict labels for multiple texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predicted labels
        """
        predictions = []
        
        for i, text in enumerate(texts):
            pred = self.predict(text)
            predictions.append(pred)
            
            # Progress indicator for large batches
            if (i + 1) % 50 == 0:
                print(f"    Classified {i+1}/{len(texts)} examples...")
            
            # Rate limiting
            time.sleep(0.1)
        
        return predictions
    
    def get_labels(self) -> List[str]:
        """Get list of known labels."""
        return sorted(list(self.labels))

