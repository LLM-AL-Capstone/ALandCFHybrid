"""
Counterfactual Quality Scoring Module

Provides quality metrics for filtering counterfactual examples (Algorithm 1):
- Label Correctness: Does the CF flip the model's prediction?
- Semantic Similarity: Does the CF preserve the original context?

Following Algorithm 1, line 14 from the paper.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import os

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CFQualityScorer:
    """Quality scorer for counterfactual examples (paper's filters only)"""
    
    def __init__(self, config: dict):
        """
        Initialize quality scorer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cf_config = config['active_learning']['counterfactuals']
        self.quality_config = self.cf_config['quality_filtering']
        self.embedding_model = None
        
        print(f"  Initialized CF Quality Scorer (Paper's Algorithm 1):")
        print(f"    - Label correctness: min_margin={self.quality_config.get('min_margin', 0.1)}")
        print(f"    - Semantic similarity: min_similarity={self.quality_config.get('min_semantic_similarity', 0.6)}")
    
    def _init_embedding_model(self):
        """Lazy initialization of embedding model"""
        if self.embedding_model is None:
            model_name = self.quality_config.get('embedding_model', 'all-MiniLM-L6-v2')
            print(f"  Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
    
    def compute_label_correctness(
        self, 
        cf_text: str, 
        target_label: str, 
        original_label: str, 
        classifier
    ) -> Tuple[float, Dict, bool]:
        """
        FILTER 1: Check if CF successfully flips the label (Paper's criterion).
        
        Criterion: p(y_target | CF) > p(y_orig | CF) + delta
        
        Args:
            cf_text: Counterfactual text
            target_label: Intended label for CF
            original_label: Original label before CF
            classifier: Trained classifier
            
        Returns:
            Tuple of (confidence_score, details_dict, passes_filter)
        """
        if not classifier:
            return 0.0, {'error': 'No classifier provided'}, False
        
        try:
            # Get probability distribution
            probs = classifier.predict_proba([cf_text])[0]
            labels = classifier.get_labels()
            label_to_idx = {label: idx for idx, label in enumerate(labels)}
            
            # Get probabilities for target and original labels
            target_idx = label_to_idx.get(target_label, -1)
            orig_idx = label_to_idx.get(original_label, -1)
            
            if target_idx == -1 or orig_idx == -1:
                return 0.0, {'error': 'Invalid labels'}, False
            
            p_target = probs[target_idx]
            p_orig = probs[orig_idx]
            
            # Calculate margin
            margin = p_target - p_orig
            
            # Get thresholds from config
            min_margin = self.quality_config.get('min_margin', 0.1)
            min_target_confidence = self.quality_config.get('min_target_confidence', 0.3)
            
            # Check both criteria
            margin_passes = margin > min_margin
            confidence_passes = p_target >= min_target_confidence
            passes_filter = margin_passes and confidence_passes
            
            details = {
                'p_target': float(p_target),
                'p_orig': float(p_orig),
                'margin': float(margin),
                'min_margin': min_margin,
                'margin_passes': bool(margin_passes),
                'confidence_passes': bool(confidence_passes),
                'passes': bool(passes_filter)
            }
            
            return float(p_target), details, bool(passes_filter)
            
        except Exception as e:
            return 0.0, {'error': str(e)}, False
    
    def compute_semantic_similarity(
        self, 
        original_text: str, 
        cf_text: str
    ) -> Tuple[float, Dict, bool]:
        """
        FILTER 2: Check if CF preserves semantic context of original.
        
        Args:
            original_text: Original text
            cf_text: Counterfactual text
            
        Returns:
            Tuple of (similarity_score, details_dict, passes_filter)
        """
        if self.embedding_model is None:
            self._init_embedding_model()
        
        try:
            # Compute embeddings
            orig_emb = self.embedding_model.encode([original_text])[0]
            cf_emb = self.embedding_model.encode([cf_text])[0]
            
            # Cosine similarity
            similarity = np.dot(orig_emb, cf_emb) / (
                np.linalg.norm(orig_emb) * np.linalg.norm(cf_emb)
            )
            
            # Require HIGH similarity to preserve semantics
            min_similarity = self.quality_config.get('min_semantic_similarity', 0.6)
            passes = similarity >= min_similarity
            
            details = {
                'similarity': float(similarity),
                'min_similarity': min_similarity,
                'passes': bool(passes)
            }
            
            return float(similarity), details, bool(passes)
            
        except Exception as e:
            return 0.0, {'error': str(e)}, False
    
    def filter_counterfactual(
        self,
        cf_text: str,
        original_text: str,
        target_label: str,
        original_label: str,
        classifier
    ) -> Tuple[bool, Dict]:
        """
        Apply both filters from Algorithm 1, line 14.
        
        CF must pass BOTH filters to be accepted.
        
        Args:
            cf_text: Counterfactual text
            original_text: Original text
            target_label: Intended label
            original_label: Original label
            classifier: Trained classifier
            
        Returns:
            Tuple of (passes_all_filters, details_dict)
        """
        all_details = {}
        
        # FILTER 1: Label Correctness
        _, label_details, passes_label = self.compute_label_correctness(
            cf_text, target_label, original_label, classifier
        )
        all_details['label_correctness'] = label_details
        
        if not passes_label:
            all_details['rejection_stage'] = 'label_correctness'
            all_details['passed'] = False
            return False, all_details
        
        # FILTER 2: Semantic Similarity
        _, semantic_details, passes_semantic = self.compute_semantic_similarity(
            original_text, cf_text
        )
        all_details['semantic_similarity'] = semantic_details
        
        if not passes_semantic:
            all_details['rejection_stage'] = 'semantic_similarity'
            all_details['passed'] = False
            return False, all_details
        
        # Passed both filters!
        all_details['passed'] = True
        return True, all_details
    
    def clear_cache(self):
        """Clear embedding model cache to free memory"""
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
