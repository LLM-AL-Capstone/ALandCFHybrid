"""
Counterfactual Quality Scoring Module (Version 3)

Provides quality metrics for filtering and scoring counterfactual examples:
- Filter 1: Label-consistency (3 explicit conditions)
- Filter 2: Semantic similarity band
- Filter 3: Length ratio constraint
- V3 Scoring: score(u') = (1 - p(y_label | u')) + β * p(y_target | u') + α * cos(E(u), E(u'))
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import os

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CFQualityScorer:
    """Quality scorer for counterfactual examples (Version 3: Enhanced filtering + scoring)"""
    
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
        
        # V3 parameters with backward compatibility
        self.tau_conf = self.quality_config.get('tau_conf', self.quality_config.get('min_target_confidence', 0.4))
        self.delta = self.quality_config.get('delta', self.quality_config.get('min_margin', 0.1))
        self.s_min = self.quality_config.get('s_min', self.quality_config.get('min_semantic_similarity', 0.7))
        self.s_max = self.quality_config.get('s_max', 0.98)
        self.r_min = self.quality_config.get('r_min', 0.7)
        self.r_max = self.quality_config.get('r_max', 1.3)
        self.alpha = self.quality_config.get('alpha', 0.3)
        self.beta = self.quality_config.get('beta', 0.5)
        
        print(f"  Initialized CF Quality Scorer (Version 3):")
        print(f"    - Label-consistency: tau_conf={self.tau_conf}, delta={self.delta}")
        print(f"    - Semantic similarity band: [{self.s_min}, {self.s_max}]")
        print(f"    - Length ratio: [{self.r_min}, {self.r_max}]")
        print(f"    - Scoring weights: alpha={self.alpha}, beta={self.beta}")
    
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
        FILTER 1: Label-consistency (Version 3: 3 explicit conditions).
        
        A CF candidate u' targeting y_target is retained only if:
        1. p(y_target | u') > p(y_label | u')  [strict inequality]
        2. p(y_target | u') >= tau_conf        [confidence threshold]
        3. p(y_target | u') - p(y_label | u') >= delta  [margin threshold]
        
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
            
            # Check all 3 conditions explicitly
            condition1 = p_target > p_orig  # Strict inequality
            condition2 = p_target >= self.tau_conf  # Confidence threshold
            condition3 = margin >= self.delta  # Margin threshold
            
            passes_filter = condition1 and condition2 and condition3
            
            details = {
                'p_target': float(p_target),
                'p_orig': float(p_orig),
                'margin': float(margin),
                'tau_conf': self.tau_conf,
                'delta': self.delta,
                'condition1_passes': bool(condition1),  # p_target > p_orig
                'condition2_passes': bool(condition2),  # p_target >= tau_conf
                'condition3_passes': bool(condition3),  # margin >= delta
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
        FILTER 2: Semantic similarity band (Version 3).
        
        Constraint: s_min <= cos(E(u), E(u')) <= s_max
        Ensures CF is similar but not identical to original.
        
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
            
            # Check band constraint: s_min <= similarity <= s_max
            passes = self.s_min <= similarity <= self.s_max
            
            details = {
                'similarity': float(similarity),
                's_min': self.s_min,
                's_max': self.s_max,
                'passes': bool(passes),
                'too_low': bool(similarity < self.s_min),
                'too_high': bool(similarity > self.s_max)
            }
            
            return float(similarity), details, bool(passes)
            
        except Exception as e:
            return 0.0, {'error': str(e)}, False
    
    def compute_length_ratio(
        self,
        original_text: str,
        cf_text: str
    ) -> Tuple[float, Dict, bool]:
        """
        FILTER 3: Length ratio constraint (Version 3).
        
        Constraint: r_min <= |u'| / |u| <= r_max
        Prevents degenerate outputs (too short or too long).
        
        Args:
            original_text: Original text
            cf_text: Counterfactual text
            
        Returns:
            Tuple of (ratio, details_dict, passes_filter)
        """
        try:
            orig_len = len(original_text)
            cf_len = len(cf_text)
            
            # Avoid division by zero
            if orig_len == 0:
                ratio = 1.0 if cf_len == 0 else float('inf')
            else:
                ratio = cf_len / orig_len
            
            # Check constraint: r_min <= ratio <= r_max
            passes = self.r_min <= ratio <= self.r_max
            
            details = {
                'ratio': float(ratio),
                'original_length': orig_len,
                'cf_length': cf_len,
                'r_min': self.r_min,
                'r_max': self.r_max,
                'passes': bool(passes),
                'too_short': bool(ratio < self.r_min),
                'too_long': bool(ratio > self.r_max)
            }
            
            return float(ratio), details, bool(passes)
            
        except Exception as e:
            return 1.0, {'error': str(e)}, False
    
    def filter_counterfactual(
        self,
        cf_text: str,
        original_text: str,
        target_label: str,
        original_label: str,
        classifier
    ) -> Tuple[bool, Dict]:
        """
        Apply all 3 filters (Version 3).
        
        CF must pass ALL filters to be accepted:
        1. Label-consistency (3 explicit conditions)
        2. Semantic similarity band
        3. Length ratio constraint
        
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
        
        # FILTER 1: Label-consistency (3 conditions)
        _, label_details, passes_label = self.compute_label_correctness(
            cf_text, target_label, original_label, classifier
        )
        all_details['label_consistency'] = label_details
        
        if not passes_label:
            all_details['rejection_stage'] = 'label_consistency'
            all_details['passed'] = False
            return False, all_details
        
        # FILTER 2: Semantic similarity band
        _, semantic_details, passes_semantic = self.compute_semantic_similarity(
            original_text, cf_text
        )
        all_details['semantic_similarity'] = semantic_details
        
        if not passes_semantic:
            all_details['rejection_stage'] = 'semantic_similarity'
            all_details['passed'] = False
            return False, all_details
        
        # FILTER 3: Length ratio constraint
        _, length_details, passes_length = self.compute_length_ratio(
            original_text, cf_text
        )
        all_details['length_ratio'] = length_details
        
        if not passes_length:
            all_details['rejection_stage'] = 'length_ratio'
            all_details['passed'] = False
            return False, all_details
        
        # Passed all 3 filters!
        all_details['passed'] = True
        return True, all_details
    
    def compute_v3_score(
        self,
        cf_text: str,
        original_text: str,
        target_label: str,
        original_label: str,
        classifier
    ) -> Tuple[float, Dict]:
        """
        Compute V3 scoring formula for ranking CFs.
        
        Formula: score(u') = (1 - p(y_label | u')) + β * p(y_target | u') + α * cos(E(u), E(u'))
        
        Args:
            cf_text: Counterfactual text
            original_text: Original text
            target_label: Intended label
            original_label: Original label
            classifier: Trained classifier
            
        Returns:
            Tuple of (score, details_dict)
        """
        if self.embedding_model is None:
            self._init_embedding_model()
        
        try:
            # Get classifier probabilities
            probs = classifier.predict_proba([cf_text])[0]
            labels = classifier.get_labels()
            label_to_idx = {label: idx for idx, label in enumerate(labels)}
            
            target_idx = label_to_idx.get(target_label, -1)
            orig_idx = label_to_idx.get(original_label, -1)
            
            if target_idx == -1 or orig_idx == -1:
                return 0.0, {'error': 'Invalid labels'}
            
            p_target = probs[target_idx]
            p_orig = probs[orig_idx]
            
            # Compute semantic similarity
            orig_emb = self.embedding_model.encode([original_text])[0]
            cf_emb = self.embedding_model.encode([cf_text])[0]
            similarity = np.dot(orig_emb, cf_emb) / (
                np.linalg.norm(orig_emb) * np.linalg.norm(cf_emb)
            )
            
            # V3 scoring formula
            score = (1.0 - p_orig) + self.beta * p_target + self.alpha * similarity
            
            details = {
                'score': float(score),
                'p_orig': float(p_orig),
                'p_target': float(p_target),
                'similarity': float(similarity),
                'term1': float(1.0 - p_orig),  # (1 - p(y_label | u'))
                'term2': float(self.beta * p_target),  # β * p(y_target | u')
                'term3': float(self.alpha * similarity),  # α * cos(E(u), E(u'))
                'alpha': self.alpha,
                'beta': self.beta
            }
            
            return float(score), details
            
        except Exception as e:
            return 0.0, {'error': str(e)}
    
    def clear_cache(self):
        """Clear embedding model cache to free memory"""
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
