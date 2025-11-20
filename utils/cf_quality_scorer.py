"""
Counterfactual Quality Scoring Module

Provides quality metrics for filtering counterfactual examples:
- Diversity: How different is the CF from existing examples?
- Confidence: How confident is the classifier about the CF's label?
- Validity: Is the CF a reasonable transformation of the original?
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import os

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CFQualityScorer:
    """Quality scorer for counterfactual examples"""
    
    def __init__(self, config: dict):
        """
        Initialize quality scorer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cf_config = config['active_learning']['counterfactuals']
        self.quality_config = self.cf_config['quality_filtering']
        
        # Initialize embedding model for diversity calculation
        self.embedding_model = None
        if self.quality_config.get('enabled', False):
            self._init_embedding_model()
        
        # Cache for labeled pool embeddings
        self.pool_embeddings_cache = None
        self.pool_texts_cache = None
    
    def _init_embedding_model(self):
        """Initialize the embedding model for diversity scoring"""
        try:
            # Use same model as retrieval if using sentence_transformers
            eval_config = self.config.get('evaluation', {})
            if eval_config.get('classifier_type') == 'retrieval':
                retrieval_config = eval_config.get('retrieval', {})
                if retrieval_config.get('embedding_backend') == 'sentence_transformers':
                    model_name = retrieval_config['sentence_transformers']['model']
                else:
                    model_name = 'all-MiniLM-L6-v2'  # Default
            else:
                model_name = 'all-MiniLM-L6-v2'  # Default
            
            self.embedding_model = SentenceTransformer(model_name)
            print(f"  Initialized embedding model for CF quality scoring: {model_name}")
        except Exception as e:
            print(f"  Warning: Could not initialize embedding model: {e}")
            self.embedding_model = None
    
    def compute_diversity_score(self, cf_text: str, labeled_pool_texts: List[str]) -> float:
        """
        Compute diversity score: how different is CF from existing examples?
        
        Higher score = more diverse (more different from existing examples)
        
        Args:
            cf_text: Counterfactual text
            labeled_pool_texts: List of texts in labeled pool
        
        Returns:
            Diversity score (0.0-1.0)
        """
        if not self.embedding_model or len(labeled_pool_texts) == 0:
            return 0.5  # Neutral score if no embedding model or empty pool
        
        try:
            # Embed the CF
            cf_embedding = self.embedding_model.encode([cf_text], show_progress_bar=False)[0]
            
            # Use cached embeddings if available
            if (self.pool_texts_cache is not None and 
                len(self.pool_texts_cache) == len(labeled_pool_texts) and
                all(a == b for a, b in zip(self.pool_texts_cache[:10], labeled_pool_texts[:10]))):
                # Cache is valid
                pool_embeddings = self.pool_embeddings_cache
            else:
                # Recompute embeddings
                pool_embeddings = self.embedding_model.encode(labeled_pool_texts, show_progress_bar=False)
                self.pool_embeddings_cache = pool_embeddings
                self.pool_texts_cache = labeled_pool_texts.copy()
            
            # Compute cosine similarities
            similarities = []
            cf_norm = np.linalg.norm(cf_embedding)
            for pool_emb in pool_embeddings:
                pool_norm = np.linalg.norm(pool_emb)
                if cf_norm > 0 and pool_norm > 0:
                    sim = np.dot(cf_embedding, pool_emb) / (cf_norm * pool_norm)
                    similarities.append(sim)
            
            if not similarities:
                return 0.5
            
            # Diversity = 1 - max_similarity (most conservative)
            max_similarity = max(similarities)
            diversity_score = 1.0 - max_similarity
            
            return max(0.0, min(1.0, diversity_score))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"    Warning: Error computing diversity score: {e}")
            return 0.5
    
    def compute_confidence_score(self, cf_text: str, target_label: str, classifier) -> Tuple[float, bool]:
        """
        Compute confidence score: how confident is classifier about CF's label?
        
        Higher score = classifier more confident
        
        Args:
            cf_text: Counterfactual text
            target_label: Intended label for the CF
            classifier: Trained classifier with predict_proba method
        
        Returns:
            Tuple of (confidence_score, passes_min_threshold)
        """
        try:
            # Get prediction probabilities
            probs = classifier.predict_proba([cf_text])[0]
            
            # Get classifier's label ordering (same as predict_proba does)
            if hasattr(classifier, 'labels'):
                # classifier.labels is a set, convert to sorted list
                label_list = sorted(list(classifier.labels))
                label_to_idx = {label: i for i, label in enumerate(label_list)}
            else:
                # Fallback: try to infer from predict
                pred = classifier.predict(cf_text)
                label_list = [pred]
                label_to_idx = {pred: 0}
            
            # Find probability for target label using label_to_idx mapping
            if target_label in label_to_idx:
                label_idx = label_to_idx[target_label]
                confidence = probs[label_idx]
            else:
                # Target label not in classifier's vocabulary
                confidence = 0.0
            
            # Check minimum threshold
            min_conf = self.quality_config['confidence'].get('min_confidence', 0.5)
            passes_threshold = confidence >= min_conf
            
            return confidence, passes_threshold
            
        except Exception as e:
            print(f"    Warning: Error computing confidence score: {e}")
            return 0.5, True  # Neutral score, pass by default
    
    def compute_validity_score(self, original_text: str, cf_text: str) -> float:
        """
        Compute validity score: is CF a reasonable transformation?
        
        Target: moderate similarity (not too similar, not too different)
        Sweet spot around 0.4-0.6 similarity
        
        Args:
            original_text: Original text
            cf_text: Counterfactual text
        
        Returns:
            Validity score (0.0-1.0)
        """
        if not self.embedding_model:
            return 0.5  # Neutral score if no embedding model
        
        try:
            # Embed both texts
            embeddings = self.embedding_model.encode([original_text, cf_text], show_progress_bar=False)
            orig_emb, cf_emb = embeddings[0], embeddings[1]
            
            # Compute cosine similarity
            orig_norm = np.linalg.norm(orig_emb)
            cf_norm = np.linalg.norm(cf_emb)
            
            if orig_norm > 0 and cf_norm > 0:
                similarity = np.dot(orig_emb, cf_emb) / (orig_norm * cf_norm)
            else:
                similarity = 0.0
            
            # Validity: penalize being too similar or too different
            # Target similarity: 0.5 (50% similar, 50% different)
            target_sim = 0.5
            deviation = abs(similarity - target_sim)
            validity_score = 1.0 - (deviation * 2.0)  # Scale to [0, 1]
            
            return max(0.0, min(1.0, validity_score))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"    Warning: Error computing validity score: {e}")
            return 0.5
    
    def compute_combined_score(
        self, 
        cf_text: str,
        original_text: str, 
        target_label: str,
        labeled_pool_texts: List[str],
        classifier
    ) -> Tuple[float, Dict, bool]:
        """
        Compute combined quality score using weighted metrics.
        
        Args:
            cf_text: Counterfactual text
            original_text: Original text
            target_label: Intended label
            labeled_pool_texts: List of texts in labeled pool
            classifier: Trained classifier
        
        Returns:
            Tuple of (combined_score, score_dict, passes_filters)
        """
        metric = self.quality_config.get('metric', 'combined')
        
        # Compute individual scores
        diversity = self.compute_diversity_score(cf_text, labeled_pool_texts)
        confidence, passes_conf = self.compute_confidence_score(cf_text, target_label, classifier)
        validity = self.compute_validity_score(original_text, cf_text)
        
        score_dict = {
            'diversity': diversity,
            'confidence': confidence,
            'validity': validity
        }
        
        # Check if passes minimum thresholds
        passes_filters = passes_conf
        
        # Compute combined score based on metric
        if metric == 'diversity':
            combined = diversity
        elif metric == 'confidence':
            combined = confidence
        elif metric == 'validity':
            combined = validity
        elif metric == 'combined':
            # Weighted combination
            w_div = self.quality_config.get('diversity_weight', 0.5)
            w_conf = self.quality_config.get('confidence_weight', 0.3)
            w_val = self.quality_config.get('validity_weight', 0.2)
            combined = w_div * diversity + w_conf * confidence + w_val * validity
        else:
            # Default to combined
            combined = 0.5 * diversity + 0.3 * confidence + 0.2 * validity
        
        # Convert all scores to Python native floats for JSON serialization
        return float(combined), {
            'diversity': float(diversity),
            'confidence': float(confidence),
            'validity': float(validity)
        }, passes_filters
    
    def clear_cache(self):
        """Clear the embeddings cache"""
        self.pool_embeddings_cache = None
        self.pool_texts_cache = None

