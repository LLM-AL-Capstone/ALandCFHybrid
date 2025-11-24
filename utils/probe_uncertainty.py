"""
Probe-based Entropy Uncertainty Estimator (ACT-ICL v2)

Implements embedding-based probe uncertainty for active learning.
Trains a Logistic Regression probe on sentence embeddings to estimate
uncertainty without requiring LLM API calls.

Formula: U_{LR-Emb}(u; L_{t-1}) = H_probe(u) = -Σ_y p̂(y|u) log p̂(y|u)
where p̂(y|u) = LR(v_u) and v_u = E_embed(u)
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import LogisticRegression
import os

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ProbeUncertaintyEstimator:
    """
    Probe-based uncertainty estimator for ACT-ICL v2.
    
    Trains a Logistic Regression probe on embeddings of labeled examples,
    then uses it to compute entropy-based uncertainty for unlabeled examples.
    """
    
    def __init__(self, config: dict):
        """
        Initialize probe uncertainty estimator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        al_config = config.get('active_learning', {})
        probe_config = al_config.get('probe_uncertainty', {})
        
        # Get embedding model configuration
        self.embedding_model_name = probe_config.get('embedding_model', 'bge-large-en-v1.5')
        self.device = probe_config.get('device', 'cpu')
        
        # LR probe training settings
        self.max_iter = probe_config.get('max_iter', 1000)
        self.C = probe_config.get('C', 1.0)
        
        # Initialize embedding model (lazy loading)
        self.embedding_model = None
        self.lr_probe = None
        self.label_list = None
        
        # Embedding cache for incremental updates
        self.embedding_cache: Dict[str, np.ndarray] = {}  # text → embedding
        self.cached_texts: set = set()  # Track which texts we've embedded
        
        print(f"  Initialized Probe Uncertainty Estimator (V2):")
        print(f"    Embedding model: {self.embedding_model_name}")
        print(f"    Device: {self.device}")
        print(f"    LR max_iter: {self.max_iter}, C: {self.C}")
        print(f"    Embedding cache: Enabled (incremental updates)")
    
    def _init_embedding_model(self):
        """Lazy initialization of embedding model"""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                # Handle special models
                if self.embedding_model_name == "E5-large-v2":
                    model_name = "intfloat/e5-large-v2"
                elif self.embedding_model_name == "bge-large-en-v1.5":
                    model_name = "BAAI/bge-large-en-v1.5"
                elif self.embedding_model_name == "gpt-5-mini-embed":
                    # This would require OpenAI API - for now, use BGE as fallback
                    print("  Warning: GPT-5-mini-embed not yet implemented, using BGE-Large")
                    model_name = "BAAI/bge-large-en-v1.5"
                else:
                    model_name = self.embedding_model_name
                
                print(f"  Loading embedding model: {model_name}")
                self.embedding_model = SentenceTransformer(model_name, device=self.device)
                print(f"  ✓ Model loaded on {self.device}")
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model {self.embedding_model_name}: {e}")
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if self.embedding_model is None:
            self._init_embedding_model()
        
        # Compute embeddings in batch
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
            normalize_embeddings=False  # Don't normalize for probe training
        )
        
        return embeddings
    
    def _train_lr_probe(self, embeddings: np.ndarray, labels: List[str]) -> LogisticRegression:
        """
        Train Logistic Regression probe on embeddings.
        
        Args:
            embeddings: Array of embeddings (n_examples, embedding_dim)
            labels: List of label strings
            
        Returns:
            Trained LogisticRegression model
        """
        # Get unique labels and create label mapping
        unique_labels = sorted(list(set(labels)))
        self.label_list = unique_labels
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Convert labels to indices
        label_indices = np.array([label_to_idx[label] for label in labels])
        
        # Train Logistic Regression
        # Note: multi_class parameter removed (deprecated in sklearn 1.5+)
        # Default behavior is correct: 'multinomial' for multi-class with 'lbfgs' solver
        lr = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            solver='lbfgs',  # Good for small-medium datasets, automatically uses multinomial for multi-class
            random_state=42,  # For reproducibility
            n_jobs=1  # Single-threaded for consistency
        )
        
        print(f"  Training LR probe on {len(embeddings)} examples with {len(unique_labels)} classes...")
        lr.fit(embeddings, label_indices)
        print(f"  ✓ LR probe trained")
        
        return lr
    
    def _compute_entropy(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute entropy from probability distributions.
        
        Formula: H(p) = -Σ_y p(y) log p(y)
        
        Args:
            probs: Array of probabilities (n_examples, n_classes)
            
        Returns:
            Array of entropy values (n_examples,)
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        return entropy
    
    def train_probe(self, labeled_pool: List[Dict]):
        """
        Train LR probe on labeled pool embeddings.
        
        This should be called at the start of each iteration with the
        updated labeled pool L_{t-1}.
        
        Uses incremental embedding caching: only computes embeddings for new texts,
        reuses cached embeddings for existing texts.
        
        Args:
            labeled_pool: List of labeled examples (dicts with 'text' and 'label' keys)
        """
        if len(labeled_pool) == 0:
            raise ValueError("Cannot train probe on empty labeled pool")
        
        # Extract texts and labels
        texts = [ex['text'] for ex in labeled_pool]
        labels = [ex['label'] for ex in labeled_pool]
        
        # Identify new texts that need embedding computation
        new_texts = [text for text in texts if text not in self.cached_texts]
        
        # Compute embeddings only for new texts
        if new_texts:
            print(f"  Computing embeddings for {len(new_texts)} new examples (cache hit: {len(texts) - len(new_texts)})...")
            new_embeddings = self._compute_embeddings(new_texts)
            
            # Update cache
            for text, emb in zip(new_texts, new_embeddings):
                self.embedding_cache[text] = emb
                self.cached_texts.add(text)
        else:
            print(f"  All {len(texts)} examples found in cache (no new embeddings needed)")
        
        # Build full embedding array in the same order as labeled_pool
        # (combining cached and new embeddings)
        embeddings = np.array([self.embedding_cache[text] for text in texts])
        
        # Train LR probe
        self.lr_probe = self._train_lr_probe(embeddings, labels)
    
    def compute_uncertainty(self, unlabeled_pool: List[Dict]) -> np.ndarray:
        """
        Compute probe-based entropy uncertainty for unlabeled examples.
        
        Formula: H_probe(u) = -Σ_y p̂(y|u) log p̂(y|u)
        where p̂(y|u) = LR(v_u) and v_u = E_embed(u)
        
        Args:
            unlabeled_pool: List of unlabeled examples (dicts with 'text' key)
            
        Returns:
            Array of uncertainty scores (entropy values) for each example
        """
        if self.lr_probe is None:
            raise ValueError("Probe not trained. Call train_probe() first.")
        
        if len(unlabeled_pool) == 0:
            return np.array([])
        
        # Extract texts
        texts = [ex['text'] for ex in unlabeled_pool]
        
        # Compute embeddings
        embeddings = self._compute_embeddings(texts)
        
        # Get probability predictions from probe
        probs = self.lr_probe.predict_proba(embeddings)
        
        # Compute entropy
        entropy_scores = self._compute_entropy(probs)
        
        return entropy_scores
    
    def get_labels(self) -> List[str]:
        """
        Get list of labels that the probe was trained on.
        
        Returns:
            List of label strings
        """
        if self.label_list is None:
            raise ValueError("Probe not trained. Call train_probe() first.")
        return self.label_list.copy()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get probability predictions from probe (for compatibility with classifier interface).
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of probabilities (n_texts, n_classes)
        """
        if self.lr_probe is None:
            raise ValueError("Probe not trained. Call train_probe() first.")
        
        # Compute embeddings
        embeddings = self._compute_embeddings(texts)
        
        # Get probabilities
        probs = self.lr_probe.predict_proba(embeddings)
        
        return probs
    
    def clear_cache(self):
        """Clear embedding model and embedding cache to free memory"""
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
        
        # Clear embedding cache
        self.embedding_cache.clear()
        self.cached_texts.clear()
        print("  ✓ Embedding cache cleared")

