"""
Retrieval-based In-Context Learning Classifier

Implements retrieval-based ICL that selects the most relevant examples
for each prediction, improving performance over static example selection.

Supports multiple embedding backends:
- Sentence Transformers (default, recommended)
- OpenAI Embeddings (future)
- TF-IDF (lightweight fallback)
"""

import numpy as np
from typing import List, Dict, Optional
import time
from sklearn.metrics.pairwise import cosine_similarity

from .classifier import SimpleICLClassifier


class RetrievalICLClassifier(SimpleICLClassifier):
    """
    Base class for retrieval-based ICL classifiers.
    
    Extends SimpleICLClassifier to retrieve relevant examples dynamically
    for each prediction instead of using a static set.
    """
    
    def __init__(self, config: dict, llm_provider):
        """
        Initialize retrieval-based classifier.
        
        Args:
            config: Configuration dictionary
            llm_provider: LLM provider instance
        """
        super().__init__(config, llm_provider)
        self.retrieval_config = config['evaluation'].get('retrieval', {})
        self.k_per_class = self.retrieval_config.get('k_per_class', 3)
        self.total_k_max = self.retrieval_config.get('total_k_max', 50)
        self.fallback_strategy = self.retrieval_config.get('fallback_strategy', 'similarity')
        
        # CF inclusion strategy (mentor directive)
        # "mixed": CFs compete with factuals in retrieval pool
        # "factual_anchored": Retrieve only factuals, then attach their CFs
        self.cf_inclusion_strategy = self.retrieval_config.get('cf_inclusion_strategy', 'mixed')
        
        self.labeled_embeddings = None
        
        # For factual_anchored strategy: separate storage
        self.factual_examples = []  # Only factuals
        self.factual_embeddings = None
        self.cf_by_original_id = {}  # Map: original_id -> list of CFs
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.
        Must be implemented by subclasses.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        raise NotImplementedError("Subclasses must implement encode_texts()")
    
    def train(self, labeled_pool: List[Dict]):
        """
        Train by storing labeled examples and encoding them.
        
        Args:
            labeled_pool: List of dicts with 'text' and 'label' keys
        """
        super().train(labeled_pool)
        
        # Separate factuals and CFs for factual_anchored strategy
        self.factual_examples = []
        self.cf_by_original_id = {}
        
        for ex in self.labeled_examples:
            if ex.get('is_counterfactual', False):
                # This is a CF - store by original_id
                original_id = ex.get('original_id')
                if original_id:
                    if original_id not in self.cf_by_original_id:
                        self.cf_by_original_id[original_id] = []
                    self.cf_by_original_id[original_id].append(ex)
            else:
                # This is a factual
                self.factual_examples.append(ex)
        
        print(f"  Factuals: {len(self.factual_examples)}, CFs: {len(self.labeled_examples) - len(self.factual_examples)}")
        print(f"  CF inclusion strategy: {self.cf_inclusion_strategy}")
        
        if self.cf_inclusion_strategy == 'factual_anchored':
            # Only encode factuals for retrieval
            texts = [ex['text'] for ex in self.factual_examples]
            print(f"  Encoding {len(texts)} factuals for retrieval (factual_anchored)...")
            self.factual_embeddings = self.encode_texts(texts)
            print(f"  ✓ Encoded factuals to {self.factual_embeddings.shape} embeddings")
            # Also encode all for backward compatibility
            all_texts = [ex['text'] for ex in self.labeled_examples]
            self.labeled_embeddings = self.encode_texts(all_texts)
        else:
            # Mixed strategy: encode all examples
            texts = [ex['text'] for ex in self.labeled_examples]
            print(f"  Encoding {len(texts)} examples for retrieval (mixed)...")
            self.labeled_embeddings = self.encode_texts(texts)
            print(f"  ✓ Encoded to {self.labeled_embeddings.shape} embeddings")
    
    def retrieve_balanced_examples(self, text: str) -> List[Dict]:
        """
        Retrieve examples using configured strategy.
        
        Strategies:
        - "mixed": CFs compete with factuals in retrieval (original behavior)
        - "factual_anchored": Retrieve only factuals, then attach their CFs
        
        Args:
            text: Query text
            
        Returns:
            List of retrieved examples
        """
        if self.cf_inclusion_strategy == 'factual_anchored':
            return self.retrieve_factual_anchored(text)
        else:
            return self.retrieve_mixed(text)
    
    def retrieve_mixed(self, text: str) -> List[Dict]:
        """
        Retrieve examples using hybrid balanced strategy (original behavior).
        CFs compete with factuals in the retrieval pool.
        
        Strategy:
        1. Try to get k_per_class similar examples from each label
        2. If insufficient examples per class, use what's available
        3. Fill remaining budget with most similar examples overall
        4. Respect total_k_max limit
        
        Args:
            text: Query text
            
        Returns:
            List of retrieved examples
        """
        if self.labeled_embeddings is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Encode query
        query_embedding = self.encode_texts([text])
        
        # Compute similarities to all examples
        similarities = cosine_similarity(query_embedding, self.labeled_embeddings)[0]
        
        # Group examples by label with their similarities
        label_groups = {}
        for i, ex in enumerate(self.labeled_examples):
            label = ex['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append((i, ex, similarities[i]))
        
        # Phase 1: Retrieve k_per_class from each label
        retrieved_examples = []
        retrieved_indices = set()
        
        for label, examples in label_groups.items():
            # Sort by similarity (descending)
            examples_sorted = sorted(examples, key=lambda x: x[2], reverse=True)
            
            # Take up to k_per_class
            k = min(self.k_per_class, len(examples_sorted))
            for idx, ex, sim in examples_sorted[:k]:
                retrieved_examples.append(ex)
                retrieved_indices.add(idx)
        
        # Phase 2: Fill remaining budget with most similar examples (if space available)
        if len(retrieved_examples) < self.total_k_max and self.fallback_strategy == 'similarity':
            # Get remaining examples sorted by similarity
            remaining_candidates = [
                (i, self.labeled_examples[i], similarities[i])
                for i in range(len(self.labeled_examples))
                if i not in retrieved_indices
            ]
            remaining_sorted = sorted(remaining_candidates, key=lambda x: x[2], reverse=True)
            
            # Add until we reach total_k_max
            needed = self.total_k_max - len(retrieved_examples)
            for idx, ex, sim in remaining_sorted[:needed]:
                retrieved_examples.append(ex)
                retrieved_indices.add(idx)
        
        # Limit to total_k_max
        retrieved_examples = retrieved_examples[:self.total_k_max]
        
        return retrieved_examples
    
    def retrieve_factual_anchored(self, text: str) -> List[Dict]:
        """
        Retrieve examples using factual-anchored strategy (mentor directive).
        
        Strategy:
        1. Retrieve only from factuals (CFs never compete with factuals)
        2. For each retrieved factual, attach its associated CFs
        3. CFs are added as paired augmentations after their parent factual
        4. Respect total_k_max limit for factuals (CFs are bonus)
        
        Args:
            text: Query text
            
        Returns:
            List of retrieved examples (factuals + their CFs)
        """
        if self.factual_embeddings is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        if len(self.factual_examples) == 0:
            # No factuals - fall back to mixed strategy
            return self.retrieve_mixed(text)
        
        # Encode query
        query_embedding = self.encode_texts([text])
        
        # Compute similarities to factuals only
        similarities = cosine_similarity(query_embedding, self.factual_embeddings)[0]
        
        # Group factuals by label with their similarities
        label_groups = {}
        for i, ex in enumerate(self.factual_examples):
            label = ex['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append((i, ex, similarities[i]))
        
        # Phase 1: Retrieve k_per_class factuals from each label
        retrieved_factuals = []
        retrieved_indices = set()
        
        for label, examples in label_groups.items():
            # Sort by similarity (descending)
            examples_sorted = sorted(examples, key=lambda x: x[2], reverse=True)
            
            # Take up to k_per_class
            k = min(self.k_per_class, len(examples_sorted))
            for idx, ex, sim in examples_sorted[:k]:
                retrieved_factuals.append(ex)
                retrieved_indices.add(idx)
        
        # Phase 2: Fill remaining budget with most similar factuals (if space available)
        if len(retrieved_factuals) < self.total_k_max and self.fallback_strategy == 'similarity':
            remaining_candidates = [
                (i, self.factual_examples[i], similarities[i])
                for i in range(len(self.factual_examples))
                if i not in retrieved_indices
            ]
            remaining_sorted = sorted(remaining_candidates, key=lambda x: x[2], reverse=True)
            
            needed = self.total_k_max - len(retrieved_factuals)
            for idx, ex, sim in remaining_sorted[:needed]:
                retrieved_factuals.append(ex)
                retrieved_indices.add(idx)
        
        # Limit factuals to total_k_max
        retrieved_factuals = retrieved_factuals[:self.total_k_max]
        
        # Phase 3: Attach CFs for each retrieved factual
        final_examples = []
        for factual in retrieved_factuals:
            # Add the factual
            final_examples.append(factual)
            
            # Find and add its CFs
            factual_id = factual.get('id')
            if factual_id and factual_id in self.cf_by_original_id:
                cfs = self.cf_by_original_id[factual_id]
                for cf in cfs:
                    final_examples.append(cf)
        
        return final_examples
    
    def predict(self, text: str) -> str:
        """
        Predict label using retrieved relevant examples.
        
        Args:
            text: Text to classify
            
        Returns:
            Predicted label
        """
        if not self.labeled_examples:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Retrieve relevant examples for this query
        icl_examples = self.retrieve_balanced_examples(text)
        
        # Build ICL prompt with retrieved examples
        labels_str = ', '.join(sorted(self.labels))
        
        messages = [
            {
                "role": "system",
                "content": f"You are a text classifier. Classify sentences into one of these labels: {labels_str}. Respond with only the label, nothing else."
            }
        ]
        
        # Add retrieved ICL examples as user/assistant pairs
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
            for label in self.labels:
                if label.lower() in predicted_label.lower():
                    predicted_label = label
                    break
            else:
                predicted_label = sorted(list(self.labels))[0]
        
        return predicted_label


class SentenceTransformerRetrieval(RetrievalICLClassifier):
    """
    Retrieval-based ICL using Sentence Transformers for embeddings.
    
    Recommended for most use cases - good balance of quality and speed.
    """
    
    def __init__(self, config: dict, llm_provider):
        """
        Initialize with Sentence Transformers model.
        
        Args:
            config: Configuration dictionary
            llm_provider: LLM provider instance
        """
        super().__init__(config, llm_provider)
        
        # Import here to avoid dependency if not using this backend
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        # Disable tokenizers parallelism to avoid fork warnings
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Get model config
        st_config = self.retrieval_config.get('sentence_transformers', {})
        model_name = st_config.get('model', 'all-MiniLM-L6-v2')
        device = st_config.get('device', 'cpu')
        
        print(f"  Loading Sentence Transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=device)
        print(f"  ✓ Model loaded on {device}")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using Sentence Transformers.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32
        )
        return embeddings


class TFIDFRetrieval(RetrievalICLClassifier):
    """
    Retrieval-based ICL using TF-IDF for embeddings.
    
    Lightweight fallback that doesn't require additional dependencies.
    Lower quality than sentence transformers but faster and simpler.
    """
    
    def __init__(self, config: dict, llm_provider):
        """
        Initialize with TF-IDF vectorizer.
        
        Args:
            config: Configuration dictionary
            llm_provider: LLM provider instance
        """
        super().__init__(config, llm_provider)
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Get TF-IDF config
        tfidf_config = self.retrieval_config.get('tfidf', {})
        max_features = tfidf_config.get('max_features', 1000)
        ngram_range = tuple(tfidf_config.get('ngram_range', [1, 2]))
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            strip_accents='unicode'
        )
        print(f"  Using TF-IDF with {max_features} features, ngrams={ngram_range}")
    
    def train(self, labeled_pool: List[Dict]):
        """
        Train by fitting TF-IDF and encoding examples.
        
        Args:
            labeled_pool: List of labeled examples
        """
        # Store labeled examples and labels
        self.labeled_examples = labeled_pool.copy()
        self.labels = set(ex['label'] for ex in labeled_pool)
        print(f"  Classifier 'trained' with {len(labeled_pool)} examples across {len(self.labels)} labels")
        
        # Fit TF-IDF on all texts
        texts = [ex['text'] for ex in self.labeled_examples]
        print(f"  Fitting TF-IDF on {len(texts)} examples...")
        self.labeled_embeddings = self.vectorizer.fit_transform(texts).toarray()
        print(f"  ✓ TF-IDF matrix: {self.labeled_embeddings.shape}")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using fitted TF-IDF vectorizer.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of TF-IDF vectors
        """
        return self.vectorizer.transform(texts).toarray()


class OpenAIEmbeddingRetrieval(RetrievalICLClassifier):
    """
    Retrieval-based ICL using OpenAI embeddings.
    
    Highest quality embeddings but requires API calls and costs money.
    Use for production or when quality is critical.
    """
    
    def __init__(self, config: dict, llm_provider):
        """
        Initialize with OpenAI embeddings.
        
        Args:
            config: Configuration dictionary
            llm_provider: LLM provider instance
        """
        super().__init__(config, llm_provider)
        
        # Get OpenAI embedding config
        openai_config = self.retrieval_config.get('openai', {})
        self.embedding_model = openai_config.get('model', 'text-embedding-3-small')
        self.batch_size = openai_config.get('batch_size', 100)
        
        # Check if we have OpenAI client
        if not hasattr(self.llm_provider, 'client'):
            raise ValueError(
                "OpenAI embeddings require OpenAI provider. "
                "Current provider does not support embeddings."
            )
        
        self.client = self.llm_provider.client
        print(f"  Using OpenAI embeddings: {self.embedding_model}")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using OpenAI embeddings API.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        
        # Batch API calls
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            
            batch_embeddings = [emb.embedding for emb in response.data]
            embeddings.extend(batch_embeddings)
            
            # Rate limiting
            if len(texts) > self.batch_size:
                time.sleep(0.5)
        
        return np.array(embeddings)


class BM25Retrieval(RetrievalICLClassifier):
    """
    Retrieval-based ICL using BM25 for keyword-based retrieval.
    
    BM25 is a ranking function used in information retrieval.
    Good for lexical matching and keyword-based search.
    """
    
    def __init__(self, config: dict, llm_provider):
        """
        Initialize with BM25 ranker.
        
        Args:
            config: Configuration dictionary
            llm_provider: LLM provider instance
        """
        super().__init__(config, llm_provider)
        
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 not installed. "
                "Install with: pip install rank-bm25"
            )
        
        self.BM25Okapi = BM25Okapi
        
        # Get BM25 config
        bm25_config = self.retrieval_config.get('bm25', {})
        self.k1 = bm25_config.get('k1', 1.5)
        self.b = bm25_config.get('b', 0.75)
        
        self.bm25 = None
        self.tokenized_corpus = None
        print(f"  Using BM25 with k1={self.k1}, b={self.b}")
    
    def train(self, labeled_pool: List[Dict]):
        """
        Train by tokenizing corpus and building BM25 index.
        
        Args:
            labeled_pool: List of labeled examples
        """
        # Store labeled examples and labels
        self.labeled_examples = labeled_pool.copy()
        self.labels = set(ex['label'] for ex in labeled_pool)
        print(f"  Classifier 'trained' with {len(labeled_pool)} examples across {len(self.labels)} labels")
        
        # Separate factuals and CFs for factual_anchored strategy
        self.factual_examples = []
        self.cf_by_original_id = {}
        
        for ex in self.labeled_examples:
            if ex.get('is_counterfactual', False):
                original_id = ex.get('original_id')
                if original_id:
                    if original_id not in self.cf_by_original_id:
                        self.cf_by_original_id[original_id] = []
                    self.cf_by_original_id[original_id].append(ex)
            else:
                self.factual_examples.append(ex)
        
        print(f"  Factuals: {len(self.factual_examples)}, CFs: {len(self.labeled_examples) - len(self.factual_examples)}")
        print(f"  CF inclusion strategy: {self.cf_inclusion_strategy}")
        
        if self.cf_inclusion_strategy == 'factual_anchored':
            # Build BM25 index on factuals only
            texts = [ex['text'] for ex in self.factual_examples]
            print(f"  Tokenizing {len(texts)} factuals for BM25 (factual_anchored)...")
            self.tokenized_corpus_factuals = [text.lower().split() for text in texts]
            self.bm25_factuals = self.BM25Okapi(self.tokenized_corpus_factuals, k1=self.k1, b=self.b)
            print(f"  ✓ BM25 factuals index built")
        
        # Also build full index for mixed strategy
        texts = [ex['text'] for ex in self.labeled_examples]
        print(f"  Tokenizing {len(texts)} examples for BM25...")
        self.tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = self.BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        print(f"  ✓ BM25 index built")
        
        # For compatibility with base class, create dummy embeddings
        self.labeled_embeddings = np.eye(len(texts))
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        BM25 doesn't use embeddings, but we need to implement this for interface compatibility.
        Returns dummy embeddings.
        
        Args:
            texts: List of texts (not used)
            
        Returns:
            Dummy array (not used for BM25)
        """
        # Return dummy embeddings (not used for BM25 retrieval)
        return np.zeros((len(texts), max(1, len(self.labeled_examples))))
    
    def retrieve_balanced_examples(self, text: str) -> List[Dict]:
        """
        Retrieve examples using configured strategy.
        
        Args:
            text: Query text
            
        Returns:
            List of retrieved examples
        """
        if self.cf_inclusion_strategy == 'factual_anchored':
            return self.retrieve_factual_anchored_bm25(text)
        else:
            return self.retrieve_mixed_bm25(text)
    
    def retrieve_mixed_bm25(self, text: str) -> List[Dict]:
        """
        Retrieve examples using BM25 scores (mixed strategy).
        
        Args:
            text: Query text
            
        Returns:
            List of retrieved examples
        """
        if self.bm25 is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Tokenize query
        tokenized_query = text.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Group examples by label with their scores
        label_groups = {}
        for i, ex in enumerate(self.labeled_examples):
            label = ex['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append((i, scores[i]))
        
        # Select top k_per_class from each label
        selected_indices = set()
        for label in label_groups:
            label_examples = sorted(label_groups[label], key=lambda x: x[1], reverse=True)
            top_k = min(self.k_per_class, len(label_examples))
            for idx, _ in label_examples[:top_k]:
                selected_indices.add(idx)
        
        # If we have room and fallback is similarity, add more top-scoring examples
        if len(selected_indices) < self.total_k_max and self.fallback_strategy == 'similarity':
            remaining = self.total_k_max - len(selected_indices)
            all_sorted = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            for idx, score in all_sorted:
                if idx not in selected_indices and len(selected_indices) < self.total_k_max:
                    selected_indices.add(idx)
        
        # Convert to list and get examples
        selected_examples = [self.labeled_examples[i] for i in sorted(selected_indices)]
        
        # Limit to total_k_max
        if len(selected_examples) > self.total_k_max:
            selected_examples = selected_examples[:self.total_k_max]
        
        return selected_examples
    
    def retrieve_factual_anchored_bm25(self, text: str) -> List[Dict]:
        """
        Retrieve examples using BM25 with factual-anchored strategy.
        
        Args:
            text: Query text
            
        Returns:
            List of retrieved examples (factuals + their CFs)
        """
        if not hasattr(self, 'bm25_factuals') or self.bm25_factuals is None:
            # Fall back to mixed if factuals index not built
            return self.retrieve_mixed_bm25(text)
        
        if len(self.factual_examples) == 0:
            return self.retrieve_mixed_bm25(text)
        
        # Tokenize query
        tokenized_query = text.lower().split()
        
        # Get BM25 scores for factuals only
        scores = self.bm25_factuals.get_scores(tokenized_query)
        
        # Group factuals by label with their scores
        label_groups = {}
        for i, ex in enumerate(self.factual_examples):
            label = ex['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append((i, ex, scores[i]))
        
        # Select top k_per_class factuals from each label
        retrieved_factuals = []
        retrieved_indices = set()
        
        for label, examples in label_groups.items():
            examples_sorted = sorted(examples, key=lambda x: x[2], reverse=True)
            top_k = min(self.k_per_class, len(examples_sorted))
            for idx, ex, score in examples_sorted[:top_k]:
                retrieved_factuals.append(ex)
                retrieved_indices.add(idx)
        
        # Fill remaining budget with top-scoring factuals
        if len(retrieved_factuals) < self.total_k_max and self.fallback_strategy == 'similarity':
            remaining_candidates = [
                (i, self.factual_examples[i], scores[i])
                for i in range(len(self.factual_examples))
                if i not in retrieved_indices
            ]
            remaining_sorted = sorted(remaining_candidates, key=lambda x: x[2], reverse=True)
            
            needed = self.total_k_max - len(retrieved_factuals)
            for idx, ex, score in remaining_sorted[:needed]:
                retrieved_factuals.append(ex)
                retrieved_indices.add(idx)
        
        # Limit factuals to total_k_max
        retrieved_factuals = retrieved_factuals[:self.total_k_max]
        
        # Attach CFs for each retrieved factual
        final_examples = []
        for factual in retrieved_factuals:
            final_examples.append(factual)
            factual_id = factual.get('id')
            if factual_id and factual_id in self.cf_by_original_id:
                for cf in self.cf_by_original_id[factual_id]:
                    final_examples.append(cf)
        
        return final_examples


class ContrieverRetrieval(RetrievalICLClassifier):
    """
    Retrieval-based ICL using Contriever embeddings.
    
    Contriever is a contrastive learning model for dense retrieval.
    Good balance between quality and efficiency.
    """
    
    def __init__(self, config: dict, llm_provider):
        """
        Initialize with Contriever model.
        
        Args:
            config: Configuration dictionary
            llm_provider: LLM provider instance
        """
        super().__init__(config, llm_provider)
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch not installed. "
                "Install with: pip install transformers torch"
            )
        
        # Get Contriever config
        contriever_config = self.retrieval_config.get('contriever', {})
        model_name = contriever_config.get('model', 'facebook/contriever')
        device = contriever_config.get('device', 'cpu')
        
        print(f"  Loading Contriever model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()
        print(f"  ✓ Contriever loaded on {device}")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using Contriever model.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        import torch
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use mean pooling of last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings


class BGELargeRetrieval(RetrievalICLClassifier):
    """
    Retrieval-based ICL using BGE-Large embeddings.
    
    BGE-Large (BAAI/bge-large-en-v1.5) is one of the best semantic retrievers.
    Recommended for final results and high-quality retrieval.
    """
    
    def __init__(self, config: dict, llm_provider):
        """
        Initialize with BGE-Large model.
        
        Args:
            config: Configuration dictionary
            llm_provider: LLM provider instance
        """
        super().__init__(config, llm_provider)
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        # Disable tokenizers parallelism
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Get BGE-Large config
        bge_config = self.retrieval_config.get('bge_large', {})
        model_name = bge_config.get('model', 'BAAI/bge-large-en-v1.5')
        device = bge_config.get('device', 'cpu')
        self.normalize_embeddings = bge_config.get('normalize_embeddings', True)
        
        print(f"  Loading BGE-Large model: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=device)
        print(f"  ✓ BGE-Large loaded on {device}")
        print(f"  Normalize embeddings: {self.normalize_embeddings}")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using BGE-Large model.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of normalized embeddings
        """
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
            normalize_embeddings=self.normalize_embeddings
        )
        return embeddings


def get_retrieval_classifier(config: dict, llm_provider):
    """
    Factory function to create appropriate retrieval classifier based on config.
    
    Args:
        config: Configuration dictionary
        llm_provider: LLM provider instance
        
    Returns:
        RetrievalICLClassifier instance
    """
    retrieval_config = config['evaluation'].get('retrieval', {})
    backend = retrieval_config.get('embedding_backend', 'sentence_transformers')
    
    if backend == 'sentence_transformers':
        return SentenceTransformerRetrieval(config, llm_provider)
    elif backend == 'openai':
        return OpenAIEmbeddingRetrieval(config, llm_provider)
    elif backend == 'tfidf':
        return TFIDFRetrieval(config, llm_provider)
    elif backend == 'bm25':
        return BM25Retrieval(config, llm_provider)
    elif backend == 'contriever':
        return ContrieverRetrieval(config, llm_provider)
    elif backend == 'bge_large':
        return BGELargeRetrieval(config, llm_provider)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}. "
                        f"Options: sentence_transformers, openai, tfidf, bm25, contriever, bge_large")

