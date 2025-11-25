"""
Retrieval-based In-Context Learning Classifier for Full-ICL Oracle Baseline

Supports BM25, Contriever, and BGE-Large retrieval methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics import accuracy_score

# Retrieval dependencies
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("WARNING: rank-bm25 not installed. Install with: pip install rank-bm25")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("WARNING: sentence-transformers not installed. Install with: pip install sentence-transformers")


class RetrievalICLClassifier:
    """
    In-Context Learning classifier with retrieval-based example selection.
    
    For each test example, retrieves k most similar training examples
    using BM25, Contriever, or BGE-Large, then uses them as few-shot examples for LLM.
    """
    
    def __init__(self, llm_provider, retrieval_method: str = 'bm25', k: int = 10):
        """
        Initialize retrieval-based ICL classifier.
        
        Args:
            llm_provider: LLM provider instance
            retrieval_method: 'bm25', 'contriever', or 'bge-large'
            k: Number of examples to retrieve for ICL
        """
        self.llm_provider = llm_provider
        self.retrieval_method = retrieval_method.lower()
        self.k = k
        
        # Storage
        self.train_texts = []
        self.train_labels = []
        self.label_set = set()
        
        # Retrieval components
        self.bm25 = None
        self.retriever_model = None
        self.train_embeddings = None
        
        # Load retrieval model
        if self.retrieval_method == 'contriever':
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
            print(f"  Loading Contriever model (facebook/contriever)...")
            self.retriever_model = SentenceTransformer('facebook/contriever')
        elif self.retrieval_method == 'bge-large':
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
            print(f"  Loading BGE-Large model (BAAI/bge-large-en-v1.5)...")
            self.retriever_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        elif self.retrieval_method == 'bm25':
            if not HAS_BM25:
                raise ImportError("rank-bm25 required. Install: pip install rank-bm25")
    
    def fit(self, train_df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label'):
        """
        Build retrieval index from training data.
        
        Args:
            train_df: Training DataFrame
            text_col: Column name for text
            label_col: Column name for label
        """
        self.train_texts = train_df[text_col].tolist()
        self.train_labels = train_df[label_col].tolist()
        self.label_set = set(self.train_labels)
        
        print(f"  Building {self.retrieval_method.upper()} index for {len(self.train_texts)} examples...")
        
        if self.retrieval_method == 'bm25':
            # Build BM25 index
            tokenized_corpus = [text.lower().split() for text in self.train_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"  BM25 index built successfully")
            
        elif self.retrieval_method in ['contriever', 'bge-large']:
            # Encode all training examples
            self.train_embeddings = self.retriever_model.encode(
                self.train_texts,
                show_progress_bar=True,
                convert_to_tensor=True,
                batch_size=32
            )
            print(f"  {self.retrieval_method.upper()} embeddings computed ({self.train_embeddings.shape})")
    
    def _retrieve_examples(self, query_text: str) -> List[int]:
        """
        Retrieve top-k most similar example indices.
        
        Args:
            query_text: Query text
            
        Returns:
            List of top-k example indices
        """
        if self.retrieval_method == 'bm25':
            # BM25 retrieval
            tokenized_query = query_text.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_k_indices = np.argsort(scores)[::-1][:self.k]
            return top_k_indices.tolist()
            
        elif self.retrieval_method in ['contriever', 'bge-large']:
            # Dense embedding retrieval (Contriever or BGE-Large)
            query_embedding = self.retriever_model.encode(
                query_text,
                convert_to_tensor=True
            )
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                self.train_embeddings
            )
            top_k_indices = torch.argsort(similarities, descending=True)[:self.k]
            return top_k_indices.cpu().numpy().tolist()
    
    def predict(self, text: str) -> str:
        """
        Predict label for a single text using retrieval-based ICL.
        
        Args:
            text: Text to classify
            
        Returns:
            Predicted label
        """
        # Retrieve k most similar examples
        retrieved_indices = self._retrieve_examples(text)
        
        # Build ICL prompt
        labels_str = ', '.join(sorted(self.label_set))
        
        messages = [
            {
                "role": "system",
                "content": f"You are a text classifier. Classify sentences into one of these labels: {labels_str}. Respond with only the label, nothing else."
            }
        ]
        
        # Add retrieved examples as ICL demonstrations
        for idx in retrieved_indices:
            messages.append({"role": "user", "content": self.train_texts[idx]})
            messages.append({"role": "assistant", "content": self.train_labels[idx]})
        
        # Add query
        messages.append({"role": "user", "content": text})
        
        # Get prediction
        response = self.llm_provider.chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=4000
        )
        
        predicted_label = response.strip()
        
        # Validate prediction
        if predicted_label not in self.label_set:
            # Try to find label as substring
            for label in self.label_set:
                if label.lower() in predicted_label.lower():
                    predicted_label = label
                    break
            else:
                # Default to first label
                predicted_label = sorted(list(self.label_set))[0]
        
        return predicted_label
    
    def predict_batch(self, texts: List[str], verbose: bool = True) -> List[str]:
        """
        Predict labels for multiple texts.
        
        Args:
            texts: List of texts to classify
            verbose: Print progress
            
        Returns:
            List of predicted labels
        """
        predictions = []
        
        for i, text in enumerate(texts):
            pred = self.predict(text)
            predictions.append(pred)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"    Processed {i+1}/{len(texts)} examples...")
        
        return predictions
