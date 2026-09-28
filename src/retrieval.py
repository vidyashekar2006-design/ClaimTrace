"""
ClaimTrace: Evidence Retrieval Module
Implements:
1. BM25 Keyword Retrieval (Okapi BM25)
2. Sentence-BERT Semantic Retrieval (Dense Vector Cosine Similarity)
3. Hybrid Retrieval using Reciprocal Rank Fusion (RRF) and Normalized Score Fusion

Returns top-k evidence sentences with source identifiers (doc_id, line_num) and retrieval scores.
"""

import math
import re
from typing import List, Dict, Any, Optional, Tuple
from src.preprocessing import clean_claim_text, tokenize_words


class DocumentSentence:
    """Represents a single evidence sentence from the FEVER corpus."""
    def __init__(self, doc_id: str, line_num: int, text: str):
        self.doc_id = doc_id
        self.line_num = line_num
        self.text = text
        self.identifier = f"{doc_id}:{line_num}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "line_num": self.line_num,
            "identifier": self.identifier,
            "text": self.text,
        }


class OkapiBM25:
    """
    Standard Okapi BM25 implementation for evidence retrieval.
    Formula:
      IDF(q_i) = ln((N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1.0)
      Score(D, Q) = sum( IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl))) )
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, tokenized_corpus: List[List[str]]):
        self.corpus_size = len(tokenized_corpus)
        if self.corpus_size == 0:
            self.avgdl = 0.0
            return

        total_length = 0
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = {}

        for doc in tokenized_corpus:
            length = len(doc)
            self.doc_lengths.append(length)
            total_length += length
            
            tf: Dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.doc_term_freqs.append(tf)

            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avgdl = total_length / self.corpus_size

        # Compute IDFs with Robertson-Spärck Jones BM25 floor protection
        for term, freq in self.doc_freqs.items():
            # Standard Lucene/BM25 IDF variant to avoid negative scores for frequent words
            val = (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            self.idf[term] = math.log(max(val, 1.0001))

    def get_scores(self, tokenized_query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        if self.corpus_size == 0 or self.avgdl == 0:
            return scores

        query_terms = set(tokenized_query)
        for term in query_terms:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            q_freq = tokenized_query.count(term)
            # Query term weight boosting (optional)
            q_weight = 1.0 + 0.1 * (q_freq - 1)

            for i in range(self.corpus_size):
                tf = self.doc_term_freqs[i].get(term, 0)
                if tf == 0:
                    continue
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths[i] / self.avgdl))
                scores[i] += q_weight * idf_val * (num / denom)
        return scores


class SemanticRetriever:
    """
    Sentence-BERT semantic embedding retriever.
    Uses sentence-transformers if available; falls back to an exact dense n-gram cosine model
    to guarantee functionality and testing without network or heavy PyTorch downloads.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.has_st = False
        self.model = None
        self.corpus_embeddings = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.has_st = True
        except Exception:
            self.has_st = False
            self.model = None

    def _fallback_vector(self, text: str) -> Dict[str, float]:
        """Compute an n-gram and subword character distribution vector."""
        cleaned = clean_claim_text(text).lower()
        words = re.findall(r"\w+", cleaned)
        vector: Dict[str, float] = {}
        for w in words:
            vector[w] = vector.get(w, 0.0) + 1.5
            # Add character tri-grams for subword morphological matching
            if len(w) >= 3:
                for j in range(len(w) - 2):
                    tri = w[j:j+3]
                    vector[f"_tri_{tri}"] = vector.get(f"_tri_{tri}", 0.0) + 0.4
        # Normalize
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            for k in vector:
                vector[k] /= norm
        return vector

    def _fallback_cosine(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Compute cosine similarity between two sparse/dense feature vectors."""
        if not vec_a or not vec_b:
            return 0.0
        # Iterate over smaller dict
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        return sum(val * vec_b.get(k, 0.0) for k, val in vec_a.items())

    def encode_corpus(self, sentences: List[str]):
        if self.has_st and self.model is not None:
            self.corpus_embeddings = self.model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)
        else:
            self.corpus_embeddings = [self._fallback_vector(s) for s in sentences]

    def get_scores(self, query: str) -> List[float]:
        if self.corpus_embeddings is None:
            return []

        if self.has_st and self.model is not None:
            import numpy as np
            q_emb = self.model.encode([query], normalize_embeddings=True)[0]
            scores = np.dot(self.corpus_embeddings, q_emb).tolist()
            return [float(s) for s in scores]
        else:
            q_vec = self._fallback_vector(query)
            return [self._fallback_cosine(q_vec, d_vec) for d_vec in self.corpus_embeddings]


class HybridRetriever:
    """
    Evidence retrieval orchestrator providing:
    - BM25 retrieval
    - Sentence-BERT semantic retrieval
    - Hybrid retrieval via Reciprocal Rank Fusion (RRF) and Normalized Weighted Score Fusion
    """
    def __init__(self, sentences: List[DocumentSentence], sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.sentences: List[DocumentSentence] = sentences
        self.bm25 = OkapiBM25(k1=1.5, b=0.75)
        self.semantic = SemanticRetriever(model_name=sbert_model_name)
        
        # Build indexes
        tokenized_corpus = [
            [t.lower() for t in tokenize_words(f"{s.doc_id.replace('_', ' ')} {s.text}")]
            for s in self.sentences
        ]
        self.bm25.fit(tokenized_corpus)
        raw_texts = [f"{s.doc_id.replace('_', ' ')}: {s.text}" for s in self.sentences]
        self.semantic.encode_corpus(raw_texts)

    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.sentences or not query or not query.strip():
            return []
        tokens = [t.lower() for t in tokenize_words(query)]
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if all(s <= 0 for s in scores):
            # No matching keywords found
            return []
        ranked_indices = [i for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True) if scores[i] > 0][:top_k]
        
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            sent = self.sentences[idx]
            results.append({
                **sent.to_dict(),
                "rank": rank,
                "score": round(float(scores[idx]), 4),
                "bm25_score": round(float(scores[idx]), 4),
                "retrieval_method": "bm25",
            })
        return results

    def retrieve_semantic(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.sentences or not query or not query.strip():
            return []
        scores = self.semantic.get_scores(query)
        if not scores:
            return []
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            sent = self.sentences[idx]
            results.append({
                **sent.to_dict(),
                "rank": rank,
                "score": round(float(scores[idx]), 4),
                "semantic_score": round(float(scores[idx]), 4),
                "retrieval_method": "semantic",
            })
        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.45,
        semantic_weight: float = 0.55,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Score fusion using Reciprocal Rank Fusion (RRF) combined with min-max normalized score fusion.
        RRF Formula: RRF_score(d) = sum_m ( w_m / (k + rank_m(d)) )
        """
        if not self.sentences or not query or not query.strip():
            return []

        query_tokens = [t.lower() for t in tokenize_words(query)]
        bm25_scores = self.bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(self.sentences)
        sem_scores = self.semantic.get_scores(query)

        num_docs = len(self.sentences)
        if num_docs == 0:
            return []

        # Ranks for BM25
        bm25_sorted_indices = sorted(range(num_docs), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_sorted_indices, start=1)}

        # Ranks for Semantic
        sem_sorted_indices = sorted(range(num_docs), key=lambda i: sem_scores[i], reverse=True)
        sem_ranks = {idx: rank for rank, idx in enumerate(sem_sorted_indices, start=1)}

        # Normalize raw scores to [0, 1]
        def min_max(arr):
            mn, mx = min(arr), max(arr)
            if mx - mn > 1e-9:
                return [(x - mn) / (mx - mn) for x in arr]
            return [0.5 for _ in arr]

        bm25_norm = min_max(bm25_scores)
        sem_norm = min_max(sem_scores)

        hybrid_scores = []
        for i in range(num_docs):
            # Reciprocal Rank Component
            rrf = (bm25_weight / (rrf_k + bm25_ranks[i])) + (semantic_weight / (rrf_k + sem_ranks[i]))
            # Weighted Normalized Score Component
            norm_blend = (bm25_weight * bm25_norm[i]) + (semantic_weight * sem_norm[i])
            # Combined hybrid score (scaled to intuitive [0, 1] range)
            combined_score = 0.6 * norm_blend + 0.4 * (rrf * (rrf_k + 1))
            hybrid_scores.append(combined_score)

        ranked_indices = sorted(range(num_docs), key=lambda i: hybrid_scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            sent = self.sentences[idx]
            results.append({
                **sent.to_dict(),
                "rank": rank,
                "score": round(float(hybrid_scores[idx]), 4),
                "bm25_score": round(float(bm25_scores[idx]), 4),
                "semantic_score": round(float(sem_scores[idx]), 4),
                "retrieval_method": "hybrid",
            })
        return results

    def retrieve(self, query: str, method: str = "hybrid", top_k: int = 5) -> List[Dict[str, Any]]:
        method_lower = method.lower().strip()
        if method_lower == "bm25":
            return self.retrieve_bm25(query, top_k=top_k)
        elif method_lower in ("semantic", "sbert", "dense"):
            return self.retrieve_semantic(query, top_k=top_k)
        else:
            return self.retrieve_hybrid(query, top_k=top_k)
