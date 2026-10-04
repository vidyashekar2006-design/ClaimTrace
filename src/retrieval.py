"""
ClaimTrace: Evidence Retrieval Module

Implements:
1. BM25 Keyword Retrieval (Okapi BM25)
2. Sentence-BERT Semantic Retrieval (Dense Vector Cosine Similarity)
3. Hybrid Retrieval using:
   - Reciprocal Rank Fusion (RRF)
   - Normalized Weighted Score Fusion

Returns top-k evidence sentences with:
- document identifiers
- sentence/line numbers
- retrieval scores
- retrieval method
"""

import math
import re
from typing import Any, Dict, List

from src.preprocessing import clean_claim_text, tokenize_words


# ============================================================================
# Document representation
# ============================================================================

class DocumentSentence:
    """
    Represents a single evidence sentence from the FEVER corpus.
    """

    def __init__(
        self,
        doc_id: str,
        line_num: int,
        text: str,
    ):
        self.doc_id = str(doc_id)
        self.line_num = int(line_num)
        self.text = str(text)

        self.identifier = f"{self.doc_id}:{self.line_num}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document sentence to a serializable dictionary.
        """
        return {
            "doc_id": self.doc_id,
            "line_num": self.line_num,
            "identifier": self.identifier,
            "text": self.text,
        }


# ============================================================================
# Okapi BM25
# ============================================================================

class OkapiBM25:
    """
    Standard Okapi BM25 implementation.

    BM25 score:

        IDF(q) =
            log(
                ((N - df + 0.5) / (df + 0.5)) + 1
            )

        Score(D, Q) =
            sum(
                IDF(q) *
                (
                    tf * (k1 + 1)
                )
                /
                (
                    tf + k1 *
                    (
                        1 - b + b * |D| / avgdl
                    )
                )
            )

    Parameters:
        k1:
            Controls term-frequency saturation.

        b:
            Controls document-length normalization.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = float(k1)
        self.b = float(b)

        self.corpus_size: int = 0
        self.avgdl: float = 0.0

        self.doc_lengths: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []

        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    # ------------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------------

    def fit(
        self,
        tokenized_corpus: List[List[str]],
    ) -> None:
        """
        Build the BM25 index from a tokenized corpus.

        Args:
            tokenized_corpus:
                List of documents, where each document is a list of tokens.
        """

        # Reset all index state so repeated calls to fit()
        # cannot retain stale information.
        self.corpus_size = len(tokenized_corpus)
        self.avgdl = 0.0

        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = {}
        self.idf = {}

        if self.corpus_size == 0:
            return

        total_length = 0

        for document in tokenized_corpus:
            # Normalize tokens to strings/lowercase.
            tokens = [
                str(token).lower()
                for token in document
                if str(token).strip()
            ]

            doc_length = len(tokens)

            self.doc_lengths.append(doc_length)
            total_length += doc_length

            # Term frequency for this document.
            term_freq: Dict[str, int] = {}

            for term in tokens:
                term_freq[term] = term_freq.get(term, 0) + 1

            self.doc_term_freqs.append(term_freq)

            # Document frequency counts each term once per document.
            for term in term_freq:
                self.doc_freqs[term] = (
                    self.doc_freqs.get(term, 0) + 1
                )

        self.avgdl = total_length / self.corpus_size

        if self.avgdl <= 0:
            return

        # BM25 IDF.
        for term, document_frequency in self.doc_freqs.items():
            numerator = (
                self.corpus_size
                - document_frequency
                + 0.5
            )

            denominator = (
                document_frequency
                + 0.5
            )

            idf_value = (
                math.log(
                    (numerator / denominator) + 1.0
                )
            )

            # Numerical protection.
            self.idf[term] = max(
                float(idf_value),
                0.0,
            )

    # ------------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------------

    def get_scores(
        self,
        tokenized_query: List[str],
    ) -> List[float]:
        """
        Calculate BM25 scores for every document.

        Returns:
            A list of scores aligned with the original corpus order.
        """

        scores = [0.0] * self.corpus_size

        if (
            self.corpus_size == 0
            or self.avgdl <= 0
            or not tokenized_query
        ):
            return scores

        query_tokens = [
            str(token).lower()
            for token in tokenized_query
            if str(token).strip()
        ]

        if not query_tokens:
            return scores

        # BM25 normally considers unique query terms.
        query_terms = set(query_tokens)

        for term in query_terms:

            if term not in self.idf:
                continue

            idf_value = self.idf[term]

            # Mild query-frequency weighting.
            query_frequency = query_tokens.count(term)

            query_weight = (
                1.0
                + 0.1 * (query_frequency - 1)
            )

            for document_index in range(self.corpus_size):

                term_frequency = (
                    self.doc_term_freqs[
                        document_index
                    ].get(term, 0)
                )

                if term_frequency == 0:
                    continue

                document_length = (
                    self.doc_lengths[
                        document_index
                    ]
                )

                numerator = (
                    term_frequency
                    * (self.k1 + 1.0)
                )

                length_normalization = (
                    1.0
                    - self.b
                    + self.b
                    * (
                        document_length
                        / self.avgdl
                    )
                )

                denominator = (
                    term_frequency
                    + self.k1
                    * length_normalization
                )

                if denominator <= 0:
                    continue

                score = (
                    query_weight
                    * idf_value
                    * (
                        numerator
                        / denominator
                    )
                )

                scores[document_index] += score

        return scores


# ============================================================================
# Semantic Retriever
# ============================================================================

class SemanticRetriever:
    """
    Sentence-BERT semantic retrieval.

    Primary method:
        SentenceTransformer embeddings + cosine similarity.

    Fallback method:
        Lightweight word + character trigram representation.

    The fallback ensures that ClaimTrace can still operate in environments
    where the Sentence-BERT model cannot be loaded.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name

        self.has_st: bool = False
        self.model = None
        self.corpus_embeddings = None

        self._init_model()

    # ------------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------------

    def _init_model(self) -> None:
        """
        Attempt to initialize Sentence-BERT.

        If the model cannot be loaded, automatically use the
        lightweight fallback implementation.
        """

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name
            )

            self.has_st = True

        except Exception:
            self.model = None
            self.has_st = False

    # ------------------------------------------------------------------------
    # Fallback vector representation
    # ------------------------------------------------------------------------

    def _fallback_vector(
        self,
        text: str,
    ) -> Dict[str, float]:
        """
        Create a lightweight sparse vector.

        Features:
        - word features
        - character trigrams

        Character trigrams provide some robustness to:
        - morphological variations
        - partial word overlap
        - names and related word forms
        """

        cleaned = clean_claim_text(
            text,
            preserve_case=False,
        )

        words = re.findall(
            r"[A-Za-z0-9]+",
            cleaned,
        )

        vector: Dict[str, float] = {}

        for word in words:

            # Word-level feature.
            vector[word] = (
                vector.get(word, 0.0)
                + 1.5
            )

            # Character trigram features.
            if len(word) >= 3:

                for index in range(
                    len(word) - 2
                ):
                    trigram = word[
                        index:index + 3
                    ]

                    key = f"_tri_{trigram}"

                    vector[key] = (
                        vector.get(key, 0.0)
                        + 0.4
                    )

        # L2 normalization.
        norm = math.sqrt(
            sum(
                value * value
                for value in vector.values()
            )
        )

        if norm > 0:

            for key in vector:
                vector[key] /= norm

        return vector

    # ------------------------------------------------------------------------
    # Fallback cosine similarity
    # ------------------------------------------------------------------------

    def _fallback_cosine(
        self,
        vec_a: Dict[str, float],
        vec_b: Dict[str, float],
    ) -> float:
        """
        Calculate cosine similarity between sparse vectors.
        """

        if not vec_a or not vec_b:
            return 0.0

        # Iterate over the smaller dictionary.
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a

        return float(
            sum(
                value * vec_b.get(
                    key,
                    0.0,
                )
                for key, value in vec_a.items()
            )
        )

    # ------------------------------------------------------------------------
    # Corpus encoding
    # ------------------------------------------------------------------------

    def encode_corpus(
        self,
        sentences: List[str],
    ) -> None:
        """
        Encode the complete evidence corpus.
        """

        if not sentences:
            self.corpus_embeddings = []
            return

        if (
            self.has_st
            and self.model is not None
        ):
            self.corpus_embeddings = (
                self.model.encode(
                    sentences,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            )

        else:
            self.corpus_embeddings = [
                self._fallback_vector(sentence)
                for sentence in sentences
            ]

    # ------------------------------------------------------------------------
    # Query scoring
    # ------------------------------------------------------------------------

    def get_scores(
        self,
        query: str,
    ) -> List[float]:
        """
        Calculate semantic similarity between a query and
        every indexed corpus sentence.
        """

        if self.corpus_embeddings is None:
            return []

        if not query or not query.strip():
            return [0.0] * len(
                self.corpus_embeddings
            )

        if (
            self.has_st
            and self.model is not None
        ):
            import numpy as np

            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True,
            )[0]

            scores = np.dot(
                self.corpus_embeddings,
                query_embedding,
            ).tolist()

            return [
                float(score)
                for score in scores
            ]

        query_vector = self._fallback_vector(
            query
        )

        return [
            self._fallback_cosine(
                query_vector,
                document_vector,
            )
            for document_vector
            in self.corpus_embeddings
        ]


# ============================================================================
# Hybrid Retriever
# ============================================================================

class HybridRetriever:
    """
    Evidence retrieval orchestrator.

    Provides:
        1. BM25 keyword retrieval
        2. Sentence-BERT semantic retrieval
        3. Hybrid retrieval using RRF + normalized score fusion
    """

    def __init__(
        self,
        sentences: List[DocumentSentence],
        sbert_model_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    ):
        self.sentences = list(sentences)

        self.bm25 = OkapiBM25(
            k1=1.5,
            b=0.75,
        )

        self.semantic = SemanticRetriever(
            model_name=sbert_model_name
        )

        # --------------------------------------------------------------------
        # Build BM25 index
        # --------------------------------------------------------------------

        tokenized_corpus = []

        for sentence in self.sentences:

            # Include document title because FEVER claims often
            # contain important entity names in the document title.
            document_text = (
                f"{sentence.doc_id.replace('_', ' ')} "
                f"{sentence.text}"
            )

            tokens = [
                token.lower()
                for token in tokenize_words(
                    document_text
                )
            ]

            tokenized_corpus.append(tokens)

        self.bm25.fit(
            tokenized_corpus
        )

        # --------------------------------------------------------------------
        # Build semantic index
        # --------------------------------------------------------------------

        semantic_texts = [
            (
                f"{sentence.doc_id.replace('_', ' ')}: "
                f"{sentence.text}"
            )
            for sentence in self.sentences
        ]

        self.semantic.encode_corpus(
            semantic_texts
        )

    # ------------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------------

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evidence using BM25.
        """

        if (
            not self.sentences
            or not query
            or not query.strip()
            or top_k <= 0
        ):
            return []

        tokens = [
            token.lower()
            for token in tokenize_words(query)
        ]

        if not tokens:
            return []

        scores = self.bm25.get_scores(
            tokens
        )

        if not scores:
            return []

        # BM25 only returns documents with a
        # positive keyword-matching score.
        ranked_indices = [
            index
            for index in sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )
            if scores[index] > 0
        ][:top_k]

        results: List[Dict[str, Any]] = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            sentence = self.sentences[index]
            score = float(scores[index])

            results.append(
                {
                    **sentence.to_dict(),
                    "rank": rank,
                    "score": round(score, 4),
                    "bm25_score": round(score, 4),
                    "retrieval_method": "bm25",
                }
            )

        return results

    # ------------------------------------------------------------------------
    # Semantic retrieval
    # ------------------------------------------------------------------------

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evidence using semantic similarity.
        """

        if (
            not self.sentences
            or not query
            or not query.strip()
            or top_k <= 0
        ):
            return []

        scores = self.semantic.get_scores(
            query
        )

        if not scores:
            return []

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: List[Dict[str, Any]] = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            sentence = self.sentences[index]
            score = float(scores[index])

            results.append(
                {
                    **sentence.to_dict(),
                    "rank": rank,
                    "score": round(score, 4),
                    "semantic_score": round(
                        score,
                        4,
                    ),
                    "retrieval_method": "semantic",
                }
            )

        return results

    # ------------------------------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------------------------------

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.45,
        semantic_weight: float = 0.55,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval using:

        1. BM25 keyword relevance
        2. Semantic relevance
        3. Reciprocal Rank Fusion
        4. Min-max normalized score fusion

        RRF:

            RRF(d) =
                Σ [
                    weight_m /
                    (rrf_k + rank_m(d))
                ]

        Final score:

            Hybrid =
                0.6 * normalized_score
                +
                0.4 * normalized_RRF

        Args:
            query:
                User claim/query.

            top_k:
                Number of evidence sentences to return.

            bm25_weight:
                Relative weight assigned to BM25.

            semantic_weight:
                Relative weight assigned to semantic retrieval.

            rrf_k:
                RRF smoothing constant.
        """

        if (
            not self.sentences
            or not query
            or not query.strip()
            or top_k <= 0
        ):
            return []

        # --------------------------------------------------------------------
        # Validate weights
        # --------------------------------------------------------------------

        if bm25_weight < 0:
            bm25_weight = 0.0

        if semantic_weight < 0:
            semantic_weight = 0.0

        total_weight = (
            bm25_weight
            + semantic_weight
        )

        if total_weight <= 0:
            bm25_weight = 0.45
            semantic_weight = 0.55
            total_weight = 1.0

        # Normalize weights.
        bm25_weight /= total_weight
        semantic_weight /= total_weight

        # RRF constant should be positive.
        rrf_k = max(
            int(rrf_k),
            1,
        )

        # --------------------------------------------------------------------
        # BM25 scores
        # --------------------------------------------------------------------

        query_tokens = [
            token.lower()
            for token in tokenize_words(query)
        ]

        if query_tokens:
            bm25_scores = self.bm25.get_scores(
                query_tokens
            )
        else:
            bm25_scores = [
                0.0
                for _ in self.sentences
            ]

        # --------------------------------------------------------------------
        # Semantic scores
        # --------------------------------------------------------------------

        semantic_scores = (
            self.semantic.get_scores(query)
        )

        # Safety fallback.
        if len(semantic_scores) != len(
            self.sentences
        ):
            semantic_scores = [
                0.0
                for _ in self.sentences
            ]

        num_documents = len(
            self.sentences
        )

        if num_documents == 0:
            return []

        # --------------------------------------------------------------------
        # Ranking helper
        # --------------------------------------------------------------------

        bm25_sorted_indices = sorted(
            range(num_documents),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )

        semantic_sorted_indices = sorted(
            range(num_documents),
            key=lambda i: semantic_scores[i],
            reverse=True,
        )

        bm25_ranks = {
            index: rank
            for rank, index
            in enumerate(
                bm25_sorted_indices,
                start=1,
            )
        }

        semantic_ranks = {
            index: rank
            for rank, index
            in enumerate(
                semantic_sorted_indices,
                start=1,
            )
        }

        # --------------------------------------------------------------------
        # Min-max normalization
        # --------------------------------------------------------------------

        def min_max_normalize(
            values: List[float],
        ) -> List[float]:
            """
            Normalize values to [0, 1].

            If all values are identical, return 0.5 for
            every document to avoid artificially favoring
            any document.
            """

            if not values:
                return []

            minimum = min(values)
            maximum = max(values)

            difference = (
                maximum - minimum
            )

            if difference <= 1e-12:
                return [
                    0.5
                    for _ in values
                ]

            return [
                (value - minimum)
                / difference
                for value in values
            ]

        bm25_normalized = min_max_normalize(
            bm25_scores
        )

        semantic_normalized = (
            min_max_normalize(
                semantic_scores
            )
        )

        # --------------------------------------------------------------------
        # Hybrid score
        # --------------------------------------------------------------------

        hybrid_scores: List[float] = []

        for index in range(
            num_documents
        ):

            # Reciprocal Rank Fusion.
            rrf_score = (
                bm25_weight
                / (
                    rrf_k
                    + bm25_ranks[index]
                )
            ) + (
                semantic_weight
                / (
                    rrf_k
                    + semantic_ranks[index]
                )
            )

            # Scale RRF approximately to [0, 1].
            normalized_rrf = (
                rrf_score
                * (rrf_k + 1)
            )

            # Weighted normalized raw-score fusion.
            normalized_score = (
                bm25_weight
                * bm25_normalized[index]
            ) + (
                semantic_weight
                * semantic_normalized[index]
            )

            # Final hybrid score.
            combined_score = (
                0.6 * normalized_score
                + 0.4 * normalized_rrf
            )

            hybrid_scores.append(
                float(combined_score)
            )

        # --------------------------------------------------------------------
        # Final ranking
        # --------------------------------------------------------------------

        ranked_indices = sorted(
            range(num_documents),
            key=lambda i: hybrid_scores[i],
            reverse=True,
        )[:top_k]

        results: List[Dict[str, Any]] = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            sentence = self.sentences[index]

            results.append(
                {
                    **sentence.to_dict(),
                    "rank": rank,
                    "score": round(
                        hybrid_scores[index],
                        4,
                    ),
                    "bm25_score": round(
                        float(
                            bm25_scores[index]
                        ),
                        4,
                    ),
                    "semantic_score": round(
                        float(
                            semantic_scores[index]
                        ),
                        4,
                    ),
                    "retrieval_method": "hybrid",
                }
            )

        return results

    # ------------------------------------------------------------------------
    # Unified retrieval interface
    # ------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Unified retrieval interface.

        Supported methods:
            bm25
            semantic / sbert / dense
            hybrid
        """

        method_lower = (
            str(method)
            .lower()
            .strip()
        )

        if method_lower == "bm25":
            return self.retrieve_bm25(
                query,
                top_k=top_k,
            )

        if method_lower in {
            "semantic",
            "sbert",
            "dense",
        }:
            return self.retrieve_semantic(
                query,
                top_k=top_k,
            )

        # Default method is hybrid.
        return self.retrieve_hybrid(
            query,
            top_k=top_k,
        )