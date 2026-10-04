"""
ClaimTrace: Integrated Backend Pipeline

Orchestrates:
1. NLP preprocessing
2. Evidence retrieval (BM25, Semantic, or Hybrid)
3. Automated Natural Language Inference claim verification
4. Diagnostic metadata, probabilities, and execution latencies
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from src.preprocessing import clean_claim_text, parse_fever_lines
from src.retrieval import DocumentSentence, HybridRetriever
from src.verification import (
    TransformerNLIVerifier,
    BaselineLogisticVerifier,
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_NEI,
)


# ---------------------------------------------------------------------------
# Corpus Loading
# ---------------------------------------------------------------------------

def load_corpus_from_jsonl(corpus_path: str) -> List[DocumentSentence]:
    """
    Load a FEVER-style JSONL corpus and convert it into individual
    DocumentSentence objects.

    Supported formats:
    1. {"id": "...", "lines": "..."}
    2. {"id": "...", "text": "..."}
    """
    sentences: List[DocumentSentence] = []

    if not corpus_path or not os.path.exists(corpus_path):
        return sentences

    with open(corpus_path, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            raw_line = raw_line.strip()

            if not raw_line:
                continue

            try:
                document = json.loads(raw_line)
            except json.JSONDecodeError:
                # Ignore malformed JSONL entries so one bad line
                # does not stop the entire corpus from loading.
                continue

            doc_id = str(document.get("id", ""))

            # FEVER-style document with numbered lines.
            lines_field = document.get("lines", "")

            if lines_field:
                try:
                    parsed_lines = parse_fever_lines(lines_field)

                    for sentence_number, sentence_text in parsed_lines:
                        sentences.append(
                            DocumentSentence(
                                doc_id=doc_id,
                                line_num=sentence_number,
                                text=sentence_text,
                            )
                        )
                except Exception:
                    # Keep loading the remaining corpus if one document
                    # contains an unexpected lines format.
                    continue

            # Simple document format.
            elif "text" in document:
                text = str(document.get("text", "")).strip()

                if text:
                    sentences.append(
                        DocumentSentence(
                            doc_id=doc_id,
                            line_num=0,
                            text=text,
                        )
                    )

    return sentences


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class ClaimTracePipeline:
    """
    End-to-end fact verification pipeline for ClaimTrace.

    Pipeline:
        Claim
          ↓
        Preprocessing
          ↓
        Evidence Retrieval
          ↓
        NLI Verification
          ↓
        Structured Result
    """

    VALID_RETRIEVAL_METHODS = {"bm25", "semantic", "hybrid"}

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model: str = "typeform/distilbert-base-uncased-mnli",
    ):
        # ---------------------------------------------------------------
        # Resolve corpus path
        # ---------------------------------------------------------------

        if corpus_path is None:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

            default_path = os.path.join(
                base_dir,
                "data",
                "fever_sample_corpus.jsonl",
            )

            if os.path.exists(default_path):
                corpus_path = default_path
            else:
                corpus_path = os.path.join(
                    "data",
                    "fever_sample_corpus.jsonl",
                )

        self.corpus_path = corpus_path
        self.sbert_model = sbert_model
        self.nli_model = nli_model

        # ---------------------------------------------------------------
        # Load corpus
        # ---------------------------------------------------------------

        self.sentences = load_corpus_from_jsonl(self.corpus_path)

        # ---------------------------------------------------------------
        # Initialize retrieval engine
        # ---------------------------------------------------------------

        self.retriever = HybridRetriever(
            self.sentences,
            sbert_model_name=self.sbert_model,
        )

        # ---------------------------------------------------------------
        # Initialize verification engines
        # ---------------------------------------------------------------

        self.nli_verifier = TransformerNLIVerifier(
            model_name=self.nli_model
        )

        self.baseline_verifier = BaselineLogisticVerifier()

    # -------------------------------------------------------------------
    # Corpus Management
    # -------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:
        """
        Dynamically add documents to the current corpus and rebuild
        the retrieval indexes.

        Supported document formats:
            {"id": "...", "lines": "..."}
            {"id": "...", "text": "..."}
        """

        if not documents:
            return

        for document in documents:
            doc_id = str(document.get("id", "Custom_Doc"))

            # FEVER-style numbered lines.
            if "lines" in document:
                try:
                    parsed_lines = parse_fever_lines(
                        document["lines"]
                    )

                    for line_number, text in parsed_lines:
                        self.sentences.append(
                            DocumentSentence(
                                doc_id=doc_id,
                                line_num=line_number,
                                text=text,
                            )
                        )
                except Exception:
                    continue

            # Simple text document.
            elif "text" in document:
                text = str(document.get("text", "")).strip()

                if text:
                    self.sentences.append(
                        DocumentSentence(
                            doc_id=doc_id,
                            line_num=0,
                            text=text,
                        )
                    )

        # Rebuild retrieval indexes while preserving the configured
        # SBERT model.
        self.retriever = HybridRetriever(
            self.sentences,
            sbert_model_name=self.sbert_model,
        )

    # -------------------------------------------------------------------
    # Claim Verification
    # -------------------------------------------------------------------

    def verify_claim(
        self,
        claim: str,
        retrieval_method: str = "hybrid",
        top_k: int = 5,
        use_baseline: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete ClaimTrace verification pipeline.

        Steps:
            1. Validate and preprocess the claim.
            2. Retrieve relevant evidence.
            3. Verify the claim against retrieved evidence.
            4. Return verdict, probabilities, evidence, and latency.

        Returns:
            Structured dictionary containing the verification result.
        """

        pipeline_start = time.perf_counter()

        # ---------------------------------------------------------------
        # Input validation
        # ---------------------------------------------------------------

        if not isinstance(claim, str):
            return self._error_response(
                claim=claim,
                retrieval_method=retrieval_method,
                error="Claim must be a string.",
            )

        if retrieval_method.lower() not in self.VALID_RETRIEVAL_METHODS:
            return self._error_response(
                claim=claim,
                retrieval_method=retrieval_method,
                error=(
                    f"Invalid retrieval method '{retrieval_method}'. "
                    f"Choose from: bm25, semantic, hybrid."
                ),
            )

        if not isinstance(top_k, int) or top_k <= 0:
            return self._error_response(
                claim=claim,
                retrieval_method=retrieval_method,
                error="top_k must be a positive integer.",
            )

        retrieval_method = retrieval_method.lower()

        # ---------------------------------------------------------------
        # Step 1: Preprocessing
        # ---------------------------------------------------------------

        cleaned_claim = clean_claim_text(claim)

        if not cleaned_claim:
            return {
                "claim": claim,
                "cleaned_claim": "",
                "verdict": LABEL_NEI,
                "confidence": 0.0,
                "probabilities": {
                    LABEL_SUPPORTED: 0.0,
                    LABEL_REFUTED: 0.0,
                    LABEL_NEI: 1.0,
                },
                "model_type": (
                    "Baseline-RuleBased"
                    if use_baseline
                    else "DistilBERT-NLI"
                ),
                "retrieval_method": retrieval_method,
                "top_k": top_k,
                "retrieved_evidence": [],
                "evidence_count": 0,
                "latencies": {
                    "retrieval_seconds": 0.0,
                    "inference_seconds": 0.0,
                    "total_seconds": round(
                        time.perf_counter() - pipeline_start,
                        4,
                    ),
                },
                "error": "Empty or invalid claim text provided.",
                "status": "error",
            }

        # ---------------------------------------------------------------
        # Step 2: Evidence Retrieval
        # ---------------------------------------------------------------

        if not self.sentences:
            return self._error_response(
                claim=claim,
                retrieval_method=retrieval_method,
                cleaned_claim=cleaned_claim,
                top_k=top_k,
                error="No documents are available in the corpus.",
                start_time=pipeline_start,
            )

        retrieval_start = time.perf_counter()

        try:
            retrieved_evidence = self.retriever.retrieve(
                cleaned_claim,
                method=retrieval_method,
                top_k=top_k,
            )
        except Exception as exc:
            return self._error_response(
                claim=claim,
                retrieval_method=retrieval_method,
                cleaned_claim=cleaned_claim,
                top_k=top_k,
                error=f"Evidence retrieval failed: {exc}",
                start_time=pipeline_start,
            )

        retrieval_latency = round(
            time.perf_counter() - retrieval_start,
            4,
        )

        # ---------------------------------------------------------------
        # Step 3: Claim Verification
        # ---------------------------------------------------------------

        verifier = (
            self.baseline_verifier
            if use_baseline
            else self.nli_verifier
        )

        evidence_texts = [
            item.get("text", "")
            for item in retrieved_evidence
            if item.get("text")
        ]

        inference_start = time.perf_counter()

        try:
            prediction = verifier.predict(
                cleaned_claim,
                evidence_texts,
            )
        except Exception as exc:
            return self._error_response(
                claim=claim,
                retrieval_method=retrieval_method,
                cleaned_claim=cleaned_claim,
                top_k=top_k,
                error=f"Claim verification failed: {exc}",
                start_time=pipeline_start,
                retrieval_latency=retrieval_latency,
                evidence=retrieved_evidence,
            )

        inference_latency = round(
            time.perf_counter() - inference_start,
            4,
        )

        # ---------------------------------------------------------------
        # Step 4: Final Response
        # ---------------------------------------------------------------

        total_latency = round(
            time.perf_counter() - pipeline_start,
            4,
        )

        return {
            "claim": claim,
            "cleaned_claim": cleaned_claim,
            "verdict": prediction.get(
                "verdict",
                LABEL_NEI,
            ),
            "confidence": prediction.get(
                "confidence",
                0.0,
            ),
            "probabilities": prediction.get(
                "probabilities",
                {
                    LABEL_SUPPORTED: 0.0,
                    LABEL_REFUTED: 0.0,
                    LABEL_NEI: 1.0,
                },
            ),
            "model_type": prediction.get(
                "model_type",
                "DistilBERT-NLI",
            ),
            "retrieval_method": retrieval_method,
            "top_k": top_k,
            "retrieved_evidence": retrieved_evidence,
            "evidence_count": len(retrieved_evidence),
            "latencies": {
                "retrieval_seconds": retrieval_latency,
                "inference_seconds": inference_latency,
                "total_seconds": total_latency,
            },
            "status": "success",
        }

    # -------------------------------------------------------------------
    # Error Response Helper
    # -------------------------------------------------------------------

    @staticmethod
    def _error_response(
        claim: Any,
        retrieval_method: str,
        error: str,
        cleaned_claim: str = "",
        top_k: int = 5,
        start_time: Optional[float] = None,
        retrieval_latency: float = 0.0,
        evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Create a consistent error response.
        """

        if start_time is None:
            total_latency = 0.0
        else:
            total_latency = round(
                time.perf_counter() - start_time,
                4,
            )

        return {
            "claim": claim,
            "cleaned_claim": cleaned_claim,
            "verdict": LABEL_NEI,
            "confidence": 0.0,
            "probabilities": {
                LABEL_SUPPORTED: 0.0,
                LABEL_REFUTED: 0.0,
                LABEL_NEI: 1.0,
            },
            "model_type": "N/A",
            "retrieval_method": retrieval_method,
            "top_k": top_k,
            "retrieved_evidence": evidence or [],
            "evidence_count": len(evidence or []),
            "latencies": {
                "retrieval_seconds": retrieval_latency,
                "inference_seconds": 0.0,
                "total_seconds": total_latency,
            },
            "error": error,
            "status": "error",
        }


# ---------------------------------------------------------------------------
# Global Pipeline Cache
# ---------------------------------------------------------------------------

_GLOBAL_PIPELINE: Optional[ClaimTracePipeline] = None
_GLOBAL_CORPUS_PATH: Optional[str] = None


def get_pipeline(
    corpus_path: Optional[str] = None,
) -> ClaimTracePipeline:
    """
    Retrieve the global ClaimTrace pipeline instance.

    The pipeline is initialized only once for a given corpus path.
    If a different corpus path is requested, a new pipeline is created.
    """

    global _GLOBAL_PIPELINE
    global _GLOBAL_CORPUS_PATH

    requested_path = corpus_path

    if (
        _GLOBAL_PIPELINE is None
        or _GLOBAL_CORPUS_PATH != requested_path
    ):
        _GLOBAL_PIPELINE = ClaimTracePipeline(
            corpus_path=corpus_path
        )

        _GLOBAL_CORPUS_PATH = requested_path

    return _GLOBAL_PIPELINE


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def verify(
    claim: str,
    retrieval_method: str = "hybrid",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Convenience functional interface for ClaimTrace verification.
    """

    pipeline = get_pipeline()

    return pipeline.verify_claim(
        claim,
        retrieval_method=retrieval_method,
        top_k=top_k,
    )