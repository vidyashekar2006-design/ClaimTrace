"""
ClaimTrace: Integrated Backend Pipeline
Orchestrates:
1. NLP Preprocessing
2. Evidence Retrieval (BM25, Semantic, or Hybrid)
3. Automated Natural Language Inference Claim Verification
4. Diagnostic Metadata, Probabilities, and Execution Latencies
"""

import json
import os
import time
from typing import List, Dict, Any, Optional

from src.preprocessing import clean_claim_text, parse_fever_lines
from src.retrieval import DocumentSentence, HybridRetriever
from src.verification import TransformerNLIVerifier, BaselineLogisticVerifier, LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI


def load_corpus_from_jsonl(corpus_path: str) -> List[DocumentSentence]:
    """
    Loads FEVER Wikipedia corpus jsonl and parses into individual DocumentSentence items.
    """
    sentences: List[DocumentSentence] = []
    if not os.path.exists(corpus_path):
        return sentences

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                doc_id = doc.get("id", "")
                lines_field = doc.get("lines", "")
                if lines_field:
                    parsed_lines = parse_fever_lines(lines_field)
                    for line_num, sent_text in parsed_lines:
                        sentences.append(DocumentSentence(doc_id=doc_id, line_num=line_num, text=sent_text))
                elif "text" in doc:
                    sentences.append(DocumentSentence(doc_id=doc_id, line_num=0, text=doc["text"]))
            except Exception:
                continue
    return sentences


class ClaimTracePipeline:
    """
    End-to-end fact verification pipeline for ClaimTrace.
    """
    def __init__(
        self,
        corpus_path: Optional[str] = None,
        sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model: str = "typeform/distilbert-base-uncased-mnli",
    ):
        if corpus_path is None:
            # Auto-detect default sample corpus
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(base_dir, "data", "fever_sample_corpus.jsonl")
            corpus_path = default_path if os.path.exists(default_path) else "data/fever_sample_corpus.jsonl"

        self.corpus_path = corpus_path
        self.sentences = load_corpus_from_jsonl(corpus_path)
        
        # Initialize retrieval engine
        self.retriever = HybridRetriever(self.sentences, sbert_model_name=sbert_model)
        
        # Initialize verification engines
        self.nli_verifier = TransformerNLIVerifier(model_name=nli_model)
        self.baseline_verifier = BaselineLogisticVerifier()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Dynamically add or extend corpus sentences."""
        for d in documents:
            doc_id = d.get("id", "Custom_Doc")
            if "lines" in d:
                for line_num, text in parse_fever_lines(d["lines"]):
                    self.sentences.append(DocumentSentence(doc_id, line_num, text))
            elif "text" in d:
                self.sentences.append(DocumentSentence(doc_id, 0, d["text"]))
        # Re-index
        self.retriever = HybridRetriever(self.sentences)

    def verify_claim(
        self,
        claim: str,
        retrieval_method: str = "hybrid",
        top_k: int = 5,
        use_baseline: bool = False,
    ) -> Dict[str, Any]:
        """
        Full verification pipeline:
        1. Preprocess & validate input claim
        2. Retrieve top-k evidence sentences
        3. Classify claim using evidence into SUPPORTED, REFUTED, or NOT ENOUGH INFO
        4. Return structured response
        """
        start_time = time.time()
        cleaned_claim = clean_claim_text(claim)

        if not cleaned_claim:
            return {
                "claim": claim,
                "verdict": LABEL_NEI,
                "confidence": 0.0,
                "probabilities": {LABEL_SUPPORTED: 0.0, LABEL_REFUTED: 0.0, LABEL_NEI: 1.0},
                "retrieval_method": retrieval_method,
                "retrieved_evidence": [],
                "error": "Empty or invalid claim text provided.",
                "latency_seconds": 0.0,
                "status": "error",
            }

        # Step 1: Retrieval
        retrieval_start = time.time()
        retrieved_evidence = self.retriever.retrieve(cleaned_claim, method=retrieval_method, top_k=top_k)
        retrieval_latency = round(time.time() - retrieval_start, 4)

        # Step 2: Verification
        verifier = self.baseline_verifier if use_baseline else self.nli_verifier
        evidence_texts = [e["text"] for e in retrieved_evidence]
        
        inference_start = time.time()
        pred = verifier.predict(cleaned_claim, evidence_texts)
        inference_latency = round(time.time() - inference_start, 4)

        total_latency = round(time.time() - start_time, 4)

        return {
            "claim": claim,
            "cleaned_claim": cleaned_claim,
            "verdict": pred["verdict"],
            "confidence": pred["confidence"],
            "probabilities": pred["probabilities"],
            "model_type": pred.get("model_type", "DistilBERT-NLI"),
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


# Global singleton cache
_GLOBAL_PIPELINE: Optional[ClaimTracePipeline] = None


def get_pipeline(corpus_path: Optional[str] = None) -> ClaimTracePipeline:
    """Retrieve or initialize the global ClaimTrace pipeline instance."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = ClaimTracePipeline(corpus_path=corpus_path)
    return _GLOBAL_PIPELINE


def verify(claim: str, retrieval_method: str = "hybrid", top_k: int = 5) -> Dict[str, Any]:
    """Convenience functional interface."""
    pipeline = get_pipeline()
    return pipeline.verify_claim(claim, retrieval_method=retrieval_method, top_k=top_k)
