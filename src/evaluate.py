"""
ClaimTrace: Research Evaluation Module
Evaluates:
1. Evidence Retrieval Recall@k (BM25 vs Sentence-BERT vs Hybrid)
2. Classification Performance (Accuracy, Macro Precision, Macro Recall, Macro F1)
3. Official FEVER Score (Label Correctness + Valid Evidence Retrieval Set)
4. Retrieval and Inference Latencies (seconds per claim)
5. Exports results to CSV and Confusion Matrix JSON.
"""

import json
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from src.preprocessing import clean_claim_text
from src.retrieval import HybridRetriever, DocumentSentence
from src.pipeline import ClaimTracePipeline, load_corpus_from_jsonl
from src.verification import LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI, ALL_LABELS


def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes Accuracy, Macro Precision, Macro Recall, Macro F1,
    and 3x3 Confusion Matrix.
    """
    labels = [LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI]
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    # Confusion matrix dict: [true_label][pred_label] = count
    cm = {t: {p: 0 for p in labels} for t in labels}
    correct = 0

    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
            if t == p:
                correct += 1

    accuracy = correct / total

    precisions = []
    recalls = []
    f1s = []

    for l in labels:
        tp = cm[l][l]
        fp = sum(cm[other][l] for other in labels if other != l)
        fn = sum(cm[l][other] for other in labels if other != l)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(labels)
    macro_recall = sum(recalls) / len(labels)
    macro_f1 = sum(f1s) / len(labels)

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
    }


def evaluate_dataset(
    pipeline: ClaimTracePipeline,
    eval_file_path: str,
    retrieval_method: str = "hybrid",
    top_k: int = 5,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs end-to-end evaluation on a FEVER dataset partition (e.g. dev set):
    - Computes Evidence Recall@top_k
    - Computes Official FEVER Score
    - Computes 3-way Label Accuracy & Macro F1
    - Measures Retrieval & Inference Latency
    - Supports max_samples for configurable quick/medium/full evaluation
    """
    if not os.path.exists(eval_file_path):
        raise FileNotFoundError(f"Evaluation file not found: {eval_file_path}")

    claims_data = []
    with open(eval_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                claims_data.append(json.loads(line))

    if max_samples and max_samples > 0:
        claims_data = claims_data[:max_samples]

    y_true: List[str] = []
    y_pred: List[str] = []
    
    verifiable_count = 0
    evidence_recall_hits = 0
    fever_score_hits = 0

    retrieval_latencies = []
    inference_latencies = []
    per_claim_results = []

    for item in claims_data:
        claim_id = item.get("id")
        claim_text = item.get("claim", "")
        gold_raw_label = item.get("label", "").upper()

        # Map to canonical labels
        if "SUPPORT" in gold_raw_label:
            gold_label = LABEL_SUPPORTED
        elif "REFUTE" in gold_raw_label:
            gold_label = LABEL_REFUTED
        else:
            gold_label = LABEL_NEI

        # Gold evidence sets: list of sets of (doc_id, line_num)
        gold_evidence_sets = []
        raw_evidence = item.get("evidence", [])
        for annot_set in raw_evidence:
            s = set()
            for ev in annot_set:
                if len(ev) >= 4:
                    doc_id = ev[2]
                    line_idx = ev[3]
                    if doc_id is not None and line_idx is not None:
                        s.add(f"{doc_id}:{line_idx}")
            if s:
                gold_evidence_sets.append(s)

        # Run pipeline
        res = pipeline.verify_claim(claim_text, retrieval_method=retrieval_method, top_k=top_k)
        pred_label = res["verdict"]

        retrieval_latencies.append(res["latencies"]["retrieval_seconds"])
        inference_latencies.append(res["latencies"]["inference_seconds"])

        y_true.append(gold_label)
        y_pred.append(pred_label)

        # Evaluate evidence recall and FEVER score
        retrieved_ids = set(e["identifier"] for e in res["retrieved_evidence"])

        is_verifiable = gold_label in (LABEL_SUPPORTED, LABEL_REFUTED)
        evidence_recalled = False
        fever_correct = False

        if is_verifiable:
            verifiable_count += 1
            # In FEVER, evidence is recalled if ANY of the complete evidence sets is a subset of retrieved evidence
            for gold_set in gold_evidence_sets:
                if gold_set.issubset(retrieved_ids):
                    evidence_recalled = True
                    break
            if evidence_recalled:
                evidence_recall_hits += 1

            if pred_label == gold_label and evidence_recalled:
                fever_correct = True
        else:
            # For NOT ENOUGH INFO, evidence is not required
            evidence_recalled = True  # N/A
            if pred_label == LABEL_NEI:
                fever_correct = True

        if fever_correct:
            fever_score_hits += 1

        per_claim_results.append({
            "claim_id": claim_id,
            "claim": claim_text,
            "gold_label": gold_label,
            "pred_label": pred_label,
            "confidence": res["confidence"],
            "evidence_recalled": evidence_recalled,
            "fever_correct": fever_correct,
            "retrieved_evidence_ids": list(retrieved_ids),
        })

    # Overall metrics
    clf_metrics = compute_classification_metrics(y_true, y_pred)
    
    recall_at_k = round(evidence_recall_hits / verifiable_count, 4) if verifiable_count > 0 else 1.0
    fever_score = round(fever_score_hits / len(claims_data), 4) if len(claims_data) > 0 else 0.0

    avg_retrieval_lat = round(sum(retrieval_latencies) / len(retrieval_latencies), 4) if retrieval_latencies else 0.0
    avg_inference_lat = round(sum(inference_latencies) / len(inference_latencies), 4) if inference_latencies else 0.0

    return {
        "dataset_size": len(claims_data),
        "verifiable_claims": verifiable_count,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "evaluation_timestamp": time.time(),
        "model_name": getattr(pipeline.nli_verifier, "model_name", "DistilBERT-NLI"),
        "evidence_recall_at_k": recall_at_k,
        "fever_score": fever_score,
        "accuracy": clf_metrics["accuracy"],
        "macro_precision": clf_metrics["macro_precision"],
        "macro_recall": clf_metrics["macro_recall"],
        "macro_f1": clf_metrics["macro_f1"],
        "avg_retrieval_latency_sec": avg_retrieval_lat,
        "avg_inference_latency_sec": avg_inference_lat,
        "confusion_matrix": clf_metrics["confusion_matrix"],
        "per_claim_results": per_claim_results,
    }
