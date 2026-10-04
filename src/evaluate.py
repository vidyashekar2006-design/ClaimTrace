"""
ClaimTrace Evaluation Module

Provides:
- Classification metrics
- Evidence Recall@K
- Official-style FEVER score
- Confusion matrix
- Per-claim evaluation results
- Dataset evaluation with optional sample limiting
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.verification import (
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_NEI,
)


LABELS = [
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_NEI,
]


def _normalise_label(label):
    """
    Convert common FEVER label names into ClaimTrace labels.
    """
    if label is None:
        return LABEL_NEI

    value = str(label).strip().upper()

    if value in {
        "SUPPORTS",
        "SUPPORTED",
        "SUPPORT",
        LABEL_SUPPORTED.upper(),
    }:
        return LABEL_SUPPORTED

    if value in {
        "REFUTES",
        "REFUTED",
        "REFUTE",
        LABEL_REFUTED.upper(),
    }:
        return LABEL_REFUTED

    if value in {
        "NOT ENOUGH INFO",
        "NOT_ENOUGH_INFO",
        "NEI",
        LABEL_NEI.upper(),
    }:
        return LABEL_NEI

    return label


def compute_classification_metrics(y_true, y_pred):
    """
    Compute accuracy, macro precision, recall, F1,
    and confusion matrix.

    Returns a dictionary suitable for JSON serialization.
    """
    if not y_true or not y_pred:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            "labels": LABELS,
        }

    y_true = [_normalise_label(x) for x in y_true]
    y_pred = [_normalise_label(x) for x in y_pred]

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
    }


def _load_jsonl_dataset(dataset_path):
    """
    Load a JSONL dataset from disk.
    """
    records = []

    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def _load_dataset(dataset):
    """
    Accept either:
    - a list of dictionaries
    - a JSONL file path
    """
    if isinstance(dataset, (str, os.PathLike)):
        return _load_jsonl_dataset(dataset)

    if isinstance(dataset, list):
        return dataset

    raise TypeError(
        "dataset must be either a list of dictionaries "
        "or a JSONL file path"
    )


def _extract_gold_evidence_sets(example):
    """
    Extract FEVER gold evidence identifiers.

    FEVER evidence format:
        [[annotation_id, page_id, page_name, line_number]]

    ClaimTrace internally identifies evidence as:
        PageName|line_number

    Returns:
        A list of evidence sets.

    Example:
        [
            {"Nikolaj_Coster-Waldau|3"}
        ]
    """
    evidence = example.get("evidence", [])

    if not evidence:
        return []

    gold_sets = []

    # FEVER can contain multiple annotation sets.
    for annotation_set in evidence:
        if not isinstance(annotation_set, list):
            continue

        current_set = set()

        for item in annotation_set:
            if not isinstance(item, list):
                continue

            if len(item) >= 4:
                page_name = item[2]
                line_number = item[3]

                if page_name is None or line_number is None:
                    continue

                try:
                    line_number = int(line_number)
                except (TypeError, ValueError):
                    continue

                current_set.add(
                    f"{page_name}|{line_number}"
                )

        if current_set:
            gold_sets.append(current_set)

    return gold_sets


def _evidence_id_from_result(result):
    """
    Convert a ClaimTrace retrieved-evidence record
    into the same Page|line format used by gold evidence.
    """
    if not isinstance(result, dict):
        return None

    doc_id = result.get("doc_id")
    line_num = result.get("line_num")

    if doc_id is not None and line_num is not None:
        try:
            line_num = int(line_num)
        except (TypeError, ValueError):
            pass

        return f"{doc_id}|{line_num}"

    identifier = result.get("identifier")

    if identifier:
        identifier = str(identifier)

        # ClaimTrace identifier format:
        # PageName:line
        if ":" in identifier:
            page_name, line_number = identifier.rsplit(":", 1)

            try:
                line_number = int(line_number)
                return f"{page_name}|{line_number}"
            except ValueError:
                pass

        return identifier

    return None


def evidence_recall_at_k(per_claim_results, k=5):
    """
    Compute evidence Recall@K over verifiable claims.

    A claim counts as retrieved if at least one of its
    gold evidence sentences appears in the top-K retrieved
    evidence sentences.

    NEI claims are excluded from evidence recall.
    """
    verifiable_claims = 0
    successful_claims = 0

    for result in per_claim_results:
        gold_sets = result.get("gold_evidence_sets", [])

        # No gold evidence means NOT ENOUGH INFO.
        if not gold_sets:
            continue

        verifiable_claims += 1

        retrieved = result.get("retrieved_evidence", [])

        retrieved_ids = set()

        for item in retrieved[:k]:
            evidence_id = _evidence_id_from_result(item)

            if evidence_id:
                retrieved_ids.add(evidence_id)

        # FEVER allows multiple acceptable evidence sets.
        # One complete set being retrieved is sufficient.
        matched = any(
            gold_set.issubset(retrieved_ids)
            for gold_set in gold_sets
        )

        if matched:
            successful_claims += 1

    if verifiable_claims == 0:
        return 0.0

    return successful_claims / verifiable_claims


def fever_score(per_claim_results):
    """
    Compute an official-style FEVER score.

    For SUPPORTED / REFUTED:
        Correct classification AND correct evidence.

    For NOT ENOUGH INFO:
        Correct classification is sufficient.
    """
    if not per_claim_results:
        return 0.0

    scores = []

    for result in per_claim_results:
        gold_label = _normalise_label(
            result.get("gold_label")
        )

        predicted_label = _normalise_label(
            result.get("predicted_label")
        )

        classification_correct = (
            gold_label == predicted_label
        )

        if not classification_correct:
            scores.append(0.0)
            continue

        # NEI does not require evidence.
        if gold_label == LABEL_NEI:
            scores.append(1.0)
            continue

        gold_sets = result.get(
            "gold_evidence_sets",
            []
        )

        retrieved = result.get(
            "retrieved_evidence",
            []
        )

        retrieved_ids = set()

        for item in retrieved:
            evidence_id = _evidence_id_from_result(item)

            if evidence_id:
                retrieved_ids.add(evidence_id)

        evidence_correct = any(
            gold_set.issubset(retrieved_ids)
            for gold_set in gold_sets
        )

        scores.append(
            1.0 if evidence_correct else 0.0
        )

    return float(np.mean(scores))


def evaluate_dataset(
    pipeline,
    dataset,
    retrieval_method="hybrid",
    top_k=5,
    max_samples=None,
):
    """
    Evaluate ClaimTrace on a dataset.

    Parameters
    ----------
    pipeline:
        ClaimTracePipeline instance.

    dataset:
        Either:
        - list of dictionaries
        - path to a JSONL dataset

    retrieval_method:
        "bm25", "semantic", or "hybrid".

    top_k:
        Number of evidence sentences to retrieve.

    max_samples:
        Optional maximum number of dataset examples.

    Returns
    -------
    dict
        Complete evaluation results.
    """
    dataset = _load_dataset(dataset)

    if max_samples is not None:
        max_samples = int(max_samples)

        if max_samples < 0:
            raise ValueError(
                "max_samples must be >= 0"
            )

        dataset = dataset[:max_samples]

    y_true = []
    y_pred = []

    per_claim_results = []

    for index, example in enumerate(dataset):
        claim = example.get("claim", "")

        gold_label = _normalise_label(
            example.get("label")
        )

        result = pipeline.verify_claim(
            claim,
            retrieval_method=retrieval_method,
            top_k=top_k,
        )

        predicted_label = _normalise_label(
            result.get("verdict")
        )

        retrieved_evidence = result.get(
            "retrieved_evidence",
            [],
        )

        gold_evidence_sets = (
            _extract_gold_evidence_sets(example)
        )

        y_true.append(gold_label)
        y_pred.append(predicted_label)

        per_claim_results.append(
            {
                "index": index,
                "id": example.get("id"),
                "claim": claim,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "classification_correct": (
                    gold_label == predicted_label
                ),
                "gold_evidence_sets": gold_evidence_sets,
                "retrieved_evidence": retrieved_evidence,
                "evidence_count": len(
                    retrieved_evidence
                ),
                "confidence": result.get(
                    "confidence",
                    0.0,
                ),
                "model_type": result.get(
                    "model_type"
                ),
                "retrieval_method": retrieval_method,
                "latencies": result.get(
                    "latencies",
                    {},
                ),
            }
        )

    classification = compute_classification_metrics(
        y_true,
        y_pred,
    )

    recall_at_5 = evidence_recall_at_k(
        per_claim_results,
        k=5,
    )

    recall_at_k = evidence_recall_at_k(
        per_claim_results,
        k=top_k,
    )

    score = fever_score(
        per_claim_results
    )

    model_name = "ClaimTrace NLI Verifier"

    if hasattr(pipeline, "verifier"):
        verifier = pipeline.verifier

        if hasattr(verifier, "model_name"):
            model_name = verifier.model_name

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    results = {
        "dataset_size": len(dataset),
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "model_name": model_name,
        "evaluation_timestamp": timestamp,

        "accuracy": classification[
            "accuracy"
        ],

        "macro_precision": classification[
            "macro_precision"
        ],

        "macro_recall": classification[
            "macro_recall"
        ],

        "macro_f1": classification[
            "macro_f1"
        ],

        "evidence_recall_at_5": float(
            recall_at_5
        ),

        "evidence_recall_at_k": float(
            recall_at_k
        ),

        "fever_score": float(score),

        "confusion_matrix": classification[
            "confusion_matrix"
        ],

        "labels": classification[
            "labels"
        ],

        "per_claim": per_claim_results,

        # Backward-compatible key used by tests
        # and earlier project scripts.
        "per_claim_results": per_claim_results,
    }

    return results


def save_evaluation_results(
    results,
    output_path,
):
    """
    Save evaluation results as JSON.
    """
    output_path = os.fspath(output_path)

    parent = os.path.dirname(
        os.path.abspath(output_path)
    )

    os.makedirs(
        parent,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


__all__ = [
    "compute_classification_metrics",
    "evidence_recall_at_k",
    "fever_score",
    "evaluate_dataset",
    "save_evaluation_results",
]