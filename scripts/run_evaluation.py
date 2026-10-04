"""
ClaimTrace: Research Experiments & Benchmark Comparison

Compares:
1. BM25 lexical retrieval
2. Sentence-BERT semantic retrieval
3. Hybrid BM25 + Sentence-BERT retrieval

The evaluation uses the bundled FEVER-style sample dataset.
It is NOT the full official FEVER benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import ClaimTracePipeline, load_corpus_from_jsonl
from src.evaluate import evaluate_dataset


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

DEFAULT_CORPUS = os.path.join(
    DATA_DIR,
    "fever_sample_corpus.jsonl",
)

DEFAULT_DEV = os.path.join(
    DATA_DIR,
    "fever_sample_dev.jsonl",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float without crashing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_latency_metrics(eval_result: dict) -> dict:
    """
    Extract latency information from the current evaluate.py structure.

    The current evaluate_dataset() stores latency information under
    the 'latency_metrics' dictionary.
    """
    latency = eval_result.get("latency_metrics", {})

    if not isinstance(latency, dict):
        latency = {}

    return latency


def get_metric(eval_result: dict, name: str, default=0.0):
    """
    Safely retrieve a metric.

    Supports the current evaluation structure and avoids crashes
    when an optional metric is unavailable.
    """
    value = eval_result.get(name, default)

    if value is None:
        return default

    return value


def count_verifiable_claims(eval_result: dict) -> int:
    """
    Count verifiable claims when per-claim results are available.

    This is metadata only and is not required for the main metrics.
    """
    per_claim = eval_result.get("per_claim", [])

    if not isinstance(per_claim, list):
        return 0

    count = 0

    for result in per_claim:
        if not isinstance(result, dict):
            continue

        label = (
            result.get("gold_label")
            or result.get("label")
            or result.get("gold")
        )

        if label in {"SUPPORTED", "REFUTED"}:
            count += 1

    return count


def build_summary(
    method: str,
    eval_result: dict,
    dataset_path: str,
) -> dict:
    """
    Build a compact summary record for one retrieval method.
    """
    latency = get_latency_metrics(eval_result)

    return {
        "method": method,
        "dataset": os.path.basename(dataset_path),
        "sample_size": eval_result.get(
            "dataset_size",
            eval_result.get("sample_size", 0),
        ),
        "verifiable_claims": count_verifiable_claims(eval_result),

        "evidence_recall_at_5": safe_float(
            eval_result.get(
                "evidence_recall_at_5",
                eval_result.get("evidence_recall", 0.0),
            )
        ),

        "accuracy": safe_float(
            eval_result.get("accuracy", 0.0)
        ),

        "macro_f1": safe_float(
            eval_result.get("macro_f1", 0.0)
        ),

        "fever_score": safe_float(
            eval_result.get("fever_score", 0.0)
        ),

        "avg_retrieval_latency_sec": safe_float(
            latency.get("avg_retrieval_latency_sec", 0.0)
        ),

        "avg_verification_latency_sec": safe_float(
            latency.get("avg_verification_latency_sec", 0.0)
        ),

        "avg_total_latency_sec": safe_float(
            latency.get(
                "avg_total_latency_sec",
                latency.get("total_latency_sec", 0.0),
            )
        ),
    }


def print_results(results: list[dict]) -> None:
    """Print the main comparison table."""
    print()
    print(
        "Method     | Recall@5   | Accuracy   | Macro F1   | "
        "FEVER Score  | Latency (s)"
    )
    print("-" * 75)

    for result in results:
        print(
            f"{result['method']:<10} | "
            f"{result['evidence_recall_at_5']:<10.4f} | "
            f"{result['accuracy']:<10.4f} | "
            f"{result['macro_f1']:<10.4f} | "
            f"{result['fever_score']:<12.4f} | "
            f"{result['avg_total_latency_sec']:<.4f}"
        )

    print()


# ---------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run ClaimTrace comparative retrieval evaluation "
            "on the bundled FEVER-style sample dataset."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "medium", "full"],
        default="quick",
        help=(
            "Evaluation mode. "
            "quick uses the bundled sample dataset. "
            "medium/full can be limited using --max-samples."
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of claims to evaluate.",
    )

    parser.add_argument(
        "--dev-file",
        type=str,
        default=None,
        help="Optional custom development/evaluation JSONL file.",
    )

    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Optional custom corpus JSONL file.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of evidence sentences retrieved per claim.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_arguments()

    # -------------------------------------------------------------
    # Select dataset
    # -------------------------------------------------------------
    dev_file = args.dev_file or DEFAULT_DEV
    corpus_file = args.corpus_file or DEFAULT_CORPUS

    if not os.path.exists(dev_file):
        raise FileNotFoundError(
            f"Evaluation dataset not found:\n{dev_file}"
        )

    if not os.path.exists(corpus_file):
        raise FileNotFoundError(
            f"Corpus not found:\n{corpus_file}"
        )

    # -------------------------------------------------------------
    # Mode handling
    # -------------------------------------------------------------
    max_samples = args.max_samples

    if args.mode == "quick":
        # Quick mode uses the bundled sample dataset.
        # No artificial sample limit unless explicitly provided.
        pass

    elif args.mode == "medium":
        if max_samples is None:
            max_samples = 50

    elif args.mode == "full":
        # IMPORTANT:
        # This project currently ships only the bundled FEVER-style
        # sample dataset. We therefore do NOT claim this is the
        # official full FEVER benchmark.
        if max_samples is None:
            max_samples = None

    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------
    print("=" * 75)
    print("ClaimTrace: Research Experiments & Benchmark Comparison")
    print(
        f"Mode: {args.mode.upper()} | "
        f"Max Samples: "
        f"{max_samples if max_samples is not None else 'All'}"
    )
    print(
        f"Evaluating Dev Set ({dev_file}) "
        f"against Corpus ({corpus_file})"
    )
    print("=" * 75)

    # -------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------
    ensure_directory(ARTIFACTS_DIR)

    # -------------------------------------------------------------
    # Load corpus
    # -------------------------------------------------------------
    corpus = load_corpus_from_jsonl(corpus_file)

    if not corpus:
        raise ValueError(
            "The evidence corpus is empty. "
            "Please check the corpus JSONL file."
        )

    # -------------------------------------------------------------
    # Initialize pipeline ONCE
    # -------------------------------------------------------------
    pipeline = ClaimTracePipeline(
        corpus_path=corpus_file,
    )

    # -------------------------------------------------------------
    # Retrieval methods
    # -------------------------------------------------------------
    methods = [
        "bm25",
        "semantic",
        "hybrid",
    ]

    all_results = []
    summary_records = []

    # -------------------------------------------------------------
    # Run experiments
    # -------------------------------------------------------------
    for method in methods:

        print()
        print(f"Running evaluation: {method.upper()}...")

        eval_result = evaluate_dataset(
            pipeline=pipeline,
            dataset=dev_file,
            retrieval_method=method,
            top_k=args.top_k,
            max_samples=max_samples,
        )

        # Store raw result
        all_results.append(
            {
                "method": method,
                "result": eval_result,
            }
        )

        # Build compact summary
        summary = build_summary(
            method=method,
            eval_result=eval_result,
            dataset_path=dev_file,
        )

        summary_records.append(summary)

    # -------------------------------------------------------------
    # Print comparison
    # -------------------------------------------------------------
    print_results(summary_records)

    # -------------------------------------------------------------
    # Save eval_results.csv
    # -------------------------------------------------------------
    results_df = pd.DataFrame(summary_records)

    eval_csv_path = os.path.join(
        ARTIFACTS_DIR,
        "eval_results.csv",
    )

    results_df.to_csv(
        eval_csv_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Save metrics_summary.json
    # -------------------------------------------------------------
    metrics_summary = {
        "project": "ClaimTrace",
        "evaluation_type": "sample_fever_style_benchmark",
        "dataset": os.path.basename(dev_file),
        "corpus": os.path.basename(corpus_file),
        "mode": args.mode,
        "max_samples": max_samples,
        "top_k": args.top_k,
        "methods": summary_records,
        "generated_at": datetime.now().isoformat(),
        "note": (
            "This evaluation uses the bundled FEVER-style sample "
            "dataset and corpus. It is not the full official FEVER "
            "benchmark."
        ),
    }

    metrics_json_path = os.path.join(
        ARTIFACTS_DIR,
        "metrics_summary.json",
    )

    with open(
        metrics_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_summary,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Save confusion matrix
    # -------------------------------------------------------------
    confusion_data = {}

    for item in all_results:
        method = item["method"]
        eval_result = item["result"]

        confusion_data[method] = {
            "confusion_matrix": eval_result.get(
                "confusion_matrix",
                [],
            ),
            "labels": eval_result.get(
                "labels",
                [],
            ),
        }

    confusion_json_path = os.path.join(
        ARTIFACTS_DIR,
        "confusion_matrix.json",
    )

    with open(
        confusion_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            confusion_data,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------
    print("=" * 75)
    print("Evaluation completed successfully.")
    print()
    print(f"Saved: {eval_csv_path}")
    print(f"Saved: {metrics_json_path}")
    print(f"Saved: {confusion_json_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()