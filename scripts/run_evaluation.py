"""
ClaimTrace: Research Evaluation Benchmark Script
Compares:
1. BM25 Keyword Retrieval
2. Sentence-BERT Semantic Retrieval
3. Hybrid Retrieval (BM25 + SBERT Score Fusion)

Calculates:
- Evidence Recall@5
- Accuracy, Macro Precision, Macro Recall, Macro F1
- Official FEVER Score
- Retrieval and Inference Latencies

Saves:
- artifacts/eval_results.csv
- artifacts/confusion_matrix.json
- artifacts/metrics_summary.json
"""

import os
import sys
import json
import csv
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ClaimTracePipeline
from src.evaluate import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description="ClaimTrace Research Evaluation Benchmark")
    parser.add_argument("--mode", choices=["quick", "medium", "full"], default="quick",
                        help="Evaluation mode: quick (default sample dev set), medium, or full official dev set")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Explicit maximum number of claim samples to evaluate")
    parser.add_argument("--dev-file", type=str, default=None,
                        help="Custom path to evaluation jsonl file")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.dev_file:
        dev_path = args.dev_file
    elif args.mode == "full":
        official_dev = os.path.join(root_dir, "data", "fever_official_dev.jsonl")
        dev_path = official_dev if os.path.exists(official_dev) else os.path.join(root_dir, "data", "fever_sample_dev.jsonl")
    else:
        dev_path = os.path.join(root_dir, "data", "fever_sample_dev.jsonl")

    corpus_path = os.path.join(root_dir, "data", "fever_sample_corpus.jsonl")
    artifacts_dir = os.path.join(root_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    max_samples = args.max_samples
    if max_samples is None and args.mode == "medium":
        max_samples = 50

    print("=" * 75)
    print("ClaimTrace: Research Experiments & Benchmark Comparison")
    print(f"Mode: {args.mode.upper()} | Max Samples: {max_samples or 'All'}")
    print(f"Evaluating Dev Set ({dev_path}) against Corpus ({corpus_path})")
    print("=" * 75)

    pipeline = ClaimTracePipeline(corpus_path=corpus_path)

    methods = ["bm25", "semantic", "hybrid"]
    summary_results = {
        "metadata": {
            "mode": args.mode,
            "max_samples": max_samples,
            "eval_file": os.path.basename(dev_path),
            "timestamp": time.time(),
            "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "model_architecture": getattr(pipeline.nli_verifier, "model_name", "DistilBERT-NLI"),
        },
        "methods": {}
    }
    csv_rows = []

    print(f"\n{'Method':<10} | {'Recall@5':<10} | {'Accuracy':<10} | {'Macro F1':<10} | {'FEVER Score':<12} | {'Latency (s)':<10}")
    print("-" * 75)

    for method in methods:
        eval_res = evaluate_dataset(pipeline, dev_path, retrieval_method=method, top_k=5, max_samples=max_samples)
        summary_results["methods"][method] = {
            "sample_size": eval_res["dataset_size"],
            "verifiable_claims": eval_res["verifiable_claims"],
            "evidence_recall_at_5": eval_res["evidence_recall_at_k"],
            "accuracy": eval_res["accuracy"],
            "macro_precision": eval_res["macro_precision"],
            "macro_recall": eval_res["macro_recall"],
            "macro_f1": eval_res["macro_f1"],
            "fever_score": eval_res["fever_score"],
            "avg_retrieval_latency_sec": eval_res["avg_retrieval_latency_sec"],
            "avg_inference_latency_sec": eval_res["avg_inference_latency_sec"],
            "total_latency_sec": round(eval_res["avg_retrieval_latency_sec"] + eval_res["avg_inference_latency_sec"], 4),
        }

        total_lat = summary_results["methods"][method]["total_latency_sec"]
        print(
            f"{method.upper():<10} | "
            f"{eval_res['evidence_recall_at_k']:<10.4f} | "
            f"{eval_res['accuracy']:<10.4f} | "
            f"{eval_res['macro_f1']:<10.4f} | "
            f"{eval_res['fever_score']:<12.4f} | "
            f"{total_lat:<10.4f}"
        )

        # Collect detailed per-claim CSV data
        for row in eval_res["per_claim_results"]:
            csv_rows.append({
                "experiment_mode": args.mode,
                "retrieval_method": method,
                "claim_id": row["claim_id"],
                "claim": row["claim"],
                "gold_label": row["gold_label"],
                "pred_label": row["pred_label"],
                "confidence": row["confidence"],
                "evidence_recalled": row["evidence_recalled"],
                "fever_correct": row["fever_correct"],
                "retrieved_evidence_ids": ";".join(row["retrieved_evidence_ids"]),
            })

        # Save confusion matrix for hybrid
        if method == "hybrid":
            cm_path = os.path.join(artifacts_dir, "confusion_matrix.json")
            with open(cm_path, "w", encoding="utf-8") as f:
                json.dump(eval_res["confusion_matrix"], f, indent=2)

    # Save summary metrics
    summary_path = os.path.join(artifacts_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    # Save CSV
    csv_path = os.path.join(artifacts_dir, "eval_results.csv")
    if csv_rows:
        keys = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_rows)

    print("-" * 75)
    print(f"Detailed CSV results saved: {csv_path}")
    print(f"Summary metrics saved: {summary_path}")
    print(f"Confusion matrix saved: {os.path.join(artifacts_dir, 'confusion_matrix.json')}")
    print("=" * 75)


if __name__ == "__main__":
    main()
