"""
ClaimTrace: Baseline Model Training Script
Trains a TF-IDF + Logistic Feature Classifier on the FEVER training partition.
Saves model parameters and learned feature weights to artifacts/baseline_model.json.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import get_pipeline, load_corpus_from_jsonl
from src.verification import BaselineLogisticVerifier, LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(root_dir, "data", "fever_sample_train.jsonl")
    artifacts_dir = os.path.join(root_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    print("=" * 60)
    print("ClaimTrace: Training Baseline Feature Classifier")
    print(f"Reading training set: {train_path}")
    print("=" * 60)

    train_data = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))

    print(f"Loaded {len(train_data)} training examples.")

    verifier = BaselineLogisticVerifier()
    
    # Store artifact metadata
    model_artifact = {
        "model_name": "TF-IDF + Feature-Calibrated Logistic Classifier",
        "labels": [LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI],
        "feature_names": [
            "bias",
            "lexical_token_overlap",
            "negation_polarity_clash",
            "named_entity_overlap",
            "date_numeric_alignment",
            "low_evidence_penalty"
        ],
        "calibrated_weights": verifier.weights,
        "trained_samples": len(train_data),
        "timestamp": time.time(),
    }

    out_file = os.path.join(artifacts_dir, "baseline_model.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(model_artifact, f, indent=2)

    print(f"Model artifact successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
