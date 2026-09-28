"""
Unit tests for research evaluation logic in src/evaluate.py.
Verifies metric calculations, confusion matrix structure, and subset limiting.
"""

import os
import unittest
from src.evaluate import compute_classification_metrics, evaluate_dataset
from src.pipeline import ClaimTracePipeline
from src.verification import LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI


class TestEvaluation(unittest.TestCase):
    def test_compute_classification_metrics_perfect(self):
        y_true = [LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI]
        y_pred = [LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI]
        res = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["macro_f1"], 1.0)
        self.assertEqual(res["macro_precision"], 1.0)
        self.assertEqual(res["macro_recall"], 1.0)
        self.assertIn("confusion_matrix", res)

    def test_compute_classification_metrics_empty(self):
        res = compute_classification_metrics([], [])
        self.assertEqual(res["accuracy"], 0.0)
        self.assertEqual(res["macro_f1"], 0.0)

    def test_evaluate_dataset_subset_limiting(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dev_path = os.path.join(root_dir, "data", "fever_sample_dev.jsonl")
        corpus_path = os.path.join(root_dir, "data", "fever_sample_corpus.jsonl")
        pipeline = ClaimTracePipeline(corpus_path=corpus_path)

        res_full = evaluate_dataset(pipeline, dev_path, retrieval_method="bm25", top_k=3)
        res_subset = evaluate_dataset(pipeline, dev_path, retrieval_method="bm25", top_k=3, max_samples=3)

        self.assertGreater(res_full["dataset_size"], 3)
        self.assertEqual(res_subset["dataset_size"], 3)
        self.assertEqual(len(res_subset["per_claim_results"]), 3)
        self.assertIn("model_name", res_subset)
        self.assertIn("evaluation_timestamp", res_subset)


if __name__ == "__main__":
    unittest.main()
