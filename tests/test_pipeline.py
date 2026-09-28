"""
Integration tests for src/pipeline.py
Verifies end-to-end claim verification pipeline.
"""

import unittest
from src.pipeline import get_pipeline, ClaimTracePipeline
from src.verification import LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = get_pipeline()

    def test_pipeline_supported_claim(self):
        claim = "Nikolaj Coster-Waldau played Jaime Lannister in Game of Thrones."
        res = self.pipeline.verify_claim(claim, retrieval_method="hybrid", top_k=5)
        self.assertEqual(res["status"], "success")
        self.assertIn("verdict", res)
        self.assertIn("confidence", res)
        self.assertIn("retrieved_evidence", res)
        self.assertEqual(len(res["retrieved_evidence"]), 5)
        self.assertEqual(res["verdict"], LABEL_SUPPORTED)

    def test_pipeline_refuted_claim(self):
        claim = "Michael Bay has never directed any action film."
        res = self.pipeline.verify_claim(claim, retrieval_method="hybrid", top_k=5)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["verdict"], LABEL_REFUTED)

    def test_pipeline_nei_claim(self):
        claim = "Nikolaj Coster-Waldau was a professional NBA basketball player."
        res = self.pipeline.verify_claim(claim, retrieval_method="hybrid", top_k=5)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["verdict"], LABEL_NEI)

    def test_pipeline_empty_claim_handled_gracefully(self):
        res = self.pipeline.verify_claim("   ")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["verdict"], LABEL_NEI)


if __name__ == "__main__":
    unittest.main()
