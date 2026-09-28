"""
Unit tests for src/verification.py
Verifies baseline and NLI verifiers, label mappings, and probability distributions.
"""

import unittest
from src.verification import (
    BaselineLogisticVerifier,
    TransformerNLIVerifier,
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_NEI,
    ALL_LABELS,
)


class TestVerification(unittest.TestCase):
    def setUp(self):
        self.verifier = BaselineLogisticVerifier()

    def test_supported_claim_classification(self):
        claim = "Nikolaj Coster-Waldau played Jaime Lannister in Game of Thrones."
        evidence = ["He played Jaime Lannister in the HBO fantasy drama series Game of Thrones."]
        res = self.verifier.predict(claim, evidence)
        self.assertEqual(res["verdict"], LABEL_SUPPORTED)
        self.assertGreater(res["confidence"], 0.5)
        self.assertEqual(set(res["probabilities"].keys()), set(ALL_LABELS))

    def test_refuted_claim_classification(self):
        claim = "Michael Bay has not directed any action films."
        evidence = ["He is best known for directing high-budget action films including Armageddon."]
        res = self.verifier.predict(claim, evidence)
        self.assertEqual(res["verdict"], LABEL_REFUTED)
        self.assertGreater(res["probabilities"][LABEL_REFUTED], 0.3)

    def test_not_enough_info_classification_on_unrelated_claim(self):
        claim = "Nikolaj Coster-Waldau played professional basketball in the NBA."
        evidence = ["Nikolaj William Coster-Waldau is a Danish actor, producer and screenwriter."]
        res = self.verifier.predict(claim, evidence)
        self.assertEqual(res["verdict"], LABEL_NEI)

    def test_empty_evidence_yields_nei(self):
        claim = "Random statement without any corpus evidence."
        nli = TransformerNLIVerifier()
        res = nli.predict(claim, [])
        self.assertEqual(res["verdict"], LABEL_NEI)
        self.assertGreater(res["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
