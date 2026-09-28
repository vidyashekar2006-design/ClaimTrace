"""
Unit tests for src/preprocessing.py
Verifies text cleaning, tokenization, entity/date extraction, and negation preservation.
"""

import unittest
from src.preprocessing import (
    clean_claim_text,
    tokenize_words,
    extract_claim_entities_and_dates,
    parse_fever_lines,
    NEGATION_WORDS,
)


class TestPreprocessing(unittest.TestCase):
    def test_clean_claim_text_normalizes_quotes_and_spaces(self):
        raw = "  “Nikolaj’s   breakthrough—role”   was in 1994.  "
        cleaned = clean_claim_text(raw)
        self.assertIn('"Nikolaj\'s', cleaned)
        self.assertIn(" - role", cleaned)
        self.assertNotIn("  ", cleaned)

    def test_tokenize_preserves_numbers_and_hyphens(self):
        claim = "In 2008, Coster-Waldau appeared in New Amsterdam."
        tokens = tokenize_words(claim)
        self.assertIn("2008", tokens)
        self.assertIn("Coster-Waldau", tokens)
        self.assertIn("Amsterdam", tokens)

    def test_extract_negation_and_dates(self):
        claim = "Michael Bay has not directed any action films in 1986."
        meta = extract_claim_entities_and_dates(claim)
        self.assertTrue(meta["has_negation"])
        self.assertIn("1986", meta["years"])
        self.assertIn("Michael Bay", meta["entities"])

    def test_parse_fever_lines_format(self):
        fever_lines = "0\tSentence zero text .\n1\tSentence one text .\n2\tSentence two text ."
        parsed = parse_fever_lines(fever_lines)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0], (0, "Sentence zero text ."))
        self.assertEqual(parsed[1], (1, "Sentence one text ."))
        self.assertEqual(parsed[2], (2, "Sentence two text ."))


if __name__ == "__main__":
    unittest.main()
