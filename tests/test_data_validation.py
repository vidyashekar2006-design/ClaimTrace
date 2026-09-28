"""
Unit tests for data integrity and schema validation on FEVER partitions.
Checks for missing files, train/dev leakage, malformed records, and valid schema.
"""

import os
import json
import unittest


class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.root_dir, "data")
        self.corpus_path = os.path.join(self.data_dir, "fever_sample_corpus.jsonl")
        self.train_path = os.path.join(self.data_dir, "fever_sample_train.jsonl")
        self.dev_path = os.path.join(self.data_dir, "fever_sample_dev.jsonl")

    def test_data_files_exist(self):
        self.assertTrue(os.path.exists(self.corpus_path), "Corpus file must exist")
        self.assertTrue(os.path.exists(self.train_path), "Train file must exist")
        self.assertTrue(os.path.exists(self.dev_path), "Dev file must exist")

    def test_corpus_lines_format(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                self.assertIn("id", record, f"Line {line_no} missing 'id'")
                self.assertTrue("lines" in record or "text" in record, f"Line {line_no} missing text or lines")

    def test_train_dev_no_id_leakage(self):
        train_ids = set()
        with open(self.train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    train_ids.add(record["id"])

        dev_ids = set()
        with open(self.dev_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    dev_ids.add(record["id"])

        overlap = train_ids & dev_ids
        self.assertEqual(len(overlap), 0, f"Train and Dev partitions must not share claim IDs! Overlap: {overlap}")

    def test_valid_labels_in_dev(self):
        valid_labels = {"SUPPORTS", "SUPPORTED", "REFUTES", "REFUTED", "NOT ENOUGH INFO"}
        with open(self.dev_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    self.assertIn(record["label"].upper(), valid_labels, f"Invalid label: {record.get('label')}")


if __name__ == "__main__":
    unittest.main()
