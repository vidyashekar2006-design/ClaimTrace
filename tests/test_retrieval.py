"""
Unit tests for src/retrieval.py
Verifies BM25, Semantic retrieval, and Hybrid RRF ranking without large model downloads.
"""

import unittest
from src.retrieval import DocumentSentence, OkapiBM25, HybridRetriever


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            DocumentSentence("Nikolaj_Coster-Waldau", 0, "Nikolaj William Coster-Waldau is a Danish actor and producer."),
            DocumentSentence("Nikolaj_Coster-Waldau", 3, "He played Jaime Lannister in the HBO fantasy drama series Game of Thrones."),
            DocumentSentence("Nikolaj_Coster-Waldau", 7, "He appeared in the Fox television movie New Amsterdam in 2008 as Detective John Amsterdam."),
            DocumentSentence("Michael_Bay", 1, "He is best known for directing high-budget action films including Armageddon and Transformers."),
            DocumentSentence("Mount_Everest", 0, "Mount Everest is Earth's highest mountain above sea level in the Himalayas."),
        ]
        self.retriever = HybridRetriever(self.sentences)

    def test_bm25_retrieval_finds_relevant_sentence(self):
        query = "Jaime Lannister Game of Thrones"
        results = self.retriever.retrieve_bm25(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["doc_id"], "Nikolaj_Coster-Waldau")
        self.assertEqual(results[0]["line_num"], 3)
        self.assertGreater(results[0]["score"], 0)

    def test_semantic_retrieval_ranks_concept_similarity(self):
        query = "highest peak in the world Himalayas"
        results = self.retriever.retrieve_semantic(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["doc_id"], "Mount_Everest")
        self.assertGreater(results[0]["score"], 0)

    def test_hybrid_retrieval_returns_top_k_with_scores(self):
        query = "Michael Bay action films Armageddon"
        results = self.retriever.retrieve_hybrid(query, top_k=3)
        self.assertEqual(len(results), 3)
        top = results[0]
        self.assertEqual(top["doc_id"], "Michael_Bay")
        self.assertIn("score", top)
        self.assertIn("bm25_score", top)
        self.assertIn("semantic_score", top)
        self.assertIn("identifier", top)


    def test_empty_query_returns_empty(self):
        self.assertEqual(self.retriever.retrieve_bm25("", top_k=3), [])
        self.assertEqual(self.retriever.retrieve_semantic("   ", top_k=3), [])
        self.assertEqual(self.retriever.retrieve_hybrid("", top_k=3), [])

    def test_empty_corpus_handled_gracefully(self):
        empty_retriever = HybridRetriever([])
        self.assertEqual(empty_retriever.retrieve_bm25("test", top_k=3), [])
        self.assertEqual(empty_retriever.retrieve_semantic("test", top_k=3), [])
        self.assertEqual(empty_retriever.retrieve_hybrid("test", top_k=3), [])


if __name__ == "__main__":
    unittest.main()
