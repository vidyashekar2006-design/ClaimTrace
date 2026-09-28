"""
ClaimTrace: Index Building Script
Builds BM25 inverted index and prepares Sentence-BERT vector store metadata.
Saves index artifacts to artifacts/ directory.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import load_corpus_from_jsonl
from src.retrieval import OkapiBM25
from src.preprocessing import tokenize_words


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_path = os.path.join(root_dir, "data", "fever_sample_corpus.jsonl")
    artifacts_dir = os.path.join(root_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    print("=" * 60)
    print("ClaimTrace: Building Evidence Retrieval Indexes")
    print(f"Loading corpus from: {corpus_path}")
    print("=" * 60)

    start = time.time()
    sentences = load_corpus_from_jsonl(corpus_path)
    print(f"Loaded {len(sentences)} evidence sentences across Wikipedia articles.")

    # Build BM25 index
    bm25 = OkapiBM25(k1=1.5, b=0.75)
    tokenized_docs = [
        [t.lower() for t in tokenize_words(f"{s.doc_id.replace('_', ' ')} {s.text}")]
        for s in sentences
    ]
    bm25.fit(tokenized_docs)

    index_data = {
        "corpus_size": bm25.corpus_size,
        "avgdl": round(bm25.avgdl, 4),
        "vocab_size": len(bm25.idf),
        "doc_identifiers": [s.identifier for s in sentences],
        "built_at_timestamp": time.time(),
        "algorithm": "Okapi BM25 (k1=1.5, b=0.75)",
    }

    output_path = os.path.join(artifacts_dir, "bm25_index.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    elapsed = round(time.time() - start, 3)
    print(f"BM25 index successfully constructed in {elapsed}s.")
    print(f"Vocabulary size: {len(bm25.idf)} unique tokens.")
    print(f"Index artifact saved to: {output_path}")


if __name__ == "__main__":
    main()
