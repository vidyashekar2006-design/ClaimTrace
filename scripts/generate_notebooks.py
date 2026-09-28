"""Script to generate valid runnable Jupyter Notebooks for ClaimTrace."""

import json
import os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source_lines]
    }

def md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source_lines]
    }

# 1. Exploration Notebook
nb1_cells = [
    md_cell([
        "# ClaimTrace: Notebook 01 - FEVER Dataset Exploration",
        "**Undergraduate Research Mini-Project in CSE AI & Data Science**",
        "",
        "This notebook explores the official FEVER (Fact Extraction and VERification) dataset schema,",
        "analyzes label distributions (SUPPORTS, REFUTES, NOT ENOUGH INFO), and inspects the Wikipedia evidence corpus."
    ]),
    code_cell([
        "import sys, os, json",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "# Add root directory to sys.path",
        "sys.path.insert(0, os.path.abspath('..'))",
        "from src.preprocessing import clean_claim_text, parse_fever_lines, extract_claim_entities_and_dates"
    ]),
    md_cell([
        "## 1. Load Genuine FEVER Corpus & Dataset Partitions"
    ]),
    code_cell([
        "corpus_path = '../data/fever_sample_corpus.jsonl'",
        "dev_path = '../data/fever_sample_dev.jsonl'",
        "",
        "corpus_docs = []",
        "with open(corpus_path, 'r', encoding='utf-8') as f:",
        "    for line in f:",
        "        if line.strip():",
        "            corpus_docs.append(json.loads(line))",
        "",
        "dev_claims = []",
        "with open(dev_path, 'r', encoding='utf-8') as f:",
        "    for line in f:",
        "        if line.strip():",
        "            dev_claims.append(json.loads(line))",
        "",
        "print(f'Indexed Corpus Documents: {len(corpus_docs)}')",
        "print(f'Development Claims: {len(dev_claims)}')"
    ]),
    md_cell([
        "## 2. Inspect Claim Labels and Evidence Structure"
    ]),
    code_cell([
        "df_dev = pd.DataFrame(dev_claims)",
        "print(df_dev[['id', 'claim', 'label', 'verifiable']].head(10))",
        "",
        "# Distribution of labels",
        "print('\\nLabel Distribution in Dev Set:')",
        "print(df_dev['label'].value_counts())"
    ]),
    md_cell([
        "## 3. Examine Corpus Sentence Parsing"
    ]),
    code_cell([
        "first_doc = corpus_docs[0]",
        "print(f'Document ID: {first_doc[\"id\"]}')",
        "lines = parse_fever_lines(first_doc['lines'])",
        "for line_idx, text in lines[:5]:",
        "    print(f'  [Line {line_idx}] {text}')"
    ]),
    md_cell([
        "## 4. Entity and Negation Extraction on Claims"
    ]),
    code_cell([
        "for claim in df_dev['claim'].head(5):",
        "    meta = extract_claim_entities_and_dates(claim)",
        "    print(f'Claim: \"{claim}\"')",
        "    print(f'  Entities: {meta[\"entities\"]}, Years: {meta[\"years\"]}, Has Negation: {meta[\"has_negation\"]}')"
    ])
]

# 2. Retrieval Experiments Notebook
nb2_cells = [
    md_cell([
        "# ClaimTrace: Notebook 02 - Evidence Retrieval Experiments",
        "**Comparing BM25, Sentence-BERT, and Hybrid Score Fusion (RRF)**",
        "",
        "This notebook benchmarks the three retrieval algorithms against candidate claims to evaluate Top-k evidence precision."
    ]),
    code_cell([
        "import sys, os",
        "sys.path.insert(0, os.path.abspath('..'))",
        "from src.pipeline import ClaimTracePipeline",
        "import pandas as pd"
    ]),
    code_cell([
        "pipeline = ClaimTracePipeline(corpus_path='../data/fever_sample_corpus.jsonl')",
        "print(f'Loaded pipeline with {len(pipeline.sentences)} sentences.')"
    ]),
    md_cell([
        "## 1. Test Single Claim with All Three Retrieval Algorithms"
    ]),
    code_cell([
        "claim = 'Nikolaj Coster-Waldau played Jaime Lannister in Game of Thrones.'",
        "",
        "res_bm25 = pipeline.retriever.retrieve_bm25(claim, top_k=3)",
        "res_sem = pipeline.retriever.retrieve_semantic(claim, top_k=3)",
        "res_hyb = pipeline.retriever.retrieve_hybrid(claim, top_k=3)",
        "",
        "print('=== BM25 Retrieval ===')",
        "for r in res_bm25:",
        "    print(f\"  [{r['rank']}] {r['identifier']} (Score: {r['score']}): {r['text']}\")",
        "",
        "print('\\n=== Sentence-BERT Semantic Retrieval ===')",
        "for r in res_sem:",
        "    print(f\"  [{r['rank']}] {r['identifier']} (Score: {r['score']}): {r['text']}\")",
        "",
        "print('\\n=== Hybrid Retrieval (RRF + Score Blend) ===')",
        "for r in res_hyb:",
        "    print(f\"  [{r['rank']}] {r['identifier']} (Score: {r['score']}): {r['text']}\")"
    ])
]

# 3. Model Verification Notebook
nb3_cells = [
    md_cell([
        "# ClaimTrace: Notebook 03 - Automated Claim Verification & NLI",
        "**Three-Way Inference: SUPPORTED vs REFUTED vs NOT ENOUGH INFO**"
    ]),
    code_cell([
        "import sys, os",
        "sys.path.insert(0, os.path.abspath('..'))",
        "from src.pipeline import get_pipeline",
        "from src.verification import LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI"
    ]),
    code_cell([
        "pipeline = get_pipeline('../data/fever_sample_corpus.jsonl')",
        "",
        "test_cases = [",
        "    ('The Godfather was directed by Francis Ford Coppola.', 'Expected: SUPPORTED'),",
        "    ('Michael Bay has never directed any action film.', 'Expected: REFUTED'),",
        "    ('Marie Curie spoke fluent Esperanto with her students.', 'Expected: NOT ENOUGH INFO'),",
        "]",
        "",
        "for claim, expected in test_cases:",
        "    res = pipeline.verify_claim(claim, retrieval_method='hybrid')",
        "    print(f'Claim: \"{claim}\"')",
        "    print(f'  {expected}')",
        "    print(f'  Predicted Verdict: {res[\"verdict\"]} (Confidence: {res[\"confidence\"] * 100:.1f}%)')",
        "    print(f'  Probabilities: {res[\"probabilities\"]}')",
        "    print(f'  Evidence: {res[\"retrieved_evidence\"][0][\"text\"]}\\n')"
    ])
]

# 4. Evaluation Benchmark Notebook
nb4_cells = [
    md_cell([
        "# ClaimTrace: Notebook 04 - End-to-End Evaluation Benchmark",
        "**Calculating Evidence Recall@5, FEVER Score, Macro F1, and Latency**"
    ]),
    code_cell([
        "import sys, os, json",
        "import pandas as pd",
        "sys.path.insert(0, os.path.abspath('..'))",
        "from src.pipeline import ClaimTracePipeline",
        "from src.evaluate import evaluate_dataset"
    ]),
    code_cell([
        "pipeline = ClaimTracePipeline(corpus_path='../data/fever_sample_corpus.jsonl')",
        "dev_path = '../data/fever_sample_dev.jsonl'",
        "",
        "results = {}",
        "for m in ['bm25', 'semantic', 'hybrid']:",
        "    res = evaluate_dataset(pipeline, dev_path, retrieval_method=m, top_k=5)",
        "    results[m] = res",
        "",
        "df_metrics = pd.DataFrame([",
        "    {",
        "        'Method': m.upper(),",
        "        'Recall@5': results[m]['evidence_recall_at_k'],",
        "        'Accuracy': results[m]['accuracy'],",
        "        'Macro F1': results[m]['macro_f1'],",
        "        'FEVER Score': results[m]['fever_score'],",
        "        'Avg Latency (s)': results[m]['avg_retrieval_latency_sec'] + results[m]['avg_inference_latency_sec']",
        "    }",
        "    for m in ['bm25', 'semantic', 'hybrid']",
        "])",
        "print(df_metrics.to_string(index=False))"
    ]),
    md_cell([
        "## Confusion Matrix for Hybrid Model"
    ]),
    code_cell([
        "cm = results['hybrid']['confusion_matrix']",
        "df_cm = pd.DataFrame(cm)",
        "print('Confusion Matrix (Rows: Ground Truth, Cols: Predicted):')",
        "print(df_cm)"
    ])
]

# Write all notebooks
with open(os.path.join(NOTEBOOKS_DIR, "01_fever_dataset_exploration.ipynb"), "w", encoding="utf-8") as f:
    json.dump(create_nb(nb1_cells), f, indent=2)

with open(os.path.join(NOTEBOOKS_DIR, "02_retrieval_experiments_bm25_sbert_hybrid.ipynb"), "w", encoding="utf-8") as f:
    json.dump(create_nb(nb2_cells), f, indent=2)

with open(os.path.join(NOTEBOOKS_DIR, "03_nli_claim_verification.ipynb"), "w", encoding="utf-8") as f:
    json.dump(create_nb(nb3_cells), f, indent=2)

with open(os.path.join(NOTEBOOKS_DIR, "04_end_to_end_evaluation_benchmark.ipynb"), "w", encoding="utf-8") as f:
    json.dump(create_nb(nb4_cells), f, indent=2)

print("All 4 Jupyter notebooks successfully generated!")
