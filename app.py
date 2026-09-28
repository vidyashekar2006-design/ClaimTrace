"""
ClaimTrace: An NLP-Based Hybrid Evidence Retrieval and Automated Claim Verification System
Streamlit Frontend Application for Fact Verification.

Run locally on Windows with:
  streamlit run app.py
"""

import os
import sys
import json
import time
import pandas as pd
import streamlit as st

# Add workspace root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import get_pipeline, ClaimTracePipeline
from src.verification import LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI

# Page configuration
st.set_page_config(
    page_title="ClaimTrace | Automated Claim Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .verdict-box {
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        font-weight: 700;
        font-size: 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .verdict-supported {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
    }
    .verdict-refuted {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .verdict-nei {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        color: #92400e;
    }
    .evidence-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        padding: 0.85rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .evidence-id {
        font-weight: 700;
        color: #1e40af;
        font-size: 0.88rem;
    }
    .evidence-score {
        float: right;
        background-color: #e0e7ff;
        color: #3730a3;
        font-size: 0.8rem;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .evidence-text {
        color: #334155;
        font-size: 0.95rem;
        margin-top: 0.35rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_pipeline():
    """Initializes and caches the ClaimTrace pipeline."""
    return get_pipeline()


pipeline = load_cached_pipeline()

# Header
st.markdown('<div class="main-title">🔍 ClaimTrace</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An NLP-Based Hybrid Evidence Retrieval & Automated Claim Verification System | FEVER Dataset</div>', unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    retrieval_method = st.selectbox(
        "Evidence Retrieval Algorithm",
        options=["hybrid", "bm25", "semantic"],
        format_func=lambda x: {
            "hybrid": "Hybrid (BM25 + SBERT Fusion)",
            "bm25": "Okapi BM25 (Lexical)",
            "semantic": "Sentence-BERT (Dense Semantic)",
        }[x],
        index=0,
        help="Choose between lexical keyword matching, dense semantic embeddings, or reciprocal rank fusion.",
    )
    
    top_k = st.slider("Top Evidence Sentences (k)", min_value=1, max_value=10, value=5)
    
    use_baseline = st.checkbox(
        "Use Baseline Classifier",
        value=False,
        help="Toggle between Neural NLI (DistilBERT) and TF-IDF calibrated feature baseline.",
    )

    st.divider()
    st.subheader("📚 Corpus Overview")
    st.info(f"Indexed Sentences: **{len(pipeline.sentences)}**\n\nIndexed Documents: **7 Wikipedia Articles**")

    # Load artifacts summary if available
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "metrics_summary.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            st.divider()
            st.subheader("📊 Research Benchmark (Dev Set)")
            methods_dict = metrics.get("methods", metrics)
            cur_metrics = methods_dict.get(retrieval_method, {})
            if cur_metrics:
                st.write(f"**Evidence Recall@5:** `{cur_metrics.get('evidence_recall_at_5', 0.0):.4f}`")
                st.write(f"**Verification Accuracy:** `{cur_metrics.get('accuracy', 0.0):.4f}`")
                st.write(f"**Macro F1:** `{cur_metrics.get('macro_f1', 0.0):.4f}`")
                st.write(f"**FEVER Score:** `{cur_metrics.get('fever_score', 0.0):.4f}`")
                if "total_latency_sec" in cur_metrics:
                    st.write(f"**Avg Latency:** `{cur_metrics.get('total_latency_sec', 0.0) * 1000:.1f} ms`")
            if "metadata" in metrics and "formatted_time" in metrics["metadata"]:
                st.caption(f"Eval Time: {metrics['metadata']['formatted_time']}")
        except Exception:
            pass

    st.markdown("---")
    st.caption("CSE AI & Data Science Undergraduate Mini-Project")


# Main Claim Input Area
st.subheader("Input Claim for Verification")

sample_claims = [
    "— Select a Sample FEVER Claim —",
    "Nikolaj Coster-Waldau played Jaime Lannister in Game of Thrones.",
    "Michael Bay has never directed any action film.",
    "The Godfather was directed by Francis Ford Coppola.",
    "Murda Beatz produced the hit single Nice for What by Drake.",
    "Mount Everest has an elevation lower than 5000 metres.",
    "Marie Curie spoke fluent Esperanto with her students.",
    "Roman Atwood owns an antique bicycle collection in Kyoto.",
]

selected_sample = st.selectbox("Quick-Load Sample Benchmark Claim:", options=sample_claims)

default_text = ""
if selected_sample != sample_claims[0]:
    default_text = selected_sample

claim_input = st.text_area(
    "Enter a factual claim statement in English:",
    value=default_text,
    height=90,
    placeholder="e.g., Mount Everest is Earth's highest mountain above sea level.",
)

col1, col2 = st.columns([1, 4])
with col1:
    verify_clicked = st.button("🔎 Verify Claim", type="primary", use_container_width=True)
with col2:
    if st.button("🧹 Clear", use_container_width=False):
        claim_input = ""
        st.rerun()

# Verification Execution
if verify_clicked or (selected_sample != sample_claims[0] and claim_input.strip()):
    if not claim_input.strip():
        st.warning("⚠️ Please provide a non-empty factual claim to verify.")
    else:
        with st.spinner("Retrieving evidence and running natural language inference..."):
            result = pipeline.verify_claim(
                claim=claim_input.strip(),
                retrieval_method=retrieval_method,
                top_k=top_k,
                use_baseline=use_baseline,
            )

        if result.get("status") == "error":
            st.error(f"Error during verification: {result.get('error')}")
        else:
            verdict = result["verdict"]
            confidence = result["confidence"]
            probs = result["probabilities"]
            evidence = result["retrieved_evidence"]
            latencies = result["latencies"]

            # Verdict box styling
            verdict_class = "verdict-supported"
            icon = "✅"
            if verdict == LABEL_REFUTED:
                verdict_class = "verdict-refuted"
                icon = "❌"
            elif verdict == LABEL_NEI:
                verdict_class = "verdict-nei"
                icon = "❓"

            st.markdown(
                f"""
                <div class="verdict-box {verdict_class}">
                    <span>{icon} Predicted Verdict: {verdict}</span>
                    <span style="font-size: 1.1rem; opacity: 0.9;">Confidence: {confidence * 100:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metrics row
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Model Architecture", result.get("model_type", "DistilBERT-NLI"))
            with mcol2:
                st.metric("Retrieval Strategy", retrieval_method.upper())
            with mcol3:
                st.metric("Retrieval Latency", f"{latencies['retrieval_seconds'] * 1000:.1f} ms")
            with mcol4:
                st.metric("Total Latency", f"{latencies['total_seconds'] * 1000:.1f} ms")

            # Probability Breakdown
            st.write("#### 🎯 Class Probability Distribution")
            prob_df = pd.DataFrame({
                "Class": [LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI],
                "Probability": [probs.get(LABEL_SUPPORTED, 0.0), probs.get(LABEL_REFUTED, 0.0), probs.get(LABEL_NEI, 0.0)]
            })
            st.bar_chart(prob_df.set_index("Class"), height=180)

            # Retrieved Evidence Section
            st.markdown("---")
            st.write(f"#### 📑 Retrieved Evidence Sentences (Top {len(evidence)})")
            
            if not evidence:
                st.warning("No evidence sentences were retrieved from the corpus.")
            else:
                for idx, ev in enumerate(evidence, start=1):
                    doc_id = ev.get("doc_id", "Unknown")
                    line_num = ev.get("line_num", 0)
                    score = ev.get("score", 0.0)
                    text = ev.get("text", "")
                    bm25_s = ev.get("bm25_score", 0.0)
                    sem_s = ev.get("semantic_score", 0.0)

                    st.markdown(
                        f"""
                        <div class="evidence-card">
                            <span class="evidence-id">[{idx}] Source: {doc_id} (Line {line_num})</span>
                            <span class="evidence-score">Score: {score:.4f} (BM25: {bm25_s:.3f} | SBERT: {sem_s:.3f})</span>
                            <div class="evidence-text">"{text}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Academic Explanations & Tabs
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🔬 Methodology & Architecture", "📈 Evaluation Metrics", "📄 Corpus Browser"])

with tab1:
    st.markdown("""
    ### System Architecture: Two-Stage Hybrid Fact Verification
    1. **Stage 1: Evidence Retrieval**
       - **Okapi BM25:** Matches exact surface keywords, proper nouns, dates, and entity identifiers.
       - **Sentence-BERT:** Dense semantic embedding cosine similarity using `all-MiniLM-L6-v2`.
       - **Reciprocal Rank Fusion (RRF):** Combines dense and sparse rank rankings without score distortion:
         $$RRF(d) = \\sum_{m \\in \\{BM25, SBERT\\}} \\frac{w_m}{k + \\text{rank}_m(d)}$$
    2. **Stage 2: Natural Language Inference (NLI)**
       - Cross-encoder evaluates (Evidence Premise, Claim Hypothesis) pairs.
       - Mapped into: `SUPPORTED` (Entailment), `REFUTED` (Contradiction), or `NOT ENOUGH INFO` (Neutral).
    """)

with tab2:
    eval_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "eval_results.csv")
    if os.path.exists(eval_csv_path):
        df_eval = pd.read_csv(eval_csv_path)
        st.write("#### Detailed Dev Set Evaluation Log")
        st.dataframe(df_eval, use_container_width=True)
        st.download_button(
            label="📥 Download Evaluation CSV",
            data=df_eval.to_csv(index=False).encode("utf-8"),
            file_name="claimtrace_eval_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Run `python scripts/run_evaluation.py` to generate the evaluation CSV log.")

with tab3:
    st.write("#### Indexed Evidence Corpus Sentences")
    corpus_df = pd.DataFrame([s.to_dict() for s in pipeline.sentences])
    st.dataframe(corpus_df, use_container_width=True)
