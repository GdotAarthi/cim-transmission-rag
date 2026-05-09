"""
CIM Transmission RAG — Streamlit UI
Interactive Q&A interface over CIM/XML transmission system data.
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from retrieval.rag_chain import CIMRagChain, EXAMPLE_QUESTIONS

st.set_page_config(
    page_title="CIM Transmission Q&A",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ CIM Transmission System Q&A")
st.caption(
    "RAG-powered assistant over IEC 61970 CIM/XML grid data · "
    "Built with LangChain + ChromaDB + OpenAI"
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "LLM model",
        ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"],
        index=0,
    )
    n_results = st.slider("Retrieved CIM objects", min_value=2, max_value=10, value=5)
    cim_class_filter = st.selectbox(
        "Filter by CIM class (optional)",
        ["All", "ACLineSegment", "PowerTransformer", "Substation",
         "Breaker", "BusbarSection", "LinearShuntCompensator"],
        index=0,
    )
    class_filter = None if cim_class_filter == "All" else cim_class_filter

    st.divider()
    st.header("Example questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["input_question"] = q

# Main area
col_main, col_sources = st.columns([3, 2])

with col_main:
    question = st.text_area(
        "Ask a question about your transmission system:",
        value=st.session_state.get("input_question", ""),
        height=100,
        placeholder="e.g. What is the reactance of the Charlotte-Gastonia 138kV line?",
    )

    run = st.button("Ask", type="primary", use_container_width=True)

    if run and question.strip():
        with st.spinner("Retrieving CIM objects and generating answer..."):
            try:
                chain = CIMRagChain(
                    model=model if not model.startswith("claude") else "gpt-4o-mini",
                    n_results=n_results,
                    cim_class_filter=class_filter,
                )
                result = chain.query(question)

                st.subheader("Answer")
                st.markdown(result["answer"])

                # Show sources in right column
                with col_sources:
                    st.subheader("Retrieved CIM sources")
                    for i, src in enumerate(result["sources"], 1):
                        relevance_pct = int(src["relevance_score"] * 100)
                        with st.expander(
                            f"[{i}] {src['cim_class']} — {src['name']} ({relevance_pct}% match)",
                            expanded=(i == 1),
                        ):
                            st.caption(f"Object ID: `{src['object_id']}`")
                            # Show raw chunk text
                            raw = result["retrieved_chunks"][i - 1]["text"]
                            st.code(raw, language=None)

            except Exception as e:
                st.error(f"Error: {e}")
                st.info(
                    "Make sure you have:\n"
                    "1. Set `OPENAI_API_KEY` in your environment\n"
                    "2. Run `python src/embeddings/vector_store.py` to build the index first"
                )

    elif run:
        st.warning("Please enter a question.")

with col_sources:
    if not (run and question.strip()):
        st.subheader("Retrieved CIM sources")
        st.caption("Sources will appear here after you ask a question.")
        st.markdown(
            """
**What this RAG system retrieves:**
- `ACLineSegment` — line impedance, length, conductor type
- `PowerTransformer` — MVA rating, impedance, winding config
- `Substation` — voltage levels, location, region
- `Breaker` — rated current, interrupting capacity
- `BusbarSection` — bus configuration, fault current
- `LinearShuntCompensator` — MVAR rating, susceptance
            """
        )
