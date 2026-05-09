"""
CIM Transmission RAG Chain
LangChain-based retrieval-augmented generation for CIM Q&A.
Returns answers grounded in retrieved CIM objects with full source citation.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any
import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from embeddings.vector_store import get_collection, similarity_search

logger = logging.getLogger(__name__)


CIM_SYSTEM_PROMPT = """You are an expert power systems engineer specialising in CIM (Common Information Model, IEC 61970/61968) and transmission grid operations.

You answer questions about transmission system assets — substations, lines, transformers, breakers, busbars, and shunt compensators — using ONLY the CIM data provided below as context.

Rules:
1. Base your answer strictly on the retrieved CIM objects. Do not invent values.
2. Always cite the asset name and CIM class for every fact you state.
3. If the data does not contain enough information to answer, say so clearly.
4. Use engineering terminology appropriate for a transmission operator or grid architect.
5. When quoting electrical parameters (R, X, B), always include units.

Retrieved CIM objects:
{context}
"""

CIM_HUMAN_PROMPT = """Question: {question}

Provide a precise, grounded answer citing the specific CIM objects above."""


def format_context(hits: List[Dict[str, Any]]) -> str:
    """Format retrieved CIM chunks into a numbered context block."""
    sections = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        sections.append(
            f"[Source {i}] {meta['cim_class']} — {meta['name']}\n"
            f"{hit['text']}\n"
        )
    return "\n---\n".join(sections)


class CIMRagChain:
    """
    End-to-end RAG chain for CIM transmission Q&A.

    Usage:
        chain = CIMRagChain()
        result = chain.query("What is the reactance of the Charlotte-Gastonia line?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n_results: int = 5,
        cim_class_filter: str = None,
    ):
        self.collection = get_collection()
        self.n_results = n_results
        self.cim_class_filter = cim_class_filter

        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=os.environ["OPENAI_API_KEY"],
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CIM_SYSTEM_PROMPT),
            ("human", CIM_HUMAN_PROMPT),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run a RAG query over the CIM vector store.

        Returns:
            dict with keys: answer, sources, retrieved_chunks
        """
        logger.info(f"Query: {question}")

        hits = similarity_search(
            self.collection,
            question,
            n_results=self.n_results,
            cim_class_filter=self.cim_class_filter,
        )

        if not hits:
            return {
                "answer": "No relevant CIM objects found for this query.",
                "sources": [],
                "retrieved_chunks": [],
            }

        context = format_context(hits)

        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })

        sources = [
            {
                "name": h["metadata"]["name"],
                "cim_class": h["metadata"]["cim_class"],
                "object_id": h["metadata"]["object_id"],
                "relevance_score": round(1 - h["distance"], 4),
            }
            for h in hits
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": hits,
        }

    def stream_query(self, question: str):
        """Stream the answer token by token (for Streamlit UI)."""
        hits = similarity_search(
            self.collection,
            question,
            n_results=self.n_results,
        )
        context = format_context(hits) if hits else "No relevant CIM data found."

        for token in self.chain.stream({"context": context, "question": question}):
            yield token

        return hits


# Example questions to showcase in your GitHub README
EXAMPLE_QUESTIONS = [
    "What is the positive-sequence reactance of the Charlotte-Gastonia 138kV line?",
    "List all 345kV equipment at Charlotte North substation.",
    "What is the rated MVA of the autotransformer at SUB_001?",
    "Which capacitor banks provide voltage support and what is their MVAR rating?",
    "Describe the zero-sequence parameters for all ACLineSegments in the model.",
    "What breakers are present and what are their interrupting ratings?",
    "Explain the transformer winding connection configuration at SUB_001.",
    "What is the nominal voltage of the busbar at Charlotte North?",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    chain = CIMRagChain()

    q = "What is the reactance of the Charlotte-Gastonia line and is it a double-circuit line?"
    result = chain.query(q)

    print(f"\nQ: {q}")
    print(f"\nA: {result['answer']}")
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['cim_class']}] {s['name']} — relevance: {s['relevance_score']}")
