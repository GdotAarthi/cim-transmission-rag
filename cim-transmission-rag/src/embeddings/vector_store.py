"""
CIM Embedding Pipeline
Converts CIMChunk objects into vector embeddings and stores in ChromaDB.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Import from sibling module
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from parser.cim_parser import CIMParser, CIMChunk


COLLECTION_NAME = "cim_transmission"
CHROMA_PATH = "./data/processed/chroma_db"

# Use OpenAI embeddings (swap to BedrockEmbeddings for AWS Bedrock)
EMBEDDING_MODEL = "text-embedding-3-small"


def build_vector_store(xml_paths: List[str], reset: bool = False) -> chromadb.Collection:
    """
    Parse CIM XML files, embed all objects, and store in ChromaDB.

    Args:
        xml_paths: List of CIM/XML file paths to ingest
        reset: If True, drops and rebuilds the collection

    Returns:
        ChromaDB collection ready for querying
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Dropped existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBEDDING_MODEL,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"description": "CIM IEC 61970 transmission system objects"},
    )

    all_chunks: List[CIMChunk] = []
    for xml_path in xml_paths:
        parser = CIMParser(xml_path)
        chunks = list(parser.parse())
        all_chunks.extend(chunks)
        logger.info(f"Parsed {len(chunks)} chunks from {xml_path}")

    if not all_chunks:
        raise ValueError("No CIM chunks parsed — check your XML files")

    # Deduplicate by chunk_id
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            unique_chunks.append(chunk)

    logger.info(f"Embedding {len(unique_chunks)} unique CIM objects...")

    # ChromaDB upsert in batches (avoid token limits)
    BATCH_SIZE = 50
    for i in range(0, len(unique_chunks), BATCH_SIZE):
        batch = unique_chunks[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        logger.info(f"  Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    logger.info(f"Vector store ready: {collection.count()} objects indexed")
    return collection


def get_collection() -> chromadb.Collection:
    """Load an existing ChromaDB collection (no re-embedding)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBEDDING_MODEL,
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
    )


def similarity_search(collection: chromadb.Collection, query: str, n_results: int = 5, cim_class_filter: str = None):
    """
    Retrieve top-k CIM objects relevant to a query.

    Args:
        collection: ChromaDB collection
        query: Natural language question
        n_results: Number of results to return
        cim_class_filter: Optional filter e.g. 'ACLineSegment'

    Returns:
        List of (text, metadata, distance) tuples
    """
    where = {"cim_class": cim_class_filter} if cim_class_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})

    return hits


if __name__ == "__main__":
    import glob

    logging.basicConfig(level=logging.INFO)

    xml_files = glob.glob("data/raw/*.xml")
    if not xml_files:
        print("No CIM XML files found in data/raw/")
    else:
        collection = build_vector_store(xml_files, reset=True)
        print(f"\nIndexed {collection.count()} CIM objects")

        # Quick test query
        hits = similarity_search(collection, "What 345kV transformers are in Charlotte?")
        print(f"\nTest query results ({len(hits)} hits):")
        for h in hits:
            print(f"  - {h['metadata']['name']} (distance: {h['distance']:.4f})")
