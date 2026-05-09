# ADR-001: Vector database selection — ChromaDB vs pgvector

**Date:** 2025-05-07
**Status:** Accepted
**Author:** Aarthi Gajendran

---

## Context

The CIM Transmission RAG system needs a vector store to index embeddings of CIM objects
(ACLineSegments, Substations, Transformers, etc.) and support semantic similarity search
at query time.

Two candidates were evaluated:

| Criterion | ChromaDB | pgvector |
|---|---|---|
| Setup complexity | Low — embedded, no server | Medium — requires Postgres |
| Local development | Excellent | Requires Docker |
| Metadata filtering | Native dict filters | SQL WHERE clauses |
| Production scale | <1M vectors (sufficient) | Unlimited (Postgres scale) |
| CIM class filtering | `where={"cim_class": "..."}` | `WHERE metadata->>'cim_class'` |
| AWS deployment | ECS + EFS volume | RDS Postgres w/ pgvector ext. |
| Observability | Limited | Full Postgres tooling |

## Decision

**Use ChromaDB** for Phase 1 (local development + GitHub demo).

Rationale:
1. A typical utility CIM model has 10k–200k objects. ChromaDB handles this comfortably.
2. Zero infrastructure overhead — the entire vector store lives in `data/processed/chroma_db/`
   and can be committed or regenerated from the raw XML in minutes.
3. Native Python metadata filtering maps cleanly to CIM class hierarchy
   (e.g. filter to `ACLineSegment` without a schema migration).
4. The abstraction layer in `vector_store.py` makes swapping to pgvector a 1-file change.

## Migration path to pgvector (Phase 2)

When deploying on AWS with multiple users or real-time CIM updates:

```python
# In vector_store.py, replace ChromaDB with:
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

vector_store = PGVector(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="cim_transmission",
    connection="postgresql+psycopg://user:pass@rds-host:5432/cimdb",
)
```

This enables:
- Multi-user concurrent access
- RDS IAM auth (no plaintext passwords)
- CloudWatch monitoring of query latency
- Incremental upserts from CIM update feeds (IEC 61968 CIM messages)

## Consequences

- All developers must run `python src/embeddings/vector_store.py` before starting the UI.
- The `data/processed/` directory is gitignored (embeddings are not source artifacts).
- The `data/raw/*.xml` files are versioned and serve as the single source of truth.
