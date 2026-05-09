# CIM Transmission System RAG

> **Retrieval-Augmented Generation over IEC 61970 CIM/XML grid data**
> Ask natural-language questions about transmission assets — substations, lines, transformers, breakers — and get grounded answers with source citations.

![CI](https://github.com/YOUR_USERNAME/cim-transmission-rag/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)

---

## Why this project exists

Utilities store grid topology in CIM (Common Information Model, IEC 61968/61970) files — RDF/XML documents that encode every substation, line segment, transformer, and breaker as a graph of interconnected objects.

Querying these files traditionally requires SPARQL expertise or bespoke tooling. This project demonstrates how LLMs + RAG can make CIM data conversationally queryable by operations engineers, without rewriting their EMS/SCADA stack.

---

## System architecture

```
CIM/XML files (IEC 61970 CGMES)
        │
        ▼
   CIM Parser (rdflib)
   Extracts: Substation, ACLineSegment, PowerTransformer,
             Breaker, BusbarSection, LinearShuntCompensator …
        │
        ▼
   Chunker — one document per CIM object, rich text format
        │
        ▼
   OpenAI text-embedding-3-small
        │
        ▼
   ChromaDB (persistent vector store)
        │
   ┌────┴────────────────────┐
   │                         │
   ▼                         ▼
Streamlit UI           Eval notebook
(Q&A + citations)   (Hit rate, MRR, faithfulness)
   │
   ▼
LLM (GPT-4o-mini / Claude / Bedrock)
```

---

## Supported CIM classes

| Class | Description |
|---|---|
| `Substation` | HV/EHV substations with region and voltage level |
| `ACLineSegment` | Transmission line impedance (R, X, B), length, voltage |
| `PowerTransformer` | Autotransformers and two-winding transformers |
| `PowerTransformerEnd` | Per-winding parameters, rated voltage/MVA, connection type |
| `Breaker` | Rated current, interrupting capacity, normal state |
| `BusbarSection` | Bus configuration, max fault current |
| `LinearShuntCompensator` | Capacitor/reactor banks, MVAR rating, susceptance |
| `VoltageLevel` | Voltage zones within a substation |
| `BaseVoltage` | Nominal kV levels |
| `ConnectivityNode` | Topology nodes |
| `Terminal` | Equipment connection points |

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/cim-transmission-rag
cd cim-transmission-rag
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Add your CIM files

Place IEC 61970 CIM/XML (CGMES format) files in `data/raw/`.
A realistic sample file is included at `data/raw/sample_transmission.xml`.

### 4. Build the vector index

```bash
python src/embeddings/vector_store.py
```

This parses all XML files in `data/raw/`, embeds each CIM object, and persists the index to `data/processed/chroma_db/`.

### 5. Launch the UI

```bash
streamlit run src/ui/app.py
```

### 6. Run evaluation

```bash
python notebooks/evaluate_rag.py
```

---

## Example questions

```
What is the positive-sequence reactance of the Charlotte-Gastonia 138kV line?
List all 345kV equipment at Charlotte North substation.
What is the rated MVA of the T1 autotransformer and its winding configuration?
Which shunt compensators provide voltage support and what is their MVAR rating?
What is the interrupting rating of the 138kV breaker on Line 001?
Describe the zero-sequence parameters of all ACLineSegments in the model.
```

---

## Evaluation results

| Metric | Value |
|---|---|
| Hit Rate @ 5 | _run eval to populate_ |
| Mean Reciprocal Rank | _run eval to populate_ |
| Answer Faithfulness | _run eval to populate_ |

Run `python notebooks/evaluate_rag.py` to populate. Results are saved to `notebooks/eval_results.json`.

---

## Project structure

```
cim-transmission-rag/
├── data/
│   ├── raw/                    # CIM/XML source files (versioned)
│   └── processed/              # ChromaDB vector store (gitignored)
├── src/
│   ├── parser/
│   │   └── cim_parser.py       # rdflib-based CIM object extractor
│   ├── embeddings/
│   │   └── vector_store.py     # ChromaDB ingestion and retrieval
│   ├── retrieval/
│   │   └── rag_chain.py        # LangChain RAG chain with CIM prompt
│   └── ui/
│       └── app.py              # Streamlit Q&A interface
├── notebooks/
│   └── evaluate_rag.py         # Hit rate, MRR, faithfulness benchmarks
├── docs/
│   └── adr/
│       └── ADR-001-vector-db-choice.md
├── tests/
│   └── test_cim_parser.py
├── .github/workflows/ci.yml
└── requirements.txt
```

---

## Extending to AWS Bedrock

To replace OpenAI embeddings with Amazon Bedrock (Titan Embeddings):

```python
from langchain_aws import BedrockEmbeddings
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)
```

To use Claude on Bedrock as the LLM:

```python
from langchain_aws import ChatBedrock
llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
```

---

## Architecture decisions

See [`docs/adr/`](docs/adr/) for architecture decision records:
- [ADR-001: Vector database selection — ChromaDB vs pgvector](docs/adr/ADR-001-vector-db-choice.md)

---

## Data notice

> All CIM data in this repository is **synthetic**, generated for demonstration purposes using standard IEC 61970 CGMES schema and textbook ACSR conductor parameters. It does not represent any real utility's network model, asset register, or operational data. The utility name "Piedmont Transmission LLC" is fictional.

---

## Author

**Aarthi Gajendran** · [LinkedIn](https://linkedin.com/in/aarthi-gajendran) · Technical Product / Data Platform Lead in the energy sector

*Built to demonstrate GenAI + CIM domain expertise for AI Architect and AI Product Owner roles.*
