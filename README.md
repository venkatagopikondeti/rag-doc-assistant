# RAG Document Assistant

Retrieval-augmented question answering over your own documents — chunking,
embeddings, a vector index, a FastAPI service, and a retrieval-quality
evaluation harness.

[![CI](https://github.com/venkatagopikondeti/rag-doc-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/venkatagopikondeti/rag-doc-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Why it is built this way

Most RAG demos hard-wire one embedding model and one LLM vendor, so they
cannot be tested without network access or an API key. Here both are behind
small interfaces with an offline default:

| Layer | Production backend | Default (offline) |
|---|---|---|
| Embeddings | `sentence-transformers` | deterministic CRC32 hashing embedder |
| Index | FAISS `IndexFlatIP` | NumPy inner-product search |
| Generation | Azure OpenAI chat completions | extractive passage return |

The consequence is that the whole retrieval path — the part that actually
decides answer quality — is unit-tested and measurable in CI without a single
external call. Swap `EMBEDDING_BACKEND=sentence` and set the Azure variables
and the same code path runs against real models.

## Quickstart

```bash
pip install -e ".[dev]"

# index a folder of .md / .txt / .pdf files
ragdoc ingest data/sample_docs --index-dir artifacts/index

# ask a question
ragdoc ask "which index does exact inner product search" --index-dir artifacts/index
```

Run the API:

```bash
uvicorn ragdoc.api:app --reload
# or
docker compose up --build
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | liveness + number of indexed chunks |
| `/ingest` | POST | multipart upload of a `.md` / `.txt` / `.pdf` file |
| `/query` | POST | `{"question": "...", "k": 4}` → answer + scored sources |

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"what does the model registry add","k":3}' | jq
```

## Measuring retrieval

Answer quality is bounded by retrieval quality, so retrieval is scored on its
own against a labelled question set (`data/eval.json`):

```bash
python scripts/evaluate_retrieval.py -k 3
```

```
questions : 6
hit@3     : 100.00%
MRR       : 1.000
```

Add your own `{"question": ..., "source": ...}` pairs and the same command
gives you a regression signal for any change to chunk size, overlap, or
embedding backend. CI runs it on every push.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_BACKEND` | `hashing` | `hashing` or `sentence` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model id |
| `EMBEDDING_DIM` | `512` | dimension of the hashing embedder |
| `INDEX_DIR` | `artifacts/index` | index loaded at API startup |
| `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT` | — | enables LLM generation |

## Layout

```
src/ragdoc/
  chunking.py     paragraph-aware splitter with overlap
  embeddings.py   Embedder interface + hashing / sentence-transformer backends
  store.py        VectorStore: FAISS when available, NumPy otherwise, with persistence
  generator.py    Generator interface + Azure OpenAI / extractive backends
  pipeline.py     ingest a path, answer a question
  api.py          FastAPI app
  cli.py          `ragdoc ingest` / `ragdoc ask`
scripts/evaluate_retrieval.py   hit-rate@k and MRR
tests/            17 tests, no network required
```

## Tests

```bash
pytest --cov=ragdoc --cov-report=term-missing
```

## License

MIT
