from ragdoc.chunking import Chunk
from ragdoc.store import VectorStore

DOCS = [
    Chunk("The mitochondria is the powerhouse of the cell.", "bio.md", 0),
    Chunk("Kubernetes schedules containers across a cluster of nodes.", "infra.md", 0),
    Chunk("MLflow tracks experiments, parameters and model artifacts.", "mlops.md", 0),
]


def build() -> VectorStore:
    store = VectorStore()
    store.add(DOCS)
    return store


def test_empty_store_returns_no_hits():
    assert VectorStore().search("anything") == []


def test_search_ranks_the_relevant_chunk_first():
    hits = build().search("how does mlflow track experiments", k=3)
    assert hits[0][0].source == "mlops.md"


def test_k_is_capped_at_corpus_size():
    assert len(build().search("cluster", k=99)) == 3


def test_roundtrip_persistence(tmp_path):
    build().save(tmp_path)
    reloaded = VectorStore.load(tmp_path)
    assert len(reloaded) == 3
    assert reloaded.search("kubernetes nodes", k=1)[0][0].source == "infra.md"
