import pytest

from ragdoc.chunking import chunk_document, split_text


def test_short_text_is_one_chunk():
    assert split_text("hello world") == ["hello world"]


def test_respects_chunk_size():
    text = "\n\n".join("word " * 60 for _ in range(10))
    chunks = split_text(text, chunk_size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_long_paragraph_is_hard_wrapped():
    chunks = split_text("x" * 2500, chunk_size=500, overlap=50)
    assert len(chunks) >= 5
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_document_carries_metadata():
    chunks = chunk_document("a\n\nb", source="notes.md", chunk_size=10, overlap=0)
    assert [c.source for c in chunks] == ["notes.md"]
    assert chunks[0].index == 0


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", chunk_size=size, overlap=overlap)
