from ragdoc.pipeline import RagPipeline


def test_ingest_directory_and_answer(tmp_path):
    (tmp_path / "a.md").write_text("Prophet models seasonality in time series forecasting.")
    (tmp_path / "b.txt").write_text("Docker packages an application with its dependencies.")
    (tmp_path / "ignore.bin").write_bytes(b"\x00")

    pipeline = RagPipeline()
    assert pipeline.ingest_path(tmp_path) == 2

    result = pipeline.answer("what does prophet do", k=1)
    assert result["sources"][0]["source"] == "a.md"
    assert result["question"] == "what does prophet do"


def test_answer_without_index_says_so():
    result = RagPipeline().answer("anything")
    assert "don't know" in result["answer"]
