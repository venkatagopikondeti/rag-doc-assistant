"""Retrieval quality harness: hit-rate@k and MRR on a labelled question set.

Usage:
    python scripts/evaluate_retrieval.py --docs data/sample_docs --questions data/eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragdoc.pipeline import RagPipeline


def evaluate(pipeline: RagPipeline, questions: list[dict], k: int) -> dict:
    hits, reciprocal = 0, 0.0
    rows = []
    for item in questions:
        results = pipeline.store.search(item["question"], k=k)
        sources = [c.source for c, _ in results]
        rank = sources.index(item["source"]) + 1 if item["source"] in sources else 0
        hits += rank > 0
        reciprocal += 1 / rank if rank else 0.0
        rows.append({"question": item["question"], "expected": item["source"], "rank": rank})
    n = len(questions) or 1
    return {"n": len(questions), "hit_rate@k": hits / n, "mrr": reciprocal / n, "detail": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--questions", default="data/eval.json")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    pipeline = RagPipeline()
    pipeline.ingest_path(args.docs)
    report = evaluate(pipeline, json.loads(Path(args.questions).read_text()), args.k)

    print(f"questions : {report['n']}")
    print(f"hit@{args.k}    : {report['hit_rate@k']:.2%}")
    print(f"MRR       : {report['mrr']:.3f}")
    for row in report["detail"]:
        status = f"rank {row['rank']}" if row["rank"] else "MISS"
        print(f"  [{status:>6}] {row['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
