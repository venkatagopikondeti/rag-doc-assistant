"""Command line interface: build an index, then ask questions."""
from __future__ import annotations

import argparse
import json

from .pipeline import RagPipeline
from .store import VectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragdoc")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index a file or directory")
    ingest.add_argument("path")
    ingest.add_argument("--index-dir", default="artifacts/index")

    ask = sub.add_parser("ask", help="query an existing index")
    ask.add_argument("question")
    ask.add_argument("--index-dir", default="artifacts/index")
    ask.add_argument("-k", type=int, default=4)

    args = parser.parse_args(argv)

    if args.command == "ingest":
        pipeline = RagPipeline()
        added = pipeline.ingest_path(args.path)
        pipeline.store.save(args.index_dir)
        print(f"indexed {added} chunks -> {args.index_dir}")
        return 0

    pipeline = RagPipeline(store=VectorStore.load(args.index_dir))
    print(json.dumps(pipeline.answer(args.question, k=args.k), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
