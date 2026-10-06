"""Compatibility entry point for the original ingestion command."""

import sys

from rag_langchain.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["ingest", *sys.argv[1:]]))
