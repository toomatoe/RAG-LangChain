"""Compatibility entry point: python queryData.py 'your question'."""

import sys

from rag_langchain.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["ask", *sys.argv[1:]]))
