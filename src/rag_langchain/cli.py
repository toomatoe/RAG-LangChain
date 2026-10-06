"""Command line entry point."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .config import Settings
from .service import RAGService


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Index local documents and ask grounded questions."
    )
    parser.add_argument("--data", type=Path, help="Document file or directory")
    parser.add_argument("--db", type=Path, help="Persistent index directory")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Synchronize the index with local documents")
    ingest.add_argument("--rebuild", action="store_true", help="Re-embed the full corpus")
    commands.add_parser("status", help="Show indexed files and configuration")
    for name in ("ask", "search"):
        command = commands.add_parser(name)
        command.add_argument("question")
        command.add_argument("--top-k", type=int)
        command.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        overrides = {
            key: value
            for key, value in (("data_path", args.data), ("db_path", args.db))
            if value is not None
        }
        service = RAGService(replace(settings, **overrides))
        if args.command == "ingest":
            print(json.dumps(service.ingest(args.rebuild), indent=2))
        elif args.command == "status":
            print(json.dumps(service.status(), indent=2))
        elif args.command == "search":
            sources = service.search(args.question, args.top_k)
            if args.json:
                print(json.dumps([asdict(s) for s in sources], indent=2))
            else:
                for source in sources:
                    print(f"[{source.number}] {source.source} (distance {source.distance:.4f})")
                    print(source.excerpt + "\n")
        else:
            result = service.ask(args.question, args.top_k)
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(result.answer)
                for source in result.sources:
                    page = f", page {source.page}" if source.page else ""
                    print(f"[{source.number}] {source.source}{page}")
        return 0
    except Exception as exc:
        # Provider exceptions can include request details: keep CLI output safe by default.
        message = (
            str(exc)
            if isinstance(exc, ValueError)
            else (
                f"Operation failed ({type(exc).__name__}). Check credentials, model access, "
                "network connectivity, and the database path."
            )
        )
        print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
