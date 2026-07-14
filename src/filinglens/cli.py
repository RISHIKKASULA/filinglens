"""Command-line entry point: filinglens fetch | sanity | run | grade | label | report."""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="filinglens",
        description="XBRL-graded evaluation harness for local-LLM financial extraction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fetch", help="Fetch and pin the corpus (writes corpus.yaml)")

    args = parser.parse_args(argv)

    if args.command == "fetch":
        print("fetch: not implemented yet", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
