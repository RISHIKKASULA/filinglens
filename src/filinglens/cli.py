"""Command-line entry point: filinglens fetch | sanity | run | grade | label | report."""

import argparse
import sys
from pathlib import Path

from filinglens import corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="filinglens",
        description="XBRL-graded evaluation harness for local-LLM financial extraction.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=corpus.DEFAULT_MANIFEST_PATH, help="corpus.yaml path"
    )
    parser.add_argument(
        "--cache", type=Path, default=corpus.DEFAULT_CACHE_DIR, help="cache directory"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Pin the corpus (writes corpus.yaml) and cache filing artifacts"
    )
    fetch_parser.add_argument(
        "tickers",
        nargs="*",
        default=list(corpus.V01_TICKERS),
        help="tickers to fetch (default: the frozen v0.1 corpus)",
    )

    sanity_parser = subparsers.add_parser(
        "sanity",
        help="Ground-truth sanity gate (§0): print XBRL fact vs filing-text pairs for review",
    )
    sanity_parser.add_argument("tickers", nargs="+", help="tickers to check, e.g. AAPL MSFT")

    args = parser.parse_args(argv)

    if args.command == "fetch":
        return _cmd_fetch(args)
    if args.command == "sanity":
        return _cmd_sanity(args)
    return 0


def _cmd_sanity(args: argparse.Namespace) -> int:
    from filinglens import sanity

    manifest = corpus.load_manifest(args.manifest)
    pins = []
    for ticker in args.tickers:
        pin = manifest.get(ticker)
        if pin is None:
            print(
                f"error: {ticker.upper()} is not pinned; run `filinglens fetch` first",
                file=sys.stderr,
            )
            return 1
        corpus.fetch_artifacts(pin, args.cache)  # cache-first no-op when already fetched
        pins.append(pin)
    kpis = corpus.load_kpis()
    print(sanity.render_report(sanity.build_pairs(pins, kpis, args.cache)))
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    tickers = tuple(t.upper() for t in args.tickers)
    manifest = corpus.pin_corpus(tickers, args.manifest)
    for ticker in tickers:
        pin = manifest.get(ticker)
        if pin is None:
            print(f"error: {ticker} could not be pinned", file=sys.stderr)
            return 1
        corpus.fetch_artifacts(pin, args.cache)
    print(f"corpus pinned in {args.manifest}; artifacts cached under {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
