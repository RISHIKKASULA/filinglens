"""Command-line entry point: filinglens fetch | sanity | grade | label | report.

§8 also lists a ``run`` subcommand; it was never exposed. See ADR-007.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

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

    grade_parser = subparsers.add_parser(
        "grade", help="Grade a run against XBRL and print a verdict summary"
    )
    grade_parser.add_argument("run_id", help="run id under runs/, e.g. v0.1")

    label_parser = subparsers.add_parser(
        "label", help="Failure-label review loop (§7): hand-label incorrect items into labels.csv"
    )
    label_parser.add_argument("run_id", help="run id under runs/, e.g. v0.1")

    report_parser = subparsers.add_parser(
        "report", help="Render the evaluation report (with the §7 taxonomy from labels.csv)"
    )
    report_parser.add_argument("run_id", help="run id under runs/, e.g. v0.1")
    report_parser.add_argument(
        "--out", type=Path, default=Path("docs/eval-report.md"), help="output markdown path"
    )

    args = parser.parse_args(argv)

    if args.command == "fetch":
        return _cmd_fetch(args)
    if args.command == "sanity":
        return _cmd_sanity(args)
    if args.command == "grade":
        return _cmd_grade(args)
    if args.command == "label":
        return _cmd_label(args)
    if args.command == "report":
        return _cmd_report(args)
    return 0


def _load_run(args: argparse.Namespace) -> tuple[Any, list[Any], list[Any], dict[str, Any]]:
    """Load a run's results frame, pins, KPIs, and config (shared by grade/label/report)."""
    import json

    import pandas as pd

    run_dir = Path("runs") / args.run_id
    frame = pd.read_parquet(run_dir / "results.parquet")
    pins = corpus.load_manifest(args.manifest).companies
    kpis = corpus.load_kpis()
    config = json.loads((run_dir / "config.json").read_text())
    return frame, pins, kpis, config


def _cmd_grade(args: argparse.Namespace) -> int:
    from collections import Counter

    from filinglens import report

    frame, pins, kpis, _ = _load_run(args)
    items = report.grade_run(frame, pins, kpis, args.cache)
    counts = Counter(i.verdict.value for i in items)
    scored = sum(1 for i in items if i.scored)
    correct = counts.get("correct", 0)
    print(f"run {args.run_id}: {len(items)} items, {scored} scored, {correct} correct")
    for verdict, n in sorted(counts.items()):
        print(f"  {verdict:16} {n}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from filinglens import label, report

    frame, pins, kpis, config = _load_run(args)
    items = report.grade_run(frame, pins, kpis, args.cache)
    run_dir = Path("runs") / args.run_id
    labels = label.load_labels(label.labels_path(run_dir))
    report.write_report(items, args.out, config=config, generated=label.today(), labels=labels)
    print(f"wrote {args.out}")
    return 0


def _cmd_label(args: argparse.Namespace) -> int:  # pragma: no cover - interactive loop
    from filinglens import label, report
    from filinglens.grade import FailureLabel

    frame, pins, kpis, _ = _load_run(args)
    items = report.grade_run(frame, pins, kpis, args.cache)
    run_dir = Path("runs") / args.run_id
    path = label.labels_path(run_dir)

    records = label.load_labels(path)
    # Seed the grader's deterministic calls so labels.csv is complete without re-typing them.
    for record in label.auto_seed(items, args.run_id):
        records.setdefault(record.key, record)
    label.save_labels(records.values(), path)

    pending = label.pending_items(items, records)
    if not pending:
        print(f"all {len(label.incorrect_items(items))} incorrect items are labeled ({path})")
        return 0

    contexts = label._ContextCache(pins, args.cache)
    print(f"{len(pending)} items to review (labels saved to {path} as you go)\n")
    for n, item in enumerate(pending, start=1):
        print(label.evidence(item, contexts.context(item)))
        while True:
            choice = label.parse_choice(input(label.prompt_label(n, len(pending))))
            if choice == "quit":
                print(f"stopped; {len(records)} labeled so far")
                return 0
            if choice == "skip":
                print()
                break
            if isinstance(choice, FailureLabel):
                note = input("  note (optional): ").strip()
                rec = label.LabelRecord(
                    run_id=args.run_id,
                    model=item.model,
                    cell=item.cell,
                    ticker=item.ticker,
                    kpi=item.kpi,
                    label=choice,
                    source="hand",
                    note=note,
                )
                records[rec.key] = rec
                label.save_labels(records.values(), path)
                print()
                break
    print(f"done; {len(records)} items labeled in {path}")
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
