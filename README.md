# filinglens

An XBRL-graded evaluation harness answering, with measured accuracy and confidence
intervals: *how reliably can a local 7–8B LLM extract financial figures from SEC filing
text?* The company's own XBRL facts are the auto-grader.

SEC filings are public — privacy is *not* a benefit for this data and this README never
claims it. The point: regulated teams that must run LLMs on-prem (their documents can't
transit third-party APIs) need to know whether small local models can be trusted on
financial text — and you can't benchmark that on confidential documents in public. Public
filings are the license-clean corpus for exactly that document class, and the one place
ground truth is free: every company files its own numbers as structured XBRL. This project
measures the question directly.

**Status:** v0.1 in progress. Design is frozen in [docs/architecture.md](docs/architecture.md);
deviations are recorded as ADRs in [docs/decisions.md](docs/decisions.md). Measured results
and the full findings write-up land with v0.1.0.

## License

MIT
