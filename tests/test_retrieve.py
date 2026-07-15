import pytest

from filinglens import retrieve

# Chunks with a clear relevance gradient for a "net income" query.
DOC = (
    "The quick brown fox jumps over the lazy dog in the meadow. "
    + ("filler words about weather and geography and history. " * 20)
    + "Consolidated net income for the fiscal year was 112,010 million dollars. "
    + ("more filler about supply chains and logistics and inventory. " * 20)
    + "Net income net earnings totaled 112,010 as reported in the statements. "
    + ("closing filler about governance and risk and compliance. " * 20)
)


def test_chunk_text_windows_and_indices() -> None:
    chunks = retrieve.chunk_text("a b c d e f g", chunk_tokens=3)
    assert [c.text for c in chunks] == ["a b c", "d e f", "g"]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_chunk_text_empty() -> None:
    assert retrieve.chunk_text("") == []


def test_retrieve_ranks_relevant_chunk_first() -> None:
    top = retrieve.retrieve(DOC, "net income net earnings", top_k=3, chunk_tokens=12)
    assert "112,010" in top[0].text


def test_retrieve_top_k_and_determinism() -> None:
    a = retrieve.retrieve(DOC, "net income", top_k=5, chunk_tokens=12)
    b = retrieve.retrieve(DOC, "net income", top_k=5, chunk_tokens=12)
    assert len(a) == 5
    assert [c.index for c in a] == [c.index for c in b]


def test_retrieve_ties_break_by_index() -> None:
    # Two identical chunks tie on score; the lower index must come first.
    doc = "alpha beta gamma alpha beta gamma"
    top = retrieve.retrieve(doc, "zzz", top_k=2, chunk_tokens=3)
    assert [c.index for c in top] == [0, 1]


def test_retrieve_empty_doc() -> None:
    assert retrieve.retrieve("", "net income") == []


def test_retrieve_for_kpi_uses_frozen_query() -> None:
    top = retrieve.retrieve_for_kpi(DOC, "net_income", top_k=2, chunk_tokens=12)
    assert "112,010" in top[0].text


def test_retrieve_for_kpi_unknown_raises() -> None:
    with pytest.raises(KeyError):
        retrieve.retrieve_for_kpi(DOC, "not_a_kpi")


def test_kpi_queries_cover_v01_kpis() -> None:
    assert set(retrieve.KPI_QUERIES) == {
        "revenue",
        "net_income",
        "eps_diluted",
        "total_assets",
        "operating_cash_flow",
    }
