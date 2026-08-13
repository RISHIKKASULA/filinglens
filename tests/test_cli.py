"""CLI error paths that must stay friendly (no tracebacks)."""

from __future__ import annotations

from pathlib import Path

import pytest

from filinglens import cli


def test_grade_without_cache_prints_fetch_instructions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_cache = tmp_path / "cache"
    empty_cache.mkdir()
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--cache", str(empty_cache), "grade", "v0.1"])
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "ground-truth cache missing" in err
    assert "filinglens fetch" in err
    assert "Traceback" not in err
