"""Model clients: OllamaClient for real inference, StubClient for deterministic tests.

The grid is run under frozen determinism settings (architecture.md §6): temperature 0,
seed 42, num_ctx 16384, one model loaded at a time. Each response records the resolved
model digest so results pin to an exact set of weights. CI never talks to Ollama — the
StubClient returns canned, deterministic responses so the full extract-grade-report
pipeline runs offline (§8).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# Frozen inference determinism (architecture.md §6).
TEMPERATURE = 0.0
SEED = 42
NUM_CTX = 16384


class ModelResponse(BaseModel):
    """One model call's result."""

    model: str
    digest: str  # weights digest for pinning; "stub" for the StubClient
    text: str
    structured: bool  # whether a JSON schema was enforced


@runtime_checkable
class ModelClient(Protocol):
    """The interface the extraction runner depends on."""

    @property
    def model(self) -> str: ...

    @property
    def digest(self) -> str: ...

    def generate(self, prompt: str, schema: dict[str, Any] | None = None) -> ModelResponse: ...


class OllamaClient:
    """Real inference via a local Ollama server (not exercised in CI)."""

    def __init__(self, model: str, host: str | None = None) -> None:  # pragma: no cover
        from ollama import Client

        self._model = model
        self._client = Client(host=host)
        self._digest = self._resolve_digest(model)

    def _resolve_digest(self, model: str) -> str:  # pragma: no cover
        for m in self._client.list().models:
            if m.model == model and m.digest:
                return m.digest
        return "unknown"

    @property
    def model(self) -> str:  # pragma: no cover
        return self._model

    @property
    def digest(self) -> str:  # pragma: no cover
        return self._digest

    def generate(
        self, prompt: str, schema: dict[str, Any] | None = None
    ) -> ModelResponse:  # pragma: no cover
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            format=schema if schema is not None else None,
            options={"temperature": TEMPERATURE, "seed": SEED, "num_ctx": NUM_CTX},
        )
        return ModelResponse(
            model=self._model,
            digest=self._digest,
            text=response.message.content or "",
            structured=schema is not None,
        )


class StubClient:
    """Deterministic canned responses for tests and offline pipeline runs.

    ``responder`` maps a prompt to raw model text; the default returns a fixed valid
    extraction JSON regardless of prompt. Because output depends only on the prompt (and
    the frozen default is constant), a grid run is byte-for-byte reproducible — the
    property the resumability test relies on.
    """

    DEFAULT_JSON = (
        '{"kpi": "revenue", "value": 0.0, "scale": "millions", '
        '"currency": "USD", "fiscal_period_end": "2025-01-01"}'
    )

    def __init__(
        self,
        model: str = "stub",
        responder: Any = None,
    ) -> None:
        self._model = model
        self._responder = responder

    @property
    def model(self) -> str:
        return self._model

    @property
    def digest(self) -> str:
        return "stub"

    def generate(self, prompt: str, schema: dict[str, Any] | None = None) -> ModelResponse:
        text = self.DEFAULT_JSON if self._responder is None else self._responder(prompt)
        return ModelResponse(
            model=self._model,
            digest="stub",
            text=text,
            structured=schema is not None,
        )
