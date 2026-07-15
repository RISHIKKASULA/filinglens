from filinglens import models


def test_stub_default_response_is_valid_json_and_deterministic() -> None:
    client = models.StubClient()
    r1 = client.generate("anything")
    r2 = client.generate("something else")
    assert r1.text == r2.text == models.StubClient.DEFAULT_JSON
    assert r1.model == "stub"
    assert r1.digest == "stub"


def test_stub_structured_flag_tracks_schema() -> None:
    client = models.StubClient()
    assert client.generate("p").structured is False
    assert client.generate("p", schema={"type": "object"}).structured is True


def test_stub_custom_responder_depends_on_prompt() -> None:
    client = models.StubClient(model="fake", responder=lambda p: f"echo:{p}")
    r = client.generate("hello")
    assert r.text == "echo:hello"
    assert r.model == "fake"
    assert r.digest == "stub"


def test_stub_satisfies_model_client_protocol() -> None:
    client = models.StubClient(model="fake")
    assert isinstance(client, models.ModelClient)
    assert client.model == "fake"
    assert client.digest == "stub"


def test_frozen_determinism_constants() -> None:
    assert models.TEMPERATURE == 0.0
    assert models.SEED == 42
    assert models.NUM_CTX == 16384
