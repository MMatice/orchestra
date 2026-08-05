import httpx
import pytest

from orchestra.backends import (
    BackendUnavailable,
    OllamaBackend,
    OpenAICompatBackend,
    create_backend,
    load_backend_config,
    normalize_base_url,
)


# --------------------------------------------------------------- configuration


def test_shipped_config_declares_litellm_and_a_default():
    default_name, entries = load_backend_config()
    assert default_name in entries
    assert "litellm" in entries
    assert entries["litellm"]["type"] == "openai"


def test_no_api_key_is_ever_stored_in_config():
    # Les cles se declarent par NOM de variable d'environnement, jamais en clair.
    _, entries = load_backend_config()
    for name, spec in entries.items():
        assert "api_key" not in (spec or {}), f"cle en clair dans le backend '{name}'"


def test_env_selects_the_backend(monkeypatch):
    monkeypatch.setenv("ORCHESTRA_BACKEND", "litellm")
    monkeypatch.delenv("ORCHESTRA_BASE_URL", raising=False)
    backend, reason = create_backend()
    assert backend.name == "litellm"
    assert isinstance(backend, OpenAICompatBackend)
    assert "ORCHESTRA_BACKEND" in reason


def test_base_url_can_be_repointed_without_editing_yaml(monkeypatch):
    monkeypatch.setenv("ORCHESTRA_BACKEND", "vllm")
    monkeypatch.setenv("ORCHESTRA_BASE_URL", "http://autre.interne:9000/v1")
    backend, _ = create_backend()
    assert backend.base_url == "http://autre.interne:9000/v1"


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("ORCHESTRA_BACKEND", "nexistepas")
    with pytest.raises(BackendUnavailable, match="inconnu"):
        create_backend()


# ------------------------------------------------------------ resolution Ollama


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", "http://127.0.0.1:11434"),
        ("127.0.0.1", "http://127.0.0.1:11434"),
        # Adresse d'ecoute, pas de connexion : doit devenir joignable.
        ("0.0.0.0", "http://127.0.0.1:11434"),
        ("0.0.0.0:11434", "http://127.0.0.1:11434"),
        ("gpu.interne:11434", "http://gpu.interne:11434"),
        ("https://gpu.interne", "https://gpu.interne:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_ollama_url_normalisation(raw, expected):
    assert normalize_base_url(raw) == expected


# ------------------------------------------------------- backend OpenAI-compat


def _mock_backend(handler, **kwargs):
    backend = OpenAICompatBackend("test", "http://passerelle/v1", **kwargs)
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def factory(*args, **kw):
        kw["transport"] = transport
        return original(*args, **kw)

    return backend, factory


@pytest.mark.asyncio
async def test_chat_translates_options_and_reads_usage(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={
                "model": "qwen-code",
                "choices": [{"message": {"content": "  resultat  "}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 34},
            },
        )

    monkeypatch.setenv("GW_KEY", "secret-de-test")
    backend, factory = _mock_backend(handler, api_key_env="GW_KEY")
    monkeypatch.setattr(httpx, "AsyncClient", factory)

    result = await backend.chat(
        "qwen-code",
        [{"role": "user", "content": "salut"}],
        options={"temperature": 0.1, "top_p": 0.8, "num_predict": 512, "num_ctx": 8192},
        fmt="json",
    )

    assert captured["url"] == "http://passerelle/v1/chat/completions"
    assert captured["auth"] == "Bearer secret-de-test"
    # num_predict -> max_tokens ; num_ctx n'a pas d'equivalent OpenAI.
    assert captured["body"]["max_tokens"] == 512
    assert captured["body"]["temperature"] == 0.1
    assert "num_ctx" not in captured["body"]
    assert captured["body"]["response_format"] == {"type": "json_object"}

    assert result.content == "resultat"
    assert result.prompt_eval_count == 120
    assert result.eval_count == 34
    assert result.backend == "test"


@pytest.mark.asyncio
async def test_missing_api_key_fails_before_any_call(monkeypatch):
    monkeypatch.delenv("GW_KEY", raising=False)
    backend = OpenAICompatBackend("test", "http://passerelle/v1", api_key_env="GW_KEY")

    with pytest.raises(BackendUnavailable, match="GW_KEY"):
        await backend.chat("m", [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_http_errors_are_actionable(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    monkeypatch.setenv("GW_KEY", "mauvaise")
    backend, factory = _mock_backend(handler, api_key_env="GW_KEY")
    monkeypatch.setattr(httpx, "AsyncClient", factory)

    with pytest.raises(BackendUnavailable, match="GW_KEY"):
        await backend.chat("m", [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_list_models_reads_the_openai_catalog(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/models")
        return httpx.Response(
            200, json={"data": [{"id": "modele-a"}, {"id": "modele-b"}]}
        )

    backend, factory = _mock_backend(handler)
    monkeypatch.setattr(httpx, "AsyncClient", factory)

    assert await backend.list_models() == ["modele-a", "modele-b"]


def test_base_url_is_mandatory():
    with pytest.raises(BackendUnavailable, match="base_url"):
        OpenAICompatBackend("test", "")


def test_local_backend_can_pull_remote_cannot():
    assert OllamaBackend().supports_pull is True
    assert OpenAICompatBackend("gw", "http://x/v1").supports_pull is False
