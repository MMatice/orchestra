"""Backend Ollama (API native /api/chat)."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx

from .base import (
    CONNECT_TIMEOUT_S,
    READ_TIMEOUT_S,
    Backend,
    BackendUnavailable,
    ChatResult,
)

DEFAULT_PORT = 11434
DEFAULT_BASE_URL = f"http://127.0.0.1:{DEFAULT_PORT}"


def normalize_base_url(raw: str) -> str:
    """Normalise schema, hote et port d'une URL Ollama.

    OLLAMA_HOST sert a la fois d'adresse d'ecoute pour `ollama serve` et
    d'adresse de connexion pour les clients ; les deux ne coincident pas.
    """
    raw = (raw or "").strip() or DEFAULT_BASE_URL
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw

    parts = urlsplit(raw)
    host = parts.hostname or "127.0.0.1"
    if host in ("0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    if ":" in host:  # IPv6 litteral
        host = f"[{host}]"

    try:
        port = parts.port or DEFAULT_PORT
    except ValueError:
        port = DEFAULT_PORT

    return f"{parts.scheme}://{host}:{port}"


class OllamaBackend(Backend):
    supports_pull = True

    def __init__(
        self,
        name: str = "ollama",
        base_url: str | None = None,
        *,
        model_overrides: dict[str, str] | None = None,
        num_ctx_cap: int | None = None,
    ) -> None:
        self.name = name
        self.base_url = normalize_base_url(base_url or DEFAULT_BASE_URL)
        self.model_overrides = model_overrides
        self.num_ctx_cap = num_ctx_cap
        self._timeout = httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)

    async def ping(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT_S) as client:
                response = await client.get(f"{self.base_url}/api/version")
                response.raise_for_status()
                return response.json().get("version", "inconnue")
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"Ollama injoignable sur {self.base_url} ({exc}). "
                "Demarre `ollama serve` ou l'application Ollama."
            ) from exc

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                return [m["name"] for m in response.json().get("models", [])]
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"Impossible de lister les modeles sur {self.base_url} ({exc})."
            ) from exc

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: dict[str, Any] | None = None,
        fmt: str | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": options or {},
        }
        if fmt:
            payload["format"] = fmt

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"Appel /api/chat impossible sur {self.base_url} ({exc})."
            ) from exc

        if response.status_code == 404:
            raise BackendUnavailable(
                f"Modele '{model}' absent du serveur Ollama. "
                "Lance `python -m orchestra.cli pull` pour installer ceux du profil."
            )
        response.raise_for_status()
        data = response.json()

        return ChatResult(
            content=(data.get("message") or {}).get("content", "").strip(),
            model=data.get("model", model),
            backend=self.name,
            total_duration_s=data.get("total_duration", 0) / 1e9,
            load_duration_s=data.get("load_duration", 0) / 1e9,
            eval_duration_s=data.get("eval_duration", 0) / 1e9,
            eval_count=data.get("eval_count", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
        )
