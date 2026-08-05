"""Client HTTP minimal pour un serveur Ollama local.

On n'utilise que trois endpoints : /api/tags (inventaire), /api/chat (inference)
et /api/version (ping). La creation des modeles derives passe par la CLI
`ollama create`, plus stable entre versions que /api/create.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from urllib.parse import urlsplit

import httpx

DEFAULT_PORT = 11434
DEFAULT_HOST = f"http://127.0.0.1:{DEFAULT_PORT}"

# Une generation locale sur 8 Go de VRAM peut prendre plusieurs minutes
# (chargement du modele + prompt long). On est genereux sur le timeout.
READ_TIMEOUT_S = 900.0
CONNECT_TIMEOUT_S = 5.0


def resolve_host() -> str:
    """URL de connexion au serveur Ollama, deduite de OLLAMA_HOST.

    OLLAMA_HOST sert a la fois d'adresse d'ECOUTE pour `ollama serve` et
    d'adresse de CONNEXION pour les clients. Les deux ne coincident pas :
    "0.0.0.0" veut dire "ecoute partout", ce qui n'est pas une cible
    joignable. On normalise donc schema, hote et port.
    """
    raw = os.environ.get("OLLAMA_HOST", "").strip() or DEFAULT_HOST
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
    except ValueError:  # port non numerique dans OLLAMA_HOST
        port = DEFAULT_PORT

    return f"{parts.scheme}://{host}:{port}"


class OllamaUnavailable(RuntimeError):
    """Le serveur Ollama ne repond pas."""


@dataclass
class ChatResult:
    content: str
    model: str
    total_duration_s: float
    load_duration_s: float
    eval_duration_s: float
    eval_count: int
    prompt_eval_count: int

    def stats_line(self) -> str:
        # Le debit se mesure sur la generation seule : total_duration inclut
        # le chargement du modele, qui fausse tout au premier appel.
        tok_s = self.eval_count / self.eval_duration_s if self.eval_duration_s > 0 else 0.0
        line = (
            f"{self.model} | {self.prompt_eval_count} tok in / "
            f"{self.eval_count} tok out | {self.total_duration_s:.1f}s total "
            f"({tok_s:.1f} tok/s)"
        )
        if self.load_duration_s >= 1.0:
            line += f" | chargement {self.load_duration_s:.1f}s"
        return line


class OllamaClient:
    def __init__(self, host: str | None = None) -> None:
        self.host = host or resolve_host()
        self._timeout = httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)

    async def ping(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT_S) as c:
                r = await c.get(f"{self.host}/api/version")
                r.raise_for_status()
                return r.json().get("version", "inconnue")
        except Exception as exc:  # noqa: BLE001 - on remonte un message lisible
            raise OllamaUnavailable(
                f"Ollama injoignable sur {self.host} ({exc}). "
                "Lance `ollama serve` ou l'application Ollama."
            ) from exc

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as c:
                r = await c.get(f"{self.host}/api/tags")
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
        except OllamaUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OllamaUnavailable(
                f"Impossible de lister les modeles sur {self.host} ({exc})."
            ) from exc

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: dict[str, Any] | None = None,
        keep_alive: str = "10m",
        fmt: str | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": options or {},
        }
        if fmt:
            payload["format"] = fmt

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as c:
                r = await c.post(f"{self.host}/api/chat", json=payload)
        except Exception as exc:  # noqa: BLE001
            raise OllamaUnavailable(
                f"Appel /api/chat impossible sur {self.host} ({exc})."
            ) from exc

        if r.status_code == 404:
            raise OllamaUnavailable(
                f"Modele '{model}' absent. Lance `python scripts/build_agents.py` "
                "pour construire les agents, ou `ollama pull` sur le modele de base."
            )
        r.raise_for_status()
        data = r.json()

        return ChatResult(
            content=(data.get("message") or {}).get("content", "").strip(),
            model=data.get("model", model),
            total_duration_s=data.get("total_duration", 0) / 1e9,
            load_duration_s=data.get("load_duration", 0) / 1e9,
            eval_duration_s=data.get("eval_duration", 0) / 1e9,
            eval_count=data.get("eval_count", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
        )
