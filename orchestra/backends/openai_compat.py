"""Backend OpenAI-compatible (/v1/chat/completions).

Un seul client couvre LiteLLM, vLLM, TGI, LM Studio, llama.cpp server,
OpenRouter, Groq, Together, Azure OpenAI et la plupart des passerelles
internes : tous exposent le meme contrat.

LiteLLM en mode proxy est le cas d'usage entreprise : une passerelle unique
qui centralise le routage, les quotas, les cles et la journalisation, derriere
laquelle on peut placer n'importe quel fournisseur.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from .base import (
    CONNECT_TIMEOUT_S,
    READ_TIMEOUT_S,
    Backend,
    BackendUnavailable,
    ChatResult,
    ToolCall,
    content_or_reasoning,
)


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Extrait les appels d'outils d'un message OpenAI.

    Les arguments arrivent en JSON serialise dans une chaine. Un modele qui
    produit du JSON casse est un cas courant, pas une exception : on remonte
    alors un dictionnaire vide et l'outil se plaindra d'arguments manquants,
    ce qui redonne la main au modele.
    """
    parsed: list[ToolCall] = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        raw = function.get("arguments")
        if isinstance(raw, dict):
            arguments = raw
        else:
            try:
                arguments = json.loads(raw or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        parsed.append(
            ToolCall(id=str(call.get("id") or f"call_{index}"), name=name, arguments=arguments)
        )
    return parsed


class OpenAICompatBackend(Backend):
    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        api_key_env: str | None = None,
        model_overrides: dict[str, str] | None = None,
        num_ctx_cap: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.base_url = (base_url or "").rstrip("/")
        self.api_key_env = api_key_env
        self.model_overrides = model_overrides
        self.num_ctx_cap = num_ctx_cap
        self.extra_headers = extra_headers or {}
        self._timeout = httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)

        if not self.base_url:
            raise BackendUnavailable(
                f"Backend '{name}' : 'base_url' est obligatoire."
            )

    # La cle n'est jamais stockee sur l'instance ni ecrite en configuration :
    # on la relit depuis l'environnement au moment de l'appel.
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "").strip()
            if not key:
                raise BackendUnavailable(
                    f"Backend '{self.name}' : la variable d'environnement "
                    f"{self.api_key_env} est vide ou absente."
                )
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _fail(self, response: httpx.Response) -> BackendUnavailable:
        detail = response.text.strip()
        if len(detail) > 400:
            detail = detail[:400] + "…"
        hint = ""
        if response.status_code in (401, 403):
            hint = (
                f" Verifie la variable {self.api_key_env}."
                if self.api_key_env
                else " Ce backend exige probablement une cle : renseigne 'api_key_env'."
            )
        elif response.status_code == 404:
            hint = " Modele inconnu de la passerelle, ou 'base_url' sans le suffixe /v1."
        elif response.status_code == 429:
            hint = " Quota ou limite de debit atteint sur la passerelle."
        return BackendUnavailable(
            f"{self.name} a repondu {response.status_code}.{hint} {detail}"
        )

    async def ping(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT_S) as client:
                response = await client.get(
                    f"{self.base_url}/models", headers=self._headers()
                )
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"Passerelle injoignable sur {self.base_url} ({exc})."
            ) from exc

        if response.status_code >= 400:
            raise self._fail(response)
        count = len(response.json().get("data", []))
        return f"OpenAI-compatible, {count} modele(s) publie(s)"

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    f"{self.base_url}/models", headers=self._headers()
                )
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"Impossible de lister les modeles sur {self.base_url} ({exc})."
            ) from exc

        if response.status_code >= 400:
            raise self._fail(response)
        return [m.get("id", "") for m in response.json().get("data", []) if m.get("id")]

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        fmt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        options = options or {}
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        # Traduction des options Ollama vers le vocabulaire OpenAI. num_ctx n'a
        # pas d'equivalent : la fenetre est une propriete du deploiement distant.
        if "temperature" in options:
            payload["temperature"] = options["temperature"]
        if "top_p" in options:
            payload["top_p"] = options["top_p"]
        if options.get("num_predict"):
            payload["max_tokens"] = options["num_predict"]
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        elif fmt == "json":
            # Exclusif : contraindre la sortie a un objet JSON empeche le
            # modele d'emettre des appels d'outils.
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"Appel /chat/completions impossible sur {self.base_url} ({exc})."
            ) from exc

        if response.status_code >= 400:
            raise self._fail(response)

        elapsed = time.perf_counter() - started
        data = response.json()
        choices = data.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        calls = _parse_tool_calls(message)
        usage = data.get("usage") or {}

        return ChatResult(
            content=content_or_reasoning(message, bool(calls)),
            tool_calls=calls,
            raw_message=message,
            model=data.get("model", model),
            backend=self.name,
            total_duration_s=elapsed,
            # Pas de decoupage charge/generation cote OpenAI : on rapporte le
            # temps mural, ce qui reste comparable d'un appel a l'autre.
            eval_duration_s=elapsed,
            eval_count=usage.get("completion_tokens", 0),
            prompt_eval_count=usage.get("prompt_tokens", 0),
        )
