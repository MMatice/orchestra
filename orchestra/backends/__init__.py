"""Selection et construction du backend d'inference."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .base import Backend, BackendUnavailable, ChatResult
from .ollama import OllamaBackend, normalize_base_url
from .openai_compat import OpenAICompatBackend

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKENDS_FILE = PROJECT_ROOT / "config" / "backends.yaml"

BACKEND_TYPES = {
    "ollama": OllamaBackend,
    "openai": OpenAICompatBackend,
}

__all__ = [
    "Backend",
    "BackendUnavailable",
    "ChatResult",
    "OllamaBackend",
    "OpenAICompatBackend",
    "create_backend",
    "load_backend_config",
    "normalize_base_url",
]


def load_backend_config(path: Path | None = None) -> tuple[str, dict[str, dict]]:
    """Retourne (nom du backend par defaut, definitions)."""
    path = path or BACKENDS_FILE
    if not path.is_file():
        # Sans fichier de configuration, on reste sur l'Ollama local.
        return "ollama", {"ollama": {"type": "ollama"}}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("backends") or {}
    if not entries:
        raise BackendUnavailable(f"Aucun backend defini dans {path}")
    return str(raw.get("default") or next(iter(entries))), entries


def create_backend(
    name: str | None = None, path: Path | None = None
) -> tuple[Backend, str]:
    """Construit le backend actif. Retourne (backend, raison du choix).

    Ordre de priorite : argument explicite > ORCHESTRA_BACKEND > `default`
    du fichier de configuration.
    """
    default_name, entries = load_backend_config(path)

    reason = "defaut du fichier de configuration"
    if name:
        reason = "demande explicitement"
    elif os.environ.get("ORCHESTRA_BACKEND", "").strip():
        name = os.environ["ORCHESTRA_BACKEND"].strip()
        reason = "force via ORCHESTRA_BACKEND"
    else:
        name = default_name

    if name not in entries:
        raise BackendUnavailable(
            f"Backend '{name}' inconnu. Definis-le dans config/backends.yaml. "
            f"Disponibles : {sorted(entries)}"
        )

    spec = entries[name] or {}
    kind = str(spec.get("type", "ollama")).strip().lower()
    if kind not in BACKEND_TYPES:
        raise BackendUnavailable(
            f"Backend '{name}' : type '{kind}' inconnu, attendu {sorted(BACKEND_TYPES)}"
        )

    # ORCHESTRA_BASE_URL permet de repointer un backend sans editer le YAML,
    # typiquement pour basculer un poste vers un serveur partage.
    base_url = os.environ.get("ORCHESTRA_BASE_URL", "").strip() or spec.get("base_url")

    models = spec.get("models") or None
    if models:
        models = {str(k): str(v) for k, v in models.items()}
    num_ctx = int(spec["num_ctx"]) if spec.get("num_ctx") else None

    if kind == "ollama":
        backend: Backend = OllamaBackend(
            name=name,
            base_url=base_url,
            model_overrides=models,
            num_ctx_cap=num_ctx,
        )
    else:
        headers = spec.get("extra_headers") or {}
        backend = OpenAICompatBackend(
            name=name,
            base_url=base_url,
            api_key_env=spec.get("api_key_env") or None,
            model_overrides=models,
            num_ctx_cap=num_ctx,
            extra_headers={str(k): str(v) for k, v in headers.items()},
        )

    return backend, reason
