"""Chargement optionnel d'un fichier .env, sans dependance externe.

Regle : l'environnement reel gagne toujours. Un .env fournit des valeurs par
defaut pour le poste de travail, il ne doit jamais ecraser ce qu'un operateur
ou un orchestrateur a explicitement exporte.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Charge le .env s'il existe. Retourne les variables effectivement posees."""
    path = path or DOTENV_PATH
    if not path.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key:
            continue

        # Une variable deja presente dans l'environnement n'est pas touchee.
        if key in os.environ:
            continue

        value = _strip_quotes(value.strip())
        os.environ[key] = value
        applied[key] = value

    return applied
