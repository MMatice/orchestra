"""Execution de commandes, verrouillee par defaut.

Donner un shell a un modele revient a lui donner la machine : le confinement
du `Workspace` ne s'applique plus des lors qu'un processus fils demarre. Cet
outil n'est donc accorde que si `ORCHESTRA_ALLOW_SHELL` est positionne, et le
filtre ci-dessous ne sert qu'a arreter les destructions accidentelles - pas un
modele qui chercherait activement a en sortir.

Le compromis est assume : c'est ce qui permet a un agent de lancer sa propre
suite de tests et d'iterer sur les echecs, ce qu'aucun outil fichier ne
remplace.
"""

from __future__ import annotations

import os
import re
import subprocess

from ..workspace import Workspace

COMMAND_TIMEOUT_S = 120
MAX_OUTPUT_CHARS = 12_000

# Motifs qui detruisent sans rattrapage possible, ou qui sortent de la racine.
BLOCKED_PATTERNS = (
    (r"\brm\s+-[a-zA-Z]*[rf]", "suppression recursive"),
    (r"\b(mkfs|fdisk|diskpart)\b", "operation disque"),
    (r"\bshutdown\b|\breboot\b", "arret machine"),
    (r"\bsudo\b|\brunas\b", "elevation de privileges"),
    (r"\bgit\s+push\b", "publication vers un depot distant"),
    (r"\bcurl\b.*\|\s*(ba)?sh", "execution de code telecharge"),
    (r"Remove-Item.*-Recurse", "suppression recursive"),
    (r">\s*/dev/[sh]d", "ecriture disque brute"),
)


def _allowed(command: str) -> str | None:
    for pattern, label in BLOCKED_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            return label
    return None


def run_command(workspace: Workspace, command: str) -> str:
    from . import ToolError

    command = (command or "").strip()
    if not command:
        raise ToolError("Commande vide.")

    blocked = _allowed(command)
    if blocked is not None:
        raise ToolError(
            f"Commande refusee ({blocked}). Les operations destructrices et les "
            "publications sortantes ne sont pas accessibles aux agents."
        )

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workspace.root,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
            # Le processus fils herite de l'environnement : on retire au moins
            # les cles d'API qu'Orchestra a chargees, pour qu'une commande ne
            # puisse pas les relire et les afficher.
            env=_sanitised_env(),
        )
    except subprocess.TimeoutExpired:
        raise ToolError(
            f"Commande interrompue apres {COMMAND_TIMEOUT_S}s. "
            "Cible un sous-ensemble plus petit."
        ) from None

    parts = [f"$ {command}", f"code de sortie : {completed.returncode}"]
    for label, stream in (("stdout", completed.stdout), ("stderr", completed.stderr)):
        text = (stream or "").strip()
        if not text:
            continue
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "\n[... sortie tronquee ...]"
        parts.append(f"--- {label} ---\n{text}")

    if len(parts) == 2:
        parts.append("(aucune sortie)")
    return "\n".join(parts)


def _sanitised_env() -> dict[str, str]:
    hidden = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    return {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in hidden)
    }
