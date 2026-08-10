"""Outils reels mis a disposition des agents.

Un agent sans outil ne peut que decrire ce qu'il ferait. Avec ces outils il
lit et modifie l'arborescence, et devient donc utilisable pour du travail
effectif plutot que pour de la suggestion.

Le catalogue est declaratif : un outil est un `ToolSpec` (schema JSON +
fonction). Les agents en selectionnent un sous-ensemble par leur YAML, ce qui
donne le controle des privileges au niveau de l'agent : un `reviewer` lit,
un `implementer` ecrit, et cela se lit dans la configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..workspace import Workspace, WorkspaceError
from .files import edit_file, list_files, read_file, search_files, write_file
from .shell import run_command

__all__ = [
    "TOOLS",
    "ToolSpec",
    "Toolset",
    "ToolError",
    "READ_ONLY_TOOLS",
    "WRITE_TOOLS",
    "build_toolset",
    "describe_catalogue",
]


class ToolError(RuntimeError):
    """Echec attendu d'un outil, renvoye au modele pour qu'il se corrige."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]
    #: modifie le disque : necessite un espace de travail ouvert en ecriture
    writes: bool = False
    #: variable d'environnement a positionner pour deverrouiller l'outil
    requires_opt_in: str | None = None

    def schema(self) -> dict[str, Any]:
        """Declaration au format function calling, commun a OpenAI et Ollama."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_STR = {"type": "string"}


TOOLS: dict[str, ToolSpec] = {
    "list_files": ToolSpec(
        name="list_files",
        description=(
            "Liste les fichiers de l'espace de travail. A appeler en premier "
            "pour decouvrir l'arborescence avant de lire ou de modifier quoi "
            "que ce soit."
        ),
        parameters=_obj(
            {
                "subdir": {
                    **_STR,
                    "description": "Sous-repertoire de depart, '.' pour la racine.",
                },
                "pattern": {
                    **_STR,
                    "description": "Motif glob, par exemple '**/*.py'. Defaut '**/*'.",
                },
            },
            [],
        ),
        handler=list_files,
    ),
    "read_file": ToolSpec(
        name="read_file",
        description=(
            "Lit le contenu integral d'un fichier. Obligatoire avant toute "
            "modification : l'edition exige le texte exact deja present."
        ),
        parameters=_obj(
            {"path": {**_STR, "description": "Chemin relatif a la racine."}},
            ["path"],
        ),
        handler=read_file,
    ),
    "search_files": ToolSpec(
        name="search_files",
        description=(
            "Cherche une expression reguliere dans les fichiers et retourne "
            "les lignes correspondantes avec leur numero."
        ),
        parameters=_obj(
            {
                "pattern": {**_STR, "description": "Expression reguliere Python."},
                "glob": {
                    **_STR,
                    "description": "Restreint aux fichiers correspondants, ex '**/*.py'.",
                },
            },
            ["pattern"],
        ),
        handler=search_files,
    ),
    "write_file": ToolSpec(
        name="write_file",
        description=(
            "Ecrit un fichier, en ecrasant son contenu s'il existe. Reserve a "
            "la creation de fichiers ; pour modifier un fichier existant, "
            "prefere edit_file qui ne touche que la zone concernee."
        ),
        parameters=_obj(
            {
                "path": {**_STR, "description": "Chemin relatif a la racine."},
                "content": {**_STR, "description": "Contenu complet du fichier."},
            },
            ["path", "content"],
        ),
        handler=write_file,
        writes=True,
    ),
    "edit_file": ToolSpec(
        name="edit_file",
        description=(
            "Remplace un fragment de texte exact dans un fichier existant. "
            "old_text doit apparaitre une seule fois : ajoute des lignes de "
            "contexte autour si le fragment est ambigu."
        ),
        parameters=_obj(
            {
                "path": {**_STR, "description": "Chemin relatif a la racine."},
                "old_text": {
                    **_STR,
                    "description": "Texte exact a remplacer, indentation comprise.",
                },
                "new_text": {**_STR, "description": "Texte de remplacement."},
            },
            ["path", "old_text", "new_text"],
        ),
        handler=edit_file,
        writes=True,
    ),
    "run_command": ToolSpec(
        name="run_command",
        description=(
            "Execute une commande dans l'espace de travail et retourne sa "
            "sortie. Utile pour lancer une suite de tests ou un linter."
        ),
        parameters=_obj(
            {"command": {**_STR, "description": "Ligne de commande a executer."}},
            ["command"],
        ),
        handler=run_command,
        writes=True,
        requires_opt_in="ORCHESTRA_ALLOW_SHELL",
    ),
}

READ_ONLY_TOOLS = tuple(n for n, s in TOOLS.items() if not s.writes)
WRITE_TOOLS = tuple(n for n, s in TOOLS.items() if s.writes)


class Toolset:
    """Les outils reellement accordes a un agent pour une execution donnee."""

    def __init__(self, specs: list[ToolSpec], workspace: Workspace) -> None:
        self.specs = {spec.name: spec for spec in specs}
        self.workspace = workspace

    def __bool__(self) -> bool:
        return bool(self.specs)

    @property
    def names(self) -> list[str]:
        return sorted(self.specs)

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self.specs.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Execute un appel d'outil. Retourne (sortie, succes).

        Un echec n'interrompt jamais la boucle : le message d'erreur repart
        vers le modele, qui dispose ainsi de quoi corriger son appel. C'est
        ce qui distingue un agent d'un script.
        """
        spec = self.specs.get(name)
        if spec is None:
            return (
                f"Erreur : outil '{name}' indisponible. "
                f"Outils accordes : {', '.join(self.names)}.",
                False,
            )
        try:
            return spec.handler(self.workspace, **arguments), True
        except TypeError as exc:
            # Arguments manquants ou inattendus : le modele a mal rempli le schema.
            return f"Erreur : arguments invalides pour {name} ({exc}).", False
        except (ToolError, WorkspaceError) as exc:
            return f"Erreur : {exc}", False
        except OSError as exc:
            return f"Erreur systeme sur {name} : {exc}", False


def build_toolset(
    names: list[str], workspace: Workspace | None
) -> Toolset | None:
    """Assemble les outils d'un agent, en filtrant ce qui n'est pas utilisable.

    Un outil d'ecriture demande sur un espace en lecture seule est retire
    silencieusement plutot que de faire echouer l'appel : l'agent travaille
    alors en lecture, ce qui reste utile.
    """
    if not names or workspace is None:
        return None

    selected: list[ToolSpec] = []
    for name in names:
        spec = TOOLS.get(name)
        if spec is None:
            continue
        if spec.writes and not workspace.writable:
            continue
        if spec.requires_opt_in and not _opted_in(spec.requires_opt_in):
            continue
        selected.append(spec)

    return Toolset(selected, workspace) if selected else None


def _opted_in(variable: str) -> bool:
    import os

    return os.environ.get(variable, "").strip().lower() in ("1", "true", "yes", "on")


def validate_tool_names(names: list[str], *, where: str = "") -> list[str]:
    """Rejette a la lecture de la configuration les outils qui n'existent pas."""
    unknown = [n for n in names if n not in TOOLS]
    if unknown:
        raise ValueError(
            f"Outil(s) inconnu(s) {unknown}{where}. Disponibles : {sorted(TOOLS)}"
        )
    return names


def describe_catalogue() -> str:
    lines = ["| Outil | Ecrit | Deverrouillage | Role |", "|---|---|---|---|"]
    for spec in TOOLS.values():
        gate = spec.requires_opt_in or "-"
        lines.append(
            f"| `{spec.name}` | {'oui' if spec.writes else 'non'} | {gate} | "
            f"{spec.description.split('.')[0]}. |"
        )
    return "\n".join(lines)


def format_arguments(arguments: dict[str, Any], limit: int = 90) -> str:
    """Rendu compact d'un appel d'outil pour les rapports d'execution."""
    rendered = json.dumps(arguments, ensure_ascii=False)
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"
