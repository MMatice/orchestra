"""Serveur MCP exposant les agents locaux a Claude Code.

Claude reste l'orchestrateur : il decide quoi deleguer, a qui, et dans quel
ordre. Ce serveur ne fait qu'offrir les agents locaux comme des outils
appelables, et se charge du routage / chainage quand on le lui demande.

Lancement : python -m orchestra.mcp_server   (transport stdio)
"""

from __future__ import annotations

import json
import logging
import sys

try:  # mcp >= 2.0
    from mcp.server import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .backends import BackendUnavailable
from .console import force_utf8_output
from .pipeline import parse_steps, run_pipeline, run_refine_loop
from .registry import AgentNotFound, Orchestra
from .tools import describe_catalogue
from .workspace import Workspace, WorkspaceError

mcp = _Server("orchestra")

_orchestra: Orchestra | None = None


def get_orchestra() -> Orchestra:
    global _orchestra
    if _orchestra is None:
        _orchestra = Orchestra.bootstrap()
    return _orchestra


def _guard(exc: Exception) -> str:
    if isinstance(exc, (AgentNotFound, BackendUnavailable, WorkspaceError)):
        return f"❌ {exc}"
    return f"❌ {type(exc).__name__}: {exc}"


@mcp.tool()
async def orchestra_status() -> str:
    """Inventaire des agents, de leurs privileges, et etat du backend.

    A appeler en premier dans une session : indique quels agents savent agir
    sur des fichiers et lesquels se contentent de produire du texte.
    """
    try:
        orch = get_orchestra()
        return (
            f"{orch.describe()}\n\n"
            f"### Outils\n\n{describe_catalogue()}\n\n"
            f"### Etat\n\n{await orch.health()}"
        )
    except Exception as exc:  # noqa: BLE001
        return _guard(exc)


@mcp.tool()
async def ask_agent(
    agent: str,
    prompt: str,
    context: str = "",
    workspace: str = "",
    allow_writes: bool = False,
) -> str:
    """Envoie une tache a un agent precis, qui peut travailler sur des fichiers.

    Args:
        agent: nom de l'agent (voir orchestra_status), ex. "reviewer", "implementer".
        prompt: l'instruction a executer.
        context: code, diff ou logs a joindre. Inutile de coller un fichier que
            l'agent peut lire lui-meme quand workspace est fourni.
        workspace: repertoire racine sur lequel l'agent travaille. Sans lui,
            l'agent n'a aucun outil et se contente de produire du texte.
        allow_writes: autorise l'agent a creer et modifier des fichiers. Second
            verrou, independant des outils declares par l'agent : les deux sont
            necessaires pour qu'une ecriture ait lieu.
    """
    try:
        run = await get_orchestra().run(
            agent,
            prompt,
            context=context or None,
            workspace=Workspace.open(workspace, writable=allow_writes),
        )
        return run.as_markdown()
    except Exception as exc:  # noqa: BLE001
        return _guard(exc)


@mcp.tool()
async def delegate(
    prompt: str,
    task_type: str = "",
    context: str = "",
    workspace: str = "",
    allow_writes: bool = False,
) -> str:
    """Delegue une tache sans choisir l'agent : le routeur selectionne le mieux place.

    Args:
        prompt: l'instruction a executer.
        task_type: indice de routage optionnel (review, explain, test, document,
            implement, summarize). Ameliore nettement la precision du routage.
        context: code, diff ou logs a joindre.
        workspace: repertoire racine sur lequel l'agent travaille.
        allow_writes: autorise la modification de fichiers.
    """
    try:
        run = await get_orchestra().delegate(
            prompt,
            task_type=task_type or None,
            context=context or None,
            workspace=Workspace.open(workspace, writable=allow_writes),
        )
        return run.as_markdown()
    except Exception as exc:  # noqa: BLE001
        return _guard(exc)


@mcp.tool()
async def pipeline(
    steps: str,
    initial_input: str = "",
    workspace: str = "",
    allow_writes: bool = False,
) -> str:
    """Enchaine plusieurs agents, la sortie de chacun alimentant le suivant.

    Args:
        steps: JSON, liste d'objets {"agent": str, "instruction": str,
            "carry_context": bool}. Exemple :
            [{"agent":"implementer","instruction":"Ecris la fonction"},
             {"agent":"reviewer","instruction":"Relis le code produit"}]
        initial_input: donnee de depart (code, spec, logs) commune aux etapes.
        workspace: repertoire racine commun a toutes les etapes.
        allow_writes: autorise la modification de fichiers.
    """
    try:
        parsed = json.loads(steps)
    except json.JSONDecodeError as exc:
        return f"❌ `steps` n'est pas du JSON valide : {exc}"
    if not isinstance(parsed, list):
        return "❌ `steps` doit etre une liste JSON d'objets."

    try:
        result = await run_pipeline(
            get_orchestra(),
            parse_steps(parsed),
            initial_input,
            workspace=Workspace.open(workspace, writable=allow_writes),
        )
        return result.as_markdown()
    except Exception as exc:  # noqa: BLE001
        return _guard(exc)


@mcp.tool()
async def refine(
    task: str,
    producer: str = "implementer",
    critic: str = "reviewer",
    max_rounds: int = 2,
    workspace: str = "",
    allow_writes: bool = False,
) -> str:
    """Boucle producteur/critique : un agent produit, un autre critique, le premier corrige.

    Avec un workspace, le producteur ecrit sur le disque et le critique relit
    les fichiers reels : les deux agents voient le meme etat, et le livrable
    ne transite plus par le contexte a chaque tour.

    Args:
        task: la tache a realiser.
        producer: agent qui produit (defaut "implementer").
        critic: agent qui critique (defaut "reviewer").
        max_rounds: nombre max de cycles critique/correction (1 a 4).
        workspace: repertoire racine sur lequel les deux agents travaillent.
        allow_writes: autorise la modification de fichiers.
    """
    try:
        rounds = max(1, min(int(max_rounds), 4))
        result = await run_refine_loop(
            get_orchestra(),
            producer,
            critic,
            task,
            max_rounds=rounds,
            workspace=Workspace.open(workspace, writable=allow_writes),
        )
        return result.as_markdown()
    except Exception as exc:  # noqa: BLE001
        return _guard(exc)


def main() -> None:
    # Les rapports renvoyes aux outils contiennent des pictogrammes d'etat.
    force_utf8_output()

    # Le transport stdio est un canal JSON-RPC : rien d'autre ne doit y ecrire.
    # httpx logge chaque requete en INFO, on le fait taire.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    try:
        get_orchestra()
    except Exception as exc:  # noqa: BLE001
        # On echoue tot et bruyamment : un serveur MCP qui demarre avec une
        # config cassee est plus penible a diagnostiquer qu'un crash net.
        print(f"[orchestra] demarrage impossible : {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
