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

mcp = _Server("orchestra")

_orchestra: Orchestra | None = None


def get_orchestra() -> Orchestra:
    global _orchestra
    if _orchestra is None:
        _orchestra = Orchestra.bootstrap()
    return _orchestra


def _guard(exc: Exception) -> str:
    if isinstance(exc, (AgentNotFound, BackendUnavailable)):
        return f"❌ {exc}"
    return f"❌ {type(exc).__name__}: {exc}"


@mcp.tool()
async def orchestra_status() -> str:
    """Inventaire des agents, backend d'inference actif et etat du service.

    A appeler en premier dans une session pour savoir ce qui est disponible.
    """
    try:
        orch = get_orchestra()
        return f"{orch.describe()}\n\n### Etat\n\n{await orch.health()}"
    except Exception as exc:  # noqa: BLE001
        return _guard(exc)


@mcp.tool()
async def ask_agent(agent: str, prompt: str, context: str = "") -> str:
    """Envoie une tache a un agent local precis.

    Args:
        agent: nom de l'agent (voir orchestra_status), ex. "reviewer", "explainer".
        prompt: l'instruction a executer.
        context: code, diff ou logs sur lesquels l'agent doit travailler.
    """
    try:
        run = await get_orchestra().run(agent, prompt, context=context or None)
        return run.as_markdown()
    except Exception as exc:  # noqa: BLE001
        return _guard(exc)


@mcp.tool()
async def delegate(prompt: str, task_type: str = "", context: str = "") -> str:
    """Delegue une tache sans choisir l'agent : le routeur selectionne le mieux place.

    Args:
        prompt: l'instruction a executer.
        task_type: indice de routage optionnel (review, explain, test, document,
            implement, summarize). Ameliore nettement la precision du routage.
        context: code, diff ou logs a joindre.
    """
    try:
        run = await get_orchestra().delegate(
            prompt, task_type=task_type or None, context=context or None
        )
        return run.as_markdown()
    except Exception as exc:  # noqa: BLE001
        return _guard(exc)


@mcp.tool()
async def pipeline(steps: str, initial_input: str = "") -> str:
    """Enchaine plusieurs agents, la sortie de chacun alimentant le suivant.

    Args:
        steps: JSON, liste d'objets {"agent": str, "instruction": str,
            "carry_context": bool}. Exemple :
            [{"agent":"implementer","instruction":"Ecris la fonction"},
             {"agent":"reviewer","instruction":"Relis le code produit"}]
        initial_input: donnee de depart (code, spec, logs) commune aux etapes.
    """
    try:
        parsed = json.loads(steps)
    except json.JSONDecodeError as exc:
        return f"❌ `steps` n'est pas du JSON valide : {exc}"
    if not isinstance(parsed, list):
        return "❌ `steps` doit etre une liste JSON d'objets."

    try:
        result = await run_pipeline(
            get_orchestra(), parse_steps(parsed), initial_input
        )
        return result.as_markdown()
    except Exception as exc:  # noqa: BLE001
        return _guard(exc)


@mcp.tool()
async def refine(
    task: str, producer: str = "implementer", critic: str = "reviewer", max_rounds: int = 2
) -> str:
    """Boucle producteur/critique : un agent produit, un autre critique, le premier corrige.

    Utile pour faire converger un livrable localement avant de le remonter,
    sans consommer de tokens cote API.

    Args:
        task: la tache a realiser.
        producer: agent qui produit (defaut "implementer").
        critic: agent qui critique (defaut "reviewer").
        max_rounds: nombre max de cycles critique/correction (1 a 4).
    """
    try:
        rounds = max(1, min(int(max_rounds), 4))
        result = await run_refine_loop(
            get_orchestra(), producer, critic, task, max_rounds=rounds
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
