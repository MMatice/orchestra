"""Boucle d'execution d'un agent outille.

C'est la piece qui separe un agent d'un appel de completion : tant que le
modele demande des outils, on les execute et on lui rend la main. Il observe
donc le resultat de ses propres actions au lieu de deviner l'etat du disque.

Trois garde-fous, parce qu'un modele en boucle coute de l'argent a chaque tour :
  - un budget de tours,
  - une detection de repetition d'appel identique,
  - une remontee des echecs d'outils vers le modele plutot qu'une exception,
    afin qu'il corrige lui-meme au lieu de faire echouer toute la tache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .backends import Backend, ChatResult
from .tools import Toolset, format_arguments

DEFAULT_MAX_TURNS = 8
MAX_TURNS_CAP = 25

# Au-dela, on considere que le modele tourne en rond sur le meme appel.
REPEAT_LIMIT = 3


@dataclass
class ToolInvocation:
    name: str
    arguments: dict[str, Any]
    output: str
    ok: bool

    def summary(self) -> str:
        mark = "✅" if self.ok else "❌"
        return f"{mark} `{self.name}({format_arguments(self.arguments)})`"


@dataclass
class LoopResult:
    content: str
    turns: int
    invocations: list[ToolInvocation] = field(default_factory=list)
    stats: str = ""
    stopped_early: str = ""

    @property
    def writes(self) -> list[ToolInvocation]:
        return [i for i in self.invocations if i.ok and i.name in ("write_file", "edit_file")]

    def trace(self) -> str:
        """Journal des actions, a rendre a l'orchestrateur.

        Sans cela, l'appelant ne peut pas savoir ce qui a reellement ete
        touche sur le disque : c'est la partie verifiable du resultat.
        """
        if not self.invocations:
            return ""
        lines = [f"**Actions ({len(self.invocations)} appels, {self.turns} tours)**", ""]
        lines += [f"- {inv.summary()}" for inv in self.invocations]
        if self.stopped_early:
            lines += ["", f"⚠️ {self.stopped_early}"]
        return "\n".join(lines)


def _fingerprint(name: str, arguments: dict[str, Any]) -> str:
    return name + json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def _merge_stats(results: list[ChatResult]) -> str:
    """Agrege le cout de tous les tours en une ligne unique.

    Le detail par tour n'interesse personne ; le total, si : c'est ce qui dit
    si l'agent a mis trois appels ou quinze pour arriver au meme resultat.
    """
    if not results:
        return ""
    last = results[-1]
    total = ChatResult(
        content="",
        model=last.model,
        backend=last.backend,
        total_duration_s=sum(r.total_duration_s for r in results),
        load_duration_s=max((r.load_duration_s for r in results), default=0.0),
        eval_duration_s=sum(r.eval_duration_s for r in results),
        eval_count=sum(r.eval_count for r in results),
        prompt_eval_count=sum(r.prompt_eval_count for r in results),
    )
    line = total.stats_line()
    return f"{line} | {len(results)} appel(s)" if len(results) > 1 else line


async def run_agent_loop(
    backend: Backend,
    model: str,
    messages: list[dict[str, Any]],
    toolset: Toolset,
    *,
    options: dict[str, Any] | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> LoopResult:
    """Fait tourner l'agent jusqu'a ce qu'il reponde sans demander d'outil."""
    budget = max(1, min(int(max_turns), MAX_TURNS_CAP))
    history = list(messages)
    schemas = toolset.schemas()

    results: list[ChatResult] = []
    invocations: list[ToolInvocation] = []
    seen: dict[str, int] = {}
    stopped_early = ""
    content = ""
    turn = 0

    while turn < budget:
        turn += 1
        result = await backend.chat(
            model, history, options=options, tools=schemas
        )
        results.append(result)
        content = result.content

        if not result.tool_calls:
            break

        # On rend le message du modele tel qu'il l'a emis : chaque fournisseur
        # sait relire sa propre representation, la reconstruire perdrait des
        # champs (identifiants d'appel, raisonnement).
        history.append(result.raw_message or {"role": "assistant", "content": content})

        for call in result.tool_calls:
            key = _fingerprint(call.name, call.arguments)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > REPEAT_LIMIT:
                output, ok = (
                    f"Erreur : appel identique a {call.name} repete "
                    f"{seen[key]} fois. Change d'approche ou termine ta reponse.",
                    False,
                )
            else:
                output, ok = toolset.execute(call.name, call.arguments)

            invocations.append(
                ToolInvocation(call.name, call.arguments, output, ok)
            )
            history.append(backend.tool_result_message(call, output))

        if seen and max(seen.values()) > REPEAT_LIMIT:
            stopped_early = "Boucle interrompue : le modele repetait le meme appel."
            break
    else:
        stopped_early = (
            f"Budget de {budget} tours epuise avant que l'agent ne conclue. "
            "Augmente max_turns ou decoupe la tache."
        )

    if not content and invocations:
        # Certains modeles terminent sur un appel d'outil sans phrase finale.
        content = "_(l'agent a termine sur une action, sans texte de conclusion)_"

    return LoopResult(
        content=content,
        turns=turn,
        invocations=invocations,
        stats=_merge_stats(results),
        stopped_early=stopped_early,
    )
