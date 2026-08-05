"""Routage d'une tache vers l'agent le plus adapte.

Deux etages :
  1. un score deterministe (type de tache declare + mots-cles), gratuit et
     instantane ;
  2. en cas d'egalite ou de score trop faible, un arbitrage par le modele
     'fast' du profil, qui coute quelques centaines de millisecondes.

L'etage 1 tranche dans la grande majorite des cas ; l'etage 2 evite de router
au hasard quand la demande est formulee librement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import AgentSpec

# En dessous de ce score, le routage deterministe n'est pas fiable.
AMBIGUITY_THRESHOLD = 6.0
_WORD_RE = re.compile(r"[a-zA-ZÀ-ſ0-9_+#.-]+")


@dataclass
class RoutingDecision:
    agent: str
    score: float
    method: str
    reason: str
    runners_up: list[tuple[str, float]]


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


def score_agent(spec: AgentSpec, task_type: str | None, text: str) -> float:
    score = 0.0
    task = (task_type or "").lower().strip()

    if task:
        if task in spec.tasks:
            score += 10.0
        else:
            # Correspondance partielle : "code-review" doit matcher "review".
            # Le score est pondere par la part du libelle effectivement couverte,
            # sinon "code-review" vaudrait autant pour l'agent declarant "code"
            # que pour celui declarant "review", et l'egalite se resoudrait au
            # hasard de l'ordre alphabetique.
            best_overlap = 0.0
            for declared in spec.tasks:
                if task in declared or declared in task:
                    ratio = len(declared) / max(len(task), len(declared))
                    best_overlap = max(best_overlap, ratio)
            score += 5.0 * best_overlap

    words = _tokens(text)
    hits = sum(1 for kw in spec.keywords if kw in words)
    score += min(hits, 4) * 1.5

    # Un mot-cle multi-mots ("code review") ne survit pas a la tokenisation :
    # on le cherche tel quel dans le texte.
    lowered = (text or "").lower()
    phrase_hits = sum(1 for kw in spec.keywords if " " in kw and kw in lowered)
    score += min(phrase_hits, 2) * 2.0

    return score


def rank_agents(
    agents: dict[str, AgentSpec], task_type: str | None, text: str
) -> list[tuple[AgentSpec, float]]:
    scored = [(spec, score_agent(spec, task_type, text)) for spec in agents.values()]
    return sorted(scored, key=lambda pair: (-pair[1], pair[0].name))


def build_triage_prompt(agents: dict[str, AgentSpec], task_type: str | None, text: str) -> str:
    catalog = "\n".join(
        f"- {spec.name}: {spec.description or spec.label} "
        f"(taches: {', '.join(spec.tasks) or 'n/a'})"
        for spec in agents.values()
    )
    task_line = f"Declared task type: {task_type}\n" if task_type else ""
    return (
        "Pick the single best agent for the request below.\n\n"
        f"Available agents:\n{catalog}\n\n"
        f"{task_line}Request:\n{text[:2000]}\n\n"
        'Answer with JSON only: {"agent": "<name>", "reason": "<10 words max>"}'
    )


def parse_triage_response(raw: str, agents: dict[str, AgentSpec]) -> tuple[str, str] | None:
    """Extrait (agent, raison) d'une reponse de triage, ou None si inexploitable."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    name = str(data.get("agent", "")).strip()
    if name not in agents:
        return None
    return name, str(data.get("reason", "")).strip() or "choisi par le triage"
