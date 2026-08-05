"""Chainage multi-agents.

C'est la brique qui fait qu'un agent en pilote un autre : la sortie de l'etape
N devient le contexte de l'etape N+1. Deux formes :

  - `run_pipeline` : sequence libre decrite par l'appelant (Claude decide
    quels agents enchainer et dans quel ordre) ;
  - `run_refine_loop` : boucle producteur/critique - un agent produit, un
    autre critique, le premier corrige - jusqu'a validation ou epuisement
    des tours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .registry import AgentRun, Orchestra

VALIDATION_MARKER = "VALIDATED"


@dataclass
class PipelineStep:
    agent: str
    instruction: str
    # Si False, l'etape ne recoit pas la sortie precedente (utile pour
    # relancer une analyse sur l'entree d'origine).
    carry_context: bool = True


@dataclass
class PipelineResult:
    runs: list[AgentRun] = field(default_factory=list)
    final_output: str = ""
    stopped_early: bool = False
    stop_reason: str = ""

    def as_markdown(self) -> str:
        parts = [f"## Pipeline - {len(self.runs)} etape(s)", ""]
        for index, run in enumerate(self.runs, start=1):
            parts.append(f"**Etape {index} - `{run.agent}` ({run.model})**")
            parts.append("")
            parts.append(run.content)
            parts.append("")
            parts.append(f"`{run.stats}`")
            parts.append("")
        if self.stopped_early:
            parts.append(f"_Arret anticipe : {self.stop_reason}_")
        return "\n".join(parts)


def parse_steps(raw_steps: list[dict]) -> list[PipelineStep]:
    steps: list[PipelineStep] = []
    for index, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict) or not item.get("agent"):
            raise ValueError(f"Etape {index} invalide : champ 'agent' requis")
        steps.append(
            PipelineStep(
                agent=str(item["agent"]).strip(),
                instruction=str(item.get("instruction", "")).strip(),
                carry_context=bool(item.get("carry_context", True)),
            )
        )
    if not steps:
        raise ValueError("Le pipeline est vide")
    return steps


async def run_pipeline(
    orchestra: Orchestra, steps: list[PipelineStep], initial_input: str
) -> PipelineResult:
    result = PipelineResult()
    previous_output: str | None = None

    for step in steps:
        # Une etape sans instruction propre reprend l'entree d'origine.
        prompt = step.instruction or initial_input
        context_parts = []
        if step.carry_context and previous_output:
            context_parts.append(
                f"Sortie de l'agent precedent :\n{previous_output}"
            )
        if step.instruction and initial_input:
            context_parts.append(f"Entree initiale :\n{initial_input}")

        run = await orchestra.run(
            step.agent,
            prompt,
            context="\n\n".join(context_parts) or None,
        )
        result.runs.append(run)
        previous_output = run.content

    result.final_output = previous_output or ""
    return result


async def run_refine_loop(
    orchestra: Orchestra,
    producer: str,
    critic: str,
    task: str,
    *,
    max_rounds: int = 2,
) -> PipelineResult:
    """Producteur -> critique -> correction, jusqu'a `max_rounds` tours.

    Le critique doit terminer sa reponse par VALIDATED quand plus rien n'est
    bloquant ; c'est ce qui permet de sortir de la boucle sans bruler un tour
    de generation supplementaire.
    """
    result = PipelineResult()

    draft_run = await orchestra.run(producer, task)
    result.runs.append(draft_run)
    draft = draft_run.content

    for _ in range(max_rounds):
        review_run = await orchestra.run(
            critic,
            "Review the work below. List blocking issues first. "
            f"If nothing is blocking, end your answer with {VALIDATION_MARKER}.",
            context=f"Tache d'origine :\n{task}\n\nTravail a reviewer :\n{draft}",
        )
        result.runs.append(review_run)

        if VALIDATION_MARKER in review_run.content.upper():
            result.stopped_early = True
            result.stop_reason = "le critique a valide la production"
            break

        fix_run = await orchestra.run(
            producer,
            "Apply the review feedback. Return the corrected work in full, "
            "not a diff or a summary of changes.",
            context=(
                f"Tache d'origine :\n{task}\n\n"
                f"Version actuelle :\n{draft}\n\n"
                f"Retour du reviewer :\n{review_run.content}"
            ),
        )
        result.runs.append(fix_run)
        draft = fix_run.content

    result.final_output = draft
    return result
