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
from .workspace import Workspace

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
            if run.trace:
                parts += ["", run.trace]
            parts += ["", f"`{run.stats}`", ""]

        touched = sorted({path for run in self.runs for path in run.changed})
        if touched:
            parts += [
                "**Fichiers modifies sur l'ensemble du pipeline** : "
                + ", ".join(f"`{p}`" for p in touched),
                "",
            ]
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
    orchestra: Orchestra,
    steps: list[PipelineStep],
    initial_input: str,
    *,
    workspace: Workspace | None = None,
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
            workspace=workspace,
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
    workspace: Workspace | None = None,
) -> PipelineResult:
    """Producteur -> critique -> correction, jusqu'a `max_rounds` tours.

    Le critique doit terminer sa reponse par VALIDATED quand plus rien n'est
    bloquant ; c'est ce qui permet de sortir de la boucle sans bruler un tour
    de generation supplementaire.

    Avec un espace de travail, la boucle change de nature : le producteur
    ecrit sur le disque et le critique relit les fichiers reels au lieu d'un
    extrait recopie. Les deux agents voient donc le meme etat, et le travail
    n'a plus besoin de transiter par le contexte a chaque tour.
    """
    result = PipelineResult()
    on_disk = workspace is not None

    draft_run = await orchestra.run(producer, task, workspace=workspace)
    result.runs.append(draft_run)
    draft = draft_run.content

    for _ in range(max_rounds):
        if on_disk:
            review_prompt = (
                "Relis le travail decrit ci-dessous en ouvrant toi-meme les "
                "fichiers concernes dans l'espace de travail. Liste d'abord ce "
                "qui est bloquant. Si rien ne l'est, termine ta reponse par "
                f"{VALIDATION_MARKER}."
            )
            review_context = (
                f"Tache d'origine :\n{task}\n\n"
                f"Compte rendu du producteur :\n{draft}"
            )
            if draft_run.changed:
                review_context += "\n\nFichiers touches : " + ", ".join(
                    draft_run.changed
                )
        else:
            review_prompt = (
                "Review the work below. List blocking issues first. "
                f"If nothing is blocking, end your answer with {VALIDATION_MARKER}."
            )
            review_context = (
                f"Tache d'origine :\n{task}\n\nTravail a reviewer :\n{draft}"
            )

        review_run = await orchestra.run(
            critic, review_prompt, context=review_context, workspace=workspace
        )
        result.runs.append(review_run)

        if VALIDATION_MARKER in review_run.content.upper():
            result.stopped_early = True
            result.stop_reason = "le critique a valide la production"
            break

        if on_disk:
            fix_prompt = (
                "Applique les corrections demandees directement dans les "
                "fichiers, avec tes outils. Conclus par ce que tu as change."
            )
            fix_context = (
                f"Tache d'origine :\n{task}\n\n"
                f"Retour du reviewer :\n{review_run.content}"
            )
        else:
            fix_prompt = (
                "Apply the review feedback. Return the corrected work in full, "
                "not a diff or a summary of changes."
            )
            fix_context = (
                f"Tache d'origine :\n{task}\n\n"
                f"Version actuelle :\n{draft}\n\n"
                f"Retour du reviewer :\n{review_run.content}"
            )

        fix_run = await orchestra.run(
            producer, fix_prompt, context=fix_context, workspace=workspace
        )
        result.runs.append(fix_run)
        draft_run = fix_run
        draft = fix_run.content

    result.final_output = draft
    return result
