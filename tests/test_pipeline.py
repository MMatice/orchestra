import pytest

from orchestra.pipeline import (
    PipelineStep,
    parse_steps,
    run_pipeline,
    run_refine_loop,
)
from orchestra.registry import AgentRun


class FakeOrchestra:
    """Orchestra factice : enregistre les appels, renvoie des reponses scriptees."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    async def run(self, agent, prompt, *, context=None, routing=None):
        self.calls.append({"agent": agent, "prompt": prompt, "context": context})
        content = self.responses.pop(0) if self.responses else f"sortie de {agent}"
        return AgentRun(agent=agent, model="fake", content=content, stats="fake")


@pytest.mark.asyncio
async def test_output_of_a_step_feeds_the_next():
    fake = FakeOrchestra(["code genere", "revue du code"])
    steps = [
        PipelineStep("implementer", "ecris la fonction"),
        PipelineStep("reviewer", "relis"),
    ]

    result = await run_pipeline(fake, steps, "spec initiale")

    assert [c["agent"] for c in fake.calls] == ["implementer", "reviewer"]
    assert "code genere" in fake.calls[1]["context"]
    assert result.final_output == "revue du code"


@pytest.mark.asyncio
async def test_carry_context_false_isolates_the_step():
    fake = FakeOrchestra(["premier", "second"])
    steps = [
        PipelineStep("implementer", "a"),
        PipelineStep("explainer", "b", carry_context=False),
    ]

    await run_pipeline(fake, steps, "entree")

    assert "premier" not in (fake.calls[1]["context"] or "")


@pytest.mark.asyncio
async def test_refine_loop_stops_when_critic_validates():
    fake = FakeOrchestra(["brouillon", "rien a signaler. VALIDATED"])

    result = await run_refine_loop(fake, "implementer", "reviewer", "tache", max_rounds=3)

    # Producteur + critique, et surtout pas de tour de correction inutile.
    assert [c["agent"] for c in fake.calls] == ["implementer", "reviewer"]
    assert result.stopped_early
    assert result.final_output == "brouillon"


@pytest.mark.asyncio
async def test_refine_loop_corrects_then_revalidates():
    fake = FakeOrchestra(
        ["brouillon", "BLOCKER: nul", "version corrigee", "ok VALIDATED"]
    )

    result = await run_refine_loop(fake, "implementer", "reviewer", "tache", max_rounds=3)

    assert [c["agent"] for c in fake.calls] == [
        "implementer", "reviewer", "implementer", "reviewer",
    ]
    assert result.final_output == "version corrigee"


@pytest.mark.asyncio
async def test_refine_loop_respects_max_rounds():
    fake = FakeOrchestra(["v0"] + ["BLOCKER: encore", "v-next"] * 5)

    await run_refine_loop(fake, "implementer", "reviewer", "tache", max_rounds=2)

    # 1 production initiale + 2 * (critique + correction)
    assert len(fake.calls) == 5


def test_parse_steps_rejects_malformed_input():
    with pytest.raises(ValueError, match="agent"):
        parse_steps([{"instruction": "sans agent"}])
    with pytest.raises(ValueError, match="vide"):
        parse_steps([])
