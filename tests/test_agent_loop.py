"""Tests de la boucle d'execution, sur backend scripte.

Ce qui est verifie ici n'est pas la qualite du modele mais la mecanique qui
l'entoure : le resultat d'un outil lui revient-il, les garde-fous tiennent-ils,
et le cout total est-il rapporte honnetement.
"""

import pytest

from orchestra.agent_loop import run_agent_loop
from orchestra.backends.base import Backend, ChatResult, ToolCall
from orchestra.tools import TOOLS, Toolset
from orchestra.workspace import Workspace


class ScriptedBackend(Backend):
    """Rejoue une suite de reponses fixee, et enregistre l'historique recu."""

    name = "scripte"
    base_url = "memoire://"

    def __init__(self, script):
        self.script = list(script)
        self.seen_histories = []

    async def ping(self):  # pragma: no cover - non sollicite
        return "ok"

    async def list_models(self):  # pragma: no cover - non sollicite
        return []

    async def chat(self, model, messages, *, options=None, fmt=None, tools=None):
        self.seen_histories.append(list(messages))
        content, calls = self.script.pop(0) if self.script else ("fini", [])
        return ChatResult(
            content=content,
            model=model,
            backend=self.name,
            total_duration_s=1.0,
            eval_duration_s=1.0,
            eval_count=10,
            prompt_eval_count=5,
            tool_calls=calls,
            raw_message={"role": "assistant", "content": content},
        )


@pytest.fixture
def toolset(tmp_path):
    (tmp_path / "note.txt").write_text("bonjour", encoding="utf-8")
    workspace = Workspace(tmp_path, writable=True)
    return Toolset(
        [TOOLS["read_file"], TOOLS["write_file"], TOOLS["edit_file"]], workspace
    )


def _call(name, **arguments):
    return ToolCall(id=f"c_{name}", name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_single_turn_when_no_tool_is_requested(toolset):
    backend = ScriptedBackend([("reponse directe", [])])
    result = await run_agent_loop(backend, "m", [], toolset)

    assert result.content == "reponse directe"
    assert result.turns == 1
    assert result.invocations == []


@pytest.mark.asyncio
async def test_tool_output_is_returned_to_the_model(toolset):
    backend = ScriptedBackend(
        [("", [_call("read_file", path="note.txt")]), ("j'ai lu le fichier", [])]
    )
    result = await run_agent_loop(backend, "m", [], toolset)

    # Le second appel doit contenir le resultat de l'outil : c'est tout
    # l'interet de la boucle.
    second = backend.seen_histories[1]
    assert any(m.get("role") == "tool" and "bonjour" in m["content"] for m in second)
    assert result.content == "j'ai lu le fichier"
    assert result.turns == 2


@pytest.mark.asyncio
async def test_writes_actually_reach_the_disk(toolset):
    backend = ScriptedBackend(
        [
            ("", [_call("write_file", path="nouveau.py", content="X = 1\n")]),
            ("fichier cree", []),
        ]
    )
    result = await run_agent_loop(backend, "m", [], toolset)

    assert (toolset.workspace.root / "nouveau.py").read_text(encoding="utf-8") == "X = 1\n"
    assert [i.name for i in result.writes] == ["write_file"]


@pytest.mark.asyncio
async def test_a_failing_tool_does_not_abort_the_task(toolset):
    """L'erreur repart vers le modele : c'est ce qui lui permet de corriger."""
    backend = ScriptedBackend(
        [
            ("", [_call("read_file", path="absent.txt")]),
            ("", [_call("read_file", path="note.txt")]),
            ("finalement trouve", []),
        ]
    )
    result = await run_agent_loop(backend, "m", [], toolset)

    assert result.content == "finalement trouve"
    assert [i.ok for i in result.invocations] == [False, True]


@pytest.mark.asyncio
async def test_turn_budget_is_enforced(toolset):
    backend = ScriptedBackend(
        [("", [_call("read_file", path="note.txt")])] * 10
    )
    result = await run_agent_loop(backend, "m", [], toolset, max_turns=3)

    assert result.turns == 3
    assert "3 tours" in result.stopped_early


@pytest.mark.asyncio
async def test_identical_repeated_call_breaks_the_loop(toolset):
    """Un modele bloque sur le meme appel coute de l'argent a chaque tour."""
    backend = ScriptedBackend(
        [("", [_call("read_file", path="note.txt")])] * 12
    )
    result = await run_agent_loop(backend, "m", [], toolset, max_turns=20)

    assert "repetait" in result.stopped_early
    assert result.turns < 20


@pytest.mark.asyncio
async def test_cost_is_summed_over_every_turn(toolset):
    backend = ScriptedBackend(
        [("", [_call("read_file", path="note.txt")]), ("fini", [])]
    )
    result = await run_agent_loop(backend, "m", [], toolset)

    # 2 appels a 10 tokens de sortie : le rapport ne doit pas n'en montrer qu'un.
    assert "20 tok out" in result.stats
    assert "2 appel(s)" in result.stats


@pytest.mark.asyncio
async def test_trace_lists_what_was_done(toolset):
    backend = ScriptedBackend(
        [("", [_call("write_file", path="x.txt", content="a")]), ("fait", [])]
    )
    result = await run_agent_loop(backend, "m", [], toolset)

    trace = result.trace()
    assert "write_file" in trace
    assert "✅" in trace


@pytest.mark.asyncio
async def test_a_silent_final_turn_still_produces_content(toolset):
    """Certains modeles terminent sur une action, sans phrase de conclusion."""
    backend = ScriptedBackend(
        [("", [_call("write_file", path="x.txt", content="a")]), ("", [])]
    )
    result = await run_agent_loop(backend, "m", [], toolset)

    assert result.content.strip()
