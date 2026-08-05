import pytest

from orchestra.config import AgentSpec, load_agents
from orchestra.profiles import Profile

PROFILE = Profile(
    id="test",
    label="test",
    min_usable_gb=0,
    num_ctx=8192,
    models={"fast": "small:1b", "code": "coder:7b", "reason": "general:8b"},
)


def _spec(**overrides):
    data = {"name": "demo", "system": "do things", "model_class": "code"}
    data.update(overrides)
    return AgentSpec.from_dict(data)


def test_shipped_agents_all_load():
    agents = load_agents()
    assert {"triage", "reviewer", "explainer", "implementer"} <= set(agents)


def test_shipped_agents_resolve_on_any_profile():
    for spec in load_agents().values():
        assert spec.resolve_model(PROFILE) in PROFILE.models.values()


def test_model_class_resolves_through_profile():
    assert _spec(model_class="reason").resolve_model(PROFILE) == "general:8b"
    assert _spec(model_class="fast").resolve_model(PROFILE) == "small:1b"


def test_pinned_model_wins_over_profile():
    spec = _spec(pinned_model="deepseek-coder-v2:16b")
    assert spec.resolve_model(PROFILE) == "deepseek-coder-v2:16b"


def test_agent_cannot_exceed_profile_context():
    # Un agent gourmand ne doit pas faire deborder la VRAM de la machine.
    assert _spec(num_ctx=99999).resolve_options(PROFILE)["num_ctx"] == 8192
    # Mais il peut demander moins pour aller plus vite.
    assert _spec(num_ctx=2048).resolve_options(PROFILE)["num_ctx"] == 2048


def test_unknown_model_class_is_rejected():
    with pytest.raises(ValueError, match="model_class"):
        _spec(model_class="gigantesque")


def test_missing_system_prompt_is_rejected():
    with pytest.raises(ValueError, match="system"):
        AgentSpec.from_dict({"name": "demo", "model_class": "code"})


def test_invalid_output_format_is_rejected():
    with pytest.raises(ValueError, match="output_format"):
        _spec(output_format="xml")
