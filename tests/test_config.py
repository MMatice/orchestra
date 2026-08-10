import pytest

from orchestra.config import AgentSpec, load_agents
from orchestra.profiles import Profile

PROFILE = Profile(
    id="test",
    label="test",
    min_usable_gb=0,
    num_ctx=8192,
    models={"fast": "small:1b", "code": "coder:7b", "reason": "general:8b"},
    max_output=2048,
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


def test_output_budget_is_capped_by_the_deployment():
    """num_predict est une ambition de role, pas une constante.

    Sans plafond, un agent regle pour une passerelle capable tronque ses
    reponses des qu'il tourne sur un petit modele local, et inversement un
    agent regle pour le local gaspille la capacite d'une passerelle.
    """
    spec = _spec(num_predict=12000)
    # Profil local : ramene a la raison.
    assert spec.resolve_options(PROFILE)["num_predict"] == 2048
    # Passerelle capable : l'agent obtient ce qu'il demande.
    assert spec.resolve_options(PROFILE, None, 32000)["num_predict"] == 12000
    # Un agent sobre n'est jamais gonfle au plafond disponible.
    assert _spec(num_predict=800).resolve_options(PROFILE, None, 32000)["num_predict"] == 800


def test_backend_output_cap_overrides_the_profile():
    spec = _spec(num_predict=9000)
    assert spec.resolve_options(PROFILE, None, 4096)["num_predict"] == 4096


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


# ------------------------------------------------------------------ outils


def test_unknown_tool_is_rejected_at_load():
    """Une faute de frappe dans un YAML doit se voir au demarrage.

    Decouvrir l'outil manquant au milieu d'une boucle facturee est le pire
    moment pour l'apprendre.
    """
    with pytest.raises(ValueError, match="Outil"):
        _spec(tools=["read_file", "delete_everything"])


def test_json_output_and_tools_are_incompatible():
    # Les deux contraignent la sortie du modele : les cumuler donne un agent
    # qui n'appelle jamais d'outil, silencieusement.
    with pytest.raises(ValueError, match="incompatibles"):
        _spec(tools=["read_file"], output_format="json")


def test_privileges_are_readable_from_the_spec():
    assert not _spec().is_agentic
    assert not _spec(tools=["read_file"]).writes
    assert _spec(tools=["read_file", "edit_file"]).writes


def test_max_turns_is_clamped():
    assert _spec(max_turns=999).max_turns <= 25
    assert _spec(max_turns=0).max_turns == 1


def test_shipped_reviewer_cannot_write():
    """Le critique doit rester un contre-pouvoir dans la boucle refine.

    S'il peut corriger lui-meme, il valide son propre travail au tour suivant.
    """
    agents = load_agents()
    assert agents["reviewer"].is_agentic
    assert not agents["reviewer"].writes
    assert not agents["explainer"].writes
    assert not agents["summarizer"].writes


def test_shipped_implementer_can_write():
    agents = load_agents()
    assert agents["implementer"].writes
    assert "edit_file" in agents["implementer"].tools


def test_editing_a_yaml_takes_effect_without_a_restart(tmp_path):
    """Un agent edite doit s'appliquer sans reconnecter le serveur MCP.

    Sans cela le fichier sur disque dit une chose et le serveur en applique
    une autre, ce qui se diagnostique tres mal : on croit que le reglage n'a
    pas d'effet, pas qu'il n'a pas ete lu.
    """
    import time

    from orchestra.config import agents_fingerprint

    agent = tmp_path / "demo.yaml"
    agent.write_text(
        "name: demo\nmodel_class: code\nnum_predict: 1000\nsystem: fais des choses\n",
        encoding="utf-8",
    )
    before = agents_fingerprint(tmp_path)
    assert load_agents(tmp_path)["demo"].num_predict == 1000

    time.sleep(0.01)
    agent.write_text(
        "name: demo\nmodel_class: code\nnum_predict: 9000\nsystem: fais des choses\n",
        encoding="utf-8",
    )
    assert agents_fingerprint(tmp_path) != before
    assert load_agents(tmp_path)["demo"].num_predict == 9000


def test_triage_stays_a_pure_json_classifier():
    # Il arbitre le routage : lui donner des outils le ferait sortir de son role
    # et casserait la contrainte JSON dont depend le parsing.
    triage = load_agents()["triage"]
    assert triage.tools == []
    assert triage.output_format == "json"
