import pytest

from orchestra.hardware import Gpu, Hardware
from orchestra.profiles import ProfileError, load_profiles, select_profile


# Fenetres de contexte publiees sur ollama.com, en tokens. Elles varient
# fortement d'un modele a l'autre, y compris au sein d'une meme famille.
MODEL_CONTEXT_WINDOWS = {
    "qwen3:0.6b": 40_960,
    "qwen3:1.7b": 40_960,
    "qwen3:4b": 262_144,
    "qwen3:8b": 40_960,
    "qwen3:14b": 40_960,
    "qwen3:30b": 262_144,
    "qwen3:32b": 40_960,
    "qwen3-coder:30b": 262_144,
    "qwen2.5-coder:1.5b": 32_768,
    "qwen2.5-coder:3b": 32_768,
    "qwen2.5-coder:7b": 32_768,
    "qwen2.5-coder:14b": 32_768,
    "gpt-oss:20b": 131_072,
    "gpt-oss:120b": 131_072,
}


def test_all_profiles_define_every_model_class():
    profiles = load_profiles()
    assert profiles, "aucun profil charge"
    for profile in profiles.values():
        assert set(profile.models) >= {"fast", "code", "reason"}


def test_profile_context_fits_every_model_it_selects():
    """num_ctx s'applique aux trois classes d'un profil.

    Il ne doit donc jamais depasser la plus petite fenetre des trois modeles,
    sinon la classe la plus limitee se voit demander un contexte qu'elle ne
    peut pas honorer.
    """
    for profile in load_profiles().values():
        for model_class, model in profile.models.items():
            window = MODEL_CONTEXT_WINDOWS.get(model)
            assert window is not None, (
                f"profil '{profile.id}' : fenetre de contexte inconnue pour "
                f"'{model}', ajoute-la a MODEL_CONTEXT_WINDOWS"
            )
            assert profile.num_ctx <= window, (
                f"profil '{profile.id}' : num_ctx={profile.num_ctx} depasse la "
                f"fenetre de {model} ({window}) sur la classe {model_class}"
            )


def test_selection_scales_with_available_memory(monkeypatch):
    monkeypatch.delenv("ORCHESTRA_PROFILE", raising=False)

    laptop = Hardware("Windows", "AMD64", 16.0, [Gpu("RTX 4060", 8.0, "cuda")])
    server = Hardware("Linux", "x86_64", 256.0, [Gpu("A100", 80.0, "cuda")])
    cpu_only = Hardware("Linux", "x86_64", 8.0, [])

    laptop_profile, _, _ = select_profile(laptop)
    server_profile, _, _ = select_profile(server)
    cpu_profile, _, _ = select_profile(cpu_only)

    # Le meme dossier agents/ doit se resoudre vers des modeles differents.
    assert laptop_profile.min_usable_gb < server_profile.min_usable_gb
    assert laptop_profile.models["code"] != server_profile.models["code"]
    assert cpu_profile.min_usable_gb <= laptop_profile.min_usable_gb


def test_multi_gpu_memory_is_summed(monkeypatch):
    monkeypatch.delenv("ORCHESTRA_PROFILE", raising=False)
    dual = Hardware(
        "Linux", "x86_64", 128.0,
        [Gpu("RTX 4090", 24.0, "cuda"), Gpu("RTX 4090", 24.0, "cuda")],
    )
    assert dual.total_vram_gb == 48.0
    profile, _, _ = select_profile(dual)
    assert profile.id == "xl"


def test_datacenter_gpu_reaches_the_top_profile(monkeypatch):
    monkeypatch.delenv("ORCHESTRA_PROFILE", raising=False)
    a100 = Hardware("Linux", "x86_64", 512.0, [Gpu("A100 80GB", 80.0, "cuda")])
    profile, _, _ = select_profile(a100)
    assert profile.id == "xxl"


def test_env_override(monkeypatch):
    monkeypatch.setenv("ORCHESTRA_PROFILE", "cpu")
    profile, _, reason = select_profile(
        Hardware("Linux", "x86_64", 256.0, [Gpu("A100", 80.0, "cuda")])
    )
    assert profile.id == "cpu"
    assert "ORCHESTRA_PROFILE" in reason


def test_unknown_env_override_is_rejected(monkeypatch):
    monkeypatch.setenv("ORCHESTRA_PROFILE", "nexistepas")
    with pytest.raises(ProfileError):
        select_profile(Hardware("Linux", "x86_64", 32.0, []))


def test_cpu_only_budget_uses_ram_not_vram():
    cpu_only = Hardware("Linux", "x86_64", 32.0, [])
    assert cpu_only.backend == "cpu"
    assert cpu_only.usable_gb == pytest.approx(19.2)
