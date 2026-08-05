import pytest

from orchestra.hardware import Gpu, Hardware
from orchestra.profiles import ProfileError, load_profiles, select_profile


def test_all_profiles_define_every_model_class():
    profiles = load_profiles()
    assert profiles, "aucun profil charge"
    for profile in profiles.values():
        assert set(profile.models) >= {"fast", "code", "reason"}


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
