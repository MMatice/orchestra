"""Selection du profil materiel et resolution classe de modele -> modele reel."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .hardware import Hardware, detect

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_FILE = PROJECT_ROOT / "config" / "profiles.yaml"

MODEL_CLASSES = ("fast", "code", "reason")


class ProfileError(RuntimeError):
    pass


@dataclass
class Profile:
    id: str
    label: str
    min_usable_gb: float
    num_ctx: int
    parallel_agents: int
    models: dict[str, str]

    def model_for(self, model_class: str) -> str:
        try:
            return self.models[model_class]
        except KeyError as exc:
            raise ProfileError(
                f"Le profil '{self.id}' ne definit pas la classe '{model_class}'. "
                f"Classes disponibles : {sorted(self.models)}"
            ) from exc

    def required_models(self) -> list[str]:
        return sorted(set(self.models.values()))


def load_profiles(path: Path | None = None) -> dict[str, Profile]:
    path = path or PROFILES_FILE
    if not path.is_file():
        raise ProfileError(f"Fichier de profils introuvable : {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("profiles") or {}
    if not entries:
        raise ProfileError(f"Aucun profil defini dans {path}")

    profiles: dict[str, Profile] = {}
    for pid, data in entries.items():
        models = data.get("models") or {}
        missing = [c for c in MODEL_CLASSES if c not in models]
        if missing:
            raise ProfileError(f"Profil '{pid}' : classes manquantes {missing}")
        profiles[pid] = Profile(
            id=pid,
            label=str(data.get("label", pid)),
            min_usable_gb=float(data.get("min_usable_gb", 0)),
            num_ctx=int(data.get("num_ctx", 8192)),
            parallel_agents=int(data.get("parallel_agents", 1)),
            models={k: str(v) for k, v in models.items()},
        )
    return profiles


def select_profile(
    hardware: Hardware | None = None, path: Path | None = None
) -> tuple[Profile, Hardware, str]:
    """Retourne (profil, materiel, raison du choix).

    ORCHESTRA_PROFILE force un profil ; sinon on prend le plus gros profil
    dont le seuil memoire est atteint.
    """
    hardware = hardware or detect()
    profiles = load_profiles(path)

    forced = os.environ.get("ORCHESTRA_PROFILE", "").strip()
    if forced:
        if forced not in profiles:
            raise ProfileError(
                f"ORCHESTRA_PROFILE='{forced}' inconnu. "
                f"Valeurs possibles : {sorted(profiles)}"
            )
        return profiles[forced], hardware, "force via ORCHESTRA_PROFILE"

    budget = hardware.usable_gb
    eligible = [p for p in profiles.values() if budget >= p.min_usable_gb]
    if not eligible:
        chosen = min(profiles.values(), key=lambda p: p.min_usable_gb)
        return chosen, hardware, "aucun seuil atteint, repli sur le profil minimal"

    chosen = max(eligible, key=lambda p: p.min_usable_gb)
    return chosen, hardware, f"budget memoire {budget} Go >= {chosen.min_usable_gb} Go"
