"""Definition et chargement des agents (agents/*.yaml).

Un agent est une COUCHE DE CONFIGURATION, pas un modele. Il porte :
  - un role et un system prompt metier,
  - une classe de modele (fast / code / reason) resolue par le profil materiel,
  - des parametres d'inference,
  - des metadonnees de routage (tasks, keywords).

Modifier un agent = editer son YAML. Aucun modele n'est recree, aucun poids
n'est duplique : tous les agents d'une meme classe partagent le meme modele
de base charge une seule fois par Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .profiles import MODEL_CLASSES, Profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_ROOT / "agents"


@dataclass
class AgentSpec:
    name: str
    label: str
    description: str
    model_class: str
    system: str
    tasks: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    temperature: float = 0.2
    top_p: float = 0.9
    num_predict: int = 2048
    num_ctx: int | None = None
    pinned_model: str | None = None
    output_format: str | None = None
    source_path: Path | None = None

    def resolve_model(self, profile: Profile) -> str:
        """Modele Ollama reel a utiliser sur ce materiel."""
        return self.pinned_model or profile.model_for(self.model_class)

    def resolve_options(self, profile: Profile) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
            # Le contexte du profil est un plafond materiel : un agent peut
            # demander moins (plus rapide) mais jamais plus.
            "num_ctx": min(self.num_ctx or profile.num_ctx, profile.num_ctx),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path | None = None) -> "AgentSpec":
        where = f" ({source.name})" if source else ""

        for key in ("name", "system"):
            if not data.get(key):
                raise ValueError(f"Agent invalide{where} : champ '{key}' manquant")

        model_class = str(data.get("model_class", "reason")).strip().lower()
        if model_class not in MODEL_CLASSES:
            raise ValueError(
                f"Agent '{data['name']}'{where} : model_class '{model_class}' "
                f"inconnue, attendu {list(MODEL_CLASSES)}"
            )

        fmt = data.get("output_format")
        if fmt not in (None, "json"):
            raise ValueError(
                f"Agent '{data['name']}'{where} : output_format doit valoir 'json' "
                "ou etre absent"
            )

        return cls(
            name=str(data["name"]).strip(),
            label=str(data.get("label") or data["name"]).strip(),
            description=str(data.get("description", "")).strip(),
            model_class=model_class,
            system=str(data["system"]).strip(),
            tasks=[str(t).lower().strip() for t in data.get("tasks", [])],
            keywords=[str(k).lower().strip() for k in data.get("keywords", [])],
            temperature=float(data.get("temperature", 0.2)),
            top_p=float(data.get("top_p", 0.9)),
            num_predict=int(data.get("num_predict", 2048)),
            num_ctx=int(data["num_ctx"]) if data.get("num_ctx") else None,
            pinned_model=(
                str(data["pinned_model"]).strip() if data.get("pinned_model") else None
            ),
            output_format=fmt,
            source_path=source,
        )


def load_agents(directory: Path | None = None) -> dict[str, AgentSpec]:
    directory = directory or AGENTS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Repertoire d'agents introuvable : {directory}")

    agents: dict[str, AgentSpec] = {}
    for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = AgentSpec.from_dict(raw, source=path)
        if spec.name in agents:
            previous = agents[spec.name].source_path
            raise ValueError(
                f"Agent '{spec.name}' defini deux fois ({previous} et {path})"
            )
        agents[spec.name] = spec

    if not agents:
        raise ValueError(f"Aucun agent trouve dans {directory}")
    return agents
