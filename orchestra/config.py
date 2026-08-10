"""Definition et chargement des agents (agents/*.yaml).

Un agent est une COUCHE DE CONFIGURATION, pas un modele. Il porte :
  - un role et un system prompt metier,
  - une classe de modele (fast / code / reason) resolue par le profil materiel,
  - des parametres d'inference,
  - des metadonnees de routage (tasks, keywords),
  - les outils qui lui sont accordes, donc ses privileges reels.

La cle `tools` decide de la nature de l'agent. Sans elle, l'agent produit du
texte en un aller-retour. Avec elle, il entre dans une boucle ou il lit et
modifie l'arborescence. Les privileges se lisent donc dans le YAML, agent par
agent : un `reviewer` qui n'a que des outils de lecture ne peut pas ecrire,
quoi que son modele decide.

Modifier un agent = editer son YAML. Aucun modele n'est recree, aucun poids
n'est duplique : tous les agents d'une meme classe partagent le meme modele
de base charge une seule fois par Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .agent_loop import DEFAULT_MAX_TURNS, MAX_TURNS_CAP
from .profiles import MODEL_CLASSES, Profile
from .tools import TOOLS, validate_tool_names

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
    #: outils accordes. Vide = agent de generation pure, un seul aller-retour.
    tools: list[str] = field(default_factory=list)
    #: budget de tours de la boucle outillee
    max_turns: int = DEFAULT_MAX_TURNS
    source_path: Path | None = None

    @property
    def is_agentic(self) -> bool:
        """L'agent peut-il agir, ou seulement produire du texte ?"""
        return bool(self.tools)

    @property
    def writes(self) -> bool:
        return any(TOOLS[name].writes for name in self.tools if name in TOOLS)

    def resolve_model(
        self, profile: Profile, model_overrides: dict[str, str] | None = None
    ) -> str:
        """Modele reel a utiliser, par ordre de priorite decroissante.

        1. `pinned_model` de l'agent - decision explicite, elle prime.
        2. Surcharge du backend - une passerelle distante publie ses propres
           noms de modeles, sans rapport avec le materiel local.
        3. Profil materiel detecte.
        """
        if self.pinned_model:
            return self.pinned_model
        if model_overrides and self.model_class in model_overrides:
            return model_overrides[self.model_class]
        return profile.model_for(self.model_class)

    def resolve_options(
        self,
        profile: Profile,
        num_ctx_cap: int | None = None,
        max_output_cap: int | None = None,
    ) -> dict[str, Any]:
        """Parametres d'inference effectifs pour cet agent.

        Contexte et sortie suivent la meme regle : le plafond vient du backend
        quand il en impose un, sinon du profil materiel, et un agent peut
        demander moins, jamais plus.

        `num_predict` est donc une ambition de role, pas une constante. Un
        implementer qui reclame de quoi ecrire un fichier entier obtient ce
        qu'il demande derriere une passerelle capable, et se voit ramene a la
        raison sur un petit modele local, ou une longue generation coute
        surtout du temps d'attente.
        """
        ctx_cap = num_ctx_cap or profile.num_ctx
        out_cap = max_output_cap or profile.max_output
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": min(self.num_predict, out_cap),
            "num_ctx": min(self.num_ctx or ctx_cap, ctx_cap),
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

        tools = [str(t).strip() for t in data.get("tools", []) if str(t).strip()]
        validate_tool_names(tools, where=f" pour l'agent '{data['name']}'{where}")

        if tools and fmt == "json":
            # Les deux contraignent la sortie du modele et s'excluent : un
            # agent qui doit rendre du JSON strict ne peut pas appeler d'outil.
            raise ValueError(
                f"Agent '{data['name']}'{where} : 'tools' et output_format "
                "'json' sont incompatibles"
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
            tools=tools,
            max_turns=max(
                1, min(int(data.get("max_turns", DEFAULT_MAX_TURNS)), MAX_TURNS_CAP)
            ),
            source_path=source,
        )


def agents_fingerprint(directory: Path | None = None) -> tuple:
    """Empreinte du dossier d'agents : noms et dates de modification.

    Permet de detecter qu'un YAML a change sans relire ni reparser les
    fichiers. Sans cela, editer un agent reste sans effet jusqu'au
    redemarrage du serveur MCP, ce qui est le genre de piege qu'on ne voit
    pas venir : le fichier sur disque dit une chose, le serveur en applique
    une autre.
    """
    directory = directory or AGENTS_DIR
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            (path.name, path.stat().st_mtime_ns)
            for path in [*directory.glob("*.yaml"), *directory.glob("*.yml")]
        )
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
