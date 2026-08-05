"""Contrat commun a tous les backends d'inference."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

# Une generation peut etre longue : gros prompt, modele froid, file d'attente
# sur une passerelle partagee.
READ_TIMEOUT_S = 900.0
CONNECT_TIMEOUT_S = 10.0


class BackendUnavailable(RuntimeError):
    """Le backend ne repond pas, ou refuse la requete."""


@dataclass
class ChatResult:
    content: str
    model: str
    backend: str
    total_duration_s: float = 0.0
    load_duration_s: float = 0.0
    eval_duration_s: float = 0.0
    eval_count: int = 0
    prompt_eval_count: int = 0

    def stats_line(self) -> str:
        parts = [f"{self.backend}:{self.model}"]
        if self.prompt_eval_count or self.eval_count:
            parts.append(
                f"{self.prompt_eval_count} tok in / {self.eval_count} tok out"
            )
        parts.append(f"{self.total_duration_s:.1f}s total")

        # Le debit se mesure sur la generation seule : le temps total inclut le
        # chargement du modele, qui fausse la lecture au premier appel.
        if self.eval_duration_s > 0 and self.eval_count:
            parts.append(f"{self.eval_count / self.eval_duration_s:.1f} tok/s")
        if self.load_duration_s >= 1.0:
            parts.append(f"chargement {self.load_duration_s:.1f}s")

        return " | ".join(parts)


class Backend(abc.ABC):
    """Un fournisseur d'inference : Ollama en local, ou toute API OpenAI-compatible."""

    #: identifiant declare dans config/backends.yaml
    name: str
    #: URL de base du service
    base_url: str
    #: surcharge classe -> modele ; prend le pas sur le profil materiel, car la
    #: capacite d'une passerelle distante n'a rien a voir avec celle du poste
    model_overrides: dict[str, str] | None = None
    #: plafond de contexte impose par le backend, si connu
    num_ctx_cap: int | None = None
    #: seul un backend local sait telecharger ses modeles
    supports_pull: bool = False

    @abc.abstractmethod
    async def ping(self) -> str:
        """Retourne une version ou un identifiant de service. Leve BackendUnavailable."""

    @abc.abstractmethod
    async def list_models(self) -> list[str]:
        """Modeles exposes par le backend. Liste vide si le service ne les publie pas."""

    @abc.abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: dict[str, Any] | None = None,
        fmt: str | None = None,
    ) -> ChatResult:
        """Une generation. `fmt="json"` contraint la sortie a un objet JSON."""

    def describe(self) -> str:
        return f"{self.name} ({type(self).__name__}) → {self.base_url}"
