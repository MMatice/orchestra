"""Contrat commun a tous les backends d'inference."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

# Une generation peut etre longue : gros prompt, modele froid, file d'attente
# sur une passerelle partagee.
READ_TIMEOUT_S = 900.0
CONNECT_TIMEOUT_S = 10.0


class BackendUnavailable(RuntimeError):
    """Le backend ne repond pas, ou refuse la requete."""


@dataclass
class ToolCall:
    """Demande d'appel d'outil emise par le modele."""

    id: str
    name: str
    arguments: dict[str, Any]


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
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: message d'origine, renvoye tel quel dans l'historique. Chaque
    #: fournisseur accepte sa propre representation : la reconstruire a la
    #: main ferait perdre les champs qui lui sont propres.
    raw_message: dict[str, Any] = field(default_factory=dict)

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


# Champs ou les modeles a raisonnement deposent leur reflexion, par ordre de
# specificite : `reasoning_content` (DeepSeek natif, vLLM), `reasoning`
# (OpenRouter), `thinking` (Ollama).
REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")


def content_or_reasoning(message: dict[str, Any], has_tool_calls: bool) -> str:
    """Contenu du message, avec repli sur le raisonnement s'il est vide.

    Un modele a raisonnement peut conclure sans rien ecrire dans `content` :
    toute sa reponse reste alors dans le champ de reflexion. Sans ce repli,
    un tour de generation complet - et facture - est purement perdu.

    Le repli ne s'applique qu'a un tour terminal : quand le modele demande un
    outil, un `content` vide est le comportement normal et non une perte.
    """
    content = (message.get("content") or "").strip()
    if content or has_tool_calls:
        return content

    for key in REASONING_KEYS:
        fallback = message.get(key)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    return ""


class Backend(abc.ABC):
    """Un fournisseur d'inference : Ollama en local, ou toute API OpenAI-compatible."""

    #: identifiant declare dans config/backends.yaml
    name: str
    #: URL de base du service
    base_url: str
    #: surcharge classe -> modele ; prend le pas sur le profil materiel, car la
    #: capacite d'une passerelle distante n'a rien a voir avec celle du poste
    model_overrides: dict[str, str] | None = None
    #: plafond de contexte impose par le backend, si connu. N'a d'effet que
    #: sur un backend local : une passerelle distante impose sa propre fenetre,
    #: que le client ne peut ni lire ni contraindre depuis l'exterieur.
    num_ctx_cap: int | None = None
    #: plafond de tokens generes par appel, si le backend en declare un
    max_output_cap: int | None = None
    #: `max_output: auto` en configuration : interroger l'endpoint
    discovers_max_output: bool = False
    #: la fenetre de contexte est-elle une propriete du deploiement distant ?
    #: Si oui, num_ctx_cap ne decrit rien d'applicable et ne doit pas etre
    #: presente comme une limite effective.
    context_is_remote: bool = False
    #: seul un backend local sait telecharger ses modeles
    supports_pull: bool = False

    async def discover_max_output(self, model: str) -> int | None:
        """Plafond de sortie annonce par l'endpoint pour ce modele, si connu.

        Retourne None quand le service ne publie pas l'information, ce qui est
        le cas de la plupart des passerelles : elles exposent un catalogue
        reduit a des identifiants. La configuration reste donc la source
        principale, la decouverte n'etant qu'un confort la ou elle existe.
        """
        return None

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
        messages: list[dict[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        fmt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        """Une generation.

        `fmt="json"` contraint la sortie a un objet JSON. `tools` publie les
        outils appelables ; le modele peut alors repondre par des `tool_calls`
        au lieu d'un contenu textuel.
        """

    def tool_result_message(self, call: ToolCall, output: str) -> dict[str, Any]:
        """Message rendant le resultat d'un outil au modele.

        Forme OpenAI par defaut, la plus repandue. Ollama attend une variante,
        d'ou la surcharge cote backend natif.
        """
        return {"role": "tool", "tool_call_id": call.id, "content": output}

    def describe(self) -> str:
        return f"{self.name} ({type(self).__name__}) → {self.base_url}"
