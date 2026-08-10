"""Coeur d'Orchestra : cablage agents + profil materiel + backend d'inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .agent_loop import run_agent_loop
from .backends import Backend, BackendUnavailable, create_backend
from .config import AgentSpec, load_agents
from .env import load_dotenv
from .hardware import Hardware
from .profiles import Profile, select_profile
from .tools import build_toolset
from .workspace import Workspace
from .router import (
    AMBIGUITY_THRESHOLD,
    RoutingDecision,
    build_triage_prompt,
    parse_triage_response,
    rank_agents,
)

# Agent consulte pour arbitrer un routage ambigu. Optionnel : s'il n'existe
# pas, on reste sur le score deterministe.
TRIAGE_AGENT = "triage"


@dataclass
class AgentRun:
    agent: str
    model: str
    content: str
    stats: str
    routing: RoutingDecision | None = None
    #: journal des outils appeles, vide pour un agent de generation pure
    trace: str = ""
    #: fichiers reellement ecrits ou modifies
    changed: list[str] = field(default_factory=list)

    def as_markdown(self) -> str:
        header = f"### {self.agent}"
        if self.routing:
            header += f"  _(routage: {self.routing.method} - {self.routing.reason})_"

        blocks = [header, "", self.content]
        if self.trace:
            blocks += ["", self.trace]
        if self.changed:
            blocks += [
                "",
                "**Fichiers modifies** : " + ", ".join(f"`{p}`" for p in self.changed),
            ]
        blocks += ["", "---", f"`{self.stats}`"]
        return "\n".join(blocks)


class AgentNotFound(KeyError):
    pass


class Orchestra:
    def __init__(
        self,
        agents: dict[str, AgentSpec],
        profile: Profile,
        hardware: Hardware,
        profile_reason: str,
        backend: Backend,
        backend_reason: str = "",
    ) -> None:
        self.agents = agents
        self.profile = profile
        self.hardware = hardware
        self.profile_reason = profile_reason
        self.backend = backend
        self.backend_reason = backend_reason

    @classmethod
    def bootstrap(cls, agents_dir: Path | None = None) -> "Orchestra":
        # Avant toute lecture de variable d'environnement.
        load_dotenv()
        backend, backend_reason = create_backend()
        profile, hardware, profile_reason = select_profile()
        return cls(
            load_agents(agents_dir),
            profile,
            hardware,
            profile_reason,
            backend,
            backend_reason,
        )

    # ------------------------------------------------------------------ infos

    def get(self, name: str) -> AgentSpec:
        try:
            return self.agents[name]
        except KeyError as exc:
            raise AgentNotFound(
                f"Agent '{name}' inconnu. Disponibles : {sorted(self.agents)}"
            ) from exc

    @property
    def remote_models(self) -> bool:
        """Le backend impose-t-il ses propres modeles ?

        Si oui, le profil materiel local ne decide plus rien : c'est la
        passerelle qui dimensionne.
        """
        return bool(self.backend.model_overrides)

    def model_for(self, spec: AgentSpec) -> str:
        return spec.resolve_model(self.profile, self.backend.model_overrides)

    def options_for(self, spec: AgentSpec) -> dict:
        return spec.resolve_options(self.profile, self.backend.num_ctx_cap)

    def describe(self) -> str:
        lines = [
            "## Orchestra",
            "",
            f"- Backend : **{self.backend.describe()}**"
            + (f" - {self.backend_reason}" if self.backend_reason else ""),
            f"- Materiel local : {self.hardware.summary()}",
        ]

        if self.remote_models:
            lines.append(
                f"- Profil : **{self.profile.id}** (ignore - les modeles sont "
                "imposes par le backend distant)"
            )
        else:
            lines.append(
                f"- Profil : **{self.profile.id}** ({self.profile.label}) - "
                f"{self.profile_reason}"
            )

        cap = self.backend.num_ctx_cap or self.profile.num_ctx
        lines += [
            f"- Contexte max : {cap} tokens",
            "",
            "| Agent | Classe | Modele resolu | Acces | Outils | Role |",
            "|---|---|---|---|---|---|",
        ]
        for spec in self.agents.values():
            if not spec.is_agentic:
                access = "texte"
            elif spec.writes:
                access = "**ecriture**"
            else:
                access = "lecture"
            lines.append(
                f"| `{spec.name}` | {spec.model_class} | `{self.model_for(spec)}` | "
                f"{access} | {', '.join(spec.tools) or '-'} | "
                f"{spec.description or spec.label} |"
            )

        lines += [
            "",
            "Les agents en ecriture n'agissent que si l'appel fournit un "
            "`workspace` **et** `allow_writes=true`. Sans espace de travail, "
            "ils retombent en generation de texte.",
        ]
        return "\n".join(lines)

    def required_models(self) -> list[str]:
        return sorted({self.model_for(spec) for spec in self.agents.values()})

    async def health(self) -> str:
        try:
            version = await self.backend.ping()
        except BackendUnavailable as exc:
            return f"❌ {exc}"

        lines = [f"✅ {self.backend.name} joignable sur {self.backend.base_url} ({version})", ""]

        try:
            published = set(await self.backend.list_models())
        except BackendUnavailable as exc:
            lines.append(f"⚠️ Inventaire des modeles indisponible : {exc}")
            return "\n".join(lines)

        if not published:
            # Certaines passerelles ne publient pas leur catalogue ; l'absence
            # d'inventaire n'est pas une erreur.
            lines.append(
                "⚠️ Le backend ne publie pas de catalogue : impossible de "
                "verifier la presence des modeles a l'avance."
            )
            return "\n".join(lines)

        missing = []
        for model in self.required_models():
            # Un modele sans tag explicite vaut ":latest" cote Ollama.
            present = model in published or f"{model}:latest" in published
            lines.append(f"{'✅' if present else '❌'} {model}")
            if not present:
                missing.append(model)

        if missing:
            if self.backend.supports_pull:
                lines += [
                    "",
                    "Modeles manquants. Installe-les avec :",
                    "```",
                    *[f"ollama pull {m}" for m in missing],
                    "```",
                ]
            else:
                lines += [
                    "",
                    "Modeles manquants sur la passerelle. Verifie les noms "
                    "declares sous `models:` pour ce backend dans "
                    "config/backends.yaml.",
                ]
        return "\n".join(lines)

    # --------------------------------------------------------------- execution

    async def run(
        self,
        agent_name: str,
        prompt: str,
        *,
        context: str | None = None,
        routing: RoutingDecision | None = None,
        workspace: Workspace | None = None,
    ) -> AgentRun:
        spec = self.get(agent_name)
        model = self.model_for(spec)

        user_content = prompt
        if context:
            user_content = (
                f"{prompt}\n\n"
                "--- CONTEXTE FOURNI PAR L'ORCHESTRATEUR ---\n"
                f"{context}\n"
                "--- FIN DU CONTEXTE ---"
            )

        toolset = build_toolset(spec.tools, workspace)
        system = spec.system
        if toolset is not None:
            system = f"{system}\n\n{_tool_briefing(toolset)}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        options = self.options_for(spec)

        if toolset is None:
            # Agent de generation pure, ou outils inutilisables faute d'espace
            # de travail : un aller-retour, comportement historique.
            result = await self.backend.chat(
                model, messages, options=options, fmt=spec.output_format
            )
            return AgentRun(
                agent=spec.name,
                model=model,
                content=result.content,
                stats=result.stats_line(),
                routing=routing,
            )

        loop = await run_agent_loop(
            self.backend,
            model,
            messages,
            toolset,
            options=options,
            max_turns=spec.max_turns,
        )
        return AgentRun(
            agent=spec.name,
            model=model,
            content=loop.content,
            stats=loop.stats,
            routing=routing,
            trace=loop.trace(),
            changed=sorted({str(i.arguments.get("path", "?")) for i in loop.writes}),
        )

    async def route(self, task_type: str | None, prompt: str) -> RoutingDecision:
        ranked = rank_agents(self.agents, task_type, prompt)
        best, best_score = ranked[0]
        runners_up = [(s.name, sc) for s, sc in ranked[1:4]]

        tie = len(ranked) > 1 and abs(best_score - ranked[1][1]) < 1e-9
        if best_score >= AMBIGUITY_THRESHOLD and not tie:
            return RoutingDecision(
                agent=best.name,
                score=best_score,
                method="deterministe",
                reason=f"score {best_score:.1f}",
                runners_up=runners_up,
            )

        triage = self.agents.get(TRIAGE_AGENT)
        if triage is not None and best.name != TRIAGE_AGENT:
            candidates = {n: s for n, s in self.agents.items() if n != TRIAGE_AGENT}
            try:
                result = await self.backend.chat(
                    self.model_for(triage),
                    [
                        {"role": "system", "content": triage.system},
                        {
                            "role": "user",
                            "content": build_triage_prompt(
                                candidates, task_type, prompt
                            ),
                        },
                    ],
                    options=self.options_for(triage),
                    fmt="json",
                )
                parsed = parse_triage_response(result.content, candidates)
                if parsed:
                    name, reason = parsed
                    return RoutingDecision(
                        agent=name,
                        score=best_score,
                        method="triage LLM",
                        reason=reason,
                        runners_up=runners_up,
                    )
            except BackendUnavailable:
                pass  # on retombe sur le score deterministe

        return RoutingDecision(
            agent=best.name,
            score=best_score,
            method="deterministe (repli)",
            reason=f"score faible {best_score:.1f}, meilleur candidat retenu",
            runners_up=runners_up,
        )

    async def delegate(
        self,
        prompt: str,
        *,
        task_type: str | None = None,
        context: str | None = None,
        workspace: Workspace | None = None,
    ) -> AgentRun:
        decision = await self.route(task_type, prompt)
        return await self.run(
            decision.agent,
            prompt,
            context=context,
            routing=decision,
            workspace=workspace,
        )


def _tool_briefing(toolset) -> str:
    """Instructions ajoutees au system prompt quand l'agent est outille.

    Le prompt metier de l'agent decrit son role ; ce bloc decrit son
    environnement. Les separer evite de dupliquer les memes consignes
    d'outillage dans chaque fichier YAML.
    """
    return (
        "ENVIRONNEMENT\n"
        f"Tu travailles dans l'arborescence : {toolset.workspace.describe()}\n"
        "Tu disposes d'outils qui agissent reellement sur ces fichiers.\n"
        "\n"
        "Methode :\n"
        "- Explore avant d'agir : list_files ou search_files pour situer, "
        "read_file avant toute modification.\n"
        "- Les chemins sont relatifs a la racine. Les chemins absolus sont refuses.\n"
        "- Modifie avec edit_file, qui exige le texte exact deja present. "
        "write_file ecrase tout le fichier : reserve-le aux creations.\n"
        "- Un outil qui echoue te renvoie la raison. Lis-la et corrige ton appel "
        "plutot que de repeter le meme.\n"
        "- Applique les changements toi-meme. Ne rends pas du code en te "
        "contentant de demander a l'utilisateur de le recopier.\n"
        "- Quand c'est fait, conclus en une ou deux phrases : ce que tu as "
        "change, et ou. Ne recopie pas le contenu des fichiers ecrits."
    )
