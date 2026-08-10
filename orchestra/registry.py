"""Coeur d'Orchestra : cablage agents + profil materiel + backend d'inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .agent_loop import run_agent_loop
from .backends import Backend, BackendUnavailable, create_backend
from .config import AgentSpec, agents_fingerprint, load_agents
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
        self.agents_dir: Path | None = None
        self._agents_stamp: tuple = ()
        self._discovered_output: dict[str, int | None] = {}

    @classmethod
    def bootstrap(cls, agents_dir: Path | None = None) -> "Orchestra":
        # Avant toute lecture de variable d'environnement.
        load_dotenv()
        backend, backend_reason = create_backend()
        profile, hardware, profile_reason = select_profile()
        instance = cls(
            load_agents(agents_dir),
            profile,
            hardware,
            profile_reason,
            backend,
            backend_reason,
        )
        instance.agents_dir = agents_dir
        instance._agents_stamp = agents_fingerprint(agents_dir)
        return instance

    def reload_agents_if_changed(self) -> list[str]:
        """Recharge les agents dont le YAML a bouge. Retourne les noms touches.

        Editer un agent doit prendre effet sans redemarrer le serveur MCP :
        regler un prompt ou un plafond est une boucle qu'on parcourt souvent,
        et devoir reconnecter a chaque essai fait perdre le fil.

        Un YAML devenu invalide n'ecrase pas la configuration en memoire : on
        conserve la derniere version qui chargeait, et l'erreur remonte.
        """
        stamp = agents_fingerprint(self.agents_dir)
        if stamp == self._agents_stamp:
            return []

        previous = self.agents
        self.agents = load_agents(self.agents_dir)
        self._agents_stamp = stamp
        self._discovered_output.clear()

        changed = [
            name
            for name, spec in self.agents.items()
            if name not in previous or previous[name] != spec
        ]
        return changed + [n for n in previous if n not in self.agents]

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

    async def max_output_for(self, model: str) -> int | None:
        """Plafond de sortie applicable, decouvert puis memorise si demande."""
        if not self.backend.discovers_max_output:
            return self.backend.max_output_cap
        if model not in self._discovered_output:
            self._discovered_output[model] = await self.backend.discover_max_output(
                model
            )
        return self._discovered_output[model] or self.backend.max_output_cap

    async def options_for(self, spec: AgentSpec, model: str | None = None) -> dict:
        model = model or self.model_for(spec)
        return spec.resolve_options(
            self.profile,
            self.backend.num_ctx_cap,
            await self.max_output_for(model),
        )

    async def describe_async(self) -> str:
        """Inventaire, avec les plafonds reellement appliques.

        Separe de `describe()` parce que le plafond de sortie peut demander
        une interrogation de l'endpoint quand il est en `auto`.
        """
        budgets = {}
        for spec in self.agents.values():
            cap = await self.max_output_for(self.model_for(spec))
            budgets[spec.name] = min(spec.num_predict, cap or self.profile.max_output)
        return self.describe(budgets)

    def describe(self, output_budgets: dict[str, int] | None = None) -> str:
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

        # Ne jamais annoncer comme limite une valeur qui n'est appliquee nulle
        # part : sur une passerelle, la fenetre appartient au deploiement et
        # Orchestra ne l'envoie pas.
        if self.backend.context_is_remote:
            lines.append(
                "- Contexte : impose par le deploiement distant, Orchestra ne "
                "le contraint pas"
            )
        else:
            cap = self.backend.num_ctx_cap or self.profile.num_ctx
            lines.append(f"- Contexte max : {cap} tokens")

        lines += [
            "",
            "| Agent | Classe | Modele resolu | Acces | Sortie max | Outils |",
            "|---|---|---|---|---|---|",
        ]
        for spec in self.agents.values():
            if not spec.is_agentic:
                access = "texte"
            elif spec.writes:
                access = "**ecriture**"
            else:
                access = "lecture"

            if output_budgets is None:
                budget = f"{spec.num_predict} demandes"
            else:
                effective = output_budgets[spec.name]
                budget = str(effective)
                if effective < spec.num_predict:
                    budget += f" (plafonne, {spec.num_predict} demandes)"

            lines.append(
                f"| `{spec.name}` | {spec.model_class} | `{self.model_for(spec)}` | "
                f"{access} | {budget} | {', '.join(spec.tools) or '-'} |"
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
        options = await self.options_for(spec, model)

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
                    options=await self.options_for(triage),
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
