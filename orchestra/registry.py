"""Coeur d'Orchestra : cablage agents + profil materiel + serveur Ollama."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import AgentSpec, load_agents
from .env import load_dotenv
from .hardware import Hardware
from .ollama_client import OllamaClient, OllamaUnavailable
from .profiles import Profile, select_profile
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

    def as_markdown(self) -> str:
        header = f"### {self.agent}"
        if self.routing:
            header += f"  _(routage: {self.routing.method} — {self.routing.reason})_"
        return f"{header}\n\n{self.content}\n\n---\n`{self.stats}`"


class AgentNotFound(KeyError):
    pass


class Orchestra:
    def __init__(
        self,
        agents: dict[str, AgentSpec],
        profile: Profile,
        hardware: Hardware,
        profile_reason: str,
        client: OllamaClient | None = None,
    ) -> None:
        self.agents = agents
        self.profile = profile
        self.hardware = hardware
        self.profile_reason = profile_reason
        self.client = client or OllamaClient()

    @classmethod
    def bootstrap(cls, agents_dir: Path | None = None) -> "Orchestra":
        # Avant toute lecture de OLLAMA_HOST / ORCHESTRA_PROFILE.
        load_dotenv()
        profile, hardware, reason = select_profile()
        return cls(load_agents(agents_dir), profile, hardware, reason)

    # ------------------------------------------------------------------ infos

    def get(self, name: str) -> AgentSpec:
        try:
            return self.agents[name]
        except KeyError as exc:
            raise AgentNotFound(
                f"Agent '{name}' inconnu. Disponibles : {sorted(self.agents)}"
            ) from exc

    def describe(self) -> str:
        lines = [
            "## Orchestra",
            "",
            f"- Materiel : {self.hardware.summary()}",
            f"- Profil : **{self.profile.id}** ({self.profile.label}) — {self.profile_reason}",
            f"- Contexte max : {self.profile.num_ctx} tokens",
            f"- Serveur Ollama : {self.client.host}",
            "",
            "| Agent | Classe | Modele resolu | Taches | Role |",
            "|---|---|---|---|---|",
        ]
        for spec in self.agents.values():
            lines.append(
                f"| `{spec.name}` | {spec.model_class} | "
                f"`{spec.resolve_model(self.profile)}` | "
                f"{', '.join(spec.tasks) or '—'} | {spec.description or spec.label} |"
            )
        return "\n".join(lines)

    def required_models(self) -> list[str]:
        return sorted({s.resolve_model(self.profile) for s in self.agents.values()})

    async def health(self) -> str:
        try:
            version = await self.client.ping()
        except OllamaUnavailable as exc:
            return f"❌ {exc}"

        installed = set(await self.client.list_models())
        lines = [f"✅ Ollama {version} sur {self.client.host}", ""]
        missing = []
        for model in self.required_models():
            # Ollama expose "name:tag" ; un modele sans tag explicite est ":latest".
            present = model in installed or f"{model}:latest" in installed
            lines.append(f"{'✅' if present else '❌'} {model}")
            if not present:
                missing.append(model)

        if missing:
            lines += [
                "",
                "Modeles manquants. Installe-les avec :",
                "```",
                *[f"ollama pull {m}" for m in missing],
                "```",
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
    ) -> AgentRun:
        spec = self.get(agent_name)
        model = spec.resolve_model(self.profile)

        user_content = prompt
        if context:
            user_content = (
                f"{prompt}\n\n"
                "--- CONTEXTE FOURNI PAR L'ORCHESTRATEUR ---\n"
                f"{context}\n"
                "--- FIN DU CONTEXTE ---"
            )

        result = await self.client.chat(
            model,
            [
                {"role": "system", "content": spec.system},
                {"role": "user", "content": user_content},
            ],
            options=spec.resolve_options(self.profile),
            fmt=spec.output_format,
        )
        return AgentRun(
            agent=spec.name,
            model=model,
            content=result.content,
            stats=result.stats_line(),
            routing=routing,
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
                result = await self.client.chat(
                    triage.resolve_model(self.profile),
                    [
                        {"role": "system", "content": triage.system},
                        {
                            "role": "user",
                            "content": build_triage_prompt(
                                candidates, task_type, prompt
                            ),
                        },
                    ],
                    options=triage.resolve_options(self.profile),
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
            except OllamaUnavailable:
                pass  # on retombe sur le score deterministe

        return RoutingDecision(
            agent=best.name,
            score=best_score,
            method="deterministe (repli)",
            reason=f"score faible {best_score:.1f}, meilleur candidat retenu",
            runners_up=runners_up,
        )

    async def delegate(
        self, prompt: str, *, task_type: str | None = None, context: str | None = None
    ) -> AgentRun:
        decision = await self.route(task_type, prompt)
        return await self.run(
            decision.agent, prompt, context=context, routing=decision
        )
