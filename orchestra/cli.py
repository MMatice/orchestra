"""CLI de test — permet d'utiliser Orchestra sans passer par Claude Code.

  python -m orchestra.cli status
  python -m orchestra.cli agents
  python -m orchestra.cli ask reviewer "Relis ce code" --file mon_module.py
  python -m orchestra.cli delegate "explique cette fonction" --task explain --file x.py
  python -m orchestra.cli pipeline steps.json --input-file spec.md
  python -m orchestra.cli refine "Ecris une fonction de parsing d'ISO-8601"
  python -m orchestra.cli pull
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from .pipeline import parse_steps, run_pipeline, run_refine_loop
from .registry import Orchestra


def _read(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8")


async def _run(args: argparse.Namespace) -> int:
    orch = Orchestra.bootstrap()

    if args.command == "status":
        print(orch.describe())
        print()
        print(await orch.health())
        return 0

    if args.command == "agents":
        for spec in orch.agents.values():
            print(
                f"{spec.name:<14} {spec.model_class:<7} "
                f"{spec.resolve_model(orch.profile):<24} {spec.description}"
            )
        return 0

    if args.command == "pull":
        models = orch.required_models()
        print(f"Modeles requis pour le profil '{orch.profile.id}' : {models}\n")
        for model in models:
            print(f"→ ollama pull {model}")
            if subprocess.run(["ollama", "pull", model]).returncode != 0:
                print(f"  echec sur {model}", file=sys.stderr)
                return 1
        return 0

    if args.command == "ask":
        run = await orch.run(args.agent, args.prompt, context=_read(args.file))
        print(run.as_markdown())
        return 0

    if args.command == "delegate":
        run = await orch.delegate(
            args.prompt, task_type=args.task, context=_read(args.file)
        )
        print(run.as_markdown())
        return 0

    if args.command == "pipeline":
        steps = parse_steps(json.loads(Path(args.steps_file).read_text(encoding="utf-8")))
        result = await run_pipeline(orch, steps, _read(args.input_file) or "")
        print(result.as_markdown())
        return 0

    if args.command == "refine":
        result = await run_refine_loop(
            orch,
            args.producer,
            args.critic,
            args.task,
            max_rounds=args.rounds,
        )
        print(result.as_markdown())
        return 0

    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orchestra", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="materiel, profil, agents, etat d'Ollama")
    sub.add_parser("agents", help="liste compacte des agents")
    sub.add_parser("pull", help="telecharge les modeles requis par le profil")

    p_ask = sub.add_parser("ask", help="appelle un agent precis")
    p_ask.add_argument("agent")
    p_ask.add_argument("prompt")
    p_ask.add_argument("--file", help="fichier joint comme contexte")

    p_del = sub.add_parser("delegate", help="laisse le routeur choisir l'agent")
    p_del.add_argument("prompt")
    p_del.add_argument("--task", default=None, help="indice de routage")
    p_del.add_argument("--file", help="fichier joint comme contexte")

    p_pipe = sub.add_parser("pipeline", help="enchaine des agents depuis un JSON")
    p_pipe.add_argument("steps_file")
    p_pipe.add_argument("--input-file", help="entree initiale")

    p_ref = sub.add_parser("refine", help="boucle producteur/critique")
    p_ref.add_argument("task")
    p_ref.add_argument("--producer", default="implementer")
    p_ref.add_argument("--critic", default="reviewer")
    p_ref.add_argument("--rounds", type=int, default=2)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
