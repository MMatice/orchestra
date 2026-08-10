"""Outils de lecture et d'ecriture de fichiers.

Chaque fonction recoit le `Workspace` en premier argument et ne manipule
jamais un chemin brut : la resolution passe systematiquement par lui.

Les messages de retour sont ecrits pour etre lus par un modele. Un echec doit
donc dire ce qui n'a pas marche ET comment reessayer, sinon l'agent boucle sur
la meme erreur jusqu'a epuiser son budget de tours.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..workspace import MAX_READ_BYTES, MAX_WRITE_BYTES, Workspace

# Un parcours d'arborescence ne doit pas noyer le contexte.
MAX_LISTED = 300
MAX_MATCHES = 60
MAX_SCANNED = 2000

# Repertoires sans interet pour un agent, et volumineux.
SKIPPED_DIRS = frozenset(
    {
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)


def _skipped(workspace: Workspace, path: Path) -> bool:
    relative = path.relative_to(workspace.root)
    return any(part in SKIPPED_DIRS for part in relative.parts)


def _read_text(path: Path) -> str:
    """Lit en UTF-8 en tolerant les fichiers mal encodes plutot qu'en echouant."""
    return path.read_text(encoding="utf-8", errors="replace")


def list_files(
    workspace: Workspace, subdir: str = ".", pattern: str = "**/*"
) -> str:
    from . import ToolError

    root = workspace.resolve(subdir or ".")
    if not root.is_dir():
        raise ToolError(f"'{subdir}' n'est pas un repertoire.")

    entries: list[str] = []
    truncated = False
    for path in sorted(root.glob(pattern or "**/*")):
        if not path.is_file():
            continue
        if not workspace.is_visible(path) or _skipped(workspace, path):
            continue
        if len(entries) >= MAX_LISTED:
            truncated = True
            break
        entries.append(f"{workspace.display(path)} ({path.stat().st_size} o)")

    if not entries:
        return (
            f"Aucun fichier sous '{workspace.display(root)}' pour le motif "
            f"'{pattern}'. Essaie un motif plus large, par exemple '**/*'."
        )

    header = f"{len(entries)} fichier(s) sous '{workspace.display(root)}'"
    if truncated:
        header += f" (tronque a {MAX_LISTED}, affine le motif)"
    return header + " :\n" + "\n".join(entries)


def read_file(workspace: Workspace, path: str) -> str:
    from . import ToolError

    target = workspace.resolve(path)
    if not target.exists():
        raise ToolError(
            f"'{path}' n'existe pas. Utilise list_files pour verifier le chemin."
        )
    if target.is_dir():
        raise ToolError(f"'{path}' est un repertoire. Utilise list_files.")

    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        raise ToolError(
            f"'{path}' fait {size} octets, au-dela de la limite de "
            f"{MAX_READ_BYTES}. Cible un fichier plus petit ou utilise "
            "search_files pour n'extraire que les lignes utiles."
        )

    content = _read_text(target)
    lines = content.count("\n") + 1
    return f"--- {workspace.display(target)} ({lines} lignes) ---\n{content}"


def write_file(workspace: Workspace, path: str, content: str) -> str:
    from . import ToolError

    target = workspace.resolve(path, for_write=True)
    if target.is_dir():
        raise ToolError(f"'{path}' est un repertoire.")

    payload = content if content is not None else ""
    encoded = payload.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        raise ToolError(
            f"Contenu de {len(encoded)} octets, au-dela de la limite de "
            f"{MAX_WRITE_BYTES}. Decoupe le fichier."
        )

    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    # newline="" : le contenu produit par le modele fait autorite, on n'y
    # ajoute pas de traduction de fins de ligne propre a la plateforme.
    target.write_text(payload, encoding="utf-8", newline="")

    verb = "Remplace" if existed else "Cree"
    return (
        f"{verb} : {workspace.display(target)} "
        f"({len(encoded)} octets, {payload.count(chr(10)) + 1} lignes)"
    )


def edit_file(
    workspace: Workspace, path: str, old_text: str, new_text: str
) -> str:
    from . import ToolError

    target = workspace.resolve(path, for_write=True)
    if not target.is_file():
        raise ToolError(
            f"'{path}' n'existe pas. Pour creer un fichier, utilise write_file."
        )
    if not old_text:
        raise ToolError(
            "old_text est vide. Pour reecrire un fichier entier, utilise write_file."
        )

    content = _read_text(target)
    occurrences = content.count(old_text)

    if occurrences == 0:
        raise ToolError(
            f"Fragment absent de '{path}'. Relis le fichier avec read_file et "
            "recopie le texte exact, espaces et indentation compris."
        )
    if occurrences > 1:
        raise ToolError(
            f"Fragment present {occurrences} fois dans '{path}'. Rallonge "
            "old_text avec les lignes qui l'entourent pour le rendre unique."
        )

    updated = content.replace(old_text, new_text)
    target.write_text(updated, encoding="utf-8", newline="")

    delta = updated.count("\n") - content.count("\n")
    change = f"{delta:+d} ligne(s)" if delta else "nombre de lignes inchange"
    return f"Modifie : {workspace.display(target)} ({change})"


def search_files(workspace: Workspace, pattern: str, glob: str = "**/*") -> str:
    from . import ToolError

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"Expression reguliere invalide : {exc}") from exc

    matches: list[str] = []
    scanned = 0
    for path in sorted(workspace.root.glob(glob or "**/*")):
        if not path.is_file():
            continue
        if not workspace.is_visible(path) or _skipped(workspace, path):
            continue
        if scanned >= MAX_SCANNED or len(matches) >= MAX_MATCHES:
            break
        scanned += 1

        try:
            content = _read_text(path)
        except OSError:
            continue
        if "\0" in content[:1024]:  # binaire
            continue

        for number, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                matches.append(
                    f"{workspace.display(path)}:{number}: {line.strip()[:200]}"
                )
                if len(matches) >= MAX_MATCHES:
                    break

    if not matches:
        return f"Aucune correspondance pour '{pattern}' dans '{glob}'."
    return f"{len(matches)} correspondance(s) :\n" + "\n".join(matches)
