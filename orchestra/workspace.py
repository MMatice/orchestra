"""Espace de travail des agents : la frontiere de securite du systeme.

Un agent outille est pilote par un modele qui n'est pas forcement local. Tout
ce qu'il lit part dans le contexte de ce modele, tout ce qu'il ecrit atterrit
sur le disque de l'utilisateur. Ce module est le seul endroit ou une chaine
fournie par un modele devient un chemin reel : c'est donc le seul endroit a
auditer serieusement.

Trois garanties, dans cet ordre :
  1. Confinement : aucun chemin ne sort de la racine, y compris via `..`,
     chemin absolu ou lien symbolique.
  2. Occultation : les fichiers de secrets sont invisibles, en lecture comme
     en ecriture. Un agent ne doit pas pouvoir lire un .env et le recopier
     dans un contexte qui part chez un fournisseur distant.
  3. Ecriture explicite : la racine est en lecture seule tant que l'appelant
     n'a pas demande le contraire.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

# Un agent qui lit un fichier enorme fait exploser le contexte et la facture.
MAX_READ_BYTES = 256 * 1024
MAX_WRITE_BYTES = 1024 * 1024

# Occultes en lecture comme en ecriture. La liste vise les porteurs de
# secrets et les repertoires dont la corruption est difficile a rattraper.
DENIED_NAMES = frozenset(
    {
        ".git",
        ".ssh",
        ".aws",
        ".gnupg",
        ".docker",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".mcp.json",
        "id_rsa",
        "id_ed25519",
        "credentials",
        "secrets.yaml",
        "secrets.yml",
    }
)
DENIED_SUFFIXES = frozenset({".pem", ".key", ".pfx", ".p12"})


class WorkspaceError(RuntimeError):
    """Chemin refuse : hors racine, occulte, ou ecriture non autorisee."""


def _is_denied(part: str) -> bool:
    low = part.lower()
    if low in DENIED_NAMES:
        return True
    # Couvre .env, .env.local, .env.production... y compris .env.example :
    # le cout de le masquer est nul, celui de le laisser fuiter ne l'est pas.
    if low.startswith(".env"):
        return True
    return PurePosixPath(low).suffix in DENIED_SUFFIXES


class Workspace:
    """Une racine sur le disque, et les regles d'acces qui vont avec."""

    def __init__(self, root: Path | str, *, writable: bool = False) -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise WorkspaceError(
                f"Espace de travail introuvable ou non repertoire : {resolved}"
            )
        self.root = resolved
        self.writable = writable

    def __repr__(self) -> str:  # pragma: no cover - confort de debogage
        mode = "rw" if self.writable else "ro"
        return f"<Workspace {self.root} ({mode})>"

    def describe(self) -> str:
        return f"{self.root} ({'lecture/ecriture' if self.writable else 'lecture seule'})"

    def resolve(self, relative: str, *, for_write: bool = False) -> Path:
        """Transforme un chemin fourni par le modele en chemin reel confine."""
        if for_write and not self.writable:
            raise WorkspaceError(
                "Espace de travail en lecture seule : l'ecriture doit etre "
                "autorisee explicitement par l'appelant."
            )

        raw = (relative or "").strip().replace("\\", "/")
        if not raw:
            raise WorkspaceError("Chemin vide.")

        # On teste les deux grammaires : un '/etc/passwd' n'est pas absolu au
        # sens de Windows, un 'C:\\...' ne l'est pas au sens POSIX. Les deux
        # doivent etre refuses quel que soit l'OS qui execute le serveur.
        if PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
            raise WorkspaceError(
                f"'{relative}' est un chemin absolu. Utilise un chemin relatif "
                "a la racine de l'espace de travail."
            )

        # .resolve() suit les liens symboliques : une racine piegee par un lien
        # vers l'exterieur echoue donc au controle de confinement ci-dessous.
        target = (self.root / raw).resolve()
        try:
            inside = target.relative_to(self.root)
        except ValueError:
            raise WorkspaceError(
                f"'{relative}' sort de l'espace de travail ({self.root})."
            ) from None

        for part in inside.parts:
            if _is_denied(part):
                raise WorkspaceError(
                    f"'{relative}' touche un emplacement protege ('{part}'). "
                    "Secrets et metadonnees de depot sont hors de portee des agents."
                )
        return target

    def display(self, path: Path) -> str:
        """Chemin relatif a la racine, pour les messages rendus au modele."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:  # pragma: no cover - ne devrait pas arriver
            return str(path)

    @classmethod
    def open(
        cls, path: str | None = None, *, writable: bool = False
    ) -> "Workspace | None":
        """Ouvre l'espace demande, sinon celui de ORCHESTRA_WORKSPACE.

        Retourne None quand aucun des deux n'est fourni : les agents non
        outilles fonctionnent sans espace de travail, et l'absence de racine
        ne doit pas les empecher de tourner.
        """
        import os

        candidate = (path or "").strip() or os.environ.get(
            "ORCHESTRA_WORKSPACE", ""
        ).strip()
        return cls(candidate, writable=writable) if candidate else None

    def is_visible(self, path: Path) -> bool:
        """Le fichier peut-il etre montre lors d'un parcours de repertoire ?"""
        try:
            inside = path.relative_to(self.root)
        except ValueError:
            return False
        return not any(_is_denied(part) for part in inside.parts)
