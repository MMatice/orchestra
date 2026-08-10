"""Tests du confinement.

C'est la surface la plus sensible du projet : un modele distant fournit les
chemins, et un seul echappement suffit a lui donner le disque entier. Chaque
voie de sortie connue a donc son test.
"""

import pytest

from orchestra.workspace import Workspace, WorkspaceError


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-secret", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]", encoding="utf-8")
    return Workspace(tmp_path, writable=True)


def test_relative_path_inside_root_resolves(ws):
    assert ws.resolve("src/app.py").read_text(encoding="utf-8") == "print('hi')"


def test_parent_traversal_is_refused(ws):
    with pytest.raises(WorkspaceError, match="sort de l'espace"):
        ws.resolve("../../../etc/passwd")


def test_traversal_hidden_mid_path_is_refused(ws):
    # Le prefixe est legitime : seul le chemin resolu compte.
    with pytest.raises(WorkspaceError, match="sort de l'espace"):
        ws.resolve("src/../../secrets.txt")


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",  # absolu POSIX
        "C:/Windows/System32/config",  # absolu Windows
        "C:\\Windows\\win.ini",  # separateurs inverses
    ],
)
def test_absolute_paths_are_refused_whatever_the_os(ws, path):
    with pytest.raises(WorkspaceError, match="chemin absolu"):
        ws.resolve(path)


def test_symlink_pointing_outside_is_refused(tmp_path):
    """Un lien est un chemin valide en apparence : .resolve() le demasque."""
    root = tmp_path / "projet"
    root.mkdir()
    outside = tmp_path / "dehors"
    outside.mkdir()
    (outside / "secret.txt").write_text("confidentiel", encoding="utf-8")

    try:
        (root / "echappe").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creation de lien symbolique indisponible sur cette machine")

    workspace = Workspace(root)
    with pytest.raises(WorkspaceError, match="sort de l'espace"):
        workspace.resolve("echappe/secret.txt")


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", ".git/config", "src/../.env", "cle.pem", ".ssh/id_rsa"],
)
def test_secret_bearing_paths_are_masked(ws, path):
    """Un agent ne doit pas pouvoir lire un secret puis le recopier ailleurs.

    Le risque n'est pas theorique : tout ce que l'agent lit part dans le
    contexte du modele, donc chez le fournisseur d'inference.
    """
    with pytest.raises(WorkspaceError, match="emplacement protege"):
        ws.resolve(path)


def test_read_only_workspace_refuses_writes(tmp_path):
    workspace = Workspace(tmp_path, writable=False)
    assert workspace.resolve("note.txt", for_write=False)
    with pytest.raises(WorkspaceError, match="lecture seule"):
        workspace.resolve("note.txt", for_write=True)


def test_empty_path_is_refused(ws):
    with pytest.raises(WorkspaceError, match="vide"):
        ws.resolve("   ")


def test_missing_root_is_refused(tmp_path):
    with pytest.raises(WorkspaceError, match="introuvable"):
        Workspace(tmp_path / "nexistepas")


def test_open_returns_none_without_a_path(monkeypatch):
    """Les agents non outilles doivent tourner sans espace de travail."""
    monkeypatch.delenv("ORCHESTRA_WORKSPACE", raising=False)
    assert Workspace.open("") is None


def test_open_falls_back_to_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRA_WORKSPACE", str(tmp_path))
    workspace = Workspace.open(None, writable=True)
    assert workspace is not None
    assert workspace.root == tmp_path.resolve()
    assert workspace.writable


def test_is_visible_hides_protected_files(ws):
    assert ws.is_visible(ws.root / "src" / "app.py")
    assert not ws.is_visible(ws.root / ".env")
    assert not ws.is_visible(ws.root / ".git" / "config")
