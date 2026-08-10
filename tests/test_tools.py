"""Tests des outils fichiers.

Deux exigences distinctes : les outils doivent faire ce qu'on attend, et leurs
messages d'echec doivent etre exploitables par un modele. Un message d'erreur
qui ne dit pas comment reessayer produit un agent qui boucle.
"""

import pytest

from orchestra.tools import TOOLS, Toolset, build_toolset
from orchestra.tools.files import edit_file, list_files, read_file, search_files, write_file
from orchestra.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "src" / "util.py").write_text("VALEUR = 3\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    return Workspace(tmp_path, writable=True)


# ------------------------------------------------------------------ lecture

def test_list_files_hides_secrets_and_noise(ws):
    listing = list_files(ws)
    assert "src/app.py" in listing
    assert ".env" not in listing
    assert "junk.pyc" not in listing


def test_read_file_returns_content_with_its_path(ws):
    out = read_file(ws, "src/app.py")
    assert "src/app.py" in out
    assert "return a + b" in out


def test_read_missing_file_says_how_to_recover(ws):
    from orchestra.tools import ToolError

    with pytest.raises(ToolError, match="list_files"):
        read_file(ws, "src/absent.py")


def test_search_files_reports_path_and_line(ws):
    out = search_files(ws, r"return", glob="**/*.py")
    assert "src/app.py:2" in out


def test_search_rejects_a_broken_regex(ws):
    from orchestra.tools import ToolError

    with pytest.raises(ToolError, match="reguliere invalide"):
        search_files(ws, "[unclosed")


# ----------------------------------------------------------------- ecriture

def test_write_file_creates_missing_parents(ws):
    out = write_file(ws, "a/b/c.txt", "contenu")
    assert "Cree" in out
    assert (ws.root / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "contenu"


def test_write_file_reports_replacement(ws):
    assert "Remplace" in write_file(ws, "src/util.py", "VALEUR = 4\n")


def test_edit_file_replaces_the_exact_fragment(ws):
    edit_file(ws, "src/app.py", "return a + b", "return a - b")
    assert "return a - b" in (ws.root / "src" / "app.py").read_text(encoding="utf-8")


def test_edit_refuses_an_ambiguous_fragment(ws):
    from orchestra.tools import ToolError

    (ws.root / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    with pytest.raises(ToolError, match="2 fois"):
        edit_file(ws, "dup.py", "x = 1", "x = 2")


def test_edit_refuses_an_absent_fragment_and_says_why(ws):
    from orchestra.tools import ToolError

    with pytest.raises(ToolError, match="read_file"):
        edit_file(ws, "src/app.py", "return a * b", "return a / b")


def test_edit_on_a_missing_file_points_to_write_file(ws):
    from orchestra.tools import ToolError

    with pytest.raises(ToolError, match="write_file"):
        edit_file(ws, "nouveau.py", "a", "b")


# ------------------------------------------------------------------ octroi

def test_write_tools_are_dropped_on_a_read_only_workspace(tmp_path):
    """Un agent en ecriture sur un espace verrouille doit rester utilisable.

    Retirer l'outil vaut mieux que le laisser echouer : l'agent travaille en
    lecture au lieu de bruler ses tours sur des refus.
    """
    toolset = build_toolset(
        ["read_file", "write_file", "edit_file"], Workspace(tmp_path, writable=False)
    )
    assert toolset.names == ["read_file"]


def test_no_workspace_means_no_tools():
    assert build_toolset(["read_file"], None) is None


def test_shell_stays_locked_without_explicit_opt_in(ws, monkeypatch):
    monkeypatch.delenv("ORCHESTRA_ALLOW_SHELL", raising=False)
    assert build_toolset(["read_file", "run_command"], ws).names == ["read_file"]

    monkeypatch.setenv("ORCHESTRA_ALLOW_SHELL", "1")
    assert "run_command" in build_toolset(["read_file", "run_command"], ws).names


def test_unknown_tool_is_reported_to_the_model_not_raised(ws):
    """Un outil hallucine ne doit pas faire echouer la tache."""
    toolset = Toolset([TOOLS["read_file"]], ws)
    output, ok = toolset.execute("delete_everything", {})
    assert not ok
    assert "read_file" in output  # on lui rappelle ce qu'il a le droit d'appeler


def test_bad_arguments_come_back_as_a_correctable_error(ws):
    toolset = Toolset([TOOLS["read_file"]], ws)
    output, ok = toolset.execute("read_file", {"chemin": "src/app.py"})
    assert not ok
    assert "arguments invalides" in output


def test_escape_attempt_is_reported_not_raised(ws):
    toolset = Toolset([TOOLS["read_file"]], ws)
    output, ok = toolset.execute("read_file", {"path": "../../../etc/passwd"})
    assert not ok
    assert "sort de l'espace" in output


def test_schemas_are_well_formed_for_function_calling():
    for spec in TOOLS.values():
        schema = spec.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == spec.name
        assert schema["function"]["parameters"]["type"] == "object"
