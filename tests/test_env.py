from orchestra.env import load_dotenv


def _write(tmp_path, content):
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    return path


def test_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "absent") == {}


def test_real_environment_always_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_HOST", "http://reel:11434")
    applied = load_dotenv(_write(tmp_path, "OLLAMA_HOST=http://dotenv:11434\n"))

    import os

    assert os.environ["OLLAMA_HOST"] == "http://reel:11434"
    assert "OLLAMA_HOST" not in applied


def test_values_are_applied_when_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("ORCHESTRA_PROFILE", raising=False)
    applied = load_dotenv(_write(tmp_path, "ORCHESTRA_PROFILE=lg\n"))
    assert applied["ORCHESTRA_PROFILE"] == "lg"


def test_comments_blanks_quotes_and_export(monkeypatch, tmp_path):
    for key in ("A_KEY", "B_KEY", "C_KEY"):
        monkeypatch.delenv(key, raising=False)

    applied = load_dotenv(
        _write(
            tmp_path,
            "# un commentaire\n"
            "\n"
            'A_KEY="entre guillemets"\n'
            "export B_KEY=avec-export\n"
            "ligne_sans_egal\n"
            "C_KEY=http://hote:11434\n",
        )
    )

    assert applied == {
        "A_KEY": "entre guillemets",
        "B_KEY": "avec-export",
        "C_KEY": "http://hote:11434",
    }


def test_shipped_example_is_parseable(monkeypatch):
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / ".env.example"
    assert example.is_file()
    # Toutes les valeurs actives de l'exemple doivent etre commentees ou valides ;
    # on verifie surtout qu'aucune ligne ne fait planter le parseur.
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    applied = load_dotenv(example)
    assert applied.get("OLLAMA_HOST") == "http://127.0.0.1:11434"
