import io
import subprocess
import sys
from pathlib import Path

from orchestra.console import force_utf8_output

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_non_reconfigurable_streams_are_tolerated(monkeypatch):
    # Un flux remplace par un StringIO n'expose pas reconfigure : ne doit pas lever.
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    force_utf8_output()


def test_cli_survives_a_non_utf8_redirected_stdout():
    """Sortie redirigee + encodage local ASCII : la commande doit aboutir.

    C'est la situation rencontree des qu'un script capture la sortie sous
    Windows, ou l'encodage par defaut n'est pas UTF-8.
    """
    result = subprocess.run(
        [sys.executable, "-m", "orchestra.cli", "backends"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        env={
            "PATH": "",
            "SYSTEMROOT": "C:\\Windows",
            "PYTHONIOENCODING": "ascii",
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert b"litellm" in result.stdout
