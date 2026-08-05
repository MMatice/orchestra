"""Sortie console robuste.

Les rapports d'Orchestra contiennent des caracteres non-ASCII (fleches,
pictogrammes d'etat). Quand la sortie est redirigee, Python ne detecte plus de
console et retombe sur l'encodage local, qui sous Windows est generalement
cp1252 : l'ecriture leve alors UnicodeEncodeError et la commande echoue alors
que le travail a bien eu lieu. On force donc UTF-8 sur stdout et stderr.
"""

from __future__ import annotations

import sys


def force_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # Flux non reconfigurable (deja remplace, ou ferme) : on n'insiste
            # pas, l'absence de couleur locale ne justifie pas un crash.
            pass
