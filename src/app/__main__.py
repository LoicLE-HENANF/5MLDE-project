"""Lance l'application : `uv run python -m app` (options Streamlit en plus, ex. `--server.port 8501`).

Le script de la page est retrouvé dans le paquet installé : la commande marche aussi dans
l'image, où le paquet est une copie figée sans arbre source. Le service de fichiers statiques
expose static/ (viz.js, chargé par la visionneuse des schémas) sous /app/static/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli


def main(argv: list[str] | None = None) -> int:
    page = Path(__file__).with_name('main.py')
    sys.argv = [
        'streamlit',
        'run',
        str(page),
        '--browser.gatherUsageStats',
        'false',
        '--server.enableStaticServing',
        'true',
        *(argv or sys.argv[1:]),
    ]
    return cli.main()


if __name__ == '__main__':
    raise SystemExit(main())
